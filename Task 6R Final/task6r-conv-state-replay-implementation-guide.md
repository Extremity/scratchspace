# Task 6R Convolution State Replay — Implementation Guidance

**Date:** 2026-08-10
**Author:** Roo (Architect Mode)
**Status:** Implementation guidance for P0 fix
**Related:** [`task6r-architectural-recommendation.md`](task6r-architectural-recommendation.md), [`task6r-audit-findings.md`](task6r-audit-findings.md)

---

## Executive Summary

**Problem:** The `dflash_custom_replay()` function in [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp:307) restores backup R/S state and replays the GDN S-state update, but **does not rebuild the convolution state (R tensor)**. Without this fix, replay produces incorrect conv state even after the `n_backup_cells` fix is applied.

**Impact:** After replay, the R tensor contains the restored pre-draft conv state instead of the conv state advanced by `n_accepted` tokens. This means the next forward pass will use stale conv state, producing incorrect outputs. The error may be subtle (the model still produces plausible output) but the results will diverge from the correct forward pass.

**Root Cause:** The current replay path handles S-state correctly via GDN replay but has no code path to advance the R (convolution) state. The tape captures `qkv_mixed` data specifically for this purpose, but no replay function consumes it.

**Solution:** Add a conv state rebuild step to `dflash_custom_replay()` that uses the captured `qkv` tape data to shift the conv window forward by `n_accepted` tokens. The algorithm is a sliding window shift, identical to the proven v0.3.2 implementation.

---

## WARNING: Architectural Debt Assessment

**Before implementation, be aware of the following:**

1. **No technical debt introduced.** This fix completes an already-designed capability. The tape already captures `qkv` data (`server_dflash_tape_gpu_layer::qkv`), and the state structure already tracks `conv_channels` and `max_tokens`. The fix uses existing infrastructure.

2. **No conflict with existing codebase.** The conv rebuild logic is self-contained within `dflash_custom_replay()`. It does not modify upstream files, does not change the graph builder, and does not alter the backup/restore path.

3. **CPU-only implementation acceptable for now.** The old v0.3.2 had both CPU and CUDA kernel implementations. The initial fix should use a CPU-based approach (read tape data, compute on CPU, write back) for simplicity and correctness. A CUDA kernel can be added later as a P3 optimization. The CPU path is ~50-80 lines and will be correct.

4. **The `qkv` tape tensor exists but may not be populated in all code paths.** Verify that the capture in [`qwen35.cpp:493-527`](src/models/qwen35.cpp:493) actually copies `qkv_mixed` to `tl.qkv`. The current code does this, so no change is needed to the capture path.

---

## Root Cause Analysis

### How Conv State Works in GDN

The GDN (Gated Delta Net) pipeline has two types of state:

| State Type | Tensor | Purpose | Update Mechanism |
|------------|--------|---------|-----------------|
| S (SSM) state | `s_l[il]` | Recurrent attention state | GDN kernel computes `s_new = g * s + k * delta` |
| R (conv) state | `r_l[il]` | Sliding conv window of qkv_mixed | Shift window forward by `n_accepted`, fill from tape |

**Conv state layout:**
- R tensor shape: `[n_embd_r, n_rows]` where `n_embd_r = (d_conv - 1) * conv_channels`
- `conv_channels = d_inner + 2 * n_group * d_state` (e.g., 10240 for Qwen3.6)
- `conv_window = d_conv - 1` (e.g., 3 for typical GDN models)
- `n_embd_r = conv_window * conv_channels`
- The conv state is a sliding window: for each channel, it stores the last `conv_window` values of the qkv input.

**During normal forward pass** ([`delta-net-base.cpp:449-524`](src/models/delta-net-base.cpp:449)):
1. Read current conv state from R tensor.
2. Reshape to `[conv_window, conv_channels, n_seqs]`.
3. Concatenate with new `qkv_mixed` along dimension 0: `[conv_window + n_seq_tokens, conv_channels, n_seqs]`.
4. Apply conv operation (produces `conv_qkv_mix`).
5. Write back the last `conv_window` columns as the new conv state.

**During replay:** The S state is updated via GDN replay. The R state must be updated by shifting the conv window forward by `n_accepted` tokens, using the captured `qkv` tape data to fill new positions.

### 1.2 What the Current Code Does

In [`dflash_custom_replay()`](common/server-dflash-custom.cpp:307):

```cpp
// Line 348: Restore backup state (R and S) to active rows
dflash_custom_restore(mem, n_cells);  // Copies R and S from backup to active

// Lines 403-490: Build GDN replay graph for S state only
for (size_t ti = 0; ti < tape_layers.size(); ++ti) {
    // ... creates q_zeros, k_view, v_view, g_view, b_view, s_backup
    // ... calls ggml_gated_delta_net() - only updates S state
    ggml_tensor * gdn_out = ggml_gated_delta_net(replay_ctx,
        q_zeros, k_view, v_view, g_view, b_view, s_backup, /* K= */ 1);
}

// Lines 500-534: Write updated S state back to active rows
for (size_t i = 0; i < gdn_outputs.size(); ++i) {
    // Copy S state from GDN output to active S row
    ggml_backend_tensor_copy(state_view, s_dst);  // Only S, no R
}

// MISSING: No conv state (R tensor) rebuild here.
```

The tape captures `qkv` data ([`qwen35.cpp:493-527`](src/models/qwen35.cpp:493)), but `dflash_custom_replay()` never uses it.

### 1.3 What the Old v0.3.2 Code Did

The old implementation had [`tape_replay_conv()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3877) called after the GDN replay. The core algorithm ([`old-versions/.../llama-context.cpp:3999-4011`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3999)):

```cpp
for (auto & d : layers) {
    for (int64_t w = 0; w < d.conv_window; ++w) {
        int src_pos = n_accepted + (int)w;
        for (int64_t ch = 0; ch < d.conv_ch; ++ch) {
            float val;
            if (src_pos < (int)d.conv_window) {
                // Position still within old window
                val = d.old_window[ch * d.conv_window + src_pos];
            } else {
                // Position comes from new qkv data
                val = d.qkv_mixed[(src_pos - d.conv_window) * d.conv_ch + ch];
            }
            d.new_conv[ch * d.conv_window + w] = val;
        }
    }
}
```

This is a **sliding window shift**: after `n_accepted` tokens, the conv window advances by `n_accepted`. Positions that still fall within the old window are kept; positions beyond the old window are filled from the tape's `qkv` data.

---

## Implementation Plan

### 2.1 Overview

Add a conv state rebuild step to `dflash_custom_replay()` after the GDN replay and S-state write-back. The step:

1. For each recurrent layer, read the current conv state from the active R row (which was restored from backup by `dflash_custom_restore`).
2. Read the `qkv` tape data for the first `n_accepted` tokens.
3. Compute the new conv state by shifting the window forward.
4. Write the new conv state back to the active R row.

### 2.2 File: `common/server-dflash-custom.cpp`

**Location:** Add conv rebuild after line 534 (after the S-state write-back loop, before `ggml_free(state->replay_ctx)`).

**New code block (after line 534, before line 536):**

```cpp
    // --- 7. Rebuild convolution state (R tensor) ---
    // After GDN replay updates S state, the conv state must also advance by n_accepted tokens.
    // The conv state is a sliding window of qkv_mixed. We shift the window forward,
    // keeping positions within the old window and filling new positions from tape qkv data.
    //
    // Algorithm (per layer):
    //   old_conv = R[backup_row][:n_embd_r]  (restored by dflash_custom_restore)
    //   qkv_tape = tape[ti].qkv[:n_accepted]  (captured during draft)
    //   for each channel ch in [0, conv_channels):
    //     for each window pos w in [0, conv_window):
    //       src_pos = n_accepted + w
    //       if src_pos < conv_window:
    //         new_conv[ch, w] = old_conv[ch, src_pos]
    //       else:
    //         new_conv[ch, w] = qkv_tape[src_pos - conv_window, ch]
    //   R[active_row][:n_embd_r] = new_conv

    uint32_t conv_channels = state->conv_channels;
    if (conv_channels > 0) {
        const auto & hp = model_hparams(model);
        uint32_t n_embd_r = hp.n_embd_r();
        uint32_t conv_window = n_embd_r / conv_channels;

        // Temporary buffers for conv rebuild
        std::vector<float> old_conv(n_embd_r);
        std::vector<float> new_conv(n_embd_r);
        std::vector<float> qkv_tape((size_t)n_accepted * (size_t)conv_channels);

        for (size_t ti = 0; ti < tape_layers.size(); ++ti) {
            const auto & tl = tape_layers[ti];
            int il = (int)layer_ids[ti];

            if (!mem->r_l[il]) {
                continue;  // Skip non-recurrent or filtered layers
            }

            ggml_tensor * r_tensor = mem->r_l[il];

            // Read current conv state from active R row (restored from backup).
            ggml_backend_tensor_get(r_tensor, old_conv.data(),
                0, n_embd_r * sizeof(float));

            // Read qkv tape data for accepted tokens.
            // Tape qkv shape: [conv_channels, max_tokens]
            // We need first n_accepted token columns.
            if (tl.qkv) {
                ggml_backend_tensor_get(tl.qkv, qkv_tape.data(),
                    0, qkv_tape.size() * sizeof(float));
            }

            // Compute new conv state: sliding window shift by n_accepted.
            for (uint32_t w = 0; w < conv_window; ++w) {
                int src_pos = n_accepted + (int)w;
                for (uint32_t ch = 0; ch < conv_channels; ++ch) {
                    float val;
                    if (src_pos < (int)conv_window) {
                        // Position still within old conv window
                        val = old_conv[ch * conv_window + (uint32_t)src_pos];
                    } else {
                        // Position comes from new qkv data
                        val = qkv_tape[(size_t)(src_pos - (int)conv_window) * (size_t)conv_channels + ch];
                    }
                    new_conv[ch * conv_window + w] = val;
                }
            }

            // Write new conv state back to active R row (row 0).
            ggml_backend_tensor_set(r_tensor, new_conv.data(),
                0, n_embd_r * sizeof(float));
        }
    }
```

**Note on placement:** This code goes between the S-state write-back loop (ending at line 534) and the `ggml_free(state->replay_ctx)` call (line 536). The comment marker is "step 7" to follow the existing "step 6" numbering.

### 2.3 File: `common/server-dflash-custom.h`

**No changes needed.** The header already defines:
- `server_dflash_custom_state::conv_channels` (line 85) — stores `conv_channels` from init.
- `server_dflash_tape_gpu_layer::qkv` (line 49) — stores captured qkv tape data.

### 2.4 File: `common/server-dflash-custom.cpp` — `dflash_custom_init()`

**Verify** that `conv_channels` is set correctly during initialization. Check [`server-dflash-custom.cpp:188-215`](common/server-dflash-custom.cpp:188):

```cpp
server_dflash_custom_state * dflash_custom_init(const llama_model * model, int n_draft_max) {
    // ... existing code ...
    state->conv_channels = d_inner + 2 * H_k * S_k;  // Should already be set
    // ...
}
```

If `conv_channels` is not already set in `dflash_custom_init()`, add it. The formula matches the tape allocation at [`server-dflash-custom.cpp:65`](common/server-dflash-custom.cpp:65).

### 2.5 File: `src/llama-hparams.cpp`

**No changes needed.** The `n_embd_r()` method already exists at [`llama-hparams.cpp:191-204`](src/llama-hparams.cpp:191) and returns `(ssm_d_conv - 1) * conv_channels`.

### 2.6 File: `src/llama-memory-recurrent.h`

**No changes needed.** The `r_l` array already exists and stores the R tensor per layer. The backup/restore path already handles R via `cell_copy()`.

---

## 3. Data Flow Verification

### 3.1 Tape Capture (No Change Needed)

The capture in [`qwen35.cpp:493-527`](src/models/qwen35.cpp:493) already copies `qkv_mixed` to `tl.qkv`:

```cpp
// qkv_mixed: [conv_channels, n_seq_tokens, n_seqs]
ggml_tensor * qkv_src = ggml_view_2d(ctx0, qkv_mixed,
    qkv_mixed->ne[0], n_seq_tokens,
    qkv_mixed->nb[1], 0);
// ...
ggml_tensor * qkv_cont = ggml_cont(ctx0, qkv_src);
// ...
ggml_tensor * q_dst = ggml_view_2d(ctx0, tl.qkv,
    tl.qkv->ne[0], (int64_t)n_seq_tokens,
    tl.qkv->nb[1], 0);
// ...
ggml_build_forward_expand(gf, ggml_cpy(ctx0, qkv_cont, q_dst));
```

This produces `tl.qkv` with shape `[conv_channels, max_tokens]` where the first `n_seq_tokens` columns contain the captured data.

### 3.2 Tape Allocation (No Change Needed)

The tape allocation at [`server-dflash-custom.cpp:106-159`](common/server-dflash-custom.cpp:106) already allocates `tl.qkv` with shape `[conv_channels, max_tokens]` in F32.

### 3.3 Conv Rebuild Data Flow

```
Draft forward pass:
    qkv_mixed[conv_channels, n_seq_tokens] --captured--> tl.qkv[conv_channels, max_tokens]

Pre-draft backup:
    R[active_row] --cell_copy--> R[backup_row]  (includes conv state)

Replay (after n_backup_cells fix):
    R[backup_row] --cell_copy--> R[active_row]  (restore pre-draft conv state)
    S[backup_row] --cell_copy--> S[active_row]  (restore pre-draft S state)
    
    GDN replay: S[active_row] updated via GDN kernel
    
    Conv rebuild (NEW):
        old_conv = R[active_row][:n_embd_r]  (pre-draft conv state)
        qkv_data = tl.qkv[:n_accepted]        (captured qkv for accepted tokens)
        new_conv = sliding_window_shift(old_conv, qkv_data, n_accepted)
        R[active_row][:n_embd_r] = new_conv   (write back)
```

### 3.4 Memory Layout

| Tensor | Shape | Layout | Device |
|--------|-------|--------|--------|
| `r_l[il]` | `[n_embd_r, n_rows]` | Column-major (ggml default) | Same GPU as model layer |
| `tl.qkv` | `[conv_channels, max_tokens]` | Column-major (ggml default) | Same GPU as model layer |
| `old_conv` | `[n_embd_r]` | Flat F32 array | CPU (temporary) |
| `new_conv` | `[n_embd_r]` | Flat F32 array | CPU (temporary) |
| `qkv_tape` | `[n_accepted * conv_channels]` | Flat F32 array | CPU (temporary) |

**Note:** The initial implementation uses CPU buffers for the conv rebuild computation. This involves two PCIe transfers per layer (read R, write R) plus one tape read. For Qwen3.6 with 48 recurrent layers and `conv_channels=10240`, `conv_window=3`:
- `n_embd_r = 30720` per layer
- `old_conv` = 122.9 KB per layer
- `new_conv` = 122.9 KB per layer
- `qkv_tape` = ~400 KB for n_accepted=10
- Total PCIe per layer: ~246 KB read + ~123 KB write = ~369 KB
- Total for 48 layers: ~17.7 MB read + ~5.9 MB write = ~23.6 MB

This is acceptable for the initial implementation. A CUDA kernel (P3 optimization) would eliminate these transfers.

---

## 4. Testing Strategy

### 4.1 Unit Test

After implementation, verify conv rebuild correctness:

1. Load a DFlash model with `--beefix-dflash-custom`.
2. Run a draft with known `n_accepted` tokens.
3. After replay, compare R tensor state with a reference forward pass that processes the same `n_accepted` tokens.
4. The R tensor should match within floating-point precision.

### 4.2 Integration Test

1. Run the existing test script ([`tests/dflash-custom-test.py`](tests/dflash-custom-test.py)).
2. Verify that output matches the baseline (stock DFlash output).
3. If conv state is correct, output should be identical. If conv state is wrong, output will diverge.

### 4.3 Edge Cases

| Case | Expected Behavior |
|------|------------------|
| `n_accepted = 0` | No conv rebuild needed (replay returns `false` at line 309) |
| `n_accepted = 1` | Conv window shifts by 1, one new position from tape |
| `n_accepted >= conv_window` | Entire conv window comes from tape data |
| `conv_window = 0` | Skip conv rebuild (no conv state to maintain) |
| Layer has no `r_l` entry | Skip (continue at the layer loop) |
| Layer has no `tl.qkv` | Skip (should not happen if capture worked) |

---

## 5. Performance Considerations

### 5.1 Initial Implementation (CPU-based)

- **PCIe transfers:** ~24 MB per replay cycle for Qwen3.6 (48 layers)
- **CPU computation:** O(conv_channels * conv_window) per layer, negligible
- **Total latency:** ~5-10 ms for PCIe transfers (dominated by read latency, not bandwidth)

### 5.2 Future Optimization (CUDA kernel, P3)

The old v0.3.2 had a CUDA kernel at [`cross-ring-interleave.cu:360-413`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml-cuda/cross-ring-interleave.cu:360) that performs the sliding window shift on GPU. This eliminates PCIe transfers entirely.

The kernel is registered via `ggml_backend_reg_get_proc_address()` as `dflash_rebuild_conv_state`. The P3 optimization would:
1. Add the kernel to `ggml/src/ggml-cuda/` (reusing the old kernel code).
2. Register it in the CUDA backend's proc address table.
3. Call it from `dflash_custom_replay()` before falling back to CPU.

**Estimated savings:** ~5-10 ms per replay cycle (the PCIe transfer time).

---

## 6. Code Change Summary

| File | Change | Lines |
|------|--------|-------|
| `common/server-dflash-custom.cpp` | Add conv rebuild step in `dflash_custom_replay()` | ~50-80 lines |
| `common/server-dflash-custom.cpp` | Verify `conv_channels` set in `dflash_custom_init()` | 0-1 line |
| `common/server-dflash-custom.h` | No changes needed | 0 |
| `src/llama-hparams.cpp` | No changes needed | 0 |
| `src/llama-memory-recurrent.h` | No changes needed | 0 |
| `src/models/qwen35.cpp` | No changes needed (capture already works) | 0 |

**Total: ~50-80 lines of new code in a single location.**

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Conv rebuild produces wrong values | Low | High | Compare R tensor against reference forward pass. The algorithm is proven in v0.3.2. |
| PCIe transfers too slow | Low | Medium | ~23 MB total is acceptable. CUDA kernel available as P3 optimization. |
| `conv_channels` formula wrong for non-Qwen models | Low | Medium | Document the formula assumption. The value comes from model hparams, so it adapts per model. |
| Tape qkv not populated | Low | High | Add assertion: if `tl.qkv == nullptr`, log error and return `false` to trigger checkpoint fallback. |
| Multi-GPU: tape and R on different devices | Low | Medium | The current tape allocation places `tl.qkv` on the same device as `model.dev_layer(il)`. The R tensor is also on that device. Same-device guarantee is already in place. |

---

## 8. Implementation Order

1. **Verify** `conv_channels` is set in `dflash_custom_init()`.
2. **Add** conv rebuild code block to `dflash_custom_replay()` after S-state write-back.
3. **Add** assertion: if `tl.qkv == nullptr`, log error and return `false`.
4. **Build** and verify compilation.
5. **Run** existing test script to verify output matches baseline.
6. **Verify** R tensor state matches reference forward pass (if test infrastructure allows).

---

*End of implementation guidance.*
