# DFlash Custom Mode — Implementation Blueprint

**Date:** 2026-08-08
**Status:** Ready for Implementation
**Based on:** Research Tasks 1-6, all source-verified

---

## Table of Contents

- [Section 1: Architecture Overview](#section-1-architecture-overview)
- [Section 2: Exact Implementation Map](#section-2-exact-implementation-map)
- [Section 3: Tape Storage Strategy](#section-3-tape-storage-strategy)
- [Section 4: Fallback Behavior](#section-4-fallback-behavior)
- [Section 5: Implementation Ordering](#section-5-implementation-ordering)
- [Section 6: Testing Plan](#section-6-testing-plan)
- [Section 7: VRAM Accounting (Corrected)](#section-7-vram-accounting-corrected)

---

## Section 1: Architecture Overview

### 1.1 The Problem

Current upstream DFlash adds **~6.2 GiB VRAM overhead** on RTX 3090 (24 GB) because `need_n_rs_seq()` includes `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`, causing the recurrent memory layer to allocate RS snapshot buffers sized `mem_size * (1 + n_rs_seq)`. For Qwen3.6-27B with `n_rs_seq=8`, that's 5,387 MiB of RS buffer plus ~800 MiB draft model.

### 1.2 The Solution: `--beefix-dflash-custom`

A new opt-in mode that:

1. **Sets `n_rs_seq=0` for DFlash** — eliminates the 5.4 GB RS snapshot buffer.
2. **Allocates backup cells** — `n_parallel × 2` extra rows in R/S tensors for pre-draft state backup (~1,200 MiB for 8 cells).
3. **Captures intermediate tensors during draft** — k, v, g, beta tensors stored in tape buffer for replay.
4. **Replays GDN for accepted tokens after verification** — restores recurrent state without full model re-decode.
5. **Falls back to checkpoint rollback** — when replay cannot execute (allocation failure, graph error, mismatch).

### 1.3 Opt-In Boundary

**Flag:** `--beefix-dflash-custom` (CLI argument, defaults to `false`).

**Struct addition** ([`common/common.h:387-424`](common/common.h:387)):
```cpp
struct common_params_speculative {
    // ... existing fields ...

    // BeeLlama custom DFlash mode: n_rs_seq=0 + backup cells + replay
    bool beefix_dflash_custom = false;
};
```

**Override location** ([`common/common.cpp:1770`](common/common.cpp:1770)):
```cpp
struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();
    // ...
    cparams.n_rs_seq = params.speculative.need_n_rs_seq();

    // BEEFIX: override for custom DFlash mode
    if (params.speculative.beefix_dflash_custom && has_dflash(params.speculative.types)) {
        cparams.n_rs_seq = 0;
    }
    // ...
}
```

This is the single point of control. Setting `n_rs_seq=0` here propagates to all downstream consumers: recurrent memory constructor, hybrid memory, batch splits, and all 12 conditional checks in [`llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp).

### 1.4 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DFlash Custom Mode Cycle                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. PRE-DRAFT PHASE                                                       │
│     ┌──────────────────┐     ┌──────────────────┐                         │
│     │ Backup active     │     │ Backup cells     │                         │
│     │ R/S state to      │────→│ (n_parallel×2    │                         │
│     │ backup cells      │     │  rows in tensor) │                         │
│     └──────────────────┘     └──────────────────┘                         │
│              │                                        │                    │
│              ▼                                        │                    │
│  2. DRAFT PHASE                                       │                    │
│     ┌──────────────────┐                              │                    │
│     │ DFlash draft      │                              │                    │
│     │ forward pass      │                              │                    │
│     │ (capture k,v,g,b  │                              │                    │
│     │  to tape buffer)  │                              │                    │
│     └──────────────────┘                              │                    │
│              │                                        │                    │
│              ▼                                        │                    │
│  3. VERIFY PHASE                                      │                    │
│     ┌──────────────────┐                              │                    │
│     │ common_sampler_   │                              │                    │
│     │ sample_and_accept │                              │                    │
│     │ _n()              │                              │                    │
│     └──────────────────┘                              │                    │
│              │                                        │                    │
│         ┌────┴────┐                                  │                    │
│         │ K accepted                              │                    │
│         │ tokens?                                  │                    │
│         └────┬────┘                                  │                    │
│              │ Yes (partial)                         │                    │
│              ▼                                       │                    │
│  4. REPLAY PHASE                                     │                    │
│     ┌──────────────────┐     ┌──────────────────┐   │                    │
│     │ Restore backup    │     │ GDN replay with   │   │                    │
│     │ state to active   │────→│ captured k,v,g,b  │   │                    │
│     │ R/S cells         │     │ + q=zeros for     │   │                    │
│     └──────────────────┘     │ K accepted tokens  │   │                    │
│                              └──────────────────┘   │                    │
│                                       │              │                    │
│                                       ▼              │                    │
│  5. CLEANUP PHASE                                     │                    │
│     ┌──────────────────┐     ┌──────────────────┐   │                    │
│     │ common_context_   │     │ Write replayed    │   │                    │
│     │ seq_rm() rejected │────→│ state to active   │   │                    │
│     │ KV beyond K       │     │ R/S tensors       │   │                    │
│     └──────────────────┘     └──────────────────┘   │                    │
│                                                                           │
│  Fallback path (any failure → checkpoint rollback):                       │
│     ┌──────────────────────────────────────────────────┐                  │
│     │ ckpt.load_tgt() → seq_rm() → restore sampler    │                  │
│     │ → return early (existing code, no changes)       │                  │
│     └──────────────────────────────────────────────────┘                  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `n_rs_seq` override point | [`common/common.cpp:1770`](common/common.cpp:1770) | Single point of control; all downstream paths handle `n_rs_seq=0` correctly |
| Backup cell storage | Extended tensor rows (Option A) | Single contiguous allocation; no cross-buffer copies |
| `cell_copy()` mechanism | Tensor views + `ggml_backend_tensor_copy()` | Device-native copy, handles same-GPU and cross-GPU |
| Tape capture | Graph-embedded `ggml_cpy` | Batched with compute, no callback overhead |
| Tape storage | **CPU buffer** (recommended) | GPU tape (~6.5 GB) WORSENS VRAM; CPU tape saves 6.5 GB GPU VRAM |
| Replay trigger | After `common_sampler_sample_and_accept_n()`, before `common_context_seq_rm()` | Existing checkpoint path remains as fallback |
| Replay mechanism | `ggml_gated_delta_net()` with q=zeros | State update is q-independent; kernel verified at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) |
| Fallback | Checkpoint rollback (always available) | Checkpoint created before every speculation cycle |
---

## Section 2: Exact Implementation Map

### 2.1 New Files to Create

| File | Purpose | Est. Lines |
|------|---------|------------|
| `common/server-dflash-custom.h` | Custom mode state struct, backup/replay declarations | ~120 |
| `common/server-dflash-custom.cpp` | Backup cell copy, tape capture setup, GDN replay orchestration | ~350 |
| `ggml/src/ggml-cuda/dflash_replay.cu` | CUDA kernel for batched GDN replay across layers | ~200 |

**Total new files: ~670 lines.**

### 2.2 Existing Files to Modify

| File | What Changes | Est. Lines Changed |
|------|-------------|-------------------|
| [`common/common.h`](common/common.h) | Add `beefix_dflash_custom` flag to `common_params_speculative` | +3 |
| [`common/arg.cpp`](common/arg.cpp) | Add `--beefix-dflash-custom` CLI argument | +8 |
| [`common/common.cpp`](common/common.cpp) | Override `n_rs_seq=0` when custom mode active | +10 |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | Add `n_backup_cells` member, `cell_copy()` declaration, `backup_offset()` helper | +15 |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | Extended tensor allocation, `cell_copy()` implementation, `backup_offset()` logic | +60 |
| [`src/llama-model.cpp`](src/llama-model.cpp) | Pass backup cell count to recurrent memory constructor | +5 |
| [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp) | Graph-embedded `ggml_cpy` for tape capture after `cb()` calls | +40 |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | Replay integration in `post_decode()` between lines 4263-4267 | +80 |
| [`tools/server/server-context.h`](tools/server/server-context.h) | Add custom mode state to `server_slot` | +10 |
| [`tools/server/server-task.h`](tools/server/server-task.h) | Add custom mode constants | +5 |

**Total modified lines: ~236 lines across 10 files.**

### 2.3 Data Structures

#### `server_dflash_custom_state` (new, in `server-dflash-custom.h`)

```cpp
struct server_dflash_custom_state {
    // Tape buffers (CPU allocation to save GPU VRAM)
    std::vector<float> tape_k;  // [n_layers][n_draft_max][S_k * H_k]
    std::vector<float> tape_v;  // [n_layers][n_draft_max][S_v * H_v]
    std::vector<float> tape_g;  // [n_layers][n_draft_max][gate_size]
    std::vector<float> tape_b;  // [n_layers][n_draft_max][beta_size]

    // Tape metadata
    uint32_t n_layers;          // number of recurrent layers
    uint32_t n_draft_max;       // max draft tokens
    uint32_t S_k;               // key state dimension (n_embd_r)
    uint32_t S_v;               // value state dimension (n_embd_s)
    uint32_t H_k;               // key head count
    uint32_t H_v;               // value head count
    uint32_t tokens_captured;   // actual tokens captured this cycle

    // Replay state
    bool enabled;               // is custom mode active
    bool replay_failed;         // permanent disable flag
    uint32_t fail_count;        // consecutive failure counter
};
```

**Lifetime:** Created at server init when `--beefix-dflash-custom` is detected. Destroyed at server shutdown. Per-slot instance (one per `server_slot`).

**VRAM impact:** Tape buffers are CPU memory (`std::vector<float>`), so they consume system RAM, not GPU VRAM. For Qwen3.6 with 15 draft tokens: 6,552 MiB in CPU RAM.

#### Extended `llama_memory_recurrent` members (in `llama-memory-recurrent.h`)

```cpp
class llama_memory_recurrent {
    // ... existing members ...

    uint32_t n_backup_cells = 0;       // extra rows for backup cells
    uint32_t backup_offset() const {   // first backup row index
        return mem_size * (1 + n_rs_seq);
    }
    // New method:
    void cell_copy(uint32_t src_row, uint32_t dst_row) const;
};
```

### 2.4 Control Flow: Complete Cycle

#### Phase 1: Pre-Draft Backup (before `common_speculative_draft()`)

**Location:** [`tools/server/server-context.cpp:3264`](tools/server/server-context.cpp:3264), before `common_speculative_draft(spec.get())`.

**Pseudocode:**
```
if custom_mode_enabled:
    for each slot:
        for cell_idx in 0 to n_parallel-1:
            src_row = cell_idx
            dst_row = backup_offset() + cell_idx
            memory.cell_copy(src_row, dst_row)
```

**Cost:** ~2.4 ms (8 cells × 149.9 MiB / 500 GB/s).

#### Phase 2: Draft with Capture (during `common_speculative_draft()`)

**Location:** [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49).

After each `cb()` call that names k, v, g, b tensors, insert `ggml_cpy` to tape buffer:

```cpp
// Existing line 50:
cb(k, "k_in", il);

// NEW: tape capture (conditional on custom mode):
if (tape_capture_enabled) {
    ggml_tensor * k_copy = ggml_cpy(ctx0, k, tape_k_buffer[il]);
    ggml_build_forward_expand(gf, k_copy);
}
```

**Cost:** ~0 ms (overlaps with forward pass compute).

#### Phase 3: Verify (existing, unchanged)

**Location:** [`tools/server/server-context.cpp:4213`](tools/server/server-context.cpp:4213).

No changes. `common_sampler_sample_and_accept_n()` returns `accepted` tokens.

#### Phase 4: Replay (new, replaces RS rollback path)

**Location:** [`tools/server/server-context.cpp:4263-4267`](tools/server/server-context.cpp:4263), between checkpoint path and RS path.

**Pseudocode:**
```
if custom_mode_enabled and n_rollback > 0 and not replay_failed:
    try:
        // Step 1: Restore backup state to active cells
        for cell_idx in 0 to n_parallel-1:
            src_row = backup_offset + cell_idx
            dst_row = cell_idx
            memory.cell_copy(src_row, dst_row)

        // Step 2: Build GDN replay graph
        for layer_il in recurrent_layers:
            q_zeros = ggml_zeros(ctx, [S_k, H_k, K, 1])
            k_tape = ggml_view(tape_k[layer_il], tokens 0..K)
            v_tape = ggml_view(tape_v[layer_il], tokens 0..K)
            g_tape = ggml_view(tape_g[layer_il], tokens 0..K)
            b_tape = ggml_view(tape_b[layer_il], tokens 0..K)
            s_backup = ggml_view(s_l[layer_il], backup_row)

            output = ggml_gated_delta_net(ctx, q_zeros, k_tape, v_tape,
                                           g_tape, b_tape, s_backup, K=1)
            ggml_build_forward_expand(replay_graph, output)

        // Step 3: Execute replay graph
        ggml_backend_sched_graph_compute(replay_sched, replay_graph)

        // Step 5: Write replayed state to active R/S rows
        for layer_il in recurrent_layers:
            copy_gdn_output_to_active(output[il], s_l[il], active_row)

    catch (replay_error):
        if structural_failure:
            replay_failed = true  // permanently disable
        fall_through_to_checkpoint()
```

#### Phase 5: Cleanup (existing, unchanged)

**Location:** [`tools/server/server-context.cpp:4319-4322`](tools/server/server-context.cpp:4319).

`common_context_seq_rm()` removes rejected KV. No changes needed.

### 2.5 New Functions/APIs Summary

| Function | File | Purpose |
|----------|------|---------|
| `cell_copy(src, dst)` | `llama-memory-recurrent.cpp` | Copy R/S data between cells using tensor views |
| `backup_offset()` | `llama-memory-recurrent.h` | Return first backup row index |
| `dflash_custom_backup(slot)` | `server-dflash-custom.cpp` | Pre-draft backup of active state |
| `dflash_custom_capture_init()` | `server-dflash-custom.cpp` | Initialize tape buffers |
| `dflash_custom_replay(slot, n_accepted)` | `server-dflash-custom.cpp` | GDN replay for accepted tokens |
| `dflash_custom_is_enabled(params)` | `server-dflash-custom.h` | Check if custom mode should be active |
| `ggml_cpy_tape_k/v/g/b()` | `delta-net-base.cpp` | Graph-embedded capture operations |

---

## Section 3: Tape Storage Strategy

### 3.1 The VRAM Reality

**Verified from Task 6.4:** The tape buffer is **~6,552 MiB** for 15 draft tokens on Qwen3.6-27B, NOT ~24 MB as Task 5 estimated.

**Root cause:** Task 5 only counted k-state (`n_embd_r = 30,720`). Actual tape captures k, v, g, AND b, where v/g/b each have `S_v = 786,432` dimensions.

**Per-token/layer breakdown:**
| Tensor | Dimensions | Size |
|--------|-----------|------|
| k | S_k × H_k = 128 × 128 × 4B | 64 KB |
| v | S_v × H_v = 786,432 × 4B | 3.0 MB |
| g | S_v × H_v = 786,432 × 4B | 3.0 MB |
| b | S_v × H_v = 786,432 × 4B | 3.0 MB |
| **Total per token/layer** | | **~9.1 MiB** |

**Total tape: 15 tokens × 9.1 MiB × 48 layers = 6,552 MiB.**

### 3.2 Options Evaluated

#### Option A: GPU Tape (Naive F32)

| Metric | Value |
|--------|-------|
| GPU VRAM | +6,552 MiB |
| Total custom mode VRAM | 9,151 MiB (WORSE than current 6,187 MiB) |
| Capture speed | ~0 ms (overlaps with compute) |
| Replay speed | ~0.1 ms (GPU-native GDN) |

**Verdict: REJECTED.** Naive GPU tape makes VRAM worse, not better.

#### Option B: CPU Tape (Recommended)

| Metric | Value |
|--------|-------|
| GPU VRAM | +0 MiB (tape in system RAM) |
| Total custom mode VRAM | 2,599 MiB (598 RS + 1,200 backup + 800 draft) |
| Capture speed | ~13 ms (GPU→CPU transfer: 6.5 GB / 500 GB/s) |
| Replay speed | ~13 ms (CPU→GPU transfer for replay) |

**Verdict: RECOMMENDED.** Achieves the primary goal (VRAM savings) with acceptable performance overhead. The ~26 ms transfer overhead is still 2-4× faster than checkpoint rollback (~50-100 ms).

**Implementation:**
```cpp
// Tape buffers allocated in CPU memory:
std::vector<float> tape_k;  // host memory
std::vector<float> tape_v;
std::vector<float> tape_g;
std::vector<float> tape_b;

// Capture: copy GPU tensor to CPU buffer:
ggml_backend_tensor_get(tape_k.data(), gpu_k_tensor, ...);

// Replay: copy CPU buffer to GPU view, then GDN:
ggml_backend_tensor_set(gpu_k_view, tape_k.data(), ...);
```

#### Option C: F16 Quantized GPU Tape

| Metric | Value |
|--------|-------|
| GPU VRAM | +3,276 MiB (half of F32) |
| Total custom mode VRAM | 5,875 MiB (slightly better than current 6,187 MiB) |
| Capture speed | ~6 ms (half bandwidth, plus conversion) |
| Replay accuracy | Potential numerical drift (F16 state updates) |

**Verdict: CONDITIONAL.** Viable if CPU tape transfer overhead is unacceptable and 5.9 GB total is acceptable. Requires F16→F32 dequantization before GDN replay (kernel expects F32 inputs).

#### Option D: Recompute Intermediates

| Metric | Value |
|--------|-------|
| GPU VRAM | +0 MiB (no tape at all) |
| Total custom mode VRAM | 599 MiB (RS base + backup only) |
| Capture speed | N/A (no capture) |
| Replay speed | Full model re-decode cost (~540 GB weight reads for K=8) |

**Verdict: REJECTED for replay.** This is Task 4's approach — no tape, just re-decode. Saves maximum VRAM but sacrifices the performance benefit of replay.

#### Option E: Hybrid (k on GPU, v/g/b on CPU)

| Metric | Value |
|--------|-------|
| GPU VRAM | +24 MiB (k only, ~1.6 KB/token/layer × 15 × 48) |
| CPU RAM | +6,528 MiB (v/g/b only) |
| Capture speed | ~12 ms (v/g/b to CPU) |
| Replay speed | ~12 ms (v/g/b from CPU) + fast k access |

**Verdict: NOT WORTH IT.** The k tensor is only 24 MiB on GPU — negligible savings over full CPU tape. Adds complexity for no meaningful benefit.

### 3.3 Recommendation: CPU Tape

**Justification:**

1. **Primary goal is VRAM savings.** CPU tape achieves ~5.6 GB GPU VRAM savings (6,187 → 2,599 MiB).
2. **Transfer overhead is acceptable.** ~26 ms total transfer per cycle vs ~50-100 ms checkpoint rollback.
3. **No numerical risk.** F32 tape preserves exact intermediate values.
4. **Simple implementation.** Uses existing `ggml_backend_tensor_get()`/`set()` APIs.
5. **Works with multi-GPU.** CPU tape avoids cross-GPU transfer issues.

**Tradeoff:** Replay is ~10× slower than GPU-native replay but still ~2-4× faster than checkpoint rollback. Benchmark to validate.

### 3.4 Tape Buffer Sizing

For Qwen3.6-27B with `n_draft_max = 15`:

| Buffer | Formula | Size |
|--------|---------|------|
| `tape_k` | 48 × 15 × 30,720 × 4B | 86 MB |
| `tape_v` | 48 × 15 × 786,432 × 4B | 2,208 MB |
| `tape_g` | 48 × 15 × 786,432 × 4B | 2,208 MB |
| `tape_b` | 48 × 15 × 786,432 × 4B | 2,208 MB |
| **Total** | | **6,710 MB** |

Allocated once at server init, reused every cycle. CPU memory only.
---

## Section 4: Fallback Behavior

### 4.1 Fallback Conditions

All failure conditions verified in Task 6.4, with fallback strategy:

| # | Failure Condition | Detection Point | Fallback Action | Scope |
|---|------------------|----------------|-----------------|-------|
| F1 | Tape buffer allocation fails (CPU `std::vector` throws `std::bad_alloc`) | Server init or first speculative cycle | Disable custom mode, use checkpoint rollback | Permanent |
| F2 | Backup cell allocation fails | `llama_memory_recurrent` constructor | Constructor throws; server catches, reduces `n_parallel` or disables replay | Permanent (startup) |
| F3 | Graph construction assertion during replay | Pre-replay graph build | Catch exception, fall back to checkpoint | Transient (context exhaustion) or Permanent (dimension mismatch) |
| F4 | Mismatch between captured tape data and backup state | Pre-replay validation | Fall back to checkpoint, permanently disable replay | Permanent (capture bug) |
| F5 | GDN kernel execution failure | During replay | Catch exception, fall back to checkpoint | Permanent |
| F6 | GPU→CPU transfer failure (tape capture) | During draft forward pass | Fall back to checkpoint for this cycle | Per-cycle |
| F7 | CPU→GPU transfer failure (tape replay) | Pre-replay | Fall back to checkpoint | Permanent |

### 4.2 Fallback Implementation

#### General Pattern

Every custom mode operation is wrapped in a try-catch that falls through to the existing checkpoint rollback path:

```cpp
// In server-context.cpp post_decode(), between lines 4263-4267:
if (custom_mode_enabled && n_rollback > 0 && !slot.dflash_custom.replay_failed) {
    try {
        // Attempt replay
        dflash_custom_replay(slot, accepted.size() - 1);
        // If successful, skip checkpoint path
    } catch (const std::exception & e) {
        SLT_WRN(slot, "dflash custom replay failed: %s — falling back to checkpoint\n", e.what());
        slot.dflash_custom.fail_count++;

        if (slot.dflash_custom.fail_count >= 3) {
            slot.dflash_custom.replay_failed = true;
            SLT_WRN(slot, "dflash custom replay permanently disabled after 3 consecutive failures\n");
        }

        // Fall through to existing checkpoint path at line 4225
        use_ckpt_tgt = true;
    }
}
```

#### F1: Tape Allocation Failure

**Detection:** `std::vector::resize()` throws `std::bad_alloc` during CPU tape allocation.

**Handling:** Catch at server init. Log warning and set `custom_mode_enabled = false`. Server continues with checkpoint rollback.

```cpp
try {
    slot.dflash_custom = server_dflash_custom_state::create(model, n_draft_max);
} catch (const std::bad_alloc & e) {
    LOG_WARN("DFlash custom replay: insufficient CPU memory for tape buffer (%.1f GB needed) — using checkpoint rollback\n",
             tape_size_bytes / 1024.0 / 1024.0 / 1024.0);
    params.speculative.beefix_dflash_custom = false;
}
```

#### F2: Backup Cell Allocation Failure

**Detection:** `llama_memory_recurrent` constructor throws `std::runtime_error` at [`llama-memory-recurrent.cpp:110-113`](src/llama-memory-recurrent.cpp:110).

**Handling:** Server catches at startup. Options:
1. Reduce `n_parallel` and retry.
2. Disable backup cells (fall back to checkpoint).
3. Exit with error.

#### F3: Graph Construction Failure

**Detection:** `ggml_gated_delta_net()` assertions fail:
- Non-contiguous inputs
- Wrong dtype (not F32)
- Dimension mismatch
- K < 1

**Handling:**
- **Context exhaustion:** Retry with larger graph context. If retry fails, permanent disable.
- **Dimension mismatch:** Permanent disable (indicates capture bug or model mismatch).

#### F4: Tape/Backup Mismatch

**Detection:** Before replay, validate:
1. `tape_k.size() >= n_accepted × S_k × H_k × n_layers`
2. Backup cell R/S rows have correct dimensions
3. `tokens_captured >= n_accepted`

**If mismatch:** Permanent disable. This indicates the capture phase failed to record all needed data.

#### F5: GDN Kernel Failure

**Detection:** CUDA error during `ggml_backend_sched_graph_compute()`.

**Handling:** Permanent disable. Fall back to checkpoint.

### 4.3 When Replay is Permanently Disabled

Replay is permanently disabled when:
1. `fail_count >= 3` consecutive failures.
2. Structural failure detected (dimension mismatch, dtype error).
3. Tape allocation failed at init.

Once disabled, the slot uses checkpoint rollback for all subsequent speculative cycles. The `--beefix-dflash-custom` flag remains set, but the replay subsystem is inactive.

### 4.4 When Replay Is Temporarily Skipped

Replay is temporarily skipped when:
1. Full acceptance (`n_rollback == 0`) — no replay needed.
2. Context exhaustion during graph build — retry with larger context next cycle.
3. Single transient GPU transfer error — retry next cycle.

---

## Section 5: Implementation Ordering

### 5.1 Subtask Dependencies

```
Subtask 1: Opt-in flag + n_rs_seq override
    ↓ (provides n_rs_seq=0 for DFlash)
Subtask 2: Backup cell infrastructure
    ↓ (provides cell_copy() and extended tensors)
Subtask 3: Tape capture mechanism
    ↓ (provides k,v,g,b data during draft)
Subtask 4: Replay orchestration
    ↓ (uses backup + tape to replay GDN)
Subtask 5: Server integration + fallback
    ↓ (wires everything into post_decode())
Subtask 6: Testing + benchmarking
```

### 5.2 Subtask 1: Opt-in Flag and n_rs_seq Override

**Prerequisites:** None
**Files:** [`common/common.h`](common/common.h), [`common/arg.cpp`](common/arg.cpp), [`common/common.cpp`](common/common.cpp)
**Est. lines:** ~20

**Changes:**
1. Add `bool beefix_dflash_custom = false` to [`common_params_speculative`](common/common.h:387).
2. Add `--beefix-dflash-custom` CLI argument in [`arg.cpp`](common/arg.cpp).
3. In [`common_context_params_to_llama()`](common/common.cpp:1770): if flag is set and DFlash detected, override `cparams.n_rs_seq = 0`.

**Expected result:** DFlash runs with `n_rs_seq=0`, eliminating 5.4 GB RS buffer. Server starts and runs with checkpoint-only rollback.

**Test strategy:**
- Start server with `--spec-type draft-dflash --beefix-dflash-custom`
- Verify log shows `n_rs_seq = 0`
- Verify VRAM usage drops from ~6.2 GB to ~0.6 GB RS overhead
- Verify DFlash still functions (checkpoint rollback works)

### 5.3 Subtask 2: Backup Cell Infrastructure

**Prerequisites:** Subtask 1 complete
**Files:** [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h), [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp), [`src/llama-model.cpp`](src/llama-model.cpp)
**Est. lines:** ~80

**Changes:**
1. Add `n_backup_cells` member to `llama_memory_recurrent` class.
2. Modify tensor allocation formula: `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells`.
3. Implement `cell_copy(src_row, dst_row)` using tensor views + `ggml_backend_tensor_copy()`.
4. Implement `backup_offset()` helper.
5. In [`llama-model.cpp`](src/llama-model.cpp:2106): pass backup cell count to constructor when custom mode active.

**Expected result:** Backup cells exist as extra rows in R/S tensors. `cell_copy()` can copy R/S data between any two rows.

**Test strategy:**
- Unit test: `cell_copy(0, backup_offset())` followed by memcmp of source and destination rows.
- Verify backup rows are not used by normal allocation (`find_slot()` skips backup range).
- Verify VRAM: RS buffer grows by `n_backup_cells × 149.9 MiB`.

### 5.4 Subtask 3: Tape Capture Mechanism

**Prerequisites:** Subtask 2 complete
**Files:** [`common/server-dflash-custom.h`](common/server-dflash-custom.h), [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp), [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp)
**Est. lines:** ~250

**Changes:**
1. Create `server_dflash_custom_state` struct with CPU tape buffers.
2. In [`delta-net-base.cpp`](src/models/delta-net-base.cpp:49): after each `cb(k, "k_in", il)` etc., insert graph-embedded `ggml_cpy` to tape buffer.
3. Implement `dflash_custom_capture_init()` to allocate CPU tape buffers.
4. Implement `dflash_custom_capture()` to trigger GPU→CPU transfer after graph execution.

**Expected result:** After draft forward pass, tape buffers contain k, v, g, b data for all draft tokens and recurrent layers.

**Test strategy:**
- Verify tape buffers are non-zero after draft pass.
- Verify tape data matches expected tensor dimensions.
- Measure capture overhead (should be ~13 ms for GPU→CPU transfer).

### 5.5 Subtask 4: Replay Orchestration

**Prerequisites:** Subtasks 2-3 complete
**Files:** [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp), [`ggml/src/ggml-cuda/dflash_replay.cu`](ggml/src/ggml-cuda/dflash_replay.cu)
**Est. lines:** ~300

**Changes:**
1. Implement `dflash_custom_replay(slot, n_accepted)`:
   a. Restore backup state to active cells.
   b. Build GDN replay graph with captured tape data and q=zeros.
   c. Execute replay graph.
   d. Write replayed state to active R/S tensors.
2. Create CUDA kernel for batched GDN replay (optional optimization).

**Expected result:** After replay, active R/S state matches the state that would have been produced by the original forward pass for accepted tokens.

**Test strategy:**
- Compare replayed state vs. original forward pass state (element-wise comparison).
- Verify numerical accuracy (should match within F32 precision).
- Measure replay overhead (~1 ms compute + ~13 ms transfer).

### 5.5 Subtask 5: Server Integration and Fallback

**Prerequisites:** Subtasks 2-4 complete
**Files:** [`tools/server/server-context.cpp`](tools/server/server-context.cpp), [`tools/server/server-context.h`](tools/server/server-context.h), [`tools/server/server-task.h`](tools/server/server-task.h)
**Est. lines:** ~100

**Changes:**
1. Add `server_dflash_custom_state` to `server_slot` struct.
2. In [`post_decode()`](tools/server/server-context.cpp:4263): insert replay attempt between checkpoint path and RS path.
3. Implement fallback logic (try-catch, fail count, permanent disable).
4. Add pre-draft backup call before [`common_speculative_draft()`](tools/server/server-context.cpp:3264).

**Expected result:** Complete custom mode cycle: backup → draft with capture → verify → replay → cleanup. Fallback to checkpoint on any failure.

**Test strategy:**
- End-to-end test: generate text with custom mode enabled.
- Verify output matches stock DFlash (same tokens generated).
- Force fallback by inducing errors.
- Measure throughput vs. stock DFlash and checkpoint-only DFlash.

### 5.7 Subtask 6: Testing and Benchmarking

**Prerequisites:** All subtasks complete
**Files:** Test scripts, benchmark commands
**Est. lines:** ~50 (test scripts)

**Changes:**
1. Create test scripts for each test scenario (see Section 6).
2. Run benchmark suite.
3. Document results.
---

## Section 6: Testing Plan

### 6.1 Baseline Test — Stock DFlash Unchanged

**Objective:** Verify that stock DFlash (without `--beefix-dflash-custom`) behaves identically before and after the implementation.

**Command:**
```bash
build/bin/llama-server -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 \
  --n_ctx 8192 --n_parallel 4 \
  --port 8080
```

**Expected:**
- Server starts successfully.
- Log shows `n_rs_seq = 8`.
- VRAM usage: ~24 GB (model 18.6 GB + RS 5.4 GB).
- Generation produces correct output.

**Pass criteria:** Identical behavior to pre-implementation baseline.

### 6.2 Custom Mode — Basic Functionality

**Objective:** Verify custom mode starts and generates correct output.

**Command:**
```bash
build/bin/llama-server -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 \
  --beefix-dflash-custom \
  --n_ctx 8192 --n_parallel 4 \
  --port 8080
```

**Expected:**
- Server starts successfully.
- Log shows `n_rs_seq = 0`.
- Log shows backup cells allocated: `n_backup_cells = 8`.
- VRAM usage: ~20 GB (model 18.6 GB + RS 0.6 GB + backup 1.2 GB).
- Generation produces **identical output** to baseline test (same prompt, same sampling).

**Pass criteria:**
1. Server starts without errors.
2. VRAM reduced by ~4 GB compared to baseline.
3. Generated tokens match baseline exactly (same prompt, seed, sampling params).

### 6.3 Custom Mode — Different Acceptance Lengths

**Objective:** Verify replay works correctly for different numbers of accepted tokens (K=0 through K=n_draft_max).

**Method:** Use controlled test prompts that produce known acceptance patterns.

| Test | Expected K | Verification |
|------|-----------|--------------|
| Full acceptance | K = n_draft (all tokens match) | No replay needed; verify path skipped |
| Partial acceptance (early mismatch) | K = 2-3 | Replay 2-3 tokens; output matches baseline |
| Partial acceptance (late mismatch) | K = n_draft - 1 | Replay n_draft-1 tokens; output matches baseline |
| Zero acceptance | K = 0 | No replay; checkpoint fallback |

**Pass criteria:** For each test, generated output matches baseline.

### 6.4 Custom Mode — Different Speculative Depths

**Objective:** Verify custom mode works with different `--spec-draft-n-max` values.

| n_max | Tape Size (CPU) | Expected Behavior |
|-------|-----------------|-------------------|
| 4 | ~1.7 GB | Smaller tape, faster transfer |
| 8 | ~3.5 GB | Medium tape |
| 15 | ~6.5 GB | Full tape, slower transfer |
| 30 | ~13 GB | Large tape (test CPU memory limits) |

**Pass criteria:** All depths produce correct output. No crashes or memory errors.

### 6.5 Custom Mode — Different Context Sizes

**Objective:** Verify custom mode works with different `--n_ctx` values.

| n_ctx | Expected Behavior |
|-------|-------------------|
| 2048 | Small context, fast |
| 8192 | Default context |
| 32768 | Large context (test KV cache pressure) |
| 131072 | Maximum context (test memory limits) |

**Pass criteria:** All context sizes produce correct output. VRAM scales as expected.

### 6.6 Fallback Tests

**Objective:** Verify fallback behavior when replay fails.

| Test | How to Trigger | Expected Fallback |
|------|---------------|-------------------|
| Tape allocation failure | Set `n_draft_max` extremely high (e.g., 1000) | Server logs warning, disables replay, uses checkpoint |
| Graph construction failure | Corrupt tape data manually | Catch exception, fall back to checkpoint |
| Dimension mismatch | Use wrong model (non-Qwen3.6) | Detect mismatch, permanently disable replay |
| Consecutive failures | Force 3 transient failures | After 3rd failure, permanently disable |

**Pass criteria:** Server never crashes. Falls back to checkpoint rollback gracefully. Output remains correct.

### 6.7 VRAM Measurement Tests

**Objective:** Measure actual VRAM usage for each configuration.

**Method:** Use `nvidia-smi` to record VRAM before and after server starts.

| Configuration | Expected VRAM | Measurement |
|--------------|---------------|-------------|
| Stock DFlash (n_rs_seq=8) | ~24 GB | Record actual |
| Custom mode (n_rs_seq=0 + backup + CPU tape) | ~20 GB | Record actual |
| Custom mode disabled fallback | ~24 GB | After permanent disable |

**Pass criteria:** Custom mode VRAM is at least 3 GB lower than baseline.

### 6.8 Performance Benchmark Tests

**Objective:** Measure throughput for each configuration.

**Method:** Use `llama-bench` or custom HTTP benchmark with fixed prompt.

| Metric | Baseline DFlash | Custom Mode | Improvement |
|--------|----------------|-------------|-------------|
| Tokens/sec | Record | Record | Calculate |
| Time to first token | Record | Record | Calculate |
| Replay overhead (ms/cycle) | N/A | Record | Compare to checkpoint |
| Checkpoint rollback frequency | Record | Record | Should be lower |

**Pass criteria:** Custom mode throughput is within 80% of baseline. If checkpoint rollback is too frequent, custom mode may be slower — document the tradeoff.

---

## Section 7: VRAM Accounting (Corrected)

### 7.1 Verified Numbers (Qwen3.6-27B)

All numbers verified against source code in Tasks 6.1-6.4.

| Component | Formula | Current (n_rs_seq=8) | Custom (n_rs_seq=0) |
|-----------|---------|---------------------|---------------------|
| RS buffer R | 30720 × n_rows × 48 × 4B | 202.5 MiB | 22.5 MiB |
| RS buffer S | 786432 × n_rows × 48 × 4B | 5,184 MiB | 576 MiB |
| **RS total** | | **5,386.5 MiB** | **598.5 MiB** |
| Backup cells | 8 × 149.9 MiB | 0 | 1,199 MiB |
| Tape buffer (CPU) | 15 × 9.1 × 48 | 0 | 0 GPU VRAM |
| Draft model | Qwen3.6-27B-DFlash | ~800 MiB | ~800 MiB |
| **DFlash overhead** | | **~6,187 MiB** | **~2,598 MiB** |
| **Total (with model)** | | **~24.8 GB** | **~22.0 GB** |

### 7.2 VRAM Savings Summary

| Metric | Value |
|--------|-------|
| Current DFlash overhead | 6,187 MiB |
| Custom mode GPU overhead | 2,598 MiB |
| **GPU VRAM saved** | **3,589 MiB (~3.5 GB)** |
| Savings percentage | 58% |
| CPU RAM used (tape) | 6,552 MiB |

### 7.3 Scaling Analysis

#### vs. Speculative Depth (n_draft_max)

| n_draft_max | Tape (CPU) | GPU VRAM | Total Overhead |
|-------------|-----------|----------|----------------|
| 4 | 1,747 MiB | 2,598 MiB | 4,345 MiB |
| 8 | 3,495 MiB | 2,598 MiB | 6,093 MiB |
| 15 | 6,552 MiB | 2,598 MiB | 9,150 MiB |

GPU VRAM is independent of `n_draft_max` (tape is on CPU). CPU RAM scales linearly.

#### vs. Context Length

| n_ctx | KV Cache | RS Base | Backup | GPU VRAM |
|-------|----------|---------|--------|----------|
| 8K | ~3.2 GB | 599 MiB | 1,200 MiB | ~5.0 GB |
| 32K | ~12.8 GB | 599 MiB | 1,200 MiB | ~14.6 GB |

RS base and backup are independent of `n_ctx`. Only KV cache scales.

#### vs. n_parallel

| n_parallel | Backup Cells | Backup VRAM | Tape (×n_parallel) | GPU VRAM |
|-----------|-------------|-------------|--------------------|----------|
| 1 | 2 | 300 MiB | 6,552 MiB | 1,700 MiB |
| 4 | 8 | 1,200 MiB | 26,208 MiB | 2,600 MiB |
| 8 | 16 | 2,400 MiB | 52,416 MiB | 3,800 MiB |

Backup VRAM scales linearly with `n_parallel`. Tape scales with `n_parallel` (one tape per slot).

### 7.4 Comparison with Alternatives

| Approach | GPU VRAM | CPU RAM | Rollback Speed | Complexity |
|----------|----------|---------|----------------|------------|
| Current upstream (n_rs_seq=8) | 6,187 MiB | 0 | Fast (RS pointer swap) | 0 (baseline) |
| Custom mode (this plan) | 2,598 MiB | 6,552 MiB | Medium (replay + transfer) | ~900 lines |
| Task 4 (backup + re-decode) | 1,800 MiB | 0 | Slow (full model re-decode) | ~120 lines |
| Hybrid (n_rs_seq=2) | 2,200 MiB | 0 | Fast for ≤2 rollback | ~10 lines |
| Old DFlash (0.3.2) | ~1,000 MiB | 0 | Fast (tape replay) | ~3,376 lines |

### 7.5 Bottom Line

Custom mode saves **~3.5 GB GPU VRAM** at the cost of **~6.5 GB CPU RAM** and **~26 ms transfer overhead per cycle**. This makes DFlash practical on 24 GB GPUs where it previously exceeded available memory. The tradeoff is: VRAM savings for transfer time. Benchmark to validate that throughput remains acceptable.

---

*End of Implementation Blueprint*
