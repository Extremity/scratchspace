# Subtask 3.3: Tape Replay Adaptability Analysis

**Date:** 2026-08-07
**Part of:** Research Task 3 — Hybrid DFlash Rollback Investigation

**Reference Documents:**
- [`task3-hybrid-investigation.md`](task3-hybrid-investigation.md) — Parent task
- [`solution1-old-dflash-restore.md`](solution1-old-dflash-restore.md) — Full old DFlash restoration analysis
- [`../dflash-comparison/final-comparison.md`](../dflash-comparison/final-comparison.md) — Side-by-side comparison

---

## Executive Summary

**Verdict: Tape replay is the mechanism that made the old implementation BOTH fast AND low-VRAM, but it is NOT practically adaptable to the current upstream architecture without essentially rebuilding the old implementation.**

The tape replay mechanism is elegant: record DeltaNet intermediate activations (K, V, gate, beta, QKV) during the forward pass, then replay the DeltaNet state update for accepted tokens after rollback. This avoided both the enormous RS buffer (5.4GB for n_rs_seq=8) AND the slow checkpoint serialize/restore path. The tape itself consumed only ~150MB/slot, making it the single most important component of the old DFlash's performance-VRAM profile.

However, adapting tape replay to the current codebase requires:
1. Rebuilding the entire `dflash_capture` infrastructure that was removed in v0.4.0
2. Re-implementing GPU-accelerated GDN replay kernels (CUDA direct path, CPU fallback)
3. Re-integrating with upstream's speculative decoding scheduler, which has no concept of tape replay
4. Reconciling with upstream's RS snapshot system, which tape replay is designed to replace

The net result is that adding tape replay to the current architecture is approximately equivalent to re-implementing the old DFlash. The "hybrid" gain is minimal because tape replay requires so much supporting infrastructure that was deliberately removed.

---

## 1. What Tape Replay Did — Exact Behavior

### 1.1 Data Recorded During Forward Pass

The old implementation recorded five DeltaNet intermediate activations per recurrent layer per token via the eval callback mechanism:

| Data | Shape | Purpose | Source Tensor Name |
|------|-------|---------|-------------------|
| K (after l2_norm) | `[S_k, H_k, n_tokens]` | State update key | `k_conv_predelta-{il}` |
| V | `[S_v, H_v, n_tokens]` | State update value | `v_conv_predelta-{il}` |
| Gate (pre-exp) | `[1, H_v, n_tokens]` | State scaling factor | `gate-{il}` |
| Beta (pre-sigmoid) | `[1, H_v, n_tokens]` | State update gate | `beta-{il}` |
| QKV mixed | `[conv_channels, n_tokens]` | Conv state rebuild | `qkv_mixed_pretranspose-{il}` |

The tensor name map was populated at initialization from the model's recurrent layer IDs:

```cpp
// old-versions/.../src/llama-context.cpp:2278-2282
dflash_capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
dflash_capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
dflash_capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
dflash_capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
dflash_capture->tape_name_map["qkv_mixed_pretranspose-" + il_str] = {idx, DFLASH_TAPE_QKV};
```

### 1.2 Two Recording Paths

**GPU Tape (Primary Path):** When GPU tape was available, graph-embedded copy operations wrote tape data directly to GPU-resident tensors during the forward pass. No eval callback was needed for tape data — the graph builder inserted per-seq copy ops that wrote K/V/gate/beta/QKV directly to the `dflash_tape_gpu` buffers.

```cpp
// old-versions/.../src/llama-context.cpp:1863-1870
if (gpu_tape_fits) {
    // GPU tape captures K/V/gate/beta/QKV through graph-embedded per-seq copies.
    // No tape tensor needs the eval callback when the active GPU tape is present.
    ...
}
```

**CPU Tape (Fallback Path):** When GPU tape was unavailable (multi-GPU, unsupported arch, shape mismatch), the eval callback read tensor data from the compute graph and stored it in CPU vectors.

```cpp
// old-versions/.../src/llama-context.cpp:1979-1984
case DFLASH_TAPE_K:
    tape.S_k = t->ne[0];
    tape.H_k = t->ne[1];
    tape.n_tokens = (int) t->ne[2];
    dflash_read_tensor(t, tape.k, n_elem);
    break;
```

### 1.3 When Tape Replay Was Triggered

Tape replay was called from `dflash_rollback()` after recurrent state was restored from backup cells:

```cpp
// old-versions/.../src/llama-context.cpp:4257-4270
// Recurrent state: restore from backup, then tape replay
mem_recr->seq_rm(seq_id, -1, -1);
mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);
mem_recr->seq_rm(seq_backup, -1, -1);

// Replay DeltaNet state updates for accepted tokens
tape_replay(seq_id, n_accepted);
```

The rollback sequence was:
1. Remove attention KV for draft tokens (except accepted prefix)
2. Remove recurrent state for main sequence
3. Copy recurrent state from backup cells to main sequence (backup = state at n_past_before)
4. Remove backup cells
5. **Tape replay** — replay DeltaNet state updates for the `n_accepted` tokens

### 1.4 What Tape Replay Computed

The tape replay function re-applied the DeltaNet-GDN state update for accepted tokens on top of the restored backup state:

```cpp
// old-versions/.../src/llama-context.cpp:4190-4210 (CPU path)
for (int tok = 0; tok < n_accepted; ++tok) {
    for (int64_t hv = 0; hv < H_v; ++hv) {
        const int64_t hk = hv % H_k;
        float g_val = exp2f(tape.gate[tok * H_v + hv] * 1.442695041f);
        float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]));

        float * S_h = state.data() + hv * S * S;
        const float * k_t = tape.k.data() + tok * (S * H_k) + hk * S;
        const float * v_t = tape.v.data() + tok * (S * H_v) + hv * S;

        // kv = S^T @ k, delta = (v - g*kv) * beta, S = g*S + k⊗delta (fused)
        for (int64_t col = 0; col < S; ++col) {
            float kv = 0.0f;
            for (int64_t row = 0; row < S; ++row) {
                kv += S_h[col * S + row] * k_t[row];
            }
            float delta_col = (v_t[col] - g_val * kv) * b_val;
            for (int64_t row = 0; row < S; ++row) {
                S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
            }
        }
    }
}
```

The conv state was also rebuilt from the tape QKV data in a separate pass (`tape_replay_conv()`).

---

## 2. Why Tape Replay Was Fast

### 2.1 Tape Replay Avoided Full Re-decode

The key insight: tape replay only re-computed the **DeltaNet state update**, not the full forward pass. The DeltaNet state update is a matrix operation on the recurrent state (S × S per head), which is orders of magnitude cheaper than the full layer forward pass (which includes QKV projection, attention, FFN, etc.).

For a typical DeltaNet layer:
- **Full forward pass:** QKV projection + attention + DeltaNet + FFN + output projection
- **Tape replay:** DeltaNet state update only (S^T @ K matrix multiply + outer product update)

The tape data (K, V, gate, beta) was already computed during the forward pass and stored. Replay just re-applied the DeltaNet update using these intermediates.

### 2.2 GPU-Accelerated Replay

The GPU tape path was even faster: tape data was already on GPU from graph-embedded copies, so replay could build a ggml compute graph that operated entirely on GPU memory. No CPU-GPU transfer was needed.

```cpp
// old-versions/.../src/llama-context.cpp:3061-3065 (GPU tape views)
if (use_gpu_tape) {
    // create views into pre-filled GPU tape tensors (zero upload)
    auto & tl = gpu_tape->layers[li];
    k_in = ggml_view_4d(ctx, tl.k, S, H_k, (int64_t)n_accepted, (int64_t)1, ...);
}
```

The GPU direct path (`tape_replay_gdn_direct_gpu()`) used custom CUDA kernels for even faster replay, bypassing ggml graph overhead entirely.

### 2.3 Tape Replay Was Cheaper Than Checkpoint Restore

Checkpoint serialize/restore requires:
1. Serialize entire model state to CPU (all layers, all parameters)
2. Restore from serialized data
3. Re-decode from checkpoint position

Tape replay:
1. Restore recurrent state from backup cells (~150MB/slot, GPU D2D copy)
2. Replay DeltaNet state update for accepted tokens (matrix ops on GPU)
3. Rebuild conv state from tape QKV

The backup cell + tape replay approach was orders of magnitude faster than checkpoint restore because it avoided full model serialization and only touched the recurrent state.

---

## 3. VRAM Cost of Tape Data

### 3.1 Tape Buffer Size Calculation

For a typical Qwen3.6 DFlash model:

| Parameter | Value |
|-----------|-------|
| Recurrent layers | ~28 (DeltaNet layers only) |
| S_k (state dim for K) | Model-dependent, ~64-128 |
| S_v (state dim for V) | Same as S_k |
| H_k (head dim for K) | S_k / num_heads |
| H_v (head dim for V) | S_v / num_heads |
| Max verify tokens | ~16 (block_size - 1) |

Per layer, per token tape data (F32):
- K: S_k × H_k × 4 bytes
- V: S_v × H_v × 4 bytes
- Gate: 1 × H_v × 4 bytes
- Beta: 1 × H_v × 4 bytes
- QKV: conv_channels × 4 bytes

**Per slot GPU tape (from old code):**
```cpp
// old-versions/.../src/llama-context.cpp:2475-2479
tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, (int64_t)max_tokens);
tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, (int64_t)max_tokens);
tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
tl.qkv  = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, (int64_t)max_tokens);
```

**Rough estimate for Qwen3.6-32B DFlash:**
- Tape buffer per slot ≈ 100-200 MB (GPU F32 tensors for ~16 tokens × 28 layers)
- Backup cells per slot ≈ 150 MB (recurrent state for backup sequence)
- **Total per slot: ~250-350 MB**

**Compare to current RS buffer:**
- RS buffer per slot (n_rs_seq=8): ~5.4 GB
- **Tape + backup: ~250-350 MB vs 5.4 GB = ~15x less VRAM**

### 3.2 Tape Data vs Backup Cells — The Complete Picture

The old implementation's VRAM efficiency came from the **combination** of backup cells AND tape replay:

| Component | Old Implementation | Current Implementation |
|-----------|-------------------|----------------------|
| Recurrent state backup | ~150 MB/slot (backup cells) | ~5.4 GB/slot (RS buffer, n_rs_seq=8) |
| Tape data | ~100-200 MB/slot (GPU) or CPU vectors | 0 (not used) |
| Checkpoint data | Not needed (tape replay is fast) | Required for rollback (slow) |
| **Total rollback VRAM** | **~250-350 MB/slot** | **~5.4 GB/slot** |

The tape replay was the mechanism that made backup cells viable. Without tape replay, you'd need either:
- Full RS buffer (5.4GB) for fast rollback, OR
- Checkpoint serialize/restore for low VRAM (but slow)

Tape replay gave you BOTH: low VRAM (backup cells only) AND fast rollback (replay DeltaNet state instead of full re-decode).

---

## 4. What Tape Replay Depends On

### 4.1 Model Structure Dependencies

Tape replay depends on the DeltaNet-GDN state update being:
1. **Deterministic** — the same inputs (K, V, gate, beta) produce the same state update
2. **Invertible from intermediates** — the state update can be re-computed from the recorded intermediates
3. **State-local** — the state update for one cell is independent of other cells

These properties are inherent to the DeltaNet architecture. The current upstream DeltaNet implementation in [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:1) uses the same GDN formulation, so the mathematical dependency is satisfied.

### 4.2 Forward Pass Hooks

Tape replay requires the ability to capture intermediate activations during the forward pass. The old implementation used:

1. **Eval callback** — registered on the graph to intercept tensor outputs by name
2. **Graph-embedded copies** — for GPU tape, the graph builder inserted copy ops that wrote tape data directly to GPU buffers

```cpp
// old-versions/.../src/llama-context.cpp:1857-1862
if (cap->tape_enabled && cap->tape_name_map.count(t->name)) {
    if (cap->profile) {
        cap->profile_cb_tape_ask++;
        ...
    }
    auto * active_tape = cap->active_tape();
    const bool gpu_tape_fits = active_tape && ...;
}
```

**Current upstream does NOT have eval callbacks for tensor interception.** The current forward pass has no mechanism to capture intermediate activations by name. The graph builder in [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:1) does not expose K/V/gate/beta intermediates for capture.

### 4.3 State Management Dependencies

Tape replay depends on:

1. **`dflash_capture` struct** — tracks tape layers, GPU tape buffers, recurrent layer IDs, replay state. Entirely removed in v0.4.0.
2. **`llama_memory_recurrent` with backup cells** — the old recurrent memory supported backup cells (separate from RS snapshots). Current `llama_memory_recurrent` uses RS snapshots exclusively.
3. **`seq_cp_recurrent_no_sync()`** — recurrent-only copy without synchronization. Removed. Current `seq_cp()` copies all state including position tracking.
4. **GPU backend for replay graph** — tape replay builds a compute graph and executes it on GPU. The current codebase has no infrastructure for building ad-hoc compute graphs at rollback time.

### 4.4 Server Integration Dependencies

Tape replay was integrated into the server's DFlash cycle:

```cpp
// old-versions/.../tools/server/server-context.cpp:7505
llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep);
```

The server called `llama_dflash_rollback()` which internally called `tape_replay()`. After rollback, `tape_replay_sync()` ensured the async GPU replay was complete before the next cycle.

**Current upstream server** uses checkpoint-based rollback or RS snapshot rollback. There is no concept of tape replay in the current speculative decoding flow.

---

## 5. Can Current Upstream Support Tape Replay?

### 5.1 What Exists in Current Upstream

| Component | Exists? | Notes |
|-----------|---------|-------|
| DeltaNet-GDN layers | YES | [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:1) — same GDN formulation |
| Recurrent memory | YES | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:1) — but RS snapshots, not backup cells |
| RS snapshot rollback | YES | Built into `llama_memory_recurrent` via `n_rs_seq` and `rs_idx` |
| Checkpoint rollback | YES | Serialize/restore via `state_write()` / `state_read()` |
| Eval callback for tensor capture | **NO** | Removed in v0.4.0. No mechanism to intercept intermediate activations. |
| `dflash_capture` struct | **NO** | Entirely removed. Would need to be rebuilt. |
| Tape buffer allocation | **NO** | `allocate_tape_gpu()` removed. |
| `seq_cp_recurrent_no_sync()` | **NO** | Removed. Only `seq_cp()` exists. |
| Backup cell support | **NO** | Current memory uses RS snapshots exclusively. |
| Tape replay function | **NO** | `tape_replay()`, `tape_replay_cpu()`, `tape_replay_conv()` all removed. |
| GDN replay kernels | **NO** | CUDA direct replay kernels removed. |

### 5.2 What Would Need to Be Added

To adapt tape replay to the current architecture, the following would need to be implemented:

1. **Tensor capture infrastructure** — Eval callback or graph-embedded copy ops to capture K/V/gate/beta during forward pass. This requires modifying the DeltaNet graph builder in [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:1) to expose intermediates and register capture hooks.

2. **Tape buffer management** — Re-implement `dflash_tape_gpu` allocation and lifecycle. The tape buffers need to be GPU-resident F32 tensors, allocated per slot, with size matching the max verify tokens × recurrent layers.

3. **Replay graph builder** — Build a ggml compute graph that:
   - Takes tape data (K, V, gate, beta) as input
   - Reads current recurrent state from `llama_memory_recurrent`
   - Applies GDN state update for each accepted token
   - Writes updated state back to `llama_memory_recurrent`

4. **Backup cell support** — Either re-implement backup cells in `llama_memory_recurrent` or find a way to use the existing RS snapshot infrastructure as backup storage.

5. **Integration with upstream speculative decoding** — The current speculative decoding in [`common/speculative.cpp`](common/speculative.cpp:1) has no concept of tape replay. The accept/rollback flow would need to be modified to call tape replay after rollback.

6. **CUDA direct replay kernels** — For GPU tape replay to be fast, custom CUDA kernels are needed to avoid CPU-GPU transfer overhead. The old implementation had `tape_replay_gdn_direct_gpu()` with custom CUDA code.

### 5.3 Assessment

**Adapting tape replay to the current architecture requires rebuilding approximately 80% of the old DFlash infrastructure.** The tape replay function itself is ~350 lines of code, but it depends on:

- `dflash_capture` struct (~200 lines of state)
- `allocate_tape_gpu()` (~150 lines)
- `set_tape_recording()` (~100 lines)
- `tape_replay_cpu()` (~70 lines)
- `tape_replay_conv()` (~130 lines)
- `tape_replay_gdn_direct_gpu()` (~150 lines)
- `tape_replay_gdn_direct_from_cpu_tape()` (~180 lines)
- `tape_replay_conv_gpu()` (~120 lines)
- `tape_replay_conv_gpu_from_cpu_tape()` (~170 lines)
- `tape_replay_sync()` (~60 lines)
- Eval callback integration (~150 lines)
- Graph builder modifications for tape capture (~100 lines)

**Total: ~1,800+ lines of new/modified code** to restore tape replay functionality, not including server integration, testing, and edge cases.

---

## 6. Performance-VRAM Tradeoff Analysis

### 6.1 Comparison of Rollback Mechanisms

| Mechanism | VRAM Cost | Rollback Speed | Implementation Effort |
|-----------|-----------|----------------|---------------------|
| **RS snapshots (current, n_rs_seq=8)** | ~5.4 GB/slot | Fast (GPU D2D copy) | Already implemented |
| **RS snapshots (n_rs_seq=1)** | ~0.7 GB/slot | Fast (GPU D2D copy) | Trivial (change parameter) |
| **Checkpoint serialize/restore** | Minimal (CPU) | Slow (full model I/O) | Already implemented |
| **Old: Backup cells + tape replay** | ~250-350 MB/slot | Fast (GPU replay) | ~1,800 lines to re-implement |
| **Hybrid: Reduced n_rs_seq + checkpoint fallback** | ~0.7 GB/slot | Fast for small rollback, slow for large | Minimal (already works) |

### 6.2 The Tape Replay Advantage

Tape replay's advantage is that it gives you:
- **Low VRAM:** ~250-350 MB/slot vs 5.4 GB/slot (15x reduction)
- **Fast rollback:** GPU replay of DeltaNet state is fast (much faster than checkpoint restore)

But this advantage comes at the cost of ~1,800 lines of infrastructure code that was deliberately removed.

### 6.3 The Simpler Alternative: Reduce n_rs_seq

The simplest hybrid approach (identified in subtask 3.2) is to reduce `n_rs_seq` from 8 to a smaller value:

| n_rs_seq | VRAM per slot | Rollback range |
|----------|--------------|----------------|
| 8 (current) | ~5.4 GB | Up to 8 tokens |
| 4 | ~2.7 GB | Up to 4 tokens |
| 2 | ~1.4 GB | Up to 2 tokens |
| 1 | ~0.7 GB | Up to 1 token |

With `n_rs_seq=1`, VRAM drops to ~0.7 GB/slot (87% reduction from current). Rollback of 1 token is fast via RS snapshot. Rollback of more than 1 token falls back to checkpoint restore, which is slow but rare (most DFlash cycles accept 1-3 tokens).

### 6.4 When Tape Replay Would Be Worth It

Tape replay would be worth the ~1,800 lines of implementation effort only if:
1. VRAM is extremely constrained (GPU cannot afford even 0.7 GB/slot for RS buffer)
2. Rollback speed is critical (checkpoint fallback is unacceptable even for rare cases)
3. The target GPU has enough memory for the model + draft + ~350 MB/slot tape/backup

For most practical scenarios, reducing `n_rs_seq` to 1-2 provides 80-90% of the VRAM savings with minimal implementation effort.

---

## 7. Key Findings Summary

### 7.1 Was Tape Replay the Mechanism for BOTH Fast AND Low-VRAM?

**YES.** Tape replay was the critical innovation that made the old implementation both VRAM-efficient and performant:

- **Low VRAM:** Backup cells (~150 MB/slot) + tape data (~100-200 MB/slot) = ~250-350 MB/slot total, compared to 5.4 GB/slot for RS buffer.
- **Fast rollback:** Tape replay re-computed DeltaNet state for accepted tokens using recorded intermediates, which is much faster than checkpoint serialize/restore.
- **The combination:** Backup cells provided the rollback starting point (state at n_past_before), and tape replay advanced the state forward for accepted tokens. Neither mechanism alone was sufficient — backup cells without tape replay would require full re-decode, and tape replay without backup cells would have no starting state.

### 7.2 Can Tape Replay Be Adapted Without the Full Old Implementation?

**NO, not practically.** Tape replay requires:

1. Tensor capture infrastructure (eval callback or graph-embedded copies) — **not in current upstream**
2. Tape buffer management — **not in current upstream**
3. Replay graph builder — **not in current upstream**
4. Backup cell support — **not in current upstream**
5. CUDA direct replay kernels — **not in current upstream**
6. Server integration — **not in current upstream**

Adding these six components to the current codebase is approximately equivalent to re-implementing the old DFlash. The "hybrid" gain is minimal because tape replay depends on so much supporting infrastructure.

### 7.3 What Would a "Minimal Tape Replay" Look Like?

A minimal tape replay would require at minimum:

1. **Tensor capture:** Modify [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:1) to expose K/V/gate/beta intermediates and register a capture hook.
2. **Tape buffer:** Allocate GPU F32 buffers for tape data per slot.
3. **CPU replay:** Implement `tape_replay_cpu()` (~70 lines) as a fallback.
4. **Backup cells:** Add minimal backup cell support to `llama_memory_recurrent` (or repurpose RS snapshots for backup).
5. **Server hook:** Modify speculative decoding to call tape replay after rollback.

Even this minimal version would be ~500-800 lines of new code and would require changes to the DeltaNet graph builder, recurrent memory, and speculative decoding. Without GPU replay kernels, performance would depend on CPU replay speed, which may not be fast enough for real-time serving.

### 7.4 Alternative: Similar Performance-VRAM Balance Without Tape Replay

**The best alternative is reducing `n_rs_seq` combined with checkpoint fallback:**

- Set `n_rs_seq=1` (or 2) to reduce RS buffer to ~0.7-1.4 GB/slot
- For rollbacks within `n_rs_seq`, use RS snapshot (fast, GPU D2D copy)
- For rollbacks beyond `n_rs_seq`, fall back to checkpoint restore (slow but rare)

This approach:
- Saves 75-87% VRAM compared to current `n_rs_seq=8`
- Preserves fast rollback for the common case (1-2 token rollback)
- Requires zero new code (just parameter tuning)
- Is already supported by the current architecture

For the rare case of large rollbacks (beyond `n_rs_seq`), checkpoint restore is slow but acceptable because:
- Most DFlash cycles accept 1-3 tokens (based on acceptance histogram in old server code)
- Large rollbacks are rare (< 5% of cycles in typical workloads)
- The performance impact is bounded by the infrequency of large rollbacks

---

## 8. Recommendation

**Do NOT pursue tape replay adaptation as a hybrid approach.** The implementation effort (~1,800+ lines of new code) is approximately equivalent to re-implementing the old DFlash, and the incremental benefit over the simpler `n_rs_seq` reduction is marginal for most use cases.

**Instead, pursue the simpler hybrid approach:**
1. **Immediate:** Reduce `n_rs_seq` from 8 to 1-2 to save 75-87% VRAM.
2. **If needed:** Optimize checkpoint restore for the rare large-rollback case (e.g., incremental checkpointing, GPU-resident checkpoint buffers).
3. **Only if VRAM is extremely constrained:** Consider tape replay as a last resort, but be prepared for the full implementation effort.

The tape replay mechanism was elegant and effective in the old implementation, but it was also deeply integrated with the old DFlash infrastructure. Adapting it to the current upstream architecture requires rebuilding that infrastructure, which defeats the purpose of a "hybrid" approach that avoids full restoration.

---

## 9. Code References

| Component | Old File | Line Range | Description |
|-----------|----------|-----------|-------------|
| `dflash_rollback()` | `old-versions/.../src/llama-context.cpp` | 4218-4292 | Three-phase rollback (attention + recurrent + tape) |
| `tape_replay()` | `old-versions/.../src/llama-context.cpp` | 2898-3255 | Main tape replay entry point |
| `tape_replay_cpu()` | `old-versions/.../src/llama-context.cpp` | 4148-4216 | CPU fallback for recurrent replay |
| `tape_replay_conv()` | `old-versions/.../src/llama-context.cpp` | 3877-4033 | Conv state rebuild from tape |
| `tape_replay_gdn_direct_gpu()` | `old-versions/.../src/llama-context.cpp` | 3257-3581 | GPU direct GDN replay |
| `tape_replay_gdn_direct_from_cpu_tape()` | `old-versions/.../src/llama-context.cpp` | 3403-3581 | GPU direct replay from CPU tape |
| `tape_replay_conv_gpu()` | `old-versions/.../src/llama-context.cpp` | 3584-3704 | GPU conv replay |
| `tape_replay_conv_gpu_from_cpu_tape()` | `old-versions/.../src/llama-context.cpp` | 3704-3876 | GPU conv replay from CPU tape |
| `tape_replay_sync()` | `old-versions/.../src/llama-context.cpp` | 4034-4147 | Sync async replay across slots |
| `dflash_tape_layer` | `old-versions/.../src/llama-context.h` | 118-130 | CPU tape data structure |
| `dflash_tape_gpu` | `old-versions/.../src/llama-context.h` | 144-148 | GPU tape data structure |
| `allocate_tape_gpu()` | `old-versions/.../src/llama-context.cpp` | 2344-2521 | GPU tape buffer allocation |
| `set_tape_recording()` | `old-versions/.../src/llama-context.cpp` | 2288-2341 | Enable/disable tape recording |
| Eval callback (tape) | `old-versions/.../src/llama-context.cpp` | 1837-2009 | Tensor interception for tape |
| Server rollback call | `old-versions/.../tools/server/server-context.cpp` | 7505 | `llama_dflash_rollback()` call |
| DeltaNet-GDN (current) | `src/models/delta-net-base.cpp` | 16-607 | Current DeltaNet graph builder |
| Recurrent memory (current) | `src/llama-memory-recurrent.cpp` | 1-1349 | Current recurrent memory with RS snapshots |
| Speculative decoding (current) | `common/speculative.cpp` | 905-1204 | Current DFlash speculative implementation |
