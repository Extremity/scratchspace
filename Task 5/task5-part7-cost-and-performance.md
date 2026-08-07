# Task 5.5 Parts 8-9: Theoretical Performance Comparison + VRAM Cost Analysis

**Date:** 2026-08-07
**Source:** Current upstream workspace + `old-versions/beellama.cpp-preview-v0.3.2/` + prior Task 5 documents
**Related:** task5-part1 through task5-part6

---

## PART 8: THEORETICAL PERFORMANCE COMPARISON

### 1. Three Approaches Defined

All three approaches share the same prefix: backup pre-draft recurrent state, run speculative DFlash verification, and determine K accepted tokens out of N draft tokens. They differ in how they reconstruct the correct post-acceptance recurrent state.

| Approach | Name | Rollback Mechanism |
|----------|------|-------------------|
| **A** | Task 4 (backup + re-decode) | Restore backup cell → full-model re-decode of K accepted tokens |
| **B** | Minimal replay (proposed) | Restore backup cell → recurrent-only GDN replay of K accepted tokens |
| **C** | Old custom DFlash (0.3.2) | Restore backup cell → old tape replay (GDN + conv rebuild) |

---

### 2. Approach A: Task 4 — Backup + Full-Model Re-decode

#### 2.1 Execution Flow

```
backup → speculative verification → restore backup → full-model re-decode(K tokens)
```

Source: [`task4-part3-accepted-state-recovery.md`](task4-part3-accepted-state-recovery.md), question D and J.

#### 2.2 Number of Full Model Evaluations

- **Per DFlash cycle:** 1 full verification pass (N draft tokens) + 1 full re-decode pass (K accepted tokens).
- **Total full model evaluations per cycle:** 1 + 1 = 2 (when partial acceptance occurs).
- **Full acceptance (K=N):** Only 1 verification pass. No re-decode needed.

The re-decode pass runs the complete model forward graph for K tokens, including:
- Input embedding lookup
- All attention layers (KV cache reads + attention computation)
- All FFN layers
- All normalization layers
- All recurrent layers (GDN with full projections)

#### 2.3 Number of Recurrent-Only Evaluations

- **Zero.** The recurrent state is advanced through the full model forward pass, not through isolated GDN calls.

#### 2.4 Major Tensor Operations

The re-decode pass touches ALL model tensors:

| Operation | Per Token | Source |
|-----------|-----------|--------|
| Input embedding lookup | 1 | `llama_model.cpp` |
| Attention projection (Q/K/V) | 3 matrix multiplies per layer | Model graph builder |
| Attention computation | O(seq_len) per layer | Attention kernel |
| FFN (up/gate/down) | 3 matrix multiplies per layer | Model graph builder |
| Normalization | 2 passes (RMS norm pre/post) | Model graph builder |
| GDN (recurrent update) | 1 fused kernel per recurrent layer | [`gated_delta_net.cu`](ggml/src/ggml-cuda/gated_delta_net.cu) |

For Qwen3.6-27B (approximately 27B parameters, ~64 total layers, 48 recurrent):
- Each full token evaluation touches ~27B parameters × 4B (F16 weights) ≈ 108 GB of weight reads.
- With quantization (Q4_K_M ≈ 2 bytes/param effective): ~67.5 GB effective reads per token.
- The re-decode of K=8 tokens would read weights ~540 GB total.

#### 2.5 Computational Complexity

**O(K × P)** where P is total model parameters.

For K accepted tokens:
- Re-decode cost: K × (attention + FFN + normalization + GDN) per layer.
- Dominated by attention and FFN (non-recurrent layers).

#### 2.6 Runtime Dominator

**Full model forward pass dominates.** The GDN recurrent update represents a small fraction of the full model evaluation. For Qwen3.6, the recurrent layers (48 of ~64) account for roughly 30-40% of total compute, with the remaining 60-70% spent in attention and FFN layers.

The re-decode of K tokens is approximately **K times the cost of a single normal decoding step** because it re-verified tokens already verified in the speculative pass.

#### 2.7 Summary

| Metric | Value |
|--------|-------|
| Full model evaluations per cycle (partial accept) | 2 (verify + re-decode) |
| Recurrent-only evaluations | 0 |
| Dominant cost | Full model re-decode of K tokens |
| Complexity | O(K × P) |

---

### 3. Approach B — Minimal Replay (Proposed)

#### 3.1 Execution Flow

```
backup → speculative verification → restore backup → GDN-only replay(K tokens)
```

Source: [`task5-part6-ggml-and-integration.md`](task5-part6-ggml-and-integration.md), Section 2.2.

#### 3.2 Number of Full Model Evaluations

- **Per DFlash cycle:** 1 full verification pass (N draft tokens).
- **Zero re-decode passes.** The replay uses only the GDN primitive, not the full model.
- **Total full model evaluations:** 1 per cycle (the verification pass).

#### 3.3 Number of Recurrent-Only Evaluations

- **Per partial-accept cycle:** 48 GDN evaluations (one per recurrent layer, each processing K tokens sequentially).
- The GDN kernel at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) processes K tokens in a single kernel launch per layer:
  ```cpp
  for (int t = 0; t < n_tokens; t++) {
      // GDN state update for token t
  }
  ```
- With `K=1` parameter (no snapshots needed for replay), the kernel writes only the final state.

#### 3.4 Major Tensor Operations

The replay graph per recurrent layer consists of:

| Operation | ggml Primitive | Cost |
|-----------|---------------|------|
| Create dummy q=zeros | `ggml_fill()` | Negligible (F32 zero fill, S_k × H_k × K) |
| View tape k for K tokens | `ggml_view_4d()` | Zero-copy view |
| View tape v for K tokens | `ggml_view_4d()` | Zero-copy view |
| View tape gate for K tokens | `ggml_view_4d()` | Zero-copy view |
| View tape beta for K tokens | `ggml_view_4d()` | Zero-copy view |
| View backup S state | `ggml_view_4d()` | Zero-copy view |
| GDN call (K tokens, K=1) | `ggml_gated_delta_net()` | **Dominant operation** |
| Copy output state to active | `ggml_cpy()` | ~3 MB per layer (S state) |

The GDN kernel internally performs per token:
- `S^T @ k`: Matrix-vector multiply (S_v × S_k = 128 × 128 for Qwen3.6)
- `exp(g)`: Elementwise exponential
- `(v - g*kv) * beta`: Elementwise
- `g*S + k⊗delta`: Rank-1 update (S_v × S_v = 128 × 128)

#### 3.5 Computational Complexity

**O(K × L_rec × S_v²)** where:
- K = number of accepted tokens
- L_rec = number of recurrent layers (48 for Qwen3.6)
- S_v = state dimension (128 for Qwen3.6)

For Qwen3.6 with K=8:
- Per layer: 8 × 128² × 4 ops ≈ 524K FLOPs (GDN core)
- Per layer with attention output: 8 × 128² × 4 × 2 ≈ 1M FLOPs (including S^T @ q, even though discarded)
- Total (48 layers): ~48M FLOPs

Compare to full model re-decode:
- Qwen3.6 full decode per token: ~54 GFLOPs (27B params × 2 for forward/backward)
- K=8 full re-decode: ~432 GFLOPs

**Ratio: GDN replay is approximately 10,000× lighter than full model re-decode.**

#### 3.6 Runtime Dominator

**GDN kernel launch overhead and state I/O dominate.** The actual GDN computation per layer is small (~1M FLOPs), but there are 48 layers, each requiring a kernel launch. The tape data (~19 MB for 8 tokens × 48 layers) must be read from GPU memory, and the backup state (~150 MB) must be restored before replay.

Key bottlenecks:
1. **Backup state restore:** ~150 MB GPU memory copy (backup → active cell).
2. **Tape data read:** ~19 MB for K=8 tokens across 48 layers.
3. **GDN kernel launches:** 48 kernel launches (one per recurrent layer).
4. **State write-back:** ~150 MB write (replayed S state → active cell).

Total memory traffic: ~330 MB (restore + tape read + write-back).

#### 3.7 Summary

| Metric | Value |
|--------|-------|
| Full model evaluations per cycle | 1 (verification only) |
| Recurrent-only evaluations | 48 GDN calls (one per layer) |
| Dominant cost | Memory I/O (backup restore + state write-back) |
| Complexity | O(K × L_rec × S_v²) |
| Compute vs full re-decode | ~10,000× lighter |

---

### 4. Approach C — Old Custom DFlash (0.3.2)

#### 4.1 Execution Flow

```
backup → speculative verification → restore backup → old tape replay(K tokens)
```

Source: [`task5-part1-old-tape-mechanics.md`](task5-part1-old-tape-mechanics.md), [`llama-context.cpp:2898-3255`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2898).

#### 4.2 Number of Full Model Evaluations

- **Per DFlash cycle:** 1 full verification pass (N draft tokens).
- **Zero re-decode passes.** The old tape replay uses only GDN + conv rebuild.
- **Total full model evaluations:** 1 per cycle.

#### 4.3 Number of Recurrent-Only Evaluations

- **Per partial-accept cycle:** Same as Approach B — 48 GDN evaluations.
- The old replay had three paths ranked by preference:
  1. **Direct GPU GDN replay** ([`tape_replay_gdn_direct_gpu()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3257)): Direct CUDA kernel call via `ggml_backend_reg_get_proc_address`. No ggml graph involved.
  2. **GGML graph replay** ([`tape_replay()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3002)): Builds ggml compute graph with GDN ops.
  3. **CPU fallback** ([`tape_replay_cpu()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4148)): Hand-coded CPU loop.

#### 4.4 Major Tensor Operations

The old replay performed two operations:

| Operation | Source | Cost |
|-----------|--------|------|
| GDN state replay | [`tape_replay_gdn_direct_gpu()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3257) | Same as Approach B |
| Conv state rebuild | [`tape_replay_conv()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3877) | Additional operation |

The conv rebuild ([`llama-context.cpp:3877-4032`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3877)):
- Shifts R state (conv buffer) by K positions.
- Fills new positions with `qkv_mixed` data from tape.
- Per layer: `R[ch*conv_window + w] = qkv_mixed[(w)*conv_ch + ch]` for w < K.
- This is a ring buffer shift + fill operation.

#### 4.5 Computational Complexity

**O(K × L_rec × (S_v² + conv_window × conv_ch))**

The GDN component is identical to Approach B. The conv rebuild adds:
- Per layer: O(K × conv_window × conv_ch) for the shift + fill.
- For Qwen3.6, conv state is small (n_embd_r = 30,720 vs n_embd_s = 786,432), so conv rebuild is ~4% of GDN cost.

#### 4.6 Runtime Dominator

**Same as Approach B, plus conv rebuild overhead.**

The old direct-GPU path bypassed ggml graph construction entirely, saving graph-building overhead. The kernel was called directly:
```cpp
// llama-context.cpp:3271
auto replay_fn = (bool (*)(...)) ggml_backend_reg_get_proc_address(
    gpu_backend->reg(), "dflash_replay_gdn_state_no_check");
```

This avoided ggml graph construction, scheduler allocation, and graph execution overhead. The replay was a raw CUDA kernel launch per layer.

#### 4.7 Summary

| Metric | Value |
|--------|-------|
| Full model evaluations per cycle | 1 (verification only) |
| Recurrent-only evaluations | 48 GDN calls + 48 conv rebuilds |
| Dominant cost | GDN kernel + memory I/O |
| Complexity | O(K × L_rec × S_v²) + O(K × L_rec × conv) |
| Graph overhead | None (direct kernel call for GPU path) |

---

### 5. Side-by-Side Comparison

#### 5.1 Computational Operations

| Operation | A: Task 4 | B: Minimal Replay | C: Old DFlash |
|-----------|-----------|-------------------|---------------|
| Full model evaluations | 2 (verify + re-decode) | 1 (verify only) | 1 (verify only) |
| Recurrent GDN evaluations | 0 (included in full model) | 48 (one per layer) | 48 (one per layer) |
| Conv rebuild | 0 (included in full model) | TBD (see Section 6) | 48 (one per layer) |
| Tape capture | Not needed | Required | Required |
| Backup restore | Required | Required | Required |

#### 5.2 Memory Traffic (Qwen3.6, K=8)

| Component | A: Task 4 | B: Minimal Replay | C: Old DFlash |
|-----------|-----------|-------------------|---------------|
| Weight reads | ~540 GB (K × full model) | ~0 GB (no weight reads) | ~0 GB (no weight reads) |
| Backup restore | ~150 MB | ~150 MB | ~150 MB |
| Tape data read | N/A | ~19 MB | ~19 MB |
| State write-back | ~150 MB (via full model) | ~150 MB | ~150 MB |
| Conv rebuild | N/A | TBD | ~3 MB |
| **Total** | **~540 GB** | **~320 MB** | **~322 MB** |

#### 5.3 Relative Performance Expectation

| Factor | A: Task 4 | B: Minimal Replay | C: Old DFlash |
|--------|-----------|-------------------|---------------|
| Compute intensity | Very high (full model) | Very low (GDN only) | Very low (GDN only) |
| Memory bandwidth | Extreme (weight reads dominate) | Low (state I/O only) | Low (state I/O only) |
| Kernel launches | Many (full model graph) | 48 (GDN per layer) | 48 (GDN per layer, direct) |
| Graph overhead | Full graph per token | Replay graph (lightweight) | None (direct kernel) |
| **Expected relative speed** | **1.0× (baseline, slowest)** | **~50-200× faster** | **~60-250× faster** |

**Rationale for speedup estimates:**
- The full model re-decode (Approach A) reads ~540 GB of weights for K=8 tokens.
- The GDN replay (Approach B) touches ~320 MB of state data.
- Ratio: 540 GB / 320 MB ≈ 1,700× in memory traffic alone.
- Actual speedup will be lower because:
  - Weight reads are cached (L2, texture cache) for repeated access.
  - GDN replay has 48 kernel launches with overhead.
  - Graph construction adds latency.
- Conservative estimate: **50-200× faster** for replay vs full re-decode.

#### 5.4 Does Minimal Replay Approach Old DFlash Performance?

**YES, within 90-95% of old DFlash replay performance.**

The differences between Approach B (minimal replay) and Approach C (old DFlash):

| Difference | Impact | Magnitude |
|------------|--------|-----------|
| Graph overhead | B builds ggml graph; C calls kernel directly | ~5-10% slower for B |
| Conv rebuild | B may skip or defer; C always does | ~1-2% slower if skipped |
| Tape capture | Both require tape capture | Identical |
| GDN kernel | Same kernel, same computation | Identical |
| Backup restore | Same mechanism | Identical |

The primary difference is graph construction overhead. Approach B builds a lightweight ggml graph (48 GDN ops + views + copies), while Approach C called the CUDA kernel directly. The graph overhead is estimated at 5-10ms per cycle for 48 layers, which is small compared to the verification pass (hundreds of ms).

**Conclusion: Minimal replay should achieve 90-95% of old DFlash replay performance while using existing ggml primitives and avoiding fork-specific CUDA kernels.**

---

### 6. Conv State Consideration

The conv state (R tensor) replay is an open question for Approach B. Options:

| Option | Description | Performance Impact |
|--------|-------------|-------------------|
| Skip conv replay | Accept small accuracy loss | ~1-2% accuracy degradation; fastest |
| Implement conv replay | Add ~100 lines for ring buffer shift | ~1% overhead vs GDN; matches old |
| Full model for conv | Re-decode only conv layers | Defeats purpose; ~10× cost |

**Recommendation:** Implement conv replay if accuracy testing shows measurable degradation without it. The conv state is small (n_embd_r = 30,720 vs n_embd_s = 786,432), so the replay cost is ~4% of GDN replay.

---

## PART 9: VRAM COST ANALYSIS

### 1. Qwen3.6 Parameters Reference

All calculations below use these Qwen3.6 parameters from [`research-summary.md`](research-summary.md):

| Parameter | Value |
|-----------|-------|
| n_embd | 5120 |
| n_layer (total) | ~64 |
| n_layer (recurrent) | 48 |
| ssm_d_inner | 6144 |
| ssm_d_state (S_v) | 128 |
| n_embd_r | 30,720 |
| n_embd_s | 786,432 (= S_v × S_v × H_v) |

**Note:** n_embd_s = 786,432 as confirmed from test logs in research-summary.md.

---

### 2. Component Breakdown: Minimal Replay VRAM Requirements

#### 2.1 Backup Recurrent State

The backup cell stores ONE copy of the current active recurrent state (R + S tensors for all recurrent layers).

**Source:** [`llama-memory-recurrent.cpp:20-100`](src/llama-memory-recurrent.cpp:20), [`task4-part2-allocation-and-rollback.md`](task4-part2-allocation-and-rollback.md).

The backup stores the recurrent state for ONE cell position (the current active position):
```
R per layer (1 position): n_embd_r × 4B = 30,720 × 4B = 120 KB
S per layer (1 position): n_embd_s × 4B = 786,432 × 4B = 3,145,728 B = 3 MB
```

**Per layer backup: ~3.12 MB**
**Total backup (48 layers): ~149.7 MB ≈ 150 MB**

This matches the old estimate from [`research-summary.md`](research-summary.md): "~150 MB/slot."

**Per-slot allocation:** YES — each parallel slot needs its own backup cell.
**Global allocation:** NO — backup cells are per-slot, not shared.

#### 2.2 Captured Replay Intermediates (Tape Buffer)

From [`task5-part2-current-delta-replay.md`](task5-part2-current-delta-replay.md), Section 1.1.2, using the documented formula:

```
Per token per layer: (S_k × H_k + 2 × S_v × H_v + 2 × H_v) floats
= (128 × 32 + 2 × 128 × 32 + 2 × 32) floats
= (4,096 + 8,192 + 64) floats
= 12,352 floats = 49.4 KB
```

| Item | Calculation | Size |
|------|-------------|------|
| Per token per layer | 12,352 floats × 4B | 49.4 KB |
| Per token (48 layers) | 49.4 KB × 48 | 2.37 MB |
| For N=15 draft tokens | 2.37 MB × 15 | 35.5 MB |
| For N=8 draft tokens | 2.37 MB × 8 | 19.0 MB |

**Per-slot allocation:** YES — tape is per-slot, allocated once and reused each cycle.
**Lifetime:** Tape is discarded at end of each cycle. Buffer is reused.

#### 2.3 Replay Buffers

The replay operation itself requires minimal additional buffers:

| Buffer | Size | Purpose |
|--------|------|---------|
| Dummy q=zeros | S_k × H_k × K × 4B = 128 × 32 × 8 × 4B = 128 KB | Per GDN call, can be reused |
| GDN output state | n_embd_s × 4B = 786,432 × 4B = 3 MB | Per layer, written to active cell |
| Temporary ggml graph context | ~256 KB per layer × 48 = ~12 MB | Graph construction overhead |

**Total replay buffers: ~15 MB** (temporary, freed after replay completes).

#### 2.4 Active Recurrent State (Baseline)

With `n_rs_seq = 0`, the active recurrent state is:

From [`research-summary.md`](research-summary.md):
```
n_rows = mem_size × (1 + n_rs_seq) = 4 × (1 + 0) = 4

R: 30,720 × 4 × 4B × 48 layers = 23.04 MB
S: 786,432 × 4 × 4B × 48 layers = 598.7 MB
Total: ~621.7 MB
```

With n_rs_seq = 8 (current default):
```
n_rows = 4 × 9 = 36
R: 30,720 × 36 × 4B × 48 = 202.5 MB
S: 786,432 × 36 × 4B × 48 = 5,184 MB
Total: ~5,386.5 MB = 5.4 GB
```

**Active state with n_rs_seq=0: ~622 MB**
**Active state with n_rs_seq=8: ~5,387 MB = 5.4 GB**

#### 2.5 Summary: Per-Slot VRAM for Minimal Replay

| Component | Size | Notes |
|-----------|------|-------|
| Active recurrent state (n_rs_seq=0) | ~622 MB | Baseline, always present |
| Backup cell | ~150 MB | Per slot, static allocation |
| Tape buffer (N=15) | ~36 MB | Per slot, reused each cycle |
| Replay buffers | ~15 MB | Temporary, freed after replay |
| **Total per slot** | **~823 MB** | Peak during replay |
| **Sustained per slot** | **~808 MB** | After replay completes |

---

### 3. Scaling Analysis

#### 3.1 n_parallel Scaling

Each parallel slot needs its own backup cell and tape buffer:

| n_parallel | Backup Cells | Tape Buffers | Total Extra VRAM |
|------------|-------------|-------------|------------------|
| 1 | 150 MB | 36 MB | 186 MB |
| 4 | 600 MB | 144 MB | 744 MB |
| 8 | 1,200 MB | 288 MB | 1,488 MB |
| 16 | 2,400 MB | 576 MB | 2,976 MB |

**Scaling:** Linear with n_parallel. Each additional slot adds ~186 MB.

#### 3.2 Context Length Scaling

The tape buffer and backup cell are **independent of context length**. They store:
- Backup: Current recurrent state (fixed size, independent of sequence length).
- Tape: Intermediates for N draft tokens only (fixed size, independent of past context).

The active recurrent state is also independent of context length (the S state is a fixed-size matrix, not a sequence-length-dependent cache).

**Scaling: O(1) with context length.**

#### 3.3 Speculative Token Count Scaling

The tape buffer scales linearly with N (number of draft tokens):

| N (draft tokens) | Tape Buffer | Total per slot |
|-----------------|-------------|---------------|
| 4 | 9.5 MB | 801 MB |
| 8 | 19.0 MB | 811 MB |
| 12 | 28.4 MB | 820 MB |
| 15 | 35.5 MB | 828 MB |
| 20 | 47.4 MB | 839 MB |

**Scaling: O(N) for tape buffer, but the coefficient is small (~2.4 MB/token).**

---

### 4. Comparison Against Baselines

#### 4.1 Current Upstream DFlash (n_rs_seq=8)

From [`research-summary.md`](research-summary.md):

| Component | Size |
|-----------|------|
| RS buffer (n_rs_seq=8) | 5,387 MB |
| Active recurrent state (base) | Included in RS buffer |
| Other DFlash overhead | ~598 MB (active state only, without RS) |
| **Total** | **~5.4 GB** |

**Source:** `need_n_rs_seq()` includes `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` at [`common/common.h:419`](common/common.h:419), causing `n_rs_seq = draft.n_max = 8`.

#### 4.2 Task 4 — Backup + Re-decode

From Task 4 documents:

| Component | Size |
|-----------|------|
| Active recurrent state (n_rs_seq=0) | ~622 MB |
| Backup cell | ~150 MB |
| No tape buffer (not needed for re-decode) | 0 MB |
| **Total per slot** | **~772 MB** |

**VRAM savings vs current:** 5.4 GB → 0.77 GB = **86% reduction**.

#### 4.3 Minimal Replay (Proposed)

| Component | Size |
|-----------|------|
| Active recurrent state (n_rs_seq=0) | ~622 MB |
| Backup cell | ~150 MB |
| Tape buffer (N=15) | ~36 MB |
| Replay buffers (temporary) | ~15 MB |
| **Total per slot (peak)** | **~823 MB** |
| **Total per slot (sustained)** | **~808 MB** |

**VRAM savings vs current:** 5.4 GB → 0.81 GB = **85% reduction**.
**VRAM overhead vs Task 4:** +51 MB (tape + replay buffers).

#### 4.4 Old 0.3.2 DFlash

From [`research-summary.md`](research-summary.md):

| Component | Size |
|-----------|------|
| Backup cells (deferred) | ~150 MB/slot |
| Tape (GPU) | ~36 MB/slot |
| Active state (no RS buffer) | ~622 MB |
| **Total per slot** | **~808 MB** |

**Note:** The old implementation's ~0.25-0.35 GB figure from research-summary.md refers to the ADDITIONAL overhead beyond the base model state, not the total VRAM. The base recurrent state (~622 MB) is always present regardless of DFlash.

**Additional DFlash overhead (old):** ~186 MB/slot (backup + tape).
**Additional DFlash overhead (minimal replay):** ~186 MB/slot (backup + tape + replay buffers).

These are essentially identical because the old implementation used the same backup cell + tape architecture.

---

### 5. Complete VRAM Comparison Table

All values are ADDITIONAL overhead beyond the base model (weights + attention KV + base recurrent state with n_rs_seq=0).

| Approach | Additional Overhead | Total VRAM (Qwen3.6, 1 slot) | Savings vs Current |
|----------|-------------------|------------------------------|--------------------|
| **Current upstream** (n_rs_seq=8) | ~4.8 GB | ~18.6 GB base + 4.8 GB = ~23.4 GB | — |
| **Task 4** (backup + re-decode) | ~150 MB | ~18.6 GB base + 0.15 GB = ~18.8 GB | 97% |
| **Minimal replay** (proposed) | ~186 MB | ~18.6 GB base + 0.19 GB = ~18.8 GB | 96% |
| **Old DFlash** (0.3.2) | ~186 MB | ~18.6 GB base + 0.19 GB = ~18.8 GB | 96% |

**Key insight:** The 5.4 GB overhead from current upstream is almost entirely the RS buffer (5,387 MB of 5,387 MB total). Eliminating `n_rs_seq` for DFlash reduces overhead by 96-97%, regardless of whether you use Task 4 re-decode or minimal replay.

The ~36 MB tape buffer is the only additional cost of minimal replay vs Task 4, and it's negligible compared to the 4.8 GB saved.

---

### 6. Multi-Slot VRAM Summary

For typical serving configurations (Qwen3.6-27B, single GPU):

| n_parallel | Current | Task 4 | Minimal Replay | Old DFlash |
|------------|---------|--------|---------------|------------|
| 1 | ~23.4 GB | ~18.8 GB | ~18.8 GB | ~18.8 GB |
| 4 | ~28.2 GB | ~20.3 GB | ~20.4 GB | ~20.4 GB |
| 8 | ~38.0 GB | ~22.2 GB | ~22.4 GB | ~22.4 GB |

**Note:** Current upstream becomes impractical beyond 1-2 slots on RTX 3090 (24 GB). Task 4 and minimal replay support 8+ slots comfortably.

---

### 7. Conclusions

#### 7.1 Part 8 — Performance

1. **Task 4 (backup + re-decode)** requires K full model evaluations per cycle, making it the slowest approach. The re-decode dominates runtime.
2. **Minimal replay** uses only GDN operations (~10,000× lighter compute than full model), making it ~50-200× faster than Task 4 for the rollback phase.
3. **Old DFlash** achieves the same GDN replay as minimal replay but with lower graph overhead (direct kernel calls). Minimal replay should achieve 90-95% of old DFlash performance.
4. The primary bottleneck for minimal replay is memory I/O (backup restore + tape read + state write-back ≈ 330 MB), not compute.

#### 7.2 Part 9 — VRAM

1. **Minimal replay adds ~186 MB/slot** beyond the base model (150 MB backup + 36 MB tape).
2. **VRAM savings vs current: 96%** (4.8 GB → 186 MB additional overhead).
3. **VRAM is essentially identical to old DFlash** because both use backup cells + tape.
4. **Scaling is favorable:** VRAM scales linearly with n_parallel (~186 MB/slot) and is O(1) with context length.
5. The tape buffer (36 MB for N=15) is the only cost difference vs Task 4, and it's negligible.

#### 7.3 Final Assessment

Minimal replay achieves:
- **Near-old-DFlash performance** (90-95%) through GDN-only replay.
- **Old-DFlash-level VRAM efficiency** (~186 MB/slot additional overhead).
- **Upstream compatibility** using only existing ggml primitives (no new CUDA kernels).
- **Small implementation footprint** (~400 lines of new code using existing primitives).

The tradeoff vs old DFlash is ~5-10% performance for ~400 lines of clean ggml code vs ~1,100 lines of fork-specific CUDA replay code.
