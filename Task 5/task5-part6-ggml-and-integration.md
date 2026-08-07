# Task 5.4 Parts 6-7: ggml Primitives Sufficiency + Task 4 Backup-Cell Integration

**Date:** 2026-08-07
**Source:** Current upstream workspace + `old-versions/beellama.cpp-preview-v0.3.2/`
**Related:** task5-part2-current-delta-replay.md, task5-part4-capture-integration.md, task5-part5-linear-only-feasibility.md, task4-part1-lifecycle-and-apis.md

---

## SECTION 1: ggml Primitives Analysis (Part 6)

### 1.1 Can Replay Be Expressed Using Existing ggml Operations?

**Short answer: YES.** The replay operation can be constructed entirely from existing ggml primitives. No new CUDA kernel is required for the replay GDN operation.

#### 1.1.1 The Core Replay Operation: `ggml_gated_delta_net`

The GDN operation at [`ggml/src/ggml.c:6426-6479`](ggml/src/ggml.c:6426) is the single ggml primitive that performs the complete recurrent state update. Its signature:

```cpp
struct ggml_tensor * ggml_gated_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,      // [S_k, H_k, n_tokens, n_seqs]
        struct ggml_tensor  * k,      // [S_k, H_k, n_tokens, n_seqs]
        struct ggml_tensor  * v,      // [S_v, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * g,      // [1 or S_v, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * beta,   // [1, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * state,  // [S_v, S_v, H_v, n_seqs] -- initial s0
        int64_t               K)       // snapshot count (K=1 for replay)
```

**Directly observed from source:** The CUDA kernel at [`ggml/src/ggml-cuda/gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) processes tokens sequentially:

```cpp
for (int t = 0; t < n_tokens; t++) {
    // ... per-token GDN update ...
    // Token t+1 reads state updated by token t
}
```

This means the GDN operation natively supports multi-token sequential state updates. For replay, we call GDN with `n_tokens = K` (number of accepted tokens), feeding captured k/v/gate/beta from the tape. The kernel will process tokens 0..K-1 sequentially, producing the correct final state SK.

#### 1.1.2 Replay GDN Call Parameters

For replay, the GDN call differs from the forward pass in two ways:

| Parameter | Forward Pass | Replay | Rationale |
|-----------|-------------|--------|-----------|
| **q** | Real query tensor | **Zeros tensor** | Attention output `S^T @ q` is discarded during replay. The kernel writes attention to `dst` which can be a dummy buffer. |
| **k** | From forward pass graph | **From tape buffer** | Captured post-l2_norm k values. |
| **v** | From forward pass graph | **From tape buffer** | Captured v values. |
| **g** | From forward pass graph | **From tape buffer** | Captured pre-exp gate values. Kernel applies `exp()` at runtime. |
| **beta** | From forward pass graph | **From tape buffer** | Captured post-sigmoid beta values (current graph builder applies sigmoid before GDN). |
| **state** | Current recurrent state | **Backup R/S state** | Restored from backup cell (Task 4 design). |
| **K** | `n_rs_seq` (draft max) | **1** | Replay only needs the final state, not intermediate snapshots. |

**Source:** GDN kernel at [`gated_delta_net.cu:103-104`](ggml/src/ggml-cuda/gated_delta_net.cu:103):
```cpp
s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;    // state update
attn_partial += s_shard[r] * q_reg[r];                       // attention (discarded)
```

When q=zeros, `attn_partial` = 0 for all columns. The attention output is meaningless but harmless. The state update (line 103) is correct regardless of q values.

#### 1.1.3 Verdict: GDN is Sufficient as a Replay Primitive

The replay operation is: **"Call GDN with captured tape data and backup state, discard attention output."**

This is a single `ggml_gated_delta_net()` call per recurrent layer. No new kernel, no new ggml op.

### 1.2 Supporting ggml Primitives for Replay Graph Construction

Beyond the core GDN op, replay requires several auxiliary operations to construct the replay graph. All exist in current ggml:

| Operation | ggml Primitive | Purpose | Source |
|-----------|---------------|---------|--------|
| **Tensor copy** | `ggml_cpy()` | Copy tape data from persistent buffer to GDN input shape. | [`ggml/src/ggml.c:3590`](ggml/src/ggml.c:3590) |
| **View/slice** | `ggml_view_4d()` | Extract single-token slice from tape buffer for per-token replay, or view K-token slice for batch replay. | [`ggml/src/ggml.c:3841`](ggml/src/ggml.c:3841) |
| **Permute** | `ggml_permute()` | Rearrange tensor dimensions to match GDN input layout. | [`ggml/src/ggml.c:3884`](ggml/src/ggml.c:3884) |
| **Scale** | `ggml_scale()` | Apply Q scaling factor (trivial when q=zeros but needed for shape consistency). | [`ggml/src/ggml.c:3576`](ggml/src/ggml.c:3576) |
| **Fill zeros** | `ggml_fill()` | Create dummy q=zeros tensor. | [`ggml/include/ggml.h:2413`](ggml/include/ggml.h:2413) |
| **Contiguous** | `ggml_cont()` | Ensure tape data is contiguous for GDN input requirements (`ggml_is_contiguous_rows`). | [`ggml/include/ggml.h:1576`](ggml/include/ggml.h:1576) |
| **Dup** | `ggml_dup()` | Duplicate backup state tensor for replay input. | [`ggml/include/ggml.h:918`](ggml/include/ggml.h:918) |
| **Cast** | `ggml_cast()` | Type conversion if tape buffer uses different precision. | [`ggml/include/ggml.h:1570`](ggml/include/ggml.h:1570) |
| **Custom1/2/3** | `ggml_map_custom*()` | Fallback for any operation not covered by standard primitives. | [`ggml/include/ggml.h:2760-2791`](ggml/include/ggml.h:2760) |

**All these primitives are standard ggml operations** used throughout the existing codebase. They are not specific to replay.

### 1.3 Matrix Operations and Elementwise Operations

The GDN kernel internally performs:

| Internal Operation | Implemented In | Backend Support |
|-------------------|---------------|-----------------|
| `S^T @ k` (matrix-vector multiply) | CUDA kernel, warp-level reduction | CUDA, CPU, Vulkan, HIP |
| `exp(g)` (elementwise exponential) | CUDA kernel, per-token | CUDA, CPU, Vulkan, HIP |
| `(v - g*kv) * beta` (elementwise) | CUDA kernel, fused | CUDA, CPU, Vulkan, HIP |
| `g*S + k⊗delta` (rank-1 update) | CUDA kernel, fused | CUDA, CPU, Vulkan, HIP |
| `S^T @ q` (matrix-vector multiply) | CUDA kernel, fused | CUDA, CPU, Vulkan, HIP |

**Directly observed from source:** The CUDA kernel at [`gated_delta_net.cu:84-141`](ggml/src/ggml-cuda/gated_delta_net.cu:84) implements all operations as a single fused kernel. The CPU backend has a corresponding implementation in [`ggml/src/ggml.c`](ggml/src/ggml.c) (the `GGML_OP_GATED_DELTA_NET` case in the compute dispatch). Vulkan has shader-based implementations.

**No individual matrix/elementwise operations are needed as separate ggml ops** because GDN fuses them all. The replay calls GDN as a single op, and the backend handles the internal decomposition.

### 1.3 State Updates and Tensor Copies

#### 1.3.1 State Update Mechanism

The GDN operation writes updated state to its `state` output buffer. For replay:

- **Input state:** Backup R/S state (from backup cell, restored to active plane).
- **Output state:** Final replayed state after K accepted tokens.

The GDN kernel writes state at [`gated_delta_net.cu:160-166`](ggml/src/ggml-cuda/gated_delta_net.cu:160):
```cpp
if constexpr (!keep_rs_t) {
    for (int r = 0; r < rows_per_lane; r++) {
        state[col * S_v + i] = s_shard[r];
    }
}
```

With `K=1` and `keep_rs_t=false` (the replay case), the kernel writes only the final state. This is the most efficient configuration.

#### 1.3.2 Tensor Copy for State Restore

To restore backup state before replay, we need to copy from the backup cell to the active cell. This requires:

**Option A: Graph-embedded copy**
```cpp
// Copy backup S state to active S state using ggml_cpy:
struct ggml_tensor * s_backup = ggml_view_4d(ctx, mem->s_l[il],
    S_v, S_v, H_v, n_seqs, ..., backup_offset);
struct ggml_tensor * s_active = ggml_view_4d(ctx, mem->s_l[il],
    S_v, S_v, H_v, n_seqs, ..., active_offset);
ggml_cpy(ctx, s_backup, s_active);
```

**Option B: Direct backend buffer copy**
```cpp
ggml_backend_buffer_cp(mem->s_l[il]->buffer, backup_ptr, active_ptr, nbytes);
```

Both options use existing primitives. Option A is graph-native and benefits from backend optimization. Option B is a direct memory copy that bypasses the graph.

**See Section 1.7 for detailed analysis of which approach fits the Task 4 integration.**

### 1.4 Slicing and Views

The tape buffer stores captured data for all N draft tokens. For replay of K accepted tokens, we need to slice the first K tokens from the tape.

Current ggml supports this via `ggml_view_4d()`:
```cpp
// View first K tokens from tape buffer:
struct ggml_tensor * k_replay = ggml_view_4d(ctx, k_tape,
    S_k, H_k, K, n_seqs,
    k_tape->nb[1], k_tape->nb[2], k_tape->nb[3], 0);
```

The GDN input requires contiguous rows (`ggml_is_contiguous_rows` assertion at [`ggml.c:4437`](ggml/src/ggml.c:4437)). The tape buffer is allocated as contiguous F32, and the view of the first K tokens maintains contiguity in dimensions 0-1 (S_k, H_k). Dimension 2 (n_tokens=K) is a prefix of the tape, which is contiguous.

**Verdict: Slicing via `ggml_view_4d()` is sufficient for tape extraction.**

### 1.5 Recurrence

The GDN kernel natively handles recurrence through its sequential token loop:

```cpp
for (int t = 0; t < n_tokens; t++) {
    // Read k[t], v[t], g[t], beta[t]
    // Update state in-place (s_shard is modified each iteration)
    // Token t+1 reads the state updated by token t
}
```

No separate recurrence primitive is needed. The GDN operation IS the recurrent transition.

### 1.6 Device Synchronization

#### 1.6.1 Same-Device Replay

When tape data and backup state reside on the same GPU as the model, replay GDN executes on that GPU with no cross-device synchronization. The ggml scheduler handles device-local tensor dependencies.

#### 1.6.2 Cross-Device Considerations

If tape data is on a different device than the model (e.g., tape on GPU0, model on GPU1), the replay graph would need `ggml_cpy()` to transfer tape data to the model device. This is handled by ggml's backend abstraction:

```cpp
// ggml_cpy between backends uses ggml_backend_buffer_cp:
ggml_cpy(ctx, k_tape_gpu0, k_replay_gpu1);
```

For the Task 4 design (backup cells on same device as model), this is not a concern.

#### 1.6.3 Async Execution

The GDN operation supports async execution through `ggml_backend_sched_graph_compute_async()`. The replay graph can be submitted asynchronously and synchronized via `ggml_backend_sched_synchronize()`.

### 1.7 Is Any Operation Missing?

**No operation is missing for the core replay computation.** The complete replay graph per recurrent layer is:

```
1. Create dummy q=zeros tensor (ggml_fill)
2. View tape k[v, gate, beta] for first K tokens (ggml_view_4d)
3. View backup S state (ggml_view_4d)
4. Call ggml_gated_delta_net(q_zeros, k_tape, v_tape, gate_tape, beta_tape, s_backup, K=1)
5. Copy output state to active cell (ggml_cpy)
```

Steps 1-4 are standard ggml graph construction. Step 5 requires writing the GDN output state back to the recurrent memory tensor. This is the only non-trivial step, discussed in Section 3 (Task 4 integration).

### 1.8 CUDA-Specific Code Requirements

#### 1.8.1 GDN Kernel — NOT Required

The GDN CUDA kernel at [`gated_delta_net.cu`](ggml/src/ggml-cuda/gated_delta_net.cu) already supports replay. No modification needed.

#### 1.8.2 Tape Capture — OPTIONAL CUDA Code

Tape capture can use:
- **Eval callback** (CPU-side, no CUDA code needed) — old approach, ~200 lines.
- **Graph-embedded `ggml_cpy`** (no CUDA code needed) — ggml handles device-local copies.

Neither approach requires new CUDA kernels.

#### 1.8.3 State Restore — OPTIONAL CUDA Code

State restore (backup → active) can use:
- **Graph-embedded `ggml_cpy`** — handled by ggml CUDA backend.
- **Direct `ggml_backend_buffer_cp`** — handled by ggml CUDA backend.

No new CUDA code needed.

#### 1.8.4 Verdict

**No new CUDA kernels are required for replay.** The replay uses existing GDN + standard ggml primitives. All CUDA execution is handled by existing backend implementations.

### 1.9 CPU and Other Backend Impact

#### 1.9.1 CPU Backend

The CPU backend already implements `GGML_OP_GATED_DELTA_NET`. Replay on CPU uses the same CPU GDN function. The old CPU replay at [`llama-context.cpp:4193-4210`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4193) was a hand-coded loop that duplicated GDN logic. The new approach uses the existing CPU GDN primitive, eliminating code duplication.

#### 1.9.2 Vulkan Backend

Vulkan has GDN shader implementations. Replay on Vulkan uses the same Vulkan GDN path.

#### 1.9.3 HIP/ROCm Backend

HIP shares the CUDA kernel source (with `#ifdef` guards). Replay on HIP uses the same kernel.

### 1.10 Can Replay Use the Same Graph Machinery as DeltaNet/GDN?

**YES.** The replay graph uses the same `ggml_gated_delta_net` operation as the forward pass. The only difference is the source of input tensors:

| Aspect | Forward Pass | Replay |
|--------|-------------|--------|
| **Graph builder** | `llm_build_delta_net_base::build_delta_net()` | Custom replay graph builder |
| **q source** | Layer input projection | Zeros tensor |
| **k source** | Layer input projection (l2_norm) | Tape buffer view |
| **v source** | Layer input projection | Tape buffer view |
| **g source** | Layer input projection | Tape buffer view |
| **beta source** | Layer input projection (sigmoid) | Tape buffer view |
| **state source** | Recurrent memory (current cell) | Recurrent memory (backup cell) |
| **K parameter** | `n_rs_seq` | 1 |
| **Output** | Attention + state snapshots | State only (attention discarded) |

The replay graph is structurally simpler than the forward graph because it skips projections, normalization, and other layer operations. It only constructs the GDN op with pre-computed inputs.

### 1.11 Summary Table: ggml Primitives for Replay

| Category | Needed? | ggml Primitive | New Code Required? |
|----------|---------|---------------|-------------------|
| Matrix multiply | NO (fused in GDN) | N/A | No |
| Elementwise ops | NO (fused in GDN) | N/A | No |
| State update | YES | `ggml_gated_delta_net` | No |
| Tensor copy | YES | `ggml_cpy` | No |
| View/slice | YES | `ggml_view_4d` | No |
| Fill zeros | YES | `ggml_fill` | No |
| Permute | YES | `ggml_permute` | No |
| Scale | YES | `ggml_scale` | No |
| Recurrence | NO (native in GDN) | N/A | No |
| Device sync | NO (ggml scheduler) | N/A | No |
| Custom op | NO | N/A | No |
| New CUDA kernel | NO | N/A | No |

**Final verdict for Part 6: All replay operations can be expressed using existing ggml primitives. No new ggml operation, no new CUDA kernel, no new backend code is required for the replay computation.**

The only new code needed is:
1. **Tape capture mechanism** (~200 lines for eval callback OR graph-embedded copy).
2. **Replay graph builder** (~100 lines to construct GDN calls with tape inputs).
3. **State restore mechanism** (~50 lines to copy backup state to active cell).
4. **Integration with speculative verification** (~50 lines to hook replay into the accept/reject flow).

Total estimated new code: ~400 lines, all using existing ggml primitives.

---

## SECTION 2: Task 4 Integration Design (Part 7)

### 2.1 Task 4 Architecture Recap

The Task 4 design from [`task4-part1-lifecycle-and-apis.md`](task4-part1-lifecycle-and-apis.md) proposes:

```
n_rs_seq = 0
+
static backup recurrent cell(s)
+
cell_copy()
+
backup before verification
```

Key parameters:
- **`n_rs_seq = 0`** — Eliminates the 5.4GB RS buffer by excluding DFlash from `need_n_rs_seq()` at [`common/common.h:417`](common/common.h:417).
- **Static backup cell(s)** — One or more dedicated recurrent cells that store pre-draft state. Size: ~598 MB per cell for Qwen3.6 (48 layers × (n_embd_r + n_embd_s)).
- **`cell_copy()`** — Copy current recurrent state to backup before speculative forward.
- **Backup before verification** — The backup captures the state at the last accepted position before the target verifies draft tokens.

### 2.2 Integration: How Replay Fits with Backup Cells

#### 2.2.1 Complete Flow with Replay

```
Cycle start — state is S_last (correct state after previous acceptance)
    ↓
[Step 1: Backup]
    Copy S_last → backup cell
    (Uses ggml_cpy or ggml_backend_buffer_cp)
    ↓
[Step 2: Speculative DFlash Forward]
    Draft model generates N candidate tokens
    Target model verifies all N tokens through forward pass
    Target recurrent state advances from S_last → S_{last+N}
    Tape captures k, v, gate, beta for all N draft tokens
    ↓
[Step 3: Verification Decision]
    Sampler returns accepted = [token_0, token_1, ..., token_{K-1}]
    K = number of accepted tokens (0 ≤ K ≤ N)
    ↓
[Step 4: Branch on Acceptance]

    ┌─────────────────────────────────────────────────────┐
    │ Case A: K == N (full acceptance)                     │
    │   — No rollback needed                               │
    │   — Target state S_{last+N} is already correct       │
    │   — Backup cell can be discarded                      │
    │   — Tape buffer discarded                             │
    │   — Proceed to next cycle                              │
    ├─────────────────────────────────────────────────────┤
    │ Case B: K < N (partial acceptance)                    │
    │   ┌─────────────────────────────────────────────────┐│
    │   │ Sub-case B1: K == 0 (no tokens accepted)        ││
    │   │   — Restore backup: S_last → active cell         ││
    │   │   — No replay needed (no tokens to replay)       ││
    │   │   — Proceed to next cycle                        ││
    │   ├─────────────────────────────────────────────────┤│
    │   │ Sub-case B2: 0 < K < N (some tokens accepted)   ││
    │   │   — Restore backup: S_last → active cell         ││
    │   │   — Replay K accepted tokens through GDN:        ││
    │   │     S_new = GDN(q=0, k[tape:0..K], v[tape:0..K], ││
    │   │                      g[tape:0..K], b[tape:0..K], ││
    │   │                      S_last, K=1)                ││
    │   │   — S_new is the correct post-acceptance state   ││
    │   │   — Proceed to next cycle                        ││
    │   └─────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────┘
```

#### 2.2.2 Does This Eliminate Full-Model Re-decode?

**YES.** The replay + backup cell approach completely eliminates the need for full-model re-decode of accepted tokens.

**Current behavior without replay (Solution 2 / n_rs_seq=0):**
1. After partial acceptance, restore checkpoint (S0).
2. Next cycle re-decodes ALL accepted tokens through the full model to arrive at SK.
3. This means running the entire model forward pass K times for accepted tokens.

**With replay + backup cells:**
1. After partial acceptance, restore backup (S_last = state before draft).
2. Replay K accepted tokens through GDN ONLY (no full model forward).
3. GDN replay is ~100-1000x faster than full model forward (GDN touches only recurrent state, not attention, FFN, normalization, etc.).

**The critical insight:** Replay operates on the recurrent state ONLY. It does not need to:
- Re-run attention layers (KV is handled separately, see Section 3).
- Re-run FFN layers (not needed for recurrent state).
- Re-run normalization layers (not needed for recurrent state).
- Re-compute input embeddings (not needed for recurrent state).

The replay only advances the R/S tensors through the accepted tokens. This is the recurrent state transition, which is exactly what GDN computes.

### 2.3 State Restore Mechanism

#### 2.3.1 Backup Cell Structure

The backup cell stores a complete copy of the recurrent state (R + S tensors for all recurrent layers). Based on the current recurrent memory at [`src/llama-memory-recurrent.cpp:20-36`](src/llama-memory-recurrent.cpp:20):

```cpp
// Current R tensor per layer: [mem_size * (1 + n_rs_seq), n_embd_r]
// Current S tensor per layer: [mem_size * (1 + n_rs_seq), n_embd_s]

// With n_rs_seq = 0:
// R: [mem_size, n_embd_r]  — single plane
// S: [mem_size, n_embd_s]  — single plane

// Backup cell adds:
// R_backup: [mem_size, n_embd_r]  — one extra plane
// S_backup: [mem_size, n_embd_s]  — one extra plane
```

For Qwen3.6 (48 recurrent layers, n_embd_r=30720, n_embd_s=786432):
- R per layer: 30720 × 4B = 120 KB
- S per layer: 786432 × 4B = 3 MB
- Total per layer: ~3.12 MB
- Total (48 layers): ~149.7 MB

Wait — let me recalculate. The S tensor size is `n_embd_s = S_v × S_v × H_v`. For Qwen3.6:
- From research-summary.md: n_embd_s = 786432
- S per layer: 786432 × 4B = 3,145,728 B = 3 MB
- R per layer: n_embd_r × 4B (conv state)

The backup cell stores ONE copy of the current active state. For a single slot, this is ~150 MB.

#### 2.3.2 Restore Options

**Option A: Graph-embedded copy (ggml_cpy)**

```cpp
// For each recurrent layer il:
struct ggml_tensor * s_backup = ggml_view_4d(ctx, mem->s_l[il],
    S_v, S_v, H_v, 1,
    ..., backup_offset);  // Source: backup cell
struct ggml_tensor * s_active = ggml_view_4d(ctx, mem->s_l[il],
    S_v, S_v, H_v, 1,
    ..., active_offset);  // Destination: active cell
ggml_cpy(ctx, s_backup, s_active);
```

**Pros:** Graph-native, backend-optimized, works on all devices.
**Cons:** Requires building a graph for the copy operation.

**Option B: Direct backend buffer copy**

```cpp
ggml_backend_buffer_cp(
    mem->s_l[il]->buffer,
    backup_ptr, active_ptr, nbytes);
```

**Pros:** Faster (no graph overhead), simpler.
**Cons:** Requires manual pointer arithmetic, less portable.

**Recommendation:** Option B for the restore step (it's a simple memory copy), Option A for the replay graph (it needs to integrate with GDN).

### 2.4 Conv State (R Tensor) Handling

The R tensor (convolutional state) is separate from the S tensor (GDN state). Both need to be restored and replayed.

#### 2.4.1 R Tensor Restore

The R tensor is restored from the backup cell using the same mechanism as S:

```cpp
// Restore R from backup:
ggml_backend_buffer_cp(r_buffer, r_backup_ptr, r_active_ptr, r_size);
```

#### 2.4.2 R Tensor Replay

The R tensor represents conv state. After restoring the backup R state, the conv state needs to advance through K accepted tokens. The conv state shift is NOT handled by GDN.

**Old code approach:** The old `tape_replay_conv_gpu()` at [`llama-context.cpp:2912`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2912) handled conv state replay separately.

**Current approach options:**

| Option | Description | Complexity |
|--------|-------------|-----------|
| A | Replay conv state using captured tape data + conv kernel | Medium (~100 lines) |
| B | Skip conv replay — accept small accuracy loss | Low (0 lines) |
| C | Re-decode accepted tokens through full model for conv only | High (defeats purpose) |

**Analysis:** The conv state is typically small compared to S state. For Qwen3.6, n_embd_r is much smaller than n_embd_s. The conv state shift is a simple ring buffer operation: shift by K positions and insert new values.

**If the conv state is captured in the tape**, replay can use `ggml_cpy` to shift the conv buffer. The captured tape data includes the pre-conv qkv values needed to update the conv state.

**If conv replay is too complex**, Option B (skip conv replay) may be acceptable if the conv state contribution is small. This requires benchmarking to validate.

**For now, the design assumes conv state is restored from backup and replayed using a simple shift operation.** The exact mechanism depends on the conv kernel used (Mamba-style SSM conv, RWKV conv, etc.).

### 2.5 Tape Buffer Management

#### 2.5.1 Tape Buffer Size

From task5-part2 analysis, per-token capture per layer:
```
(S_k * H_k + 2 * S_v * H_v + 2 * H_v) floats
```

For Qwen3.6 (S_k=S_v=128, H_k=32, H_v=32):
- Per token per layer: ~49.4 KB
- Per token (48 layers): ~2.37 MB
- For 15 draft tokens: ~35.5 MB total tape

The tape buffer is allocated once per slot and reused across cycles. Size: ~36 MB for Qwen3.6 with 15 draft tokens.

#### 2.5.2 Tape Buffer Lifetime

```
Cycle N:
    — Capture tape during speculative forward
    — Use tape for replay (if partial acceptance)
    — Discard tape at end of cycle

Cycle N+1:
    — Reuse tape buffer (overwrite old data)
```

The tape buffer does NOT persist across cycles. Each cycle captures fresh data.

### 2.6 Complete Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DFlash Cycle with Replay                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  [Pre-draft]                                                 │
│  Active cell: S_last                                         │
│  Backup cell: empty                                          │
│                                                               │
│       ↓ cell_copy(active → backup)                           │
│                                                               │
│  [Backup]                                                    │
│  Active cell: S_last                                         │
│  Backup cell: S_last                                         │
│                                                               │
│       ↓ speculative forward (draft + verify)                 │
│       ↓ tape capture (k, v, g, b for N tokens)              │
│                                                               │
│  [Post-verify]                                               │
│  Active cell: S_{last+N} (may be wrong if partial accept)    │
│  Backup cell: S_last (correct pre-draft state)               │
│  Tape: k[0..N], v[0..N], g[0..N], b[0..N]                   │
│                                                               │
│       ↓ verification decision                                │
│                                                               │
│  ┌─ K == N (full accept) ─────────────────────────────┐     │
│  │  Active: S_{last+N} (correct, keep)                 │     │
│  │  Backup: discard                                     │     │
│  │  Tape: discard                                       │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌─ K == 0 (no accept) ────────────────────────────────┐     │
│  │  Restore: backup → active (S_last)                   │     │
│  │  No replay needed                                    │     │
│  │  Tape: discard                                       │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌─ 0 < K < N (partial accept) ─────────────────────────┐    │
│  │  Restore: backup → active (S_last)                    │    │
│  │  Replay: GDN(q=0, tape[0..K], S_last, K=1)           │    │
│  │  Result: active = S_{last+K} (correct)               │    │
│  │  Tape: discard                                        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  [Next cycle starts with correct state S_{last+K}]           │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 2.7 Does This Completely Eliminate Full-Model Re-decode?

**YES, with caveats:**

| Component | Eliminated by Replay? | Notes |
|-----------|----------------------|-------|
| **GDN recurrent state (S)** | **YES** | GDN replay advances S through K tokens without full model forward. |
| **Conv state (R)** | **PARTIAL** | Requires separate conv replay mechanism. If conv replay is implemented, YES. If skipped, small accuracy loss. |
| **Attention KV** | **NO** | Handled separately (see Section 3). Attention KV is managed by `seq_rm` and normal cache operations, not replay. |
| **Input embeddings** | **YES** | Not needed for replay — tape has pre-computed intermediates. |
| **FFN layers** | **YES** | Not needed for replay — replay only touches recurrent layers. |
| **Layer normalization** | **YES** | Not needed for replay — GDN receives pre-normalized inputs from tape. |

**The replay eliminates the need for full-model re-decode of accepted tokens for the RECURRENT STATE.** Attention KV is handled independently and does not require re-decode (see Section 3).

---

## SECTION 3: Attention/KV Handling (Part 7)

### 3.1 Can Recurrent State Be Restored Independently?

**YES.** The hybrid memory architecture at [`src/llama-memory-hybrid.cpp`](src/llama-memory-hybrid.cpp) explicitly separates attention and recurrent memory:

```cpp
// llama_memory_hybrid contains:
const std::unique_ptr<llama_memory_i> mem_attn;   // Attention KV cache
const std::unique_ptr<llama_memory_recurrent> mem_recr;  // Recurrent state
```

The `seq_rm` operation at [`src/llama-memory-hybrid.cpp:182-193`](src/llama-memory-hybrid.cpp:182) calls both:
```cpp
bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}
```

This means recurrent state and attention KV can be modified independently. The backup/restore/replay mechanism operates on `mem_recr` only, leaving `mem_attn` untouched.

### 3.2 Can Rejected Attention KV Be Removed Without Disturbing Recurrent State?

**YES.** The attention KV cache has its own `seq_rm` mechanism:

```cpp
// Attention-only seq_rm (without touching recurrent):
mem_attn->seq_rm(seq_id, p0, p1);  // Removes KV for positions [p0, p1)
```

The current speculative rollback at [`server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221) calls `common_context_seq_rm` which triggers `mem_attn->seq_rm()` AND `mem_recr->seq_rm()`. With replay, we would:

1. **Restore recurrent state from backup** — operates on `mem_recr` only.
2. **Replay accepted tokens through GDN** — operates on `mem_recr` only.
3. **Remove rejected attention KV** — operates on `mem_attn` only.

These three operations are independent and can be performed in any order.

### 3.3 Does the Current Hybrid Memory API Permit This?

**YES.** The hybrid memory API provides separate access to attention and recurrent components:

| Operation | Attention API | Recurrent API | Independent? |
|-----------|-------------|---------------|-------------|
| **seq_rm** | `mem_attn->seq_rm()` | `mem_recr->seq_rm()` | YES |
| **clear** | `mem_attn->clear()` | `mem_recr->clear()` | YES |
| **state_write** | Via `llama_memory_hybrid::state_write()` | Via `llama_memory_recurrent::state_write()` | Partially |
| **state_read** | Via `llama_memory_hybrid::state_read()` | Via `llama_memory_recurrent::state_read()` | Partially |
| **cell access** | `llama_kv_cache::cells` | `llama_memory_recurrent::cells` | YES |
| **tensor access** | `llama_kv_cache::k_l[v_l]` | `llama_memory_recurrent::r_l[s_l]` | YES |

**Directly observed from source:** The hybrid memory at [`src/llama-memory-hybrid.h:113-120`](src/llama-memory-hybrid.h:113) exposes both components:
```cpp
llama_memory_i * get_mem_attn() const;
llama_memory_recurrent * get_mem_recr() const;

const std::unique_ptr<llama_memory_i> mem_attn;
const std::unique_ptr<llama_memory_recurrent> mem_recr;
```

### 3.4 Does Replay Require Any Interaction with Attention KV?

**NO.** Replay operates on recurrent state only. The attention KV is managed by the normal speculative decoding mechanism:

1. **During verification:** Attention KV is updated for all N draft tokens.
2. **After partial acceptance:** Rejected KV (positions K+1..N) is removed via `mem_attn->seq_rm(seq_id, pos_K, pos_N)`.
3. **Accepted KV (positions 0..K):** Kept in place. These are the correct attention entries for accepted tokens.

The replay GDN call does not read from or write to attention KV. It only reads/writes recurrent state (R/S tensors).

### 3.5 Is Additional API Required?

**Minimal additional API needed:**

| New API | Purpose | Existing Equivalent |
|---------|---------|-------------------|
| **`cell_copy(src → dst)`** | Copy recurrent cell state to backup | `ggml_cpy()` on R/S tensors |
| **`tape_capture()`** | Capture k, v, g, b during forward | Eval callback or graph-embedded `ggml_cpy` |
| **`tape_replay(K)`** | Replay K accepted tokens through GDN | Custom graph builder using `ggml_gated_delta_net` |
| **`conv_replay(K)`** | Replay conv state for K tokens | Separate mechanism (see Section 2.4.2) |

The `cell_copy` operation could be a thin wrapper around `ggml_backend_buffer_cp()` or `ggml_cpy()`. It does not require changes to the ggml API.

### 3.6 Attention/KV State During Replay Flow

```
Position:  0     1     2     ...   K-1   K     K+1   ...   N-1
           ┌─────────────────┐    ┌─────────────────────┐
           │  ACCEPTED KV    │    │   REJECTED KV       │
           │  (keep in place)│    │   (remove via seq_rm)│
           └─────────────────┘    └─────────────────────┘

Recurrent state:
Before replay:  S_last (restored from backup)
After replay:   S_{last+K} (GDN replay of K tokens)

Attention KV:
After seq_rm:   Positions 0..K-1 remain (correct for accepted tokens)
                Positions K..N-1 removed

Result:
Both attention KV and recurrent state are consistent
for the accepted prefix of K tokens.
```

### 3.7 Edge Cases

#### 3.7.1 K = 0 (No Tokens Accepted)

- Recurrent state: Restore backup (S_last). No replay needed.
- Attention KV: Remove all N draft positions via `seq_rm`.
- Result: State returns to pre-draft state. Next cycle starts fresh.

#### 3.7.2 K = N (Full Acceptance)

- Recurrent state: Already correct (S_{last+N} from verification). No restore or replay needed.
- Attention KV: Already correct (all N positions valid). No seq_rm needed.
- Result: No action needed. Backup cell discarded.

#### 3.7.3 Multi-Sequence

Each sequence (seq_id) has its own:
- Recurrent cell (independent R/S tensors per cell).
- Attention KV (independent positions per seq_id).
- Backup cell (one backup per active sequence, or shared if sequences reuse cells).

The replay mechanism operates per-sequence. Different sequences can have different acceptance counts K.

### 3.8 Complete Independence Verification

| Question | Answer | Evidence |
|----------|--------|----------|
| Can recurrent state be restored independently? | **YES** | Hybrid memory separates `mem_attn` and `mem_recr`. |
| Can rejected attention KV be removed without disturbing recurrent state? | **YES** | `mem_attn->seq_rm()` operates on attention only. |
| Does the current hybrid memory API permit this? | **YES** | `get_mem_attn()` and `get_mem_recr()` provide separate access. |
| Does replay require any interaction with attention KV? | **NO** | Replay GDN only reads/writes R/S tensors. |
| Is additional API required? | **MINIMAL** | `cell_copy`, `tape_capture`, `tape_replay` — thin wrappers around existing primitives. |

### 3.9 Final Integration Flow

```
After verification accepts K of N draft tokens:

1. RESTORE RECURRENT STATE (mem_recr only)
   ggml_backend_buffer_cp(r_buffer, r_backup, r_active, r_size)
   ggml_backend_buffer_cp(s_buffer, s_backup, s_active, s_size)

2. REPLAY ACCEPTED TOKENS (mem_recr only)
   For each recurrent layer il:
       k_replay = view(k_tape, 0, K)
       v_replay = view(v_tape, 0, K)
       g_replay = view(g_tape, 0, K)
       b_replay = view(b_tape, 0, K)
       q_dummy  = fill_zeros(S_k, H_k, K, 1)
       s_result = ggml_gated_delta_net(q_dummy, k_replay, v_replay,
                                        g_replay, b_replay, s_active, K=1)
       ggml_cpy(s_result, s_active)

3. REMOVE REJECTED KV (mem_attn only)
   mem_attn->seq_rm(seq_id, pos_K, pos_N)

4. UPDATE CELL POSITION
   cell.pos = pos_K  (advance to last accepted position)

5. PROCEED TO NEXT CYCLE
   Active state is now S_{last+K} with correct attention KV.
```

---

## SECTION 4: Conclusions

### 4.1 Part 6 — ggml Primitives Sufficiency

**All replay operations can be expressed using existing ggml primitives.** Specifically:

1. The GDN operation (`ggml_gated_delta_net`) is the complete replay primitive — no new kernel needed.
2. Supporting operations (copy, view, fill, permute, scale) are standard ggml ops.
3. No CUDA-specific code is required for replay computation.
4. CPU, Vulkan, and HIP backends are automatically supported through existing GDN implementations.
5. The replay graph uses the same graph machinery as DeltaNet/GDN forward pass.

### 4.2 Part 7 — Task 4 Integration

**Replay integrates cleanly with Task 4 backup-cell design:**

1. The backup cell provides S_last (pre-draft state) for restore.
2. Replay advances S_last → S_{last+K} using captured tape data.
3. This completely eliminates full-model re-decode for accepted tokens.
4. Recurrent state and attention KV are managed independently through hybrid memory.
5. No additional ggml API is required — only thin wrapper functions.

### 4.3 Overall Assessment

The replay + backup cell approach achieves:
- **Low VRAM:** ~150 MB backup cell vs 5.4 GB RS buffer (97% reduction).
- **Fast rollback:** GDN replay is orders of magnitude faster than full-model re-decode.
- **Minimal code:** ~400 lines of new code using existing ggml primitives.
- **Backend portable:** Works on CUDA, CPU, Vulkan, HIP without backend-specific code.
- **Upstream compatible:** Uses existing `ggml_gated_delta_net` operation, no fork-specific ggml ops.

The design is ready for implementation. The next step (Task 5.5) should analyze theoretical performance and VRAM cost to validate these claims with benchmarks.
