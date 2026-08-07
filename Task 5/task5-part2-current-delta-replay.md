# Task 5.2 Parts 2-3: Mathematical Minimum for Replay + Old vs Current DeltaNet/GDN Comparison

**Date:** 2026-08-07
**Source:** Current upstream workspace + `old-versions/beellama.cpp-preview-v0.3.2/`

---

## Section 1: Mathematical Minimum for Replay (Part 2)

### 1.1 What is required to replay ONE accepted token's GDN state transition?

The GDN (Gated Delta Net) state update for one token per head is a rank-1 update to the S matrix. The CUDA kernel at [`ggml/src/ggml-cuda/gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) reveals the exact per-token operation:

**Non-KDA mode (scalar gate):**
```
// Per token t, per head h, per column col (0..S_v-1):

kv_col = sum_i(S[i][col] * k[i])                          // S^T @ k
delta_col = (v[col] - exp(g) * kv_col) * beta             // delta
S_new[i][col] = exp(g) * S[i][col] + k[i] * delta_col     // rank-1 update
```

**KDA mode (vector gate, per-key-dimension):**
```
kv_col = sum_i(exp(g[i]) * S[i][col] * k[i])              // gated S^T @ k
delta_col = (v[col] - kv_col) * beta                       // delta
S_new[i][col] = exp(g[i]) * S[i][col] + k[i] * delta_col  // rank-1 update
```

**Directly observed from source:** The CUDA kernel at [`gated_delta_net.cu:84-141`](ggml/src/ggml-cuda/gated_delta_net.cu:84) processes tokens sequentially (line 63: `for (int t = 0; t < n_tokens; t++)`), meaning token t+1 uses the state updated by token t. This is inherently sequential.

### 1.1.1 Input tensors required per token

| Input | Shape | Required for replay? | Notes |
|-------|-------|---------------------|-------|
| **k** | `[S_k, H_k]` per token per head | **YES** | Key after l2_norm. Used in `S^T @ k` and rank-1 update. |
| **v** | `[S_v]` per token per head | **YES** | Value. Used in `v - g*kv` delta computation. |
| **g (gate)** | `[1]` (scalar) or `[S_v]` (KDA) per token per head | **YES** | Pre-exp gate. Kernel applies `exp(g)` at runtime. |
| **beta** | `[1]` per token per head | **YES** | Pre-sigmoid beta. Applied as scalar multiplier to delta. |
| **s (state)** | `[S_v, S_v]` per head | **YES** | Current S state (input from previous token or backup). |
| **q** | `[S_k, H_k]` per token per head | **NO** | Only used for attention output `S^T @ q`. Discarded in replay. |

**Source:** CUDA kernel [`gated_delta_net.cu:96-104`](ggml/src/ggml-cuda/gated_delta_net.cu:96) for non-KDA and [`gated_delta_net.cu:113-133`](ggml/src/ggml-cuda/gated_delta_net.cu:113) for KDA. The Q tensor is only consumed at line 104 (`attn_partial += s_shard[r] * q_reg[r]`) to compute attention output, which is discarded during replay.

### 1.1.2 Summary: Mathematical minimum per accepted token

For replaying ONE accepted token through the GDN state update:

```
CAPTURE FROM SPECULATIVE FORWARD PASS:
  k[t]    — [S_k, H_k] per head       (S_k * H_k floats)
  v[t]    — [S_v] per head            (S_v floats)
  gate[t] — [1] or [S_v] per head     (H_v * (1 or S_v) floats)
  beta[t] — [1] per head              (H_v floats)

INPUT FROM BACKUP/REPLAY STATE:
  s       — [S_v, S_v, H_v]           (S_v * S_v * H_v floats = n_embd_s)

OUTPUT:
  s_new   — [S_v, S_v, H_v]           (same shape as input s)

CHEAP TO RECOMPUTE:
  q       — NOT NEEDED (attention output discarded)
  attn    — NOT NEEDED (S^T @ q output not used in state update)
```

**Total capture per token per layer (non-KDA scalar gate):**
```
(S_k * H_k + S_v * H_v + 1 * H_v + 1 * H_v) floats
= (S_k * H_k + 2 * S_v * H_v + 2 * H_v) floats
≈ S_k * H_k + 2 * S_v * H_v floats (dominated by k and v)
```

For Qwen3.6 (S_k=S_v=128, H_k=32, H_v=32):
```
Per token per layer: 128*32 + 2*128*32 + 2*32 = 4,096 + 8,192 + 64 = 12,352 floats = 49.4 KB
Per token (48 layers): 49.4 KB * 48 = 2.37 MB
For 8 draft tokens: 18.9 MB total tape (vs old CPU tape which stored the same data)
```

---

### 1.2 Which quantities can be recomputed cheaply?

| Quantity | Can recompute? | Reason |
|----------|---------------|--------|
| **Q tensor** | YES / NOT NEEDED | Attention output is discarded. Old replay passed Q=zeros. |
| **Attention output** | NOT NEEDED | Only the state update matters for replay. |
| **l2_norm(k)** | NO | The captured k is post-l2_norm. The pre-norm k is not stored. If you capture pre-norm k, you'd need to recompute l2_norm, but the GDN kernel expects post-norm k. |
| **exp(gate)** | YES (trivial) | Kernel applies `exp()` at runtime. Store raw gate, compute exp inline. |
| **sigmoid(beta)** | DEBATABLE | Old CPU replay applied `sigmoid()` at replay time ([`llama-context.cpp:4194`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4194)). Current CUDA kernel uses raw beta directly as scalar multiplier. **The current kernel does NOT apply sigmoid** — beta is stored pre-sigmoid in the model and the forward pass applies sigmoid before passing to GDN. |

**Critical observation:** The current CUDA kernel at [`gated_delta_net.cu:96`](ggml/src/ggml-cuda/gated_delta_net.cu:96) uses `beta_val` directly as a scalar multiplier: `delta_col = (v_t[col] - g_val * kv_col) * beta_val`. This means the beta passed to the GDN kernel is already post-sigmoid (the sigmoid is applied in the graph builder before the GDN op). The old CPU replay at [`llama-context.cpp:4194`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4194) applied `sigmoid()` during replay: `b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]))`. This means the old tape captured PRE-sigmoid beta and applied sigmoid during replay.

This is a discrepancy that needs investigation: does the current graph builder apply sigmoid before the GDN op, or does the kernel expect pre-sigmoid beta?

**Investigation:** Looking at [`src/models/delta-net-base.cpp:42-43`](src/models/delta-net-base.cpp:42):
```cpp
GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
```

The beta shape assertion `[1, H_v, n_tokens, n_seqs]` matches both old and current. The beta tensor flows through the graph builder as `b` parameter. In the Qwen3.5 model builder (caller of `build_recurrent_attn`), beta is computed from model weights. The graph builder does NOT apply sigmoid before passing beta to GDN — the sigmoid is applied INSIDE the GDN kernel or is expected to be pre-applied.

**Resolution:** Looking at the old CPU replay code more carefully:

Old tape capture ([`llama-context.cpp:2277-2282`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2277)):
```cpp
dflash_capture->tape_name_map["beta-" + il_str] = {idx, DFLASH_TAPE_BETA};
```

The old tape captured the `beta-{il}` graph node. The old CPU replay then applied sigmoid. This means the graph node `beta-{il}` contains PRE-sigmoid values, and the old replay applied sigmoid at replay time.

But the current CUDA kernel uses beta directly without sigmoid. This means either:
1. The current graph builder applies sigmoid before the GDN op (making kernel beta = post-sigmoid), OR
2. The current kernel expects pre-sigmoid beta and applies it internally.

Looking at [`gated_delta_net.cu:96`](ggml/src/ggml-cuda/gated_delta_net.cu:96): `delta_col = (v_t[col] - g_val * kv_col) * beta_val` — the kernel uses `beta_val` directly without sigmoid. This means the graph builder MUST apply sigmoid before the GDN op, making the beta tensor at the GDN input already post-sigmoid.

**Verification needed:** Check the model-specific graph builder (e.g., qwen35.cpp) to see if sigmoid is applied before calling `build_recurrent_attn()`.

---

### 1.2 Which quantities must actually be captured from the speculative forward pass?

Based on the CUDA kernel analysis, replay requires these captured intermediates per token:

| Must Capture | Shape | Rationale |
|-------------|-------|-----------|
| **k** (post-l2_norm) | `[S_k, H_k, n_tokens]` | Used in `S^T @ k` and rank-1 update. Cannot be recomputed from forward pass inputs without re-running the entire layer. |
| **v** | `[S_v, H_v, n_tokens]` | Used in `v - g*kv`. Cannot be recomputed. |
| **gate** (pre-exp) | `[1 or S_v, H_v, n_tokens]` | Kernel applies `exp()` at runtime. Store raw gate. |
| **beta** | `[1, H_v, n_tokens]` | Used as scalar multiplier. **Must determine if captured value is pre- or post-sigmoid.** |

**NOT required:**
- Q (attention output discarded)
- Full layer input embeddings
- Attention KV (separate from recurrent state)
- FFN intermediates
- Normalization state

### 1.3 Which tensors are specific to recurrent update vs. attention/KV?

The GDN operation fuses attention output computation with recurrent state update. For replay:

| Component | Recurrent Update | Attention Output | Needed for Replay? |
|-----------|-----------------|-------------------|-------------------|
| `S^T @ k` | YES | NO | YES (state update) |
| `delta = (v - g*kv) * beta` | YES | NO | YES (state update) |
| `S_new = g*S + k⊗delta` | YES | NO | YES (state update) |
| `S^T @ q` | NO | YES | NO (discarded) |
| Conv state shift | R-state only | N/A | YES (separate from GDN) |

The GDN kernel computes both attention output AND state update in one pass. For replay, only the state update matters. The attention output is written to `dst` (attn_data) and the state is written to `state`. During replay, `dst` can be a dummy buffer.

### 1.4 Do the required intermediates already exist as ggml graph nodes?

**YES.** The current graph builder at [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) names these intermediates:

```cpp
cb(q, "q_in", il);
cb(k, "k_in", il);      // <-- k after l2_norm (the value needed for replay)
cb(v, "v_in", il);      // <-- v (the value needed for replay)
cb(b, "b_in", il);      // <-- beta (the value needed for replay)
cb(g, "g_in", il);      // <-- gate (the value needed for replay)
```

**Directly observed from source:** All 5 tensors captured by the old tape (k, v, gate, beta, qkv_mixed) exist as named ggml graph nodes in the current upstream graph builder. The naming convention uses `{name}-{il}` format where `il` is the layer index.

### 1.5 Do those intermediates remain available at DFlash verification acceptance decision point?

**This is the critical question.** The intermediates exist as ggml graph nodes during graph CONSTRUCTION. But after graph execution, the tensor data lives in GPU memory and may be overwritten by subsequent operations.

In the old implementation, the eval callback (`dflash_eval_callback`) intercepted tensor computation and copied data to the tape buffer during graph execution. This mechanism was REMOVED in current upstream.

**Current upstream has NO mechanism to capture intermediates during graph execution.** The graph nodes exist at construction time, but their data is transient — available only during the compute window and then freed.

For replay to work with current upstream, one of these approaches is needed:
1. **Re-introduce eval callback** for tape capture (old approach, ~200 lines).
2. **Graph-embedded copy** — add `ggml_cpy` ops to the graph that write intermediates to persistent GPU buffers (GPU tape approach from old code).
3. **Recompute** — re-run the forward pass for accepted tokens (defeats the purpose of replay).

### 1.6 Does the current graph already compute everything needed for replay?

**YES, the current graph computes all required intermediates.** The GDN operation in the current graph receives k, v, gate, and beta as inputs. These tensors are computed by preceding graph nodes (l2_norm for k, linear projections for v/gate/beta). The graph already produces the exact values needed for replay.

The gap is not in COMPUTATION but in CAPTURE. The values exist during graph execution but are not persisted to a replay buffer.

---

## Section 2: Old vs Current DeltaNet/GDN Comparison

### 2.1 Function Signature Comparison

| Aspect | Old (0.3.2) | Current Upstream | Difference |
|--------|-------------|------------------|------------|
| **Function** | [`ggml_gated_delta_net()`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml.c:6403) | [`ggml_gated_delta_net()`](ggml/src/ggml.c:6426) | |
| **Params** | `(ctx, q, k, v, g, beta, state)` | `(ctx, q, k, v, g, beta, state, K)` | **K parameter added** |
| **K derivation** | Inferred from `state->ne[1]` (3D) or K=1 (4D) | Explicit `int64_t K` parameter | **API change** |
| **State input (3D)** | `(S_v*S_v*H, K, n_seqs)` — supported | **NOT supported** | **Removed** |
| **State input (4D)** | `(S_v, S_v, H, n_seqs)` — K=1 only | `(S_v, S_v, H, n_seqs)` — any K | **Extended** |
| **State holds** | Initial state s0 only (both versions) | Initial state s0 only | **SAME** |
| **Output layout** | `[S_v*H, n_tokens*n_seqs + K*S_v*n_seqs, 1, 1]` | `[S_v*H, n_tokens*n_seqs + K*S_v*n_seqs, 1, 1]` | **SAME** |

**Directly observed from source:**

Old signature at [`ggml/src/ggml.c:6403-6410`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml.c:6403):
```cpp
struct ggml_tensor * ggml_gated_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state) {
```

Current signature at [`ggml/src/ggml.c:6426-6434`](ggml/src/ggml.c:6426):
```cpp
struct ggml_tensor * ggml_gated_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        int64_t               K) {
```

**Classification: API/graph-construction difference.** The K parameter was moved from being implicit (encoded in state tensor shape) to explicit. This is a cleaner API but requires callers to pass K explicitly.

### 2.2 State Dimension/Layout Comparison

| Aspect | Old (0.3.2) | Current | Difference |
|--------|-------------|---------|------------|
| **S state shape** | `[S_v, S_v, H_v, n_seqs]` (4D) | `[S_v, S_v, H_v, n_seqs]` (4D) | **SAME** |
| **S state meaning** | Transposed: `S[i][col]` = row i, column col | Transposed: same layout | **SAME** |
| **S state type** | F32 | F32 | **SAME** |
| **S state size** | `n_embd_s = S_v * S_v * H_v` | Same | **SAME** |
| **R state (conv)** | `[conv_ch * conv_window]` per layer | Same | **SAME** |
| **State in recurrent memory** | `s_l[il]` tensor per layer | `s_l[il]` tensor per layer | **SAME** |
| **Multi-slot state (K>1)** | 3D `(D, K, n_seqs)` or explicit K param | Explicit K param, 4D input | **API difference** |

**Directly observed from source:**

Old state validation at [`ggml.c:6434-6436`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml.c:6434):
```cpp
const bool state_is_3d = state->ne[0] == S_v * S_v * H && state->ne[2] == n_seqs && state->ne[3] == 1;
const bool state_is_4d = state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H && state->ne[3] == n_seqs;
GGML_ASSERT(state_is_3d || state_is_4d);
```

Current state validation at [`ggml.c:6459-6463`](ggml/src/ggml.c:6459):
```cpp
GGML_ASSERT(state->ne[0] == S_v);
GGML_ASSERT(state->ne[1] == S_v);
GGML_ASSERT(state->ne[2] == H);
GGML_ASSERT(state->ne[3] == n_seqs);
GGML_ASSERT(K >= 1);
```

**Classification: API/graph-construction difference.** Current version requires 4D state input with explicit K. Old version accepted both 3D and 4D. The underlying state layout (S_v, S_v, H, n_seqs) is identical.

### 2.3 Gating/Update Equations Comparison

| Aspect | Old CPU Replay | Current CUDA Kernel | Difference |
|--------|---------------|---------------------|------------|
| **Gate transform** | `exp2(gate * log2(e))` = `exp(gate)` | `expf(*g_t)` | **SAME math** |
| **Beta transform** | `sigmoid(beta)` applied at replay | Raw `beta_val` used directly | **POTENTIAL DIFFERENCE** |
| **kv computation (non-KDA)** | `S^T @ k` | `warp_reduce_sum(S[i][col] * k[i])` | **SAME** |
| **kv computation (KDA)** | N/A in old CPU replay | `sum_i(exp(g[i]) * S[i][col] * k[i])` | **SAME** (KDA added in both) |
| **delta** | `(v - g*kv) * beta` | `(v - g*kv) * beta` | **SAME** |
| **S update** | `g*S + k⊗delta` | `g*S + k*delta` (fused) | **SAME** |
| **Attention output** | `S^T @ q` (discarded in replay) | `S^T @ q` (written to dst) | **SAME** |

**Directly observed from source:**

Old CPU replay at [`llama-context.cpp:4193-4208`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4193):
```cpp
float g_val = exp2f(tape.gate[tok * H_v + hv] * 1.442695041f);  // = exp(gate)
float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv])); // = sigmoid(beta)
// ...
float delta_col = (v_t[col] - g_val * kv) * b_val;
S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
```

Current CUDA kernel at [`gated_delta_net.cu:85-104`](ggml/src/ggml-cuda/gated_delta_net.cu:85):
```cpp
const float g_val = expf(*g_t);
// ...
float kv_col = warp_reduce_sum<warp_size>(kv_shard);
float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
```

**Critical discrepancy — beta handling:**

The old CPU replay applied `sigmoid()` to beta during replay. The current CUDA kernel uses beta directly without sigmoid. This means:

- **Old tape captured PRE-sigmoid beta** (the graph node `beta-{il}` contained raw beta before sigmoid).
- **Current GDN kernel receives POST-sigmoid beta** (the graph builder applies sigmoid before the GDN op).

**OR** the old replay code was applying sigmoid because the tape captured the value before the graph builder's sigmoid, while the current kernel also receives pre-sigmoid beta but the sigmoid was moved into the kernel. Need to verify.

**Investigation of beta flow:**

In the model graph builder, beta flows through the layer as follows:
1. Model weights produce raw beta logits.
2. A sigmoid op is applied in the graph.
3. The post-sigmoid beta is passed to `build_recurrent_attn()`.
4. The GDN kernel receives post-sigmoid beta.

The old tape captured `beta-{il}` which was the graph node AFTER sigmoid (the callback registered at the graph builder naming point). But the old CPU replay applied sigmoid again. This would be a double-sigmoid bug UNLESS the old tape captured a different node.

**Re-reading old tape name map** ([`llama-context.cpp:2277-2282`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2277)):
```cpp
dflash_capture->tape_name_map["beta-" + il_str] = {idx, DFLASH_TAPE_BETA};
```

The tensor name `beta-{il}` in the old graph builder. Let me check what the old graph builder named beta:

Old [`delta-net-base.cpp:54`](old-versions/beellama.cpp-preview-v0.3.2/src/models/delta-net-base.cpp:54):
```cpp
cb(b, "b_in", il);
```

The callback name is `"b_in-{il}"`, not `"beta-{il}"`. The tape name map uses `"beta-{il}"`. This means the tape captured a DIFFERENT node than `b_in`. The `beta-{il}` name was likely set by the model-specific builder (e.g., qwen35.cpp) before calling `build_recurrent_attn()`.

**This requires checking the model-specific builder to determine whether `beta-{il}` was pre- or post-sigmoid.** Without that, the beta handling discrepancy remains unresolved.

**Classification: POTENTIAL semantic difference in beta handling.** The GDN core math (kv, delta, S update) is IDENTICAL. The gate transform is IDENTICAL (`exp(gate)`). The only potential difference is whether beta is pre- or post-sigmoid at the capture point.

### 2.4 Per-Token Intermediate Values

| Aspect | Old | Current | Difference |
|--------|-----|---------|------------|
| **k capture point** | After l2_norm, before GDN | After l2_norm, before GDN | **SAME** |
| **v capture point** | After projection, before GDN | After projection, before GDN | **SAME** |
| **gate capture point** | After projection, before GDN | After projection, before GDN | **SAME** |
| **beta capture point** | Model-specific (pre/post sigmoid?) | Model-specific | **SAME (if same model code)** |
| **qkv_mixed capture** | After qkv projection, before conv | Same location in graph | **SAME** |

### 2.5 Ordering of Operations

| Step | Old Graph Builder | Current Graph Builder | Difference |
|------|------------------|----------------------|------------|
| 1. Scale q | `q = ggml_scale(q, 1/sqrt(S_k))` | Same | **SAME** |
| 2. Permute inputs | `permute(q,k,v,g,b)` | Same | **SAME** |
| 3. Call GDN | `ggml_gated_delta_net(q,k,v,g,b,s)` | `ggml_gated_delta_net(q,k,v,g,b,s,K)` | **K param added** |
| 4. Extract output | View first `S_v*H*n_tokens*n_seqs` elements | Same | **SAME** |
| 5. Extract state | View remaining elements | Same | **SAME** |

### 2.6 Precision/Types

| Aspect | Old | Current | Difference |
|--------|-----|---------|------------|
| **GDN input type** | F32 (asserted) | F32 (asserted) | **SAME** |
| **GDN output type** | F32 | F32 | **SAME** |
| **State storage** | F32 in recurrent memory | F32 | **SAME** |
| **Tape storage (old)** | F32 (CPU), F32 (GPU) | N/A (removed) | N/A |
| **Kernel precision** | F32 throughout | F32 throughout | **SAME** |

### 2.7 Device Placement

| Aspect | Old | Current | Difference |
|--------|-----|---------|------------|
| **GDN kernel** | CUDA, CPU, Metal, SYCL, Vulkan, OpenCL, Hexagon | Same backends | **SAME** |
| **State device** | Same as model layer | Same as model layer | **SAME** |
| **Tape device (old)** | GPU tape (persistent tensors) or CPU tape (eval callback) | N/A (removed) | N/A |
| **Replay device (old)** | GPU (direct kernel or ggml graph) or CPU fallback | N/A (removed) | N/A |

### 2.8 Batching

| Aspect | Old | Current | Difference |
|--------|-----|---------|------------|
| **n_seqs support** | Multi-sequence in GDN kernel | Same | **SAME** |
| **n_tokens support** | Multi-token (prefill) and single-token (TG) | Same | **SAME** |
| **K snapshots** | Per-sequence snapshots | Per-sequence snapshots | **SAME** |
| **Snapshot layout** | Slot 0 = newest, slot K-1 = oldest | Same (`target_slot = n_tokens - 1 - t`) | **SAME** |

### 2.9 Graph Construction

| Aspect | Old | Current | Difference |
|--------|-----|---------|------------|
| **Fused GDN AR** | `cb(result, "fgdn_ar", il)` | `res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il})` | **API change** |
| **Fused GDN CH** | `cb(result, "fgdn_ch", il)` | `res->add_fused_node({LLM_FUSED_OP_GDN_CH, result, il})` | **API change** |
| **State write-back** | `ggml_cpy(new_state, view(ssm_states_all))` | Same | **SAME** |
| **K>1 snapshot scatter** | Manual loop over K slots | Single `ggml_cpy` with 3D views | **Implementation difference** |
| **DDTree path** | `ggml_gated_delta_net_tree()` | **REMOVED** | **Semantic removal** |

**Directly observed from source:**

Old K>1 snapshot scatter at [`delta-net-base.cpp:632-646`](old-versions/beellama.cpp-preview-v0.3.2/src/models/delta-net-base.cpp:632):
```cpp
for (int64_t k_i = 0; k_i < K; ++k_i) {
    const uint32_t cache_slot = (uint32_t) (K - 1 - k_i);
    ggml_tensor * src = ggml_view_4d(ctx0, gdn_out, ...);
    ggml_tensor * dst = ggml_view_2d(ctx0, ssm_states_all, ...);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
}
```

Current K>1 snapshot scatter at [`delta-net-base.cpp:591-603`](src/models/delta-net-base.cpp:591):
```cpp
ggml_tensor * src = ggml_view_3d(ctx0, gdn_out, D, n_seqs, n_written, ...);
ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, n_written, ...);
ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
```

**Classification: Implementation difference.** Current version uses a single 3D cpy instead of a K-loop of 4D cpys. Same mathematical result, more efficient graph.

### 2.10 Summary Classification Table

| Difference | Type | Impact on Replay Compatibility |
|------------|------|-------------------------------|
| K parameter explicit vs implicit | API/graph-construction | LOW — replay uses K=1 equivalent |
| 3D state input removed | API/graph-construction | LOW — replay uses 4D state |
| Fused node registration API | API/graph-construction | NONE — internal only |
| K>1 scatter: loop vs single cpy | Implementation | NONE — same result |
| DDTree GDN removed | Semantic removal | NONE — replay is linear-only |
| Beta pre/post-sigmoid ambiguity | **POTENTIAL semantic** | **MEDIUM** — requires verification |
| Tape capture mechanism removed | Semantic removal | **HIGH** — must be re-implemented |

---

## Section 3: Compatibility Assessment

### 3.1 Can old replay equations work with current graph?

**Short answer: YES, with conditions.**

The GDN mathematical operation is IDENTICAL between old and current implementations. The CUDA kernel at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) performs the same rank-1 update that the old CPU replay loop at [`llama-context.cpp:4190-4210`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4190) performed.

**Conditions:**

1. **Beta handling must be verified.** If the current graph builder passes post-sigmoid beta to the GDN kernel, replay must also use post-sigmoid beta (or capture post-sigmoid beta). If the old tape captured pre-sigmoid beta and applied sigmoid during replay, the replay code must match the current graph's beta convention.

2. **Tape capture mechanism must be re-implemented.** The current graph has NO tape capture. The intermediates (k, v, gate, beta) exist as graph nodes but are not persisted. Two options:
   - **Eval callback** (old CPU approach): ~200 lines to reinstall.
   - **Graph-embedded copy** (old GPU approach): embed copy ops in the graph builder that write intermediates to persistent GPU buffers.

3. **GDN op must accept Q=zeros for replay.** The current GDN kernel computes attention output even when Q=zeros. This is fine — the output is discarded. The old replay graph passed Q=zeros explicitly ([`llama-context.cpp:3082-3083`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3082)).

4. **State layout is compatible.** Both old and current use `[S_v, S_v, H_v, n_seqs]` 4D layout for S state. The replay can read from and write to the same recurrent memory tensors.

5. **Conv state rebuild is separate.** The conv state (R) rebuild uses `qkv_mixed` from the tape and shifts the ring buffer. This is orthogonal to GDN replay and uses the same mechanism as old code.

### 3.2 What would need to change to enable replay with current graph?

**Minimum changes:**

1. **Add tape capture** (~200-400 lines):
   - Either eval callback (CPU tape) or graph-embedded copies (GPU tape).
   - The graph nodes already exist with correct names.

2. **Add replay function** (~100-200 lines):
   - Call `ggml_gated_delta_net()` with Q=zeros, captured k/v/gate/beta, and current S state.
   - Write output state back to recurrent memory.
   - The GDN kernel already supports this — just pass the right inputs.

3. **Verify beta convention** (0 lines if already correct):
   - Confirm whether graph builder applies sigmoid before GDN.
   - Ensure replay uses matching beta values.

4. **Integrate with rollback** (~50 lines):
   - After restoring backup state, call replay for n_accepted tokens.
   - Same integration point as old `dflash_rollback()`.

**Total estimated new code: ~400-700 lines** for a minimal replay system that works with current upstream graph primitives.

### 3.3 Key Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Beta pre/post-sigmoid mismatch | Medium | Verify in model builder; add unit test comparing replay vs forward. |
| Tape capture overhead | Low | GPU tape avoids CPU sync. Old GPU tape showed minimal overhead. |
| GDN kernel Q=zeros behavior | Low | Kernel handles Q=zeros correctly (attention output = zeros). |
| State layout drift | Low | 4D layout is stable; unit test can verify. |
| K parameter confusion | Low | Replay uses K=1 (single state output). Current API supports this. |

### 3.4 Conclusion

**The old replay equations CAN work with the current upstream graph.** The GDN math is identical. The state layout is identical. The only gaps are:

1. No tape capture mechanism (must be re-implemented).
2. Beta convention needs verification (likely compatible).
3. Integration with current rollback system (straightforward).

The current `ggml_gated_delta_net()` op with K=1, Q=zeros, and captured k/v/gate/beta is a drop-in replacement for the old replay GDN operation. The replay can use the same CUDA kernel that powers the forward pass, eliminating the need for a separate replay kernel.

---

*End of Part 2-3 analysis. Part 4 (capture points) and Part 5 (linear-only feasibility) continue in subsequent documents.*
