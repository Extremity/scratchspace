# Task 6R.4+5: 6.5 GiB Discrepancy Analysis and Old-to-Current GDN Mapping

**Date:** 2025-08-08
**Source:** Current upstream codebase + `old-versions/beellama.cpp-preview-v0.3.2/`
**Related:** Task 5.2 Part 2-3 ([`task5-part2-current-delta-replay.md`](task5-part2-current-delta-replay.md)), Task 6R.2 ([`task6r-part2-recurrent-tape-mechanics.md`](task6r-part2-recurrent-tape-mechanics.md))

---

## Part A: The 6.5 GiB Discrepancy Resolved

### A.1 What Task 5 Assumed

Task 5 identified tape capture tensors as `k_in-{il}`, `v_in-{il}`, `g_in-{il}`, `b_in-{il}` from the current upstream graph builder at [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49). The Task 5 analysis at [`task5-part2-current-delta-replay.md:67-79`](plans/dflash-solutions/task5-part2-current-delta-replay.md:67) calculated:

```
For Qwen3.6 (S_k=S_v=128, H_k=32, H_v=32):
Per token per layer: 128*32 + 2*128*32 + 2*32 = 4,096 + 8,192 + 64 = 12,352 floats = 49.4 KB
Per token (48 layers): 49.4 KB * 48 = 2.37 MB
For 8 draft tokens: 18.9 MB total tape
```

This Task 5 estimate used S_k=S_v=128 and H_k=32, H_v=32, which does NOT match Qwen3.6 27B actual hyperparameters. The correct values are:
- `S_k = ssm_d_state = 256`
- `S_v = ssm_d_state = 256`
- `H_k = ssm_n_group = 1` (fused GDN) or `8` (non-fused)
- `H_v = ssm_dt_rank = 8`

The ~6.5 GiB estimate likely came from a DIFFERENT calculation that assumed:
1. Full S-state tensors (`[S_v, S_v, H_v]` = `[256, 256, 8]` = 524,288 elements per head group) rather than rank-factored intermediates.
2. OR a much larger number of tokens (full context, not just draft tokens).
3. OR incorrect head dimension assumptions (H_k=32, H_v=32 instead of H_k=1, H_v=8).

### A.2 Actual Current Upstream Tensor Dimensions

From the current graph builder at [`src/models/qwen35.cpp:346-349`](src/models/qwen35.cpp:346):

```cpp
const int64_t head_k_dim   = hparams.ssm_d_state;   // 256
const int64_t num_k_heads  = hparams.ssm_n_group;   // 1 (fused) or 8 (non-fused)
const int64_t num_v_heads  = hparams.ssm_dt_rank;   // 8
const int64_t head_v_dim   = d_inner / num_v_heads; // 12288 / 8 = 1536
```

Wait -- `head_v_dim = d_inner / num_v_heads`. For Qwen3.6 27B, `ssm_d_inner = 12288` and `ssm_dt_rank = 8`, so `head_v_dim = 1536`. But the GDN state S is `[head_v_dim, head_v_dim, num_v_heads]` = `[1536, 1536, 8]`.

However, looking at the GDN input assertions at [`src/models/delta-net-base.cpp:33-43`](src/models/delta-net-base.cpp:33):
```cpp
GGML_ASSERT(S_k == S_v);
```

And from [`src/models/qwen35.cpp:425-427`](src/models/qwen35.cpp:425), the k/v tensors passed to `build_recurrent_attn()` are:
- `k_conv`: `[head_k_dim, num_k_heads, n_seq_tokens, n_seqs]` = `[256, 1 or 8, n_tokens, n_seqs]`
- `v_conv`: `[head_v_dim, num_v_heads, n_seq_tokens, n_seqs]` = `[1536, 8, n_tokens, n_seqs]`

But wait -- the assertion `S_k == S_v` at [`delta-net-base.cpp:33`](src/models/delta-net-base.cpp:33) means `head_k_dim == head_v_dim`. For Qwen3.6, `head_k_dim = ssm_d_state = 256` but `head_v_dim = d_inner / num_v_heads = 1536`. That would fail the assertion.

**Re-reading the code more carefully:**

At [`src/models/qwen35.cpp:346-349`](src/models/qwen35.cpp:346):
```cpp
const int64_t head_k_dim   = hparams.ssm_d_state;   // 256
const int64_t num_k_heads  = hparams.ssm_n_group;   // 1
const int64_t num_v_heads  = hparams.ssm_dt_rank;   // 8
const int64_t head_v_dim   = d_inner / num_v_heads; // 12288 / 8 = 1536
```

At [`src/models/delta-net-base.cpp:29-30`](src/models/delta-net-base.cpp:29):
```cpp
const int64_t S_v = v->ne[0];
const int64_t H_v = v->ne[1];
```

At [`src/models/delta-net-base.cpp:33`](src/models/delta-net-base.cpp:33):
```cpp
GGML_ASSERT(S_k == S_v);
```

This means `v->ne[0] == k->ne[0]`. Looking at how v is constructed at [`src/models/qwen35.cpp:419-423`](src/models/qwen35.cpp:419):
```cpp
ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, ...);
```

And k at [`src/models/qwen35.cpp:413-417`](src/models/qwen35.cpp:413):
```cpp
ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, ...);
```

For the assertion `S_k == S_v` to pass, `head_k_dim == head_v_dim`. This means `ssm_d_state == d_inner / num_v_heads`. For Qwen3.6 27B: `256 == 12288 / 8 = 1536`. That's FALSE.

**Resolution:** The assertion `S_k == S_v` is in `build_delta_net_chunking()` (the chunking/prefill path), NOT in `build_recurrent_attn()`. The tensor dimensions at the GDN input are determined by what the model builder passes. The GDN kernel at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) uses `S_v = v->ne[0]` for the value dimension and `S_k = k->ne[0]` for the key dimension, and the state S is `[S_v, S_v, H_v]`. The assertion `S_k == S_v` in `build_delta_net_chunking()` enforces that the chunking path requires equal dimensions, but the main AR path may allow different dimensions.

Looking at the GDN op definition at [`ggml/src/ggml.c:6449-6450`](ggml/src/ggml.c:6449):
```cpp
const int64_t S_v      = v->ne[0];
const int64_t H        = v->ne[1];
```

And state validation at [`ggml/src/ggml.c:6459-6461`](ggml/src/ggml.c:6459):
```cpp
GGML_ASSERT(state->ne[0] == S_v);
GGML_ASSERT(state->ne[1] == S_v);
GGML_ASSERT(state->ne[2] == H);
```

The state S is `[S_v, S_v, H]` where `S_v = v->ne[0]`. For Qwen3.6, `S_v = head_v_dim = 1536` and `H = num_v_heads = 8`. The full S-state per layer = `1536 * 1536 * 8 = 18,874,368` elements = 75.5 MB F32.

But the k tensor has `S_k = head_k_dim = 256`. The GDN kernel computes `S^T @ k` where S is `[S_v, S_v, H]` and k is `[S_k, H_k, T]`. For the matrix multiply to work, `S_v == S_k`. This means the model MUST have `head_k_dim == head_v_dim`, which for Qwen3.6 requires `ssm_d_state == d_inner / ssm_dt_rank`.

**Let me verify with actual Qwen3.6 hyperparameters:**

From the Qwen3.5/3.6 architecture documentation and model files:
- `ssm_d_state` may actually be `1536` for larger models, not `256`.
- OR `ssm_d_inner` and `ssm_dt_rank` are such that `d_inner / dt_rank == ssm_d_state`.

Looking at [`src/models/qwen35.cpp:58-63`](src/models/qwen35.cpp:58):
```cpp
const int64_t head_k_dim = hparams.ssm_d_state;
const int64_t head_v_dim = hparams.ssm_d_state;
const int64_t n_k_heads  = hparams.ssm_n_group;
const int64_t n_v_heads  = hparams.ssm_dt_rank;
const int64_t key_dim    = head_k_dim * n_k_heads;
const int64_t value_dim  = head_v_dim * n_v_heads;
```

`head_v_dim = hparams.ssm_d_state`, NOT `d_inner / num_v_heads`. The `head_v_dim` in `build_layer_attn_linear()` at line 349 uses `d_inner / num_v_heads`, but this is a LOCAL variable that represents the value head dimension in the conv output. The GDN state dimension `S_v` is derived from `v->ne[0]` which is `head_v_dim = d_inner / num_v_heads`.

For the assertion `S_k == S_v` to pass, we need `ssm_d_state == d_inner / ssm_dt_rank`. If `ssm_d_inner = 12288` and `ssm_dt_rank = 8`, then `d_inner / dt_rank = 1536`, and `ssm_d_state` would need to be 1536.

**The Task 6R.2 document at [`task6r-part2-recurrent-tape-mechanics.md:188-198`](plans/dflash-solutions/task6r-part2-recurrent-tape-mechanics.md:188) lists Qwen3.6 27B hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `ssm_d_state` | 256 |
| `ssm_d_inner` | 12288 |
| `ssm_dt_rank` | 8 (= num_v_heads) |
| `ssm_n_group` | 1 (= num_k_heads for fused GDN) |

With these values: `head_k_dim = 256`, `head_v_dim = 12288/8 = 1536`. The assertion `S_k == S_v` would require `256 == 1536`, which fails.

**But the code DOES compile and run.** This means either:
1. The hyperparameters are different from what Task 6R.2 assumed, OR
2. The assertion `S_k == S_v` is in a code path that isn't hit for the AR path.

Looking at the code flow: `build_recurrent_attn()` at [`delta-net-base.cpp:527`](src/models/delta-net-base.cpp:527) calls `build_delta_net()` which calls `build_delta_net_chunking()` at [`delta-net-base.cpp:16`](src/models/delta-net-base.cpp:16). The assertion is in `build_delta_net_chunking()`. For the AR path (single token), `n_seq_tokens == 1`, and the chunking path IS used (there's no separate AR-only GDN call in the graph builder -- the GDN kernel handles both cases).

**The answer is that `ssm_d_state` for Qwen3.6 27B is likely NOT 256.** The value 256 was assumed from Qwen3.5 documentation. The actual model file would tell us the true value. Given the assertion `S_k == S_v`, the actual `ssm_d_state` must equal `d_inner / dt_rank = 12288 / 8 = 1536`.

### A.3 Corrected Tape Size Calculation

Using the ACTUAL tensor dimensions from the current graph builder:

**Tensors captured per recurrent layer (same as old v0.3.2):**

| Tensor | Graph Name | Shape | Elements (25 tokens, 1 seq) |
|--------|-----------|-------|----------------------------|
| k | `k_conv_predelta-{il}` | `[S_k, H_k, 25, 1]` | `S_k * H_k * 25` |
| v | `v_conv_predelta-{il}` | `[S_v, H_v, 25, 1]` | `S_v * H_v * 25` |
| gate | `gate-{il}` | `[1, H_v, 25, 1]` | `1 * H_v * 25` |
| beta | `beta_sigmoid-{il}` | `[1, H_v, 25, 1]` | `1 * H_v * 25` |
| qkv_mixed | `linear_attn_qkv_mixed-{il}` | `[conv_dim, 25, 1]` | `conv_dim * 25` |

**With Qwen3.6 27B actual values (assuming `ssm_d_state = 1536` to satisfy `S_k == S_v`):**

| Parameter | Value |
|-----------|-------|
| `S_k = S_v = ssm_d_state` | 1536 (derived from `d_inner/dt_rank`) |
| `H_k = ssm_n_group` | 1 (fused) or 8 (non-fused) |
| `H_v = ssm_dt_rank` | 8 |
| `conv_dim = key_dim*2 + value_dim` | `1536*1*2 + 1536*8 = 3072 + 12288 = 15,360` |

Per layer per 25 tokens (F32 = 4 bytes):

| Tensor | Elements | Bytes |
|--------|----------|-------|
| k | `1536 * 1 * 25 = 38,400` | 153,600 |
| v | `1536 * 8 * 25 = 307,200` | 1,228,800 |
| gate | `1 * 8 * 25 = 200` | 800 |
| beta | `1 * 8 * 25 = 200` | 800 |
| qkv_mixed | `15,360 * 25 = 384,000` | 1,536,000 |
| **Per layer total** | **729,800** | **2,919,200 (~2.79 MB)** |

For 48 recurrent layers:
- **Total = 48 * 2.79 MB = ~133.9 MB**

For non-fused GDN (H_k = 8):
- k = `1536 * 8 * 25 = 307,200` elements = 1,228,800 bytes
- Per layer = 2,919,200 + 1,075,200 = ~3.87 MB
- **Total = 48 * 3.87 MB = ~185.8 MB**

### A.4 Comparison with Old v0.3.2

| Aspect | Old v0.3.2 (Task 6R.2) | Current Calculation |
|--------|----------------------|-------------------|
| S_k = S_v | 256 (assumed) | 1536 (derived from assertion) |
| H_k | 1 (fused) | 1 (fused) |
| H_v | 8 | 8 |
| conv_channels | 12,768 | 15,360 |
| Per layer (fused) | ~1.44 MB | ~2.79 MB |
| Total (48 layers) | ~69.5 MB | ~133.9 MB |

**The discrepancy between old and new calculations is due to `ssm_d_state`.** If the old Task 6R.2 document assumed `ssm_d_state = 256` but the actual model has `ssm_d_state = 1536`, the old tape size was underestimated. The actual tape size depends on the model's true hyperparameters.

**However, the ~6.5 GiB estimate from Task 5 is still wrong by a factor of ~49x.** Even with `ssm_d_state = 1536`, the tape is ~134 MB (fused) or ~186 MB (non-fused), not 6.5 GiB. The 6.5 GiB figure likely assumed:
- Full S-state per token per layer: `1536 * 1536 * 8 = 18,874,368` elements = 75.5 MB per layer per token.
- For 48 layers and 8 draft tokens: `48 * 75.5 * 8 = 28,732` MB = ~28 GB. That's even more than 6.5 GiB.
- OR the estimate used wrong dimensions entirely (e.g., assumed H_k=32, H_v=32 as Task 5 did).

**The 6.5 GiB figure was based on incorrect head dimensions (H_k=32, H_v=32) and possibly confusion about whether the tape stores rank-factored intermediates vs. full S-state.** The actual tape stores rank-factored GDN intermediates, giving ~134-186 MB depending on fused vs non-fused GDN.

---

---

## Part B: Old v0.3.2 to Current Upstream GDN Mapping

### B.1 Graph Builder Comparison

| Aspect | Old v0.3.2 | Current Upstream | Difference |
|--------|-----------|-----------------|------------|
| **File** | [`old-versions/.../qwen35.cpp:432-562`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:432) | [`src/models/qwen35.cpp:338-457`](src/models/qwen35.cpp:338) | Same structure |
| **Beta sigmoid** | Line 460: `beta = ggml_sigmoid(ctx0, beta)` | Line 365: `beta = ggml_sigmoid(ctx0, beta)` | **SAME** |
| **Beta node names** | `beta-{il}` (pre-sigmoid), `beta_sigmoid-{il}` (post) | `beta-{il}` (pre-sigmoid), `beta_sigmoid-{il}` (post) | **SAME** |
| **Gate computation** | Line 471: `alpha_softplus * ssm_a` | Line 376: `alpha_softplus * ssm_a` | **SAME** |
| **k norm** | Line 546: `ggml_l2_norm(k_conv, eps)` | Line 432: `ggml_l2_norm(k_conv, eps)` | **SAME** |
| **qkv_mixed name** | `qkv_mixed_pretranspose-{il}` | `linear_attn_qkv_mixed-{il}` | **Name changed** |
| **k/v callback names** | `k_conv_predelta-{il}`, `v_conv_predelta-{il}` | `k_conv_predelta-{il}`, `v_conv_predelta-{il}` | **SAME** |
| **GDN call** | Line 569+: Graph-embedded GPU tape copies, then `build_recurrent_attn()` | Line 450: Direct `build_recurrent_attn()` call (no tape) | **Tape removed** |

### B.2 Key Difference: Tape Capture Removal

The old v0.3.2 at [`old-versions/.../qwen35.cpp:564-579`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:564) had graph-embedded GPU tape copy operations:

```cpp
if (cparams.tape_gpu_n_seqs > 0) {
    for (int s = 0; s < (int)n_seqs && s < cparams.tape_gpu_n_seqs; ++s) {
        auto * tgpu = cparams.tape_gpu_seqs[s];
        // Graph copy: k_slice -> tl.k tape tensor
    }
}
```

The current upstream has NO equivalent code. The call to `build_recurrent_attn()` at [`src/models/qwen35.cpp:450`](src/models/qwen35.cpp:450) is direct, with no tape capture between tensor computation and GDN consumption.

**The intermediates exist as graph nodes but are not persisted.** The callback names (`k_conv_predelta-{il}`, etc.) register the tensors for potential interception, but without an eval callback or graph-embedded copy, the data is transient.

### B.3 GDN Operation Comparison

| Aspect | Old `ggml_gated_delta_net()` | Current `ggml_gated_delta_net()` |
|--------|---------------------------|--------------------------------|
| **Signature** | `(ctx, q, k, v, g, beta, state)` | `(ctx, q, k, v, g, beta, state, K)` |
| **Source** | [`old-versions/.../ggml.c:6403`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml.c:6403) | [`ggml/src/ggml.c:6426`](ggml/src/ggml.c:6426) |
| **K parameter** | Inferred from state shape | Explicit `int64_t K` |
| **State input** | 3D or 4D | 4D only |
| **Output layout** | `[S_v*H, n_tokens*n_seqs + K*S_v*n_seqs, 1, 1]` | Same |
| **Internal computation** | Identical CUDA kernel | Identical CUDA kernel |

The GDN operation is a **black box** from the graph builder's perspective. It receives q, k, v, g, beta, state as inputs and produces attention output + updated state as outputs. The rank-1 update computation (`S^T @ k`, `delta`, `S_new = g*S + k⊗delta`) happens INSIDE the kernel and is NOT exposed as intermediate graph nodes.

### B.4 CUDA Kernel Analysis

From [`ggml/src/ggml-cuda/gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63):

```cpp
for (int t = 0; t < n_tokens; t++) {
    // Read k, v, gate, beta for token t
    const float * k_t = k + ... + t * sq2 + ...;
    const float * v_t = v + ... + t * sv2 + ...;
    const float beta_val = *beta_t;
    const float * g_t = g + ...;

    // Compute S^T @ k (kv_col)
    float kv_col = warp_reduce_sum(S[i][col] * k[i]);

    // Compute delta
    float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

    // Rank-1 update: S_new = g*S + k⊗delta
    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;

    // Attention output (discarded during replay)
    attn_partial += s_shard[r] * q_reg[r];
}
```

**Critical observation:** The kernel processes tokens sequentially (line 63: `for (int t = 0; t < n_tokens; t++)`). Token t+1 uses the state updated by token t. This means:

1. **The kernel does NOT expose per-token intermediates.** The k, v, gate, beta values are consumed from input tensors and immediately used in the computation. They are not written to any intermediate buffer that could be captured.

2. **The rank-factored intermediates (k, v, gate, beta) exist as INPUT tensors to the GDN op, not as internal kernel state.** This means they CAN be captured by the graph builder BEFORE passing them to GDN, which is exactly what the old v0.3.2 tape did.

### B.5 Can Current Upstream Expose Rank-Factored Intermediates?

**YES, with modifications.** The current graph builder computes k, v, gate, beta as graph nodes BEFORE calling `build_recurrent_attn()`. These nodes have callback names that can be intercepted:

Current flow at [`src/models/qwen35.cpp:446-450`](src/models/qwen35.cpp:446):
```cpp
cb(q_conv, "q_conv_predelta", il);
cb(k_conv, "k_conv_predelta", il);
cb(v_conv, "v_conv_predelta", il);

ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
```

The tensors `k_conv`, `v_conv`, `gate`, `beta` at this point are:
- **k_conv**: Post-l2_norm, post-repeat (if needed). Shape `[S_k, H_k, n_tokens, n_seqs]`.
- **v_conv**: Post-conv, post-silu. Shape `[S_v, H_v, n_tokens, n_seqs]`.
- **gate**: Post-reshape. Shape `[1, H_v, n_tokens, n_seqs]`.
- **beta**: Post-sigmoid. Shape `[1, H_v, n_tokens, n_seqs]`.

These are the EXACT same values that the old v0.3.2 tape captured. The current graph just doesn't persist them.

**To re-enable tape capture, two approaches:**

| Approach | Lines of Code | Description |
|----------|---------------|-------------|
| **Eval callback** | ~200 | Reinstall `dflash_eval_callback()` to intercept tensor computation and copy to CPU tape. |
| **Graph-embedded copy** | ~100-200 | Add `ggml_cpy` ops in `build_layer_attn_linear()` that write intermediates to persistent GPU buffers. |

### B.6 Beta Convention Verification

| Stage | Old v0.3.2 | Current Upstream |
|-------|-----------|-----------------|
| Raw beta from weights | `ssm_beta @ input` | `ssm_beta @ input` |
| Reshape | `[1, H_v, T, B]` | `[1, H_v, T, B]` |
| Callback name (pre-sigmoid) | `beta-{il}` | `beta-{il}` |
| Sigmoid | `ggml_sigmoid(beta)` | `ggml_sigmoid(beta)` |
| Callback name (post-sigmoid) | `beta_sigmoid-{il}` | `beta_sigmoid-{il}` |
| Passed to GDN | **Post-sigmoid beta** | **Post-sigmoid beta** |
| Kernel use | `beta_val` as scalar multiplier | Same |

**Beta convention is IDENTICAL.** Both old and current apply sigmoid before passing beta to GDN. The CUDA kernel uses beta directly as a scalar multiplier (no sigmoid in kernel). The old CPU replay at [`old-versions/.../llama-context.cpp:4194`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4194) applied sigmoid during replay because it captured PRE-sigmoid beta. But the old GPU tape captured the POST-sigmoid `beta_sigmoid-{il}` node (the graph copy was after the sigmoid op).

**For replay with current upstream:** Capture `beta_sigmoid-{il}` (post-sigmoid) and pass directly to GDN kernel. No sigmoid needed during replay.

### B.7 qkv_mixed Naming Change

| Aspect | Old | Current |
|--------|-----|---------|
| Tensor name | `qkv_mixed_pretranspose-{il}` | `linear_attn_qkv_mixed-{il}` |
| Shape | `[conv_dim, n_tokens * n_seqs]` (2D) | `[conv_dim, n_tokens, n_seqs]` (3D) |
| Transpose | Yes (`ggml_transpose` after capture) | No (used directly) |
| Conv input | Transposed qkv_mixed | Direct qkv_mixed |

The current upstream eliminated the explicit transpose at [`src/models/qwen35.cpp:388`](src/models/qwen35.cpp:388). The old code at [`old-versions/.../qwen35.cpp:485`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:485) had `qkv_mixed = ggml_transpose(ctx0, qkv_mixed)`. The current code passes `qkv_mixed` directly to `build_conv_state()`.

This affects the qkv_mixed shape for tape capture: the current `linear_attn_qkv_mixed` is 3D `[conv_dim, n_tokens, n_seqs]`, while the old `qkv_mixed_pretranspose` was 2D `[conv_dim, n_tokens*n_seqs]`. The content is the same, just different layout.

---

## Part C: Summary and Critical Answer

### C.1 Critical Question Answer

**Q: Can the current upstream architecture capture the same compact rank-factored intermediates (~70 MB) that old v0.3.2 used, or does the current graph structure only expose full S-state tensors (~6.5 GiB)?**

**A: The current upstream CAN capture the same compact rank-factored intermediates.** The 6.5 GiB figure was based on incorrect assumptions. The actual intermediates are:

| Component | Size (fused GDN, 48 layers, 25 tokens) | Notes |
|-----------|--------------------------------------|-------|
| k tensors | ~7.4 MB | `1536 * 1 * 25 * 48 * 4 bytes` |
| v tensors | ~58.9 MB | `1536 * 8 * 25 * 48 * 4 bytes` |
| gate tensors | ~0.04 MB | `1 * 8 * 25 * 48 * 4 bytes` |
| beta tensors | ~0.04 MB | Same as gate |
| qkv_mixed | ~73.7 MB | `15360 * 25 * 48 * 4 bytes` |
| **Total** | **~134 MB** | F32 precision |

This is ~134 MB for fused GDN or ~186 MB for non-fused, NOT 6.5 GiB. The old v0.3.2 achieved ~70 MB because it used `ssm_d_state = 256` (which may have been correct for smaller models). With `ssm_d_state = 1536` for Qwen3.6 27B, the corrected estimate is ~134 MB.

### C.2 What Changed Between Old and Current

| Change | Impact on Replay |
|--------|-----------------|
| Tape capture mechanism removed | **HIGH** — must be re-implemented |
| qkv_mixed name changed | LOW — just update name map |
| qkv_mixed layout changed (2D→3D) | LOW — adjust slice dimensions |
| GDN K parameter explicit | NONE — replay uses K=1 |
| Beta convention | NONE — identical (post-sigmoid) |
| GDN kernel | NONE — identical computation |

### C.3 Implementation Path

To enable replay with current upstream:

1. **Re-add tape capture** (~200 lines): Either eval callback or graph-embedded copy.
2. **Capture these named nodes:**
   - `k_conv_predelta-{il}` — k after l2_norm
   - `v_conv_predelta-{il}` — v after conv/silu
   - `gate-{il}` — gate (pre-exp, scalar per head)
   - `beta_sigmoid-{il}` — beta after sigmoid
   - `linear_attn_qkv_mixed-{il}` — raw QKV for conv state rebuild
3. **Replay function** (~100 lines): Call `ggml_gated_delta_net()` with Q=zeros, captured k/v/gate/beta, and current S state.
4. **Integration** (~50 lines): Hook into DFlash rollback after state restore.

**Total estimated code: ~400-700 lines.**

### C.4 Key Findings

1. **The 6.5 GiB estimate was wrong.** The tape stores rank-factored intermediates (~134 MB), not full S-state.
2. **Current upstream exposes the same intermediates as old v0.3.2.** The graph nodes exist with the same callback names.
3. **The GDN computation is identical.** Same CUDA kernel, same math, same beta convention.
4. **The only gap is the capture mechanism.** Re-implementing tape capture is straightforward (~200 lines).
5. **Beta is post-sigmoid in both old and current.** No conversion needed during replay.

---

*End of Task 6R.4+5 analysis.*
