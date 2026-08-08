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
- `S_k = ssm_d_state = 128`
- `S_v = ssm_d_state = 128`
- `H_k = ssm_n_group = 16` (fused GDN) or `48` (non-fused)
- `H_v = ssm_dt_rank = 48`

**CORRECTION (2026-08-08):** The values above were previously listed as S_k=256, H_k=1/8, H_v=8. Those were carried over from old Qwen3.5 documentation and v0.3.2 code comments. The actual Qwen3.6 GGUF metadata shows `qwen35.ssm.state_size=128`, `qwen35.ssm.group_count=16`, `qwen35.ssm.time_step_rank=48`. See [`task6r-correction-part1-dimensions.md`](task6r-correction-part1-dimensions.md) for the full analysis.

The ~6.5 GiB estimate likely came from a DIFFERENT calculation that assumed:
1. Full S-state tensors (`[S_v, S_v, H_v]` = `[256, 256, 8]` = 524,288 elements per head group) rather than rank-factored intermediates.
2. OR a much larger number of tokens (full context, not just draft tokens).
3. OR incorrect head dimension assumptions (H_k=32, H_v=32 instead of H_k=1, H_v=8).

### A.2 Actual Current Upstream Tensor Dimensions

**CORRECTION (2026-08-08):** The entire Section A.2 analysis below was based on incorrect hyperparameter assumptions (`ssm_d_state=256`, `ssm_d_inner=12288`, `ssm_dt_rank=8`, `ssm_n_group=1/8`). These values were carried over from old Qwen3.5 documentation and v0.3.2 code comments, not verified against the actual Qwen3.6 GGUF. The **actual GGUF metadata** shows:

| GGUF Key | Actual Value | Old Assumed Value |
|----------|-------------|-------------------|
| `qwen35.ssm.state_size` (`ssm_d_state`) | **128** | 256 |
| `qwen35.ssm.group_count` (`ssm_n_group`) | **16** | 1 (fused) / 8 (non-fused) |
| `qwen35.ssm.time_step_rank` (`ssm_dt_rank`) | **48** | 8 |
| `qwen35.ssm.inner_size` (`ssm_d_inner`) | **6144** | 12288 |

With these actual values, the `S_k == S_v` assertion is satisfied: `head_k_dim = ssm_d_state = 128` and `head_v_dim = d_inner / num_v_heads = 6144 / 48 = 128`. So `128 == 128` — **SATISFIED**. No need for `ssm_d_state = 1536`.

The corrected tensor dimensions from the current graph builder at [`src/models/qwen35.cpp:346-349`](src/models/qwen35.cpp:346):

```cpp
const int64_t head_k_dim   = hparams.ssm_d_state;   // 128 (ACTUAL)
const int64_t num_k_heads  = hparams.ssm_n_group;   // 16 (fused) or 48 (non-fused)
const int64_t num_v_heads  = hparams.ssm_dt_rank;   // 48
const int64_t head_v_dim   = d_inner / num_v_heads; // 6144 / 48 = 128
```

And from the model loader at [`src/models/qwen35.cpp:58-63`](src/models/qwen35.cpp:58):
```cpp
const int64_t head_k_dim = hparams.ssm_d_state;   // 128
const int64_t head_v_dim = hparams.ssm_d_state;   // 128 (SAME as head_k_dim)
const int64_t n_k_heads  = hparams.ssm_n_group;   // 16
const int64_t n_v_heads  = hparams.ssm_dt_rank;   // 48
```

Both sources agree: `head_k_dim == head_v_dim == 128`. The `S_k == S_v` assertion at [`src/models/delta-net-base.cpp:33`](src/models/delta-net-base.cpp:33) is satisfied.

**Derived values with actual GGUF metadata:**
- `conv_channels = d_inner + 2 * ssm_n_group * ssm_d_state = 6144 + 2*16*128 = 10,240`
- `n_embd_s = ssm_d_state * ssm_d_inner = 128 * 6144 = 786,432`
- S-state shape: `[128, 128, 48]` = 786,432 elements = 3.0 MB F32 per layer

See [`task6r-correction-part1-dimensions.md`](task6r-correction-part1-dimensions.md) for the complete derivation and source code verification.

### A.3 Corrected Tape Size Calculation

**CORRECTION (2026-08-08):** This section has been completely rewritten using actual Qwen3.6 GGUF metadata. The previous calculation used `ssm_d_state=1536`, `H_k=1/8`, `H_v=8`, and `conv_channels=15,360`, all derived from incorrect assumptions. The corrected values are `ssm_d_state=128`, `H_k=16/48`, `H_v=48`, and `conv_channels=10,240`.

Using the ACTUAL tensor dimensions from the current graph builder with verified GGUF metadata:

**Tensors captured per recurrent layer (same as old v0.3.2):**

| Tensor | Graph Name | Shape | Elements (25 tokens, 1 seq) |
|--------|-----------|-------|----------------------------|
| k | `k_conv_predelta-{il}` | `[S_k, H_k, 25, 1]` | `S_k * H_k * 25` |
| v | `v_conv_predelta-{il}` | `[S_v, H_v, 25, 1]` | `S_v * H_v * 25` |
| gate | `gate-{il}` | `[1, H_v, 25, 1]` | `1 * H_v * 25` |
| beta | `beta_sigmoid-{il}` | `[1, H_v, 25, 1]` | `1 * H_v * 25` |
| qkv_mixed | `linear_attn_qkv_mixed-{il}` | `[conv_channels, 25, 1]` | `conv_channels * 25` |

**With Qwen3.6 27B actual values (from GGUF metadata):**

| Parameter | Value |
|-----------|-------|
| `S_k = S_v = ssm_d_state` | **128** |
| `H_k = ssm_n_group` | **16** (fused) or **48** (non-fused) |
| `H_v = ssm_dt_rank` | **48** |
| `conv_channels = d_inner + 2*ssm_n_group*ssm_d_state` | **10,240** |

Per layer per 25 tokens (F32 = 4 bytes):

**Fused GDN (H_k = 16):**

| Tensor | Elements | Bytes |
|--------|----------|-------|
| k | `128 * 16 * 25 = 51,200` | 204,800 |
| v | `128 * 48 * 25 = 153,600` | 614,400 |
| gate | `1 * 48 * 25 = 1,200` | 4,800 |
| beta | `1 * 48 * 25 = 1,200` | 4,800 |
| qkv_mixed | `10,240 * 25 = 256,000` | 1,024,000 |
| **Per layer total** | **463,200** | **1,852,800 (~1.77 MB)** |

**Non-Fused GDN (H_k = 48):**

| Tensor | Elements | Bytes |
|--------|----------|-------|
| k | `128 * 48 * 25 = 153,600` | 614,400 |
| v | `128 * 48 * 25 = 153,600` | 614,400 |
| gate | `1 * 48 * 25 = 1,200` | 4,800 |
| beta | `1 * 48 * 25 = 1,200` | 4,800 |
| qkv_mixed | `10,240 * 25 = 256,000` | 1,024,000 |
| **Per layer total** | **638,400** | **2,537,200 (~2.42 MB)** |

For 48 recurrent layers:
- **Fused: Total = 48 * 1.77 MB = ~84.9 MB**
- **Non-fused: Total = 48 * 2.42 MB = ~116.5 MB**

### A.4 Comparison with Previous Estimates

| Estimate | Source | Fused | Non-Fused | Status |
|----------|--------|-------|-----------|--------|
| ~70 MB | Old v0.3.2 runtime logs | ~70 MB | ~74 MB | **Wrong dimensions but coincidentally close** — larger S (256) offset by smaller H_v (8 vs 48) |
| ~134 MB | task6r-part4 (old) | ~134 MB | ~186 MB | **NEEDS CORRECTION** — used S=1536 derived from incorrect d_inner/dt_rank |
| **~85 MB** | **This correction** | **~85 MB** | **~117 MB** | **CORRECTED with actual GGUF metadata** |
| ~6.5 GiB | Task 5 | — | — | **Still wrong by ~55-77x** |

**The old v0.3.2 ~70 MB estimate used wrong dimensions (S=256, H_v=8, H_k=1) but happened to produce a reasonable answer because the errors partially canceled out:** larger S (256 vs 128) was offset by smaller H_v (8 vs 48) and smaller conv_ch (12,768 vs 10,240).

**The ~134-186 MB estimate from the original task6r-part4 does NOT survive.** It needs correction DOWN to ~85-117 MB because the actual `ssm_d_state` is 128 (not 1536), making the k and v tensors 12x smaller in the S dimension. However, H_v is 6x larger (48 vs 8), which partially offsets the reduction.

**The ~6.5 GiB estimate from Task 5 is wrong by a factor of ~55-77x.** The tape stores rank-factored GDN intermediates (~85-117 MB), not full S-state. Even with the old S=1536 assumption, full S-state would be `1536*1536*8 = 18,874,368` elements per layer = 75.5 MB, which for 48 layers and 8 tokens = ~28 GB, far exceeding 6.5 GiB.

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

**Q: Can the current upstream architecture capture the same compact rank-factored intermediates that old v0.3.2 used, or does the current graph structure only expose full S-state tensors (~6.5 GiB)?**

**A: The current upstream CAN capture the same compact rank-factored intermediates.** The 6.5 GiB figure was based on incorrect assumptions. The actual intermediates are:

**CORRECTION (2026-08-08):** The table below has been updated with actual Qwen3.6 GGUF metadata (S=128, H_k=16/48, H_v=48, conv_ch=10,240). The previous values used S=1536, H_k=1/8, H_v=8, conv_ch=15,360.

| Component | Size (fused GDN, 48 layers, 25 tokens) | Notes |
|-----------|--------------------------------------|-------|
| k tensors | ~9.8 MB | `128 * 16 * 25 * 48 * 4 bytes` |
| v tensors | ~29.7 MB | `128 * 48 * 25 * 48 * 4 bytes` |
| gate tensors | ~0.2 MB | `1 * 48 * 25 * 48 * 4 bytes` |
| beta tensors | ~0.2 MB | Same as gate |
| qkv_mixed | ~49.2 MB | `10240 * 25 * 48 * 4 bytes` |
| **Total** | **~85 MB** | F32 precision |

This is ~85 MB for fused GDN or ~117 MB for non-fused, NOT 6.5 GiB. The old v0.3.2 achieved ~70 MB because it used wrong dimensions (S=256, H_v=8) that partially canceled out. With actual GGUF metadata (S=128, H_v=48), the corrected estimate is ~85 MB (fused) / ~117 MB (non-fused).

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

1. **The 6.5 GiB estimate was wrong.** The tape stores rank-factored intermediates (~85 MB fused, ~117 MB non-fused), not full S-state. **CORRECTION (2026-08-08):** Updated from ~134-186 MB to ~85-117 MB using actual GGUF metadata.
2. **Current upstream exposes the same intermediates as old v0.3.2.** The graph nodes exist with the same callback names.
3. **The GDN computation is identical.** Same CUDA kernel, same math, same beta convention.
4. **The only gap is the capture mechanism.** Re-implementing tape capture is straightforward (~200 lines).
5. **Beta is post-sigmoid in both old and current.** No conversion needed during replay.

---

*End of Task 6R.4+5 analysis.*
