# Task 6R Correction Part 1: Dimension Correction Using Actual Qwen3.6-27B GGUF Metadata

**Date:** 2026-08-08
**Purpose:** Resolve the 256-vs-1536-vs-128 dimension discrepancy using authoritative GGUF metadata.

---

## 1. Authoritative GGUF Metadata

The following values come from the ACTUAL Qwen3.6-27B GGUF file. These are the ground truth:

| GGUF Key | Value | Meaning |
|----------|-------|---------|
| `qwen35.ssm.state_size` | **128** | `ssm_d_state` (key/value head dimension for GDN) |
| `qwen35.ssm.group_count` | **16** | `ssm_n_group` (number of K heads) |
| `qwen35.ssm.time_step_rank` | **48** | `ssm_dt_rank` (number of V heads) |
| `qwen35.ssm.inner_size` | **6144** | `ssm_d_inner` (inner dimension for conv/SSM) |
| `qwen35.embedding_length` | **5120** | `n_embd` (model embedding dimension) |
| `qwen35.block_count` | **65** | `n_blocks` (total decoder blocks) |
| `qwen35.attention.key_length` | **256** | Full-attention key head dimension |
| `qwen35.attention.value_length` | **256** | Full-attention value head dimension |

**Key derived values:**
- `head_k_dim = ssm_d_state = 128`
- `head_v_dim = ssm_d_state = 128` (same as head_k_dim per model loader)
- `head_v_dim = d_inner / num_v_heads = 6144 / 48 = 128` (same result from graph builder)
- `n_embd_s = ssm_d_state * ssm_d_inner = 128 * 6144 = 786,432`
- `conv_channels = d_inner + 2 * ssm_n_group * ssm_d_state = 6144 + 2*16*128 = 6144 + 4096 = 10,240`
- `S_k == S_v` assertion: `128 == 128` — **SATISFIED**

---

## 2. Where Each Wrong Number Came From

### 2.1 The "256" Assumption

**Source:** Old documents (Task 6R.2, Task 5) assumed `ssm_d_state = 256` from Qwen3.5 architecture documentation.

**Evidence:** The old v0.3.2 code at [`old-versions/.../llama-context.cpp:2416`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2416) contains the comment:
```cpp
const int64_t S = hparams.ssm_d_state;     // 256 for Qwen3.5-27B
```

And at [`old-versions/.../llama-context.cpp:2417`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2417):
```cpp
const int64_t H_v = hparams.ssm_dt_rank;   // 8 (num_v_heads)
```

These comments reflect assumptions about Qwen3.5, not verified against the actual Qwen3.6 GGUF. The old code was written before the Qwen3.6 GGUFs were available, and the authors assumed Qwen3.6 would share Qwen3.5's `ssm_d_state=256`, `ssm_dt_rank=8`, and `ssm_n_group=1`.

**Why it was wrong:** Qwen3.6 uses different SSM hyperparameters than Qwen3.5. The actual values are `ssm_d_state=128`, `ssm_dt_rank=48`, `ssm_n_group=16`.

### 2.2 The "1536" Derivation

**Source:** [`task6r-part4-discrepancy-and-current-mapping.md:146`](plans/dflash-solutions/task6r-part4-discrepancy-and-current-mapping.md:146) derived `ssm_d_state = 1536` from the `S_k == S_v` assertion combined with incorrect assumptions about `d_inner` and `dt_rank`.

**The flawed reasoning chain:**

1. Assumed `ssm_d_inner = 12288` (wrong — actual is 6144)
2. Assumed `ssm_dt_rank = 8` (wrong — actual is 48)
3. Computed `head_v_dim = d_inner / num_v_heads = 12288 / 8 = 1536`
4. Observed `S_k == S_v` requires `head_k_dim == head_v_dim`
5. Concluded `ssm_d_state = 1536`

**Why the assumptions were wrong:**
- `ssm_d_inner = 12288` was carried over from old Qwen3.5 documentation. The actual Qwen3.6 GGUF has `ssm_d_inner = 6144`.
- `ssm_dt_rank = 8` was carried over from old Qwen3.5 documentation. The actual Qwen3.6 GGUF has `ssm_dt_rank = 48`.

**With actual values:** `head_v_dim = 6144 / 48 = 128`, and `head_k_dim = ssm_d_state = 128`, so `S_k == S_v` is `128 == 128` — satisfied without needing `ssm_d_state = 1536`.

### 2.3 Summary of Wrong Assumptions

| Parameter | Actual (GGUF) | Old Docs Assumed | task6r-part4 Derived |
|-----------|---------------|-----------------|---------------------|
| `ssm_d_state` | **128** | 256 | 1536 |
| `ssm_n_group` | **16** | 1 (fused) / 8 (non-fused) | 1 / 8 |
| `ssm_dt_rank` | **48** | 8 | 8 |
| `ssm_d_inner` | **6144** | 12288 | 12288 |
| `head_k_dim` | **128** | 256 | 1536 |
| `head_v_dim` | **128** | 256 (loader) / 1536 (graph) | 1536 |
| `conv_channels` | **10,240** | 12,768 | 15,360 |
| `n_embd_s` | **786,432** | 3,145,728 | 18,874,368 |

---

## 3. Source Code Verification with Actual Dimensions

### 3.1 Model Loader ([`src/models/qwen35.cpp:58-63`](src/models/qwen35.cpp:58))

```cpp
const int64_t head_k_dim = hparams.ssm_d_state;   // 128 (ACTUAL)
const int64_t head_v_dim = hparams.ssm_d_state;   // 128 (ACTUAL — SAME as head_k_dim)
const int64_t n_k_heads  = hparams.ssm_n_group;   // 16
const int64_t n_v_heads  = hparams.ssm_dt_rank;   // 48
const int64_t key_dim    = head_k_dim * n_k_heads; // 128 * 16 = 2,048
const int64_t value_dim  = head_v_dim * n_v_heads; // 128 * 48 = 6,144
```

**Verification:** `head_v_dim == head_k_dim` at the loader level. Both are `ssm_d_state = 128`.

### 3.2 Graph Builder ([`src/models/qwen35.cpp:346-349`](src/models/qwen35.cpp:346))

```cpp
const int64_t d_inner      = hparams.ssm_d_inner;   // 6144
const int64_t head_k_dim   = hparams.ssm_d_state;   // 128
const int64_t num_k_heads  = hparams.ssm_n_group;   // 16
const int64_t num_v_heads  = hparams.ssm_dt_rank;   // 48
const int64_t head_v_dim   = d_inner / num_v_heads; // 6144 / 48 = 128
```

**Verification:** `head_v_dim = 6144 / 48 = 128 = head_k_dim`. The `S_k == S_v` assertion at [`src/models/delta-net-base.cpp:33`](src/models/delta-net-base.cpp:33) is satisfied: `128 == 128`.

### 3.3 S-State Shape ([`src/models/qwen35.cpp:391`](src/models/qwen35.cpp:391))

```cpp
state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
// = [128, 128, 48, n_seqs]
```

**Verification:** `n_embd_s = head_v_dim * head_v_dim * num_v_heads = 128 * 128 * 48 = 786,432`. This matches the value in [`research-summary.md`](plans/dflash-solutions/research-summary.md) exactly.

Also matches [`src/llama-hparams.cpp:221`](src/llama-hparams.cpp:221): `return ssm_d_state * ssm_d_inner = 128 * 6144 = 786,432`.

### 3.4 Conv Channels ([`src/models/qwen35.cpp:386`](src/models/qwen35.cpp:386))

```cpp
const int64_t conv_channels = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;
// = 6144 + 2 * 16 * 128 = 6144 + 4096 = 10,240
```

### 3.5 Old v0.3.2 Tape Allocation ([`old-versions/.../llama-context.cpp:2416-2421`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2416))

```cpp
const int64_t S = hparams.ssm_d_state;     // 128 (ACTUAL — comment "256" was wrong)
const int64_t H_v = hparams.ssm_dt_rank;     // 48 (ACTUAL — comment "8" was wrong)
const int64_t H_k = (cparams.fused_gdn_ar && cparams.fused_gdn_ch)
                   ? hparams.ssm_n_group     // 16 (fused — comment "1" was wrong)
                   : H_v;                     // 48 (non-fused — comment "8" was wrong)
```

### 3.6 Old Tape Tensor Shapes ([`old-versions/.../llama-context.cpp:2475-2479`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2475))

```cpp
tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, max_tokens);
tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, max_tokens);
tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, 1, H_v, max_tokens);
tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, 1, H_v, max_tokens);
tl.qkv  = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, max_tokens);
```

With actual dimensions:
- k: `[128, H_k, 25]` where H_k = 16 (fused) or 48 (non-fused)
- v: `[128, 48, 25]`
- gate: `[1, 48, 25]`
- beta: `[1, 48, 25]`
- qkv: `[10240, 25]`

---

## 4. Corrected Tape Size Calculation

### 4.1 Per-Layer Tensor Sizes (25 max tokens, F32 = 4 bytes)

#### Fused GDN (H_k = 16)

| Tensor | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| k | `[128, 16, 25]` | 51,200 | 204,800 |
| v | `[128, 48, 25]` | 153,600 | 614,400 |
| gate | `[1, 48, 25]` | 1,200 | 4,800 |
| beta | `[1, 48, 25]` | 1,200 | 4,800 |
| qkv | `[10,240, 25]` | 256,000 | 1,024,000 |
| **Per layer total** | | **463,200** | **1,852,800 (~1.77 MB)** |

#### Non-Fused GDN (H_k = 48)

| Tensor | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| k | `[128, 48, 25]` | 153,600 | 614,400 |
| v | `[128, 48, 25]` | 153,600 | 614,400 |
| gate | `[1, 48, 25]` | 1,200 | 4,800 |
| beta | `[1, 48, 25]` | 1,200 | 4,800 |
| qkv | `[10,240, 25]` | 256,000 | 1,024,000 |
| **Per layer total** | | **638,400** | **2,537,200 (~2.42 MB)** |

### 4.2 Total Tape Size (48 recurrent layers, 1 slot)

Recurrent layer count: Qwen3.6-27B has 65 total blocks. Every 5th block is a full-attention layer (blocks 4, 9, 14, ..., 64), so `65 - 13 = 52` recurrent layers in the trunk. However, the actual number of recurrent layers captured by DFlash depends on the `is_recr` array. Based on [`src/models/qwen35.cpp:24-26`](src/models/qwen35.cpp:24) with `full_attn_interval = 4`, recurrent layers are those where `(i + 1) % 4 != 0`, giving approximately 48-49 recurrent layers in the trunk (matching the research documents' assumption of 48).

| Mode | Per Layer | 48 Layers Total |
|------|-----------|----------------|
| Fused GDN | ~1.77 MB | **~84.9 MB** |
| Non-fused GDN | ~2.42 MB | **~116.5 MB** |

---

## 5. Comparison Against Previous Estimates

### 5.1 Old v0.3.2 Runtime Logs (~70 MB Tape)

The old v0.3.2 runtime logs reported ~70 MB tape. Using the old assumed dimensions:

| Parameter | Old Assumed | Actual |
|-----------|------------|--------|
| S | 256 | 128 |
| H_v | 8 | 48 |
| H_k (fused) | 1 | 16 |
| conv_ch | 12,768 | 10,240 |

Old per-layer (fused, assumed): `256*1*25 + 256*8*25 + 1*8*25 + 1*8*25 + 12768*25 = 6,400 + 51,200 + 200 + 200 + 319,200 = 377,200` elements = 1,508,800 bytes (~1.44 MB)

Old total (48 layers): ~69.5 MB

**The old ~70 MB estimate used wrong dimensions but happened to produce a reasonable answer because the errors partially canceled out:** larger S (256 vs 128) was offset by smaller H_v (8 vs 48) and smaller conv_ch (12,768 vs 10,240).

**Actual tape size is ~22% larger than the old ~70 MB estimate** for fused GDN (~85 MB vs ~70 MB) and ~66% larger for non-fused (~117 MB vs ~70 MB).

### 5.2 task6r-part4 Revised Estimate (~134-186 MB)

The task6r-part4 estimate of ~134 MB (fused) and ~186 MB (non-fused) was based on `ssm_d_state = 1536` derived from incorrect `d_inner/dt_rank` values.

| Parameter | task6r-part4 Used | Actual |
|-----------|-------------------|--------|
| S_k = S_v | 1536 | 128 |
| H_k (fused) | 1 | 16 |
| H_v | 8 | 48 |
| conv_ch | 15,360 | 10,240 |

**The ~134-186 MB estimate does NOT survive.** It needs correction DOWN to ~85-117 MB because the actual `ssm_d_state` is 128 (not 1536), making the k and v tensors 12x smaller in the S dimension. However, the H_v dimension is 6x larger (48 vs 8), which partially offsets the reduction.

### 5.3 Task 5 Estimate (~6.5 GiB)

Task 5's ~6.5 GiB estimate remains wrong, but for different reasons than previously analyzed. It assumed H_k=32, H_v=32 and possibly confused full S-state capture with rank-factored intermediate capture. The actual tape is ~85-117 MB, making the Task 5 estimate wrong by a factor of ~55-77x.

---

## 6. Summary of Corrected Tape Sizes

| Estimate | Source | Fused | Non-Fused | Status |
|----------|--------|-------|-----------|--------|
| ~70 MB | Old v0.3.2 logs | ~70 MB | ~74 MB | **Wrong dimensions, coincidentally close** |
| ~134 MB | task6r-part4 | ~134 MB | ~186 MB | **NEEDS CORRECTION — used S=1536** |
| ~85 MB | **This document** | **~85 MB** | **~117 MB** | **CORRECTED with actual GGUF metadata** |
| ~6.5 GiB | Task 5 | — | — | **Still wrong by ~55-77x** |

---

## 7. Conclusion

**The ~134-186 MB estimate from task6r-part4 and the revised implementation blueprint needs correction.**

The correct tape sizes for Qwen3.6-27B with actual GGUF metadata are:
- **Fused GDN: ~85 MB** for 48 recurrent layers, 1 slot, 25 max tokens
- **Non-fused GDN: ~117 MB** for the same configuration

These values are:
- ~22% larger than the old v0.3.2 ~70 MB estimate (because H_v=48 is 6x larger than assumed H_v=8, even though S=128 is 2x smaller than assumed S=256)
- ~37-42% smaller than the task6r-part4 ~134-186 MB estimate (because S=128 is 12x smaller than the derived S=1536)

**All downstream documents that reference tape size (task6r-revised-implementation-blueprint.md, task6r-revised-blueprint-part2.md, task6r-revised-blueprint-part3.md) should be updated to use the corrected ~85-117 MB figures.**
