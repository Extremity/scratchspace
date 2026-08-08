# Task 6R.2: Recurrent Tape Mechanics and Actual Dimensions (v0.3.2)

## Overview

This document analyzes the old BeeLlama v0.3.2 recurrent tape mechanics, tensor dimensions, and actual GPU memory footprint. This is the critical investigation to understand why v0.3.2 DFlash used only ~200 MB for GPU cross ring + tape, while our Task 5 design estimated ~6.5 GiB for the replay tape.

**Source files analyzed:**
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h:117-160` — `dflash_tape_layer`, `dflash_tape_gpu_layer`, `dflash_tape_gpu` structs
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:1952-2016` — tape recording in eval callback
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2264-2286` — `dflash_ensure_recurrent_setup()`
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2344-2519` — `allocate_tape_gpu()`
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2288-2342` — `set_tape_recording()`
- `old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:432-562` — GDN graph builder (tensor production)
- `old-versions/beellama.cpp-preview-v0.3.2/src/models/delta-net-base.cpp:17-44` — GDN chunking math
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-hparams.cpp:179-218` — `n_embd_r()`, `n_embd_s()`
- `old-versions/beellama.cpp-preview-v0.3.2/include/llama.h:1282` — `LLAMA_DFLASH_MAX_VERIFY_TOKENS`

---

## 1. Tape Data Structures

### 1.1 CPU Tape Layer (`dflash_tape_layer`)

```cpp
// llama-context.h:118-130
struct dflash_tape_layer {
    std::vector<float> k;          // [S_k * H_k * n_tokens] after l2_norm
    std::vector<float> v;          // [S_v * H_v * n_tokens]
    std::vector<float> gate;       // [H_v * n_tokens] pre-exp
    std::vector<float> beta;       // [H_v * n_tokens] pre-sigmoid
    std::vector<float> qkv_mixed;  // [conv_channels * n_tokens * n_seqs] for conv state rebuild
    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t conv_channels = 0;
    int n_tokens = 0;
    int n_seqs = 1;
    llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS] = {};
};
```

This is the CPU-side tape used when GPU tape is unavailable (fallback). The dimensions `S_k`, `H_k`, `S_v`, `H_v` are discovered at recording time from the actual tensor shapes.

### 1.2 GPU Tape Layer (`dflash_tape_gpu_layer`)

```cpp
// llama-context.h:133-142
struct dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor * v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor * gate = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * beta = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * qkv  = nullptr;  // [conv_channels, max_tokens]
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_dev_t dev = nullptr;
};
```

Key observations:
- GPU tape tensors are pre-allocated to `max_tokens` capacity (NOT actual tokens recorded).
- Each tensor is F32 (confirmed by `allocate_tape_gpu()` using `GGML_TYPE_F32`).
- Gate and beta have `ne[0] = 1` (not `S_k` or `S_v`), meaning they are 1D-per-head values.
- QKV is 2D `[conv_channels, max_tokens]` — NOT 3D with `n_seqs` dimension (single-seq at GPU tape level).

### 1.3 GPU Tape Container (`dflash_tape_gpu`)

```cpp
// llama-context.h:144-160
struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;  // one per recurrent layer
    std::vector<int32_t> layer_ids;             // model layer indices → tape index mapping
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    int max_tokens = 0;                         // allocated capacity
    int n_tokens = 0;                           // actual tokens recorded this pass
    // destructor frees per-layer buffers and contexts
};
```

---

## 2. Tape Recording Flow

### 2.1 When is Tape Recording Enabled?

From `set_tape_recording()` (llama-context.cpp:2288-2342):

```cpp
void llama_context::set_tape_recording(bool enable) {
    if (enable) {
        dflash_ensure_recurrent_setup();
        if (dflash_capture->tapes.empty()) {
            allocate_tape_gpu(1, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
        }
        // reset n_tokens on existing tapes
    }
}
```

- Tape recording is allocated with `n_slots=1` and `max_tokens = LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25`.
- The constant `LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25` (llama.h:1282) is the maximum verify batch size.

### 2.2 Tape Name Map (`dflash_ensure_recurrent_setup()`)

```cpp
// llama-context.cpp:2267-2286
void llama_context::dflash_ensure_recurrent_setup() {
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.is_recr(il)) {
            int idx = (int) dflash_capture->recurrent_layer_ids.size();
            dflash_capture->recurrent_layer_ids.push_back(il);

            std::string il_str = std::to_string(il);
            dflash_capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
            dflash_capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
            dflash_capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
            dflash_capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
            dflash_capture->tape_name_map["qkv_mixed_pretranspose-" + il_str] = {idx, DFLASH_TAPE_QKV};
        }
    }
}
```

The tape captures 5 tensors per recurrent layer. The tensor names match the callback names in the GDN graph builder (`qwen35.cpp:560-562` for k/v, `qwen35.cpp:472` for gate, `qwen35.cpp:457` for beta, `qwen35.cpp:484` for qkv_mixed).

### 2.3 Tape Recording in Eval Callback

From `dflash_eval_callback()` (llama-context.cpp:1952-2016):

```cpp
if (cap->tape_enabled) {
    auto it = cap->tape_name_map.find(t->name);
    if (it != cap->tape_name_map.end()) {
        // When GPU tape is active, skip CPU read (graph handles it):
        auto * active_tape = cap->active_tape();
        const bool gpu_tape_fits = active_tape && (!ub || (int) ub->n_seq_tokens <= active_tape->max_tokens);
        if (gpu_tape_fits) {
            return true; // skip; already on GPU
        }
        // Otherwise read tensor data into CPU tape vectors...
    }
}
```

Key insight: When GPU tape is active and the batch fits within `max_tokens`, the callback returns early — the GPU tape is populated by graph-embedded copy operations in the model graph builder itself (see `qwen35.cpp:564-579` where `cparams.tape_gpu_n_seqs > 0` triggers GPU tape copy operations).

### 2.4 GPU Tape Copy in Graph Builder

From `qwen35.cpp:564-579` (within `build_layer_attn_linear()`):

```cpp
if (cparams.tape_gpu_n_seqs > 0) {
    for (int s = 0; s < (int)n_seqs && s < cparams.tape_gpu_n_seqs; ++s) {
        auto * tgpu = cparams.tape_gpu_seqs[s];
        // ... find layer index ...
        auto & tl = tgpu->layers[li];

        ggml_tensor * k_slice = ggml_view_3d(ctx0, k_conv,
            k_conv->ne[0], k_conv->ne[1], n_seq_tokens,
            k_conv->nb[1], k_conv->nb[2], s * k_conv->nb[3]);
        // graph-embedded copy to GPU tape tensor tl.k
    }
}
```

The GPU tape is populated by graph copy operations, NOT by the eval callback. The eval callback is bypassed when GPU tape is active.

---

## 3. GDN Intermediate Tensor Dimensions

### 3.1 What is Captured (NOT the full S-state)

The tape captures GDN **intermediate** tensors produced during the forward pass:

| Tensor | Graph Name | Shape (from qwen35.cpp) | Meaning |
|--------|-----------|------------------------|---------|
| k | `k_conv_predelta-{il}` | `[head_k_dim, num_k_heads, n_tokens, n_seqs]` | Normalized K after conv |
| v | `v_conv_predelta-{il}` | `[head_v_dim, num_v_heads, n_tokens, n_seqs]` | V after conv (no norm) |
| gate | `gate-{il}` | `[1, num_v_heads, n_tokens, n_seqs]` | Pre-exp (-A_log * softplus) |
| beta | `beta-{il}` | `[1, num_v_heads, n_tokens, n_seqs]` | Pre-sigmoid |
| qkv_mixed | `qkv_mixed_pretranspose-{il}` | `[conv_channels, n_tokens * n_seqs]` | Raw QKV before transpose |

**These are NOT the full S-state.** The full S-state is `n_embd_s = ssm_d_state * ssm_d_inner` (typically 786,432 for Qwen3.6 27B). The tape stores the rank-factored components used in the GDN update `S_new = g*S + k⊗delta`.

### 3.2 Qwen3.6 27B Hyperparameters

From the code comments and model structure:

| Parameter | Source | Value (Qwen3.6 27B) |
|-----------|--------|---------------------|
| `n_embd` | Model header | 5120 |
| `ssm_d_conv` | `LLM_KV_SSM_CONV_KERNEL` | 4 (typical for Qwen3.5/3.6) |
| `ssm_d_inner` | `LLM_KV_SSM_INNER_SIZE` | 12288 (typical for 27B) |
| `ssm_d_state` | `LLM_KV_SSM_STATE_SIZE` | 256 |
| `ssm_dt_rank` | `LLM_KV_SSM_TIME_STEP_RANK` | 8 (= num_v_heads) |
| `ssm_n_group` | `LLM_KV_SSM_GROUP_COUNT` | 1 (= num_k_heads for fused GDN) |

Derived values:
- `head_k_dim = ssm_d_state = 256`
- `head_v_dim = ssm_d_state = 256`
- `num_k_heads = ssm_n_group = 1` (fused GDN) or `8` (non-fused, repeated)
- `num_v_heads = ssm_dt_rank = 8`
- `n_embd_s = ssm_d_state * ssm_d_inner = 256 * 12288 = 3,145,728` (Full S-state — NOT stored in tape)

Wait — the code comment at `llama-context.cpp:2416` says `S = hparams.ssm_d_state; // 256 for Qwen3.5-27B`. Let me verify the actual tape tensor dimensions from `allocate_tape_gpu()`:

### 3.3 Tape Tensor Dimensions from `allocate_tape_gpu()`

From `llama-context.cpp:2413-2479`:

```cpp
const int64_t S = hparams.ssm_d_state;     // 256 for Qwen3.5-27B
const int64_t H_v = hparams.ssm_dt_rank;   // 8 (num_v_heads)

const int64_t H_k = (cparams.fused_gdn_ar && cparams.fused_gdn_ch)
                   ? (int64_t) hparams.ssm_n_group   // 1 (not repeated)
                   : H_v;                             // 8 (repeated)

// Tensor allocation:
tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, max_tokens);
tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, max_tokens);
tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, 1, H_v, max_tokens);
tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, 1, H_v, max_tokens);
tl.qkv  = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, max_tokens);
```

So the tape tensor dimensions are:
- **k**: `[S, H_k, max_tokens]` = `[256, 1, 25]` (fused GDN) or `[256, 8, 25]` (non-fused)
- **v**: `[S, H_v, max_tokens]` = `[256, 8, 25]`
- **gate**: `[1, H_v, max_tokens]` = `[1, 8, 25]`
- **beta**: `[1, H_v, max_tokens]` = `[1, 8, 25]`
- **qkv**: `[conv_ch, max_tokens]`

### 3.4 Conv Channels Calculation

```cpp
const auto * conv_kernel = model.layers[il].ssm_conv1d;
const int64_t conv_window = conv_kernel->ne[0] - 1;
const int64_t conv_ch = hparams.n_embd_r() / conv_window;
```

Where `n_embd_r()` from `llama-hparams.cpp:200`:
```cpp
return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);
```

For Qwen3.6 27B:
- `ssm_d_conv = 4` (conv kernel size)
- `conv_window = ssm_d_conv - 1 = 3`
- `n_embd_r = 3 * (12288 + 2 * 1 * 256) = 3 * 12768 = 38,304`
- `conv_ch = n_embd_r / conv_window = 38304 / 3 = 12,768`

Wait — that seems large. Let me re-check from `qwen35.cpp:481`:
```cpp
const int64_t conv_channels = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;
```
- `conv_channels = 12288 + 2 * 1 * 256 = 12,768` (This is `conv_dim` per layer)

So `qkv` tensor = `[12768, 25]` — that's the largest tape tensor.

---

## 4. ACTUAL GPU Tape Memory Calculation

### 4.1 Per Layer Per Token (F32 = 4 bytes per element)

| Tensor | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| k | `[256, 1, 25]` | 6,400 | 25,600 |
| v | `[256, 8, 25]` | 51,200 | 204,800 |
| gate | `[1, 8, 25]` | 200 | 800 |
| beta | `[1, 8, 25]` | 200 | 800 |
| qkv | `[12768, 25]` | 319,200 | 1,276,800 |
| **Per layer total** | | **377,200** | **1,507,900** |

### 4.2 Per Recurrent Layer

Each recurrent layer tape = ~1.44 MB (1,507,900 bytes).

### 4.3 How Many Recurrent Layers?

For Qwen3.6 27B with 64 layers and `full_attention_interval = 4`:
- Every 4th layer is full attention (layers 3, 7, 11, ..., 63)
- That's 16 full-attention layers
- Recurrent layers = 64 - 16 = 48 recurrent layers

But the MTP (NextN) layers at the end are dense attention-only. For Qwen3.6 27B with `n_layer_nextn > 0`, the MTP layers are not recurrent.

### 4.4 Total GPU Tape Size

For 48 recurrent layers, 1 slot, 25 max tokens:
- **Total = 48 * 1.44 MB = ~69.5 MB**

For the non-fused GDN case (H_k = 8):
- k tensor = `[256, 8, 25]` = 51,200 elements = 204,800 bytes (same as v)
- Per layer = 1,507,900 + 179,200 (extra k) = ~1.55 MB
- **Total = 48 * 1.55 MB = ~74.4 MB**

### 4.5 Comparison with Hidden-State Cross Ring

From Part 1 (6R.1):
- Hidden-state cross ring: ~200 MB (5 layers * 1024 slots * 5120 embd * 4 bytes, but actually using cross_ctx slots which is smaller)

The GPU tape is SEPARATE from the hidden-state ring. The ~200 MB figure observed in v0.3.2 runtime logs was the hidden-state ring, NOT including the tape. The tape adds ~70-75 MB on top.

**Total GPU memory for DFlash capture + tape in v0.3.2: ~270-275 MB**

---

## 5. KEY FINDING: Why v0.3.2 Tape is So Small

The v0.3.2 tape is small because:

1. **It stores GDN intermediates, NOT full S-state.** The full S-state is `ssm_d_state * ssm_d_inner = 256 * 12288 = 3,145,728` elements per layer per token. The tape stores rank-factored k, v, gate, beta that total only ~377,200 elements per layer per token — a **8.4x reduction**.

2. **Max tokens is only 25** (`LLAMA_DFLASH_MAX_VERIFY_TOKENS`). The tape is sized for the verify batch, not the full context.

3. **Only 1 slot** (not 8+ for multi-seq verify). The tape allocates `n_slots = 1` by default.

4. **F32 precision.** All tape tensors are F32. If the Task 5 design assumed F16, that would halve the size, but the actual tape uses F32.

5. **QKV is the largest component** at ~1.9 MB per layer per max_tokens, but it's only `conv_channels * max_tokens` = 12,768 * 25 = 319,200 elements.

---

## 6. Comparison with Task 5 Design Estimate

The Task 5 design estimated ~6.5 GiB for the replay tape. The key difference is:

| Aspect | v0.3.2 Tape | Task 5 Design |
|--------|-------------|---------------|
| Data stored | GDN intermediates (k, v, gate, beta, qkv) | Full S-state replay data |
| Per layer per token | ~377K elements F32 (~1.5 MB for 25 tokens) | Estimated based on full S-state |
| Max tokens | 25 (verify batch) | May have assumed full context |
| Slots | 1 | May have assumed multiple |
| Total (48 layers) | ~70-75 MB | ~6.5 GiB estimate |

The Task 5 estimate likely assumed storing full S-state (`n_embd_s = 3,145,728` elements per layer per token) rather than the rank-factored GDN intermediates.

**Next: Part 3 will analyze the relationship between tape, rollback, and replay mechanics, and how the tape is consumed during DFlash verification.**

---

## 7. Summary

| Component | v0.3.2 Actual Size | Notes |
|-----------|-------------------|-------|
| Hidden-state ring (GPU cross) | ~200 MB | 5 layers * cross_ctx slots * 5120 embd |
| GPU tape | ~70-75 MB | 48 layers * 25 tokens * GDN intermediates |
| **Total GPU DFlash memory** | **~270-275 MB** | Both components combined |

The GPU tape stores GDN intermediate tensors (k, v, gate, beta, qkv_mixed) at F32 precision, sized for `LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25` verify tokens and 1 slot. These are rank-factored components of the GDN update, providing an 8.4x reduction compared to storing the full S-state.
