# Task 6R.1: Hidden-State Capture and GPU Cross Ring Analysis (v0.3.2)

## Overview

This document analyzes the old BeeLlama v0.3.2 hidden-state capture and GPU cross ring mechanisms to understand why the GPU cross ring + tape used only ~200 MB, compared to our Task 5 design estimate of ~6.5 GiB for the replay tape.

**Source files analyzed:**
- `old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp` — `common_speculative_impl_dflash` class
- `old-versions/beellama.cpp-preview-v0.3.2/include/llama.h` — public API declarations
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp` — `dflash_cross_ring_handle`, `init_cross_ring_gpu()`, write/read functions
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h` — `dflash_hidden_gpu`, `dflash_capture_data`, `dflash_tape_type`
- `old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml-cuda/cross-ring-interleave.cu` — CUDA implementation of GPU cross ring
- `old-versions/beellama.cpp-preview-v0.3.2/docs/beellama-features.md` — v0.3.2 feature documentation

---

## A. Hidden-State Capture

### What is Captured

The target model's hidden states (internal layer activations, i.e., the output of the residual stream / layer normalization at specific layers) are captured during target model evaluation. These hidden states are consumed by the DFlash drafter model for cross-attention conditioning.

### At Which Layers

The capture layers are specified in the DFlash drafter GGUF metadata as `target_layer_ids`. The number of capture layers is `n_target_layers`. At construction time:

```cpp
// speculative.cpp:2472-2476
capture_layers.assign(n_target_layers, 0);
const int n_read = llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
```

For Qwen3.6 DFlash, typical configurations capture 5 layers from the target model. The layer IDs are validated against the target model's layer range (speculative.cpp:2480-2484).

### Tensor Dimensions

Each captured layer produces hidden states of dimension `[n_embd]` per token. The drafter's `n_embd` must match the target's hidden size (enforced at speculative.cpp:2465-2467):

```cpp
if (n_embd != target_n_embd) {
    fail_contract("drafter n_embd must match target hidden size");
}
```

The total feature dimension per token across all captured layers is `n_target_features = n_embd * n_target_layers` (speculative.cpp:2468-2470).

For the runtime evidence showing `5 layers x 5120 embd`:
- `n_target_layers = 5`
- `n_embd = 5120`
- `n_target_features = 25,600`

### Where is it Stored

#### CPU Ring Buffer (`ring_buf`)

```cpp
// speculative.cpp:2097-2104
static constexpr int RING_SIZE = 4096;

// ring_buf[layer][slot * n_embd ... (slot+1) * n_embd - 1], slot = pos % RING_SIZE
std::vector<std::vector<float>> ring_buf; // [n_target_layers][RING_SIZE * n_embd]
int ring_write_pos = 0;    // next write slot (0..RING_SIZE-1)
int ring_filled = 0;       // how many valid slots (0..RING_SIZE)
int committed_len = 0;     // total tokens committed (unbounded counter)
bool cpu_ring_valid = true;
```

- Fixed-size circular buffer: 4096 tokens per captured layer.
- Allocation: `ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f)` (speculative.cpp:2573-2575).
- Memory footprint (CPU): `n_target_layers * RING_SIZE * n_embd * sizeof(float)` = 5 * 4096 * 5120 * 4 = **400 MB**.

#### GPU Cross Ring (`gpu_ring_handle`)

```cpp
// speculative.cpp:2119
void * gpu_ring_handle = nullptr;
```

- Opaque handle returned by `llama_dflash_cross_ring_gpu_init()` (speculative.cpp:2596).
- Points to `dflash_cross_ring_handle` struct containing GPU device pointers.
- The GPU ring uses `cross_ctx` slots (NOT `RING_SIZE`).

#### GPU Hidden Capture (`dflash_hidden_gpu`)

```cpp
// llama-context.h:162-179
struct dflash_hidden_gpu {
    std::vector<ggml_tensor *> layers;  // one [n_embd, max_tokens] tensor per captured layer
    std::vector<int32_t> layer_ids;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<ggml_context *> ctxs;
    int64_t n_embd = 0;
    int max_tokens = 0;
    int n_tokens = 0;
    // ...
};
```

- Per-slot GPU tensors holding captured hidden states from target model evaluation.
- One tensor per captured layer, sized `[n_embd, max_tokens]`.
- Accessed via `llama_get_layer_hidden()`, `llama_get_layer_hidden_n_tokens()`, `llama_get_layer_hidden_n_embd()`.
- When GPU capture is enabled, hidden states remain on GPU (device pointers), and `ring_write()` uses direct D2D copies via `llama_dflash_cross_ring_gpu_write_hidden()`.

### What `cross_ctx` Controls

`cross_ctx` (default 512, configurable via `--spec-dflash-cross-ctx`) controls:

1. **GPU cross ring size:** The GPU ring buffer is allocated for `cross_ctx` slots, NOT `RING_SIZE`. This is the key distinction.
2. **Cross-attention window:** The drafter sees at most `cross_ctx` recent hidden states. In `build_cross_data()`:
   ```cpp
   int cross_len = std::min(ring_filled, cross_ctx > 0 ? cross_ctx : ring_filled);
   ```
3. **GPU ring write position:** The GPU ring write position wraps at `cross_ctx`, not `RING_SIZE`:
   ```cpp
   int gpu_write_pos = ring_write_pos % cross_ctx;
   int gpu_filled = std::min(ring_filled, cross_ctx);
   ```

### When Hidden-State Data is Copied

Hidden states are copied during `ring_write()` (speculative.cpp:3426-3562):

1. **CPU path:** If `cpu_ring_should_track` is true (either no GPU ring or force-CPU), data is memcpy'd from the target's hidden buffer into `ring_buf[layer]`.
2. **GPU path:** If `gpu_ring_handle` exists:
   - If source data is on GPU (no CPU pointer available): use `llama_dflash_cross_ring_gpu_write_hidden()` for direct D2D copy.
   - If source data is on CPU: use `llama_dflash_cross_ring_gpu_write()` for H2D copy.
3. **Synchronization:** After GPU writes, `llama_dflash_cross_ring_gpu_synchronize()` is called.

The copy happens:
- After prefill (`capture_target_hiddens()` — speculative.cpp:3650).
- After each verification decode for accepted tokens (`append_target_hiddens()` — speculative.cpp:3687).
- During prefill flush (`flush_prefill()` — speculative.cpp:2698).

### Who Consumes the Hidden States

The DFlash drafter model consumes the hidden states through its cross-attention mechanism. The cross-attention data is built by `build_cross_data()` (speculative.cpp:2362-2394):

- **GPU path:** Calls `llama_dflash_cross_ring_gpu_set_cross()` which uses the GPU ring's interleave kernel to produce the drafter's cross-attention tensor directly on GPU.
- **CPU path:** Builds `cross_buf` from `ring_buf` and uploads via `llama_set_cross_data_seq()`.

---

## B. GPU Cross Ring

### Allocation

**Entry point:** `llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, cross_ctx)` (speculative.cpp:2596).

**Implementation** (`llama-context.cpp:9022-9082`):

1. Resolves CUDA/ROCm backend registry and function pointers via `ggml_backend_reg_get_proc_address()`.
2. Calls `dflash_cross_ring_gpu_alloc(n_layers, n_embd, ring_size)` where `ring_size = cross_ctx` (NOT 4096).

**CUDA allocation** (`cross-ring-interleave.cu:126-178`):

```cpp
struct dflash_cross_ring_gpu {
    int device;
    int n_layers;
    int n_embd;
    int ring_size;
    float ** d_layer_rings;   // per-layer ring buffers on device
    float *  d_staging;       // interleaved output staging buffer
    float ** h_layer_ptrs;    // host copy of per-layer device pointers
};
```

Allocation breakdown:
- Per-layer ring: `n_layers * ring_size * n_embd * sizeof(float)` bytes.
- Staging buffer: `ring_size * n_layers * n_embd * sizeof(float)` bytes.
- Device pointer array: `n_layers * sizeof(float*)` bytes (negligible).

### Writes

Three write functions:

1. **`llama_dflash_cross_ring_gpu_write(handle, layer, ring_pos, data, n_tokens, n_embd)`** — Host-to-device copy. Called when hidden-state data is available on CPU.

2. **`llama_dflash_cross_ring_gpu_write_hidden(handle, ctx, layer, ring_pos, src_offset, n_tokens, n_embd)`** — Direct device-to-device copy from target model's GPU hidden tensors. Falls back to D2H+H2D if D2D unavailable.

3. **`llama_dflash_prefill_gpu_write_hidden(handle, ctx, slot, layer, ring_pos, src_offset, n_tokens, n_embd)`** — D2D copy from prefill GPU staging buffer (used during suffix prefill capture).

All writes handle ring wrap-around (if `ring_pos + n_tokens > ring_size`, the copy is split into two segments).

### Reads

**`llama_dflash_cross_ring_gpu_set_cross(ctx, handle, seq_id, ring_write_pos, ring_filled, n_layers, n_embd, ctx_window)`** (llama-context.cpp:9291-9303):

1. Calls `fn_interleave(gpu_ring, ring_write_pos, ring_filled, ctx_window)` to launch the CUDA interleave kernel.
2. The kernel reads per-layer ring buffers and writes interleaved output to `d_staging`.
3. The drafter model's cross-attention tensor is set to point to `d_staging` via `ctx->set_cross_data_gpu()`.

**Interleave kernel** (`cross-ring-interleave.cu:104-124`):
```cpp
__global__ static void k_cross_ring_interleave(
    const float * const * __restrict__ d_rings,
    float * __restrict__ d_out,
    const int ring_size, const int read_start, const int cross_len,
    const int n_layers, const int n_embd) {
    const int t = blockIdx.x; // token index
    const int l = blockIdx.y; // layer index
    const int slot = (read_start + t) % ring_size;
    const float * src = d_rings[l] + (size_t)slot * n_embd;
    float * dst = d_out + (size_t)t * n_layers * n_embd + (size_t)l * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        dst[i] = src[i];
}
```

### Lifetime and Cleanup

- Allocated in constructor: `gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, cross_ctx)`.
- Freed in destructor: `llama_dflash_cross_ring_gpu_free(gpu_ring_handle)` (speculative.cpp:2612).
- The `dflash_cross_ring_handle` struct and its GPU buffers are freed by `dflash_cross_ring_gpu_free()` which calls `fn_free(gpu_ring)` (cross-ring-interleave.cu:206-218).

### Relation to DFlash Speculation

The GPU cross ring is the communication channel between target and drafter:

1. Target model evaluates, producing hidden states at capture layers.
2. Hidden states are written to GPU cross ring (D2D if on GPU, H2D if on CPU).
3. When drafter needs to generate tokens, `build_cross_data()` interleaves the ring into the drafter's cross-attention format.
4. Drafter runs cross-attention using the interleaved hidden states.
5. After verification, accepted tokens' hidden states are appended to the ring.
6. The ring position advances circularly.

---

## C. Memory Footprint: Why ~200 MB is Sufficient

### The Runtime Evidence

```
dflash gpu ring: allocated 5 layers x 1024 slots x 5120 embd + staging (~200 MB)
```

### The Calculation

From `cross-ring-interleave.cu:172-175`:
```cpp
size_t total_mb = ((size_t)ring_size * n_embd * sizeof(float) * n_layers +
                   (size_t)ring_size * n_layers * n_embd * sizeof(float)) / (1024 * 1024);
```

With `n_layers = 5`, `ring_size = 1024` (cross_ctx), `n_embd = 5120`:

| Component | Formula | Bytes | MB |
|---|---|---|---|
| Per-layer ring (5 layers) | 5 * 1024 * 5120 * 4 | 104,857,600 | 100 |
| Staging buffer | 1024 * 5 * 5120 * 4 | 104,857,600 | 100 |
| **Total GPU cross ring** | | **209,715,200** | **~200** |

### Key Insight: `cross_ctx` vs `RING_SIZE`

The GPU cross ring uses `cross_ctx` slots (default 512, in this runtime log shows 1024), NOT `RING_SIZE` (4096). This is the critical design decision:

- **CPU ring:** 4096 tokens per layer. Memory: 5 * 4096 * 5120 * 4 = ~400 MB.
- **GPU ring:** `cross_ctx` tokens per layer → 5 * 1024 * 5120 * 4 * 2 (ring + staging) = ~200 MB.

The GPU ring only holds the recent `cross_ctx` tokens that the drafter actually needs for cross-attention. The CPU ring holds the full 4096-token history as a fallback and for building cross data when GPU ring is unavailable.

### What About the Tape?

The v0.3.2 GPU cross ring does NOT include the replay tape. The tape (recurrent state recording for rollback) is a separate system:

- **Tape types** (llama-context.h:181-187): K, V, GATE, BETA, QKV for DeltaNet recurrent state.
- **Tape storage** (llama-context.h:234-235): `dflash_tape_gpu` per slot, plus `dflash_hidden_gpu` for hidden states.
- The tape records recurrent state at each token position for rollback after token acceptance/rejection.

The ~200 MB figure from the runtime log specifically refers to the GPU cross ring allocation. The tape would be allocated separately and its size depends on the number of DeltaNet layers, block size, and context length. In v0.3.2, the tape was likely much smaller because:

1. It only records recurrent state (conv state, gate/beta parameters) for DeltaNet layers, not full KV cache.
2. The tape is per-token (not per-context-position), so it only grows during active draft tokens.
3. The Drafter model (not the target) has the recurrent state, and the drafter is typically a small model.

---

## D. Summary

| Component | Size | Purpose |
|---|---|---|
| CPU ring buffer | `n_layers * 4096 * n_embd * 4` bytes (~400 MB) | Full hidden-state history, fallback path |
| GPU cross ring | `n_layers * cross_ctx * n_embd * 4 * 2` bytes (~200 MB) | Recent hidden states for drafter cross-attention |
| GPU hidden capture | `n_layers * [n_embd, max_tokens]` | Target model hidden states on GPU (eval-time buffer) |
| Replay tape | Per-DeltaNet-layer recurrent state | Rollback after accept/reject (separate from cross ring) |

The ~200 MB GPU cross ring is sufficient because it only stores the recent `cross_ctx` tokens of hidden states that the drafter needs for cross-attention. It does NOT store the replay tape, which is a separate allocation.
