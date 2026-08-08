# DFlash Custom Mode — Revised Implementation Blueprint (Part 2 of 3)

**Date:** 2026-08-08
**Continuation of:** `task6r-revised-implementation-blueprint.md`
**Based on:** Task 6R.1-6R.4 findings

---

## PART C: Revised Implementation Blueprint

### C.1 Every File That Needs Modification or Creation

#### New Files to Create

| File | Purpose | Est. Lines |
|------|---------|------------|
| `common/server-dflash-custom.h` | Custom mode state struct, tape GPU layer declarations, backup/replay function declarations | ~80 |
| `common/server-dflash-custom.cpp` | Backup cell copy, GPU tape allocation, GDN replay orchestration, capture integration | ~250 |

**Total new files: ~330 lines.**

#### Existing Files to Modify

| File | What Changes | Est. Lines Changed |
|------|-------------|-------------------|
| [`common/common.h`](common/common.h) | Add `beefix_dflash_custom` flag to `common_params_speculative` | +3 |
| [`common/arg.cpp`](common/arg.cpp) | Add `--beefix-dflash-custom` CLI argument | +8 |
| [`common/common.cpp`](common/common.cpp) | Override `n_rs_seq=0` when custom mode active | +10 |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | Add `n_backup_cells` member, `cell_copy()` declaration, `backup_offset()` helper | +15 |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | Extended tensor allocation, `cell_copy()` implementation | +60 |
| [`src/llama-model.cpp`](src/llama-model.cpp) | Pass backup cell count to recurrent memory constructor | +5 |
| [`src/models/qwen35.cpp`](src/models/qwen35.cpp) | Graph-embedded `ggml_cpy` for tape capture after tensor computation (THE key change) | +35 |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | Replay integration in `post_decode()`, pre-draft backup call | +70 |
| [`tools/server/server-context.h`](tools/server/server-context.h) | Add custom mode state to `server_slot` | +10 |

**Total modified lines: ~216 lines across 9 files.**

**Grand total: ~546 lines of new/modified code.**

---

### C.2 Every Function — Existing and New

#### Existing Functions Used (No Modification Required)

| Function | Location | How Used |
|----------|----------|----------|
| `ggml_gated_delta_net()` | [`ggml/src/ggml.c:6426`](ggml/src/ggml.c:6426) | Core replay operation — receives q=zeros, captured k/v/gate/beta, backup state; produces updated state |
| `ggml_backend_tensor_copy()` | ggml backend API | Used by `cell_copy()` for device-native row-to-row copies |
| `ggml_new_tensor_3d()` / `ggml_new_tensor_2d()` | ggml core | Allocate GPU tape tensors |
| `ggml_cpy()` | ggml core | Graph-embedded copy operations for tape capture |
| `ggml_view_3d()` / `ggml_view_2d()` | ggml core | Create views into tape tensors for specific token ranges |
| `ggml_backend_alloc_ctx_tensors_from_buft()` | ggml backend | Allocate tape tensors on device-specific buffer |
| `ggml_backend_dev_buffer_type()` | ggml backend | Resolve device → buffer type for device-aware tape placement |
| `model.dev_layer(il)` | [`src/llama-model.cpp`](src/llama-model.cpp) | Get device for layer il (for device-aware tape placement) |
| `common_sampler_sample_and_accept_n()` | sampler | Returns accepted tokens after verification (unchanged) |
| `common_context_seq_rm()` | context | Remove rejected KV after replay (unchanged) |
| `need_n_rs_seq()` | [`common/common.h:417`](common/common.h:417) | Returns n_rs_seq for spec type — overridden to 0 for custom DFlash |

#### New Functions Required

| Function | File | Signature | Purpose |
|----------|------|-----------|---------|
| `dflash_custom_backup()` | `server-dflash-custom.cpp` | `void dflash_custom_backup(llama_memory_recurrent * mem, uint32_t n_cells)` | Pre-draft backup: copy active R/S rows to backup rows |
| `dflash_custom_restore()` | `server-dflash-custom.cpp` | `void dflash_custom_restore(llama_memory_recurrent * mem, uint32_t n_cells)` | Post-replay restore: copy backup rows back to active rows |
| `dflash_custom_tape_alloc()` | `server-dflash-custom.cpp` | `server_dflash_tape_gpu * dflash_custom_tape_alloc(llama_model * model, ggml_backend_t backend, int max_tokens)` | Allocate GPU tape tensors with device-aware placement |
| `dflash_custom_tape_free()` | `server-dflash-custom.cpp` | `void dflash_custom_tape_free(server_dflash_tape_gpu * tape)` | Free GPU tape tensors |
| `dflash_custom_replay()` | `server-dflash-custom.cpp` | `bool dflash_custom_replay(server_dflash_custom_state * state, llama_context * ctx, int n_accepted)` | Build and execute GDN replay graph for accepted tokens |
| `dflash_custom_init()` | `server-dflash-custom.cpp` | `server_dflash_custom_state * dflash_custom_init(llama_model * model, int n_draft_max, int n_parallel)` | Initialize custom mode state at server startup |
| `dflash_custom_free()` | `server-dflash-custom.cpp` | `void dflash_custom_free(server_dflash_custom_state * state)` | Cleanup custom mode state |
| `dflash_custom_is_enabled()` | `server-dflash-custom.h` | `bool dflash_custom_is_enabled(const common_params_speculative & spec)` | Check if custom mode should be active |

---

### C.3 Every Data Structure

#### `server_dflash_tape_gpu_layer` (new, in `server-dflash-custom.h`)

Mirrors old [`dflash_tape_gpu_layer`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h:133):

```cpp
struct server_dflash_tape_gpu_layer {
    ggml_tensor *      k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor *      v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor *      gate = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor *      beta = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor *      qkv  = nullptr;  // [conv_channels, max_tokens]
    ggml_backend_buffer_t buf = nullptr;
    ggml_context *     ctx = nullptr;
    ggml_backend_dev_t dev = nullptr;
};
```

**Purpose:** Per-layer GPU tape storage for one recurrent layer. Each tensor is pre-allocated to `max_tokens` capacity and F32 precision.

**Allocation:** Device-aware — placed on the same GPU as the corresponding model layer.

#### `server_dflash_tape_gpu` (new, in `server-dflash-custom.h`)

```cpp
struct server_dflash_tape_gpu {
    std::vector<server_dflash_tape_gpu_layer> layers;  // one per recurrent layer
    std::vector<uint32_t>                     layer_ids; // model layer indices → tape index
    int                                       max_tokens = 0;
    int                                       n_tokens = 0;
};
```

**Purpose:** Container for all recurrent layer tapes. The `layer_ids` vector maps model layer index to tape layer index.

#### `server_dflash_custom_state` (new, in `server-dflash-custom.h`)

```cpp
struct server_dflash_custom_state {
    // Tape
    server_dflash_tape_gpu * tape = nullptr;

    // Tape metadata
    uint32_t n_layers = 0;           // number of recurrent layers
    uint32_t S_k = 0;                // key state dimension
    uint32_t S_v = 0;                // value state dimension
    uint32_t H_k = 0;                // key head count
    uint32_t H_v = 0;                // value head count
    uint32_t conv_channels = 0;      // conv output channels
    uint32_t max_tokens = 0;         // max draft tokens
    uint32_t tokens_captured = 0;    // actual tokens captured this cycle

    // Replay state
    bool enabled = false;            // is custom mode active
    bool replay_failed = false;      // permanent disable flag
    uint32_t fail_count = 0;         // consecutive failure counter

    // Graph context for replay
    ggml_context * replay_ctx = nullptr;
    ggml_backend_sched_t replay_sched = nullptr;
};
```

**Lifetime:** Created at server init when `--beefix-dflash-custom` is detected. Destroyed at server shutdown. Per-slot instance.

#### Extended `llama_memory_recurrent` Members (in `llama-memory-recurrent.h`)

```cpp
class llama_memory_recurrent {
    // ... existing members ...

    uint32_t n_backup_cells = 0;  // extra rows for backup cells (0 = disabled)

    uint32_t backup_offset() const {
        return mem_size * (1 + n_rs_seq);
    }

    void cell_copy(uint32_t src_row, uint32_t dst_row) const;
};
```

---

### C.4 Memory Accounting

#### C.4.1 GPU Tape Tensors

**CORRECTION (2026-08-08):** The table below has been updated with actual Qwen3.6 GGUF metadata. Previous values used S_k=1536, H_k=1/8, H_v=8, conv_channels=15,360. Corrected values are S_k=128, H_k=16/48, H_v=48, conv_channels=10,240.

| Tensor | Shape | Elements (Qwen3.6, fused) | Bytes (F32) | Per 48 Layers |
|--------|-------|--------------------------|-------------|---------------|
| k | `[S_k, H_k, max_tokens]` = `[128, 16, 25]` | 51,200 | 204,800 | 9.8 MB |
| v | `[S_v, H_v, max_tokens]` = `[128, 48, 25]` | 153,600 | 614,400 | 29.7 MB |
| gate | `[1, H_v, max_tokens]` = `[1, 48, 25]` | 1,200 | 4,800 | 0.2 MB |
| beta | `[1, H_v, max_tokens]` = `[1, 48, 25]` | 1,200 | 4,800 | 0.2 MB |
| qkv | `[conv_channels, max_tokens]` = `[10240, 25]` | 256,000 | 1,024,000 | 49.2 MB |
| **Per layer total** | | **463,200** | **1,852,800 (~1.77 MB)** | **~85 MB** |

**For non-fused GDN (H_k = 48):** k tensor grows to `[128, 48, 25]` = 153,600 elements = 614,400 bytes. Per layer = ~2.42 MB. Total = ~117 MB.

**Allocation:** Pre-allocated once at server init. Reused every cycle. Device-aware placement (each layer's tape on the same GPU as the model layer).

**Lifetime:** Server lifetime. No per-cycle allocation.

#### C.4.2 Backup Cells

**CORRECTION (2026-08-08):** This section has been revised based on [`task6r-correction-part2-backup-cells.md`](task6r-correction-part2-backup-cells.md). The previous estimate used `n_backup_cells = n_parallel * 2 = 8` extra cells. The old v0.3.2 code proved that `n_backup_cells = n_parallel = 4` is sufficient (1 backup cell per slot). This reduces backup cell VRAM from ~1,246 MB to ~612 MB.

The R tensor has shape `[n_embd_r, n_rows]` and the S tensor has shape `[n_embd_s, n_rows]`. The current allocation is `n_rows = mem_size * (1 + n_rs_seq)`. With `n_rs_seq=0` and `mem_size = n_ctx * n_parallel`, that's `n_rows = n_ctx * n_parallel`.

Backup cells add `n_backup_cells` extra rows. For `n_backup_cells = n_parallel = 4` (CORRECTED from `n_parallel * 2 = 8`):

| Component | Per Row | Total (4 backup rows) |
|-----------|---------|----------------------|
| R per row | `n_embd_r * n_layers * 4B` = `30720 * 48 * 4` = 5.9 MB | 23.5 MB |
| S per row | `n_embd_s * n_layers * 4B` = `786432 * 48 * 4` = 149.9 MB | 599.6 MB |

**Total backup cells: ~623 MB for 4 backup cells (CORRECTED from ~1,246 MB for 8 cells).**

The old v0.3.2 used `mem_size = 2 * n_parallel = 8` total cells (4 normal + 4 backup), giving 4 backup cells. The 6R proposal incorrectly used `n_backup_cells = n_parallel * 2 = 8`, doubling the necessary backup cells. The old code proved 1 backup cell per slot is sufficient for the rollback-then-replay pattern.

**Scaling behavior:**
- Linear with `n_parallel` (backup cells = `n_parallel`, NOT `n_parallel * 2`).
- Linear with `n_layers` (each layer has R and S state).
- Independent of `n_ctx` (backup cells are per-cell, not per-context).

#### C.4.3 Replay Graph Context

| Component | Size | Notes |
|-----------|------|-------|
| `replay_ctx` | ~1-2 MB | ggml context for replay graph tensors (views into tape + state) |
| `replay_sched` | Minimal | Reuses existing backend scheduler |
| Per-layer replay graph | ~10 tensors | q_zeros, k_view, v_view, g_view, b_view, s_view, output |

**Lifetime:** Allocated once at init, reused every cycle. Graph is rebuilt each cycle but context is persistent.

#### C.4.4 Total Memory Summary

**CORRECTION (2026-08-08):** Updated tape size from ~134-186 MB to ~85-117 MB, and backup cells from ~1,246 MB (8 cells) to ~623 MB (4 cells).

| Component | Location | Size (Qwen3.6, n_parallel=4) | Lifetime |
|-----------|----------|--------------------------------|----------|
| GPU tape (fused) | GPU | ~85 MB | Server lifetime |
| GPU tape (non-fused) | GPU | ~117 MB | Server lifetime |
| Backup cells (4) | GPU | ~623 MB | Server lifetime |
| Base RS (n_rs_seq=0) | GPU | ~599 MB | Server lifetime |
| Replay graph context | GPU | ~1-2 MB | Server lifetime |
| Draft model | GPU | ~800 MB | Server lifetime |
| **Total DFlash GPU overhead** | | **~2,108 - 2,140 MB** | |
| **vs Current upstream** | | **~6,187 MB** | |
| **VRAM saved** | | **~4,047 - 4,079 MB (~4.0-4.1 GB)** | |

**Note (CORRECTED):** The revised design has LOWER GPU overhead than the previous estimate (~2,865 MB) due to two corrections: (1) tape size reduced from ~134-186 MB to ~85-117 MB using actual GGUF metadata, and (2) backup cells reduced from ~1,246 MB (8 cells) to ~623 MB (4 cells) matching old v0.3.2's approach. The total DFlash overhead is now ~2.1 GB vs current upstream's ~6.2 GB, saving ~4 GB VRAM.

---

### C.5 Control Flow — Complete Custom DFlash Lifecycle

#### Phase 0: Initialization (Server Startup)

```
1. Parse --beefix-dflash-custom flag
2. Override n_rs_seq = 0 for DFlash
3. Construct recurrent memory with n_backup_cells = n_parallel * 2
4. Allocate GPU tape with device-aware placement
5. Initialize replay graph context and scheduler
```

**Entry point:** [`common/common.cpp:1770`](common/common.cpp:1770) for n_rs_seq override.
**Tape allocation:** `dflash_custom_init()` called from server initialization.

#### Phase 1: Pre-Draft Backup

```
For each parallel cell:
    cell_copy(src=active_row, dst=backup_row)
```

**Entry point:** Before `common_speculative_draft()` at [`server-context.cpp:3264`](tools/server/server-context.cpp:3264).
**Cost:** ~1.2 ms (4 cells × 156 MB / 500 GB/s device-native copy). **CORRECTION (2026-08-08):** Updated from 8 cells to 4 cells.

#### Phase 2: Draft with Capture

```
For each draft token:
    Forward pass through target model
    At each recurrent layer (qwen35.cpp):
        After computing k_conv, v_conv, gate, beta, qkv_mixed:
            ggml_cpy(k_conv -> tape_k[layer][token])
            ggml_cpy(v_conv -> tape_v[layer][token])
            ggml_cpy(gate   -> tape_g[layer][token])
            ggml_cpy(beta   -> tape_b[layer][token])
            ggml_cpy(qkv    -> tape_qkv[layer][token])
    GDN consumes k, v, gate, beta (unchanged)
```

**Entry point:** [`src/models/qwen35.cpp:446-450`](src/models/qwen35.cpp:446) — after tensor computation, before `build_recurrent_attn()`.
**Cost:** ~0 ms (graph-embedded copies overlap with forward pass compute).

#### Phase 3: Verify (Unchanged)

```
common_sampler_sample_and_accept_n() returns accepted tokens.
```

**Entry point:** [`server-context.cpp:4213`](tools/server/server-context.cpp:4213).

#### Phase 4: Replay (New)

```
If partial acceptance (n_accepted > 0 and n_accepted < n_draft):
    1. Restore backup state to active cells:
        cell_copy(src=backup_row, dst=active_row)

    2. Build GDN replay graph:
        For each recurrent layer:
            q_zeros = ggml_zeros([S_k, H_k, n_accepted, 1])
            k_view  = view(tape_k[layer], tokens 0..n_accepted)
            v_view  = view(tape_v[layer], tokens 0..n_accepted)
            g_view  = view(tape_g[layer], tokens 0..n_accepted)
            b_view  = view(tape_b[layer], tokens 0..n_accepted)
            s_backup = view(s_l[layer], backup_row)

            output = ggml_gated_delta_net(q_zeros, k_view, v_view,
                                          g_view, b_view, s_backup, K=n_accepted)
            ggml_build_forward_expand(replay_graph, output)

    3. Execute replay graph:
        ggml_backend_sched_graph_compute(replay_sched, replay_graph)

    4. Write replayed state to active R/S rows:
        For each layer:
            copy_gdn_output_to_active(output[layer], s_l[layer], active_row)
```

**Entry point:** Between checkpoint path and RS path at [`server-context.cpp:4263-4267`](tools/server/server-context.cpp:4263).
**Cost:** ~0.1 ms (48 layers × GDN replay with n_accepted tokens, GPU-native).

#### Phase 5: Cleanup (Unchanged)

```
common_context_seq_rm() removes rejected KV beyond n_accepted.
```

**Entry point:** [`server-context.cpp:4319-4322`](tools/server/server-context.cpp:4319).

#### Special Cases

| Case | Behavior |
|------|----------|
| Full acceptance (all draft tokens accepted) | No replay needed. Backup state discarded. KV retained. |
| Zero acceptance (no draft tokens accepted) | No replay needed. Restore backup state (cell_copy only). Checkpoint rollback as fallback. |
| Replay failure | Fall back to checkpoint rollback (`ckpt.load_tgt()` → `seq_rm()` → restore sampler). |

---

### C.6 Performance Analysis

#### C.6.1 GPU Computation

| Operation | GPU Cycles | Estimated Time |
|-----------|-----------|----------------|
| Pre-draft backup (cell_copy) | Memory copy, 4 cells × 156 MB | ~1.2 ms (500 GB/s) |
| Draft forward pass | Unchanged from baseline | Baseline |
| Tape capture (ggml_cpy) | Memory copy, batched with compute | ~0 ms (overlaps) |
| GDN replay | 48 layers × n_accepted tokens × rank-1 update | ~0.1 ms |
| State write-back | Memory copy, 48 layers × S-state | ~0.3 ms |
| **Total custom overhead** | | **~1.6 ms per cycle** |

**CORRECTION (2026-08-08):** Updated from ~2.8 ms to ~1.6 ms due to backup cells reduced from 8 to 4.

#### C.6.2 CPU Computation

| Operation | Estimated Time |
|-----------|---------------|
| Graph construction (replay) | ~0.5 ms (48 layers × view + GDN node creation) |
| Failure handling | ~0.1 ms (exception catch, counter increment) |
| **Total CPU overhead** | **~0.6 ms per cycle** |

#### C.6.3 Synchronization

| Sync Point | Type | Estimated Time |
|------------|------|---------------|
| Backup cell_copy completion | GPU sync (implicit in graph scheduling) | ~0 ms |
| Replay graph execution | GPU sync (explicit after sched_graph_compute) | ~0.1 ms |
| **Total sync overhead** | | **~0.1 ms** |

#### C.6.4 PCIe Transfers

| Transfer | Size | Time |
|----------|------|------|
| None (GPU-native tape) | 0 | 0 ms |

**This is the key improvement over the previous blueprint**, which required ~13 GB/cycle of PCIe transfers.

#### C.6.5 Graph Construction

| Graph | Tensors | Build Time |
|-------|---------|-----------|
| Replay graph | ~10 tensors per layer × 48 layers = ~480 tensors | ~0.5 ms |

#### C.6.6 Memory Allocation

| Allocation | When | Size | Frequency |
|-----------|------|------|-----------|
| GPU tape | Server init | ~134-186 MB | Once |
| Backup cells | Server init | ~1,246 MB | Once |
| Replay graph context | Server init | ~1-2 MB | Once |
| Per-cycle tensors | Each cycle | Views (no allocation) | Every cycle |

**No per-cycle GPU memory allocation** — all buffers pre-allocated at init.

---

### C.7 VRAM Calculation

#### C.7.1 Upstream DFlash Overhead (Current)

| Component | Size |
|-----------|------|
| RS buffer (n_rs_seq=8) | 5,387 MB |
| Draft model | ~800 MB |
| Hidden-state capture | Current upstream mechanism |
| **Total** | **~6,187 MB** |

#### C.7.2 Custom DFlash Overhead (Revised)

**CORRECTION (2026-08-08):** Updated tape size from ~134-186 MB to ~85-117 MB, and backup cells from ~1,246 MB (8 cells) to ~623 MB (4 cells).

| Component | Size |
|-----------|------|
| Base RS (n_rs_seq=0) | 599 MB |
| Backup cells (4) | 623 MB |
| GPU tape (fused) | 85 MB |
| GPU tape (non-fused) | 117 MB |
| Replay graph context | 2 MB |
| Draft model | ~800 MB |
| **Total (fused)** | **~2,109 MB** |
| **Total (non-fused)** | **~2,141 MB** |

#### C.7.3 Removed Allocations

| Component | Size Removed |
|-----------|-------------|
| RS snapshot buffer (n_rs_seq=8 → 0) | 4,788 MB |

#### C.7.4 Added Allocations

| Component | Size Added |
|-----------|-----------|
| Backup cells (4 rows) | 623 MB |
| GPU tape | 85-117 MB |
| Replay graph context | 2 MB |
| **Total added** | **~710-742 MB** |

#### C.7.5 Net VRAM Savings

| Metric | Value |
|--------|-------|
| Removed | 4,788 MB |
| Added | 710-742 MB |
| **Net savings** | **4,046 - 4,078 MB (~4.0-4.1 GB)** |
| **Savings %** | **84-85%** |

**Note (CORRECTED):** The revised design saves ~4.0-4.1 GB VRAM vs current upstream (up from the previous ~3.3 GB estimate). The improvement comes from two corrections: (1) tape size reduced from ~134-186 MB to ~85-117 MB using actual GGUF metadata, and (2) backup cells reduced from ~1,246 MB (8 cells) to ~623 MB (4 cells) matching old v0.3.2's approach. The revised design also eliminates the ~6.5 GB CPU RAM requirement and ~26 ms PCIe transfer overhead, making it significantly better in practice.

---

*End of Part 2 (Parts C-D section C). Section D follows below.*

---

## PART D: Changes from Previous Task 6 Blueprint

### D.1 What Remains Valid from Previous Blueprint

| Component | Status | Notes |
|-----------|--------|-------|
| `--beefix-dflash-custom` opt-in flag | ✅ VALID | Same mechanism, same location (`common/common.cpp:1770`) |
| `n_rs_seq=0` override | ✅ VALID | Same approach, same effect |
| Backup cells in recurrent memory | ✅ VALID | Same `cell_copy()` mechanism, same extended tensor rows |
| `llama_memory_recurrent` modifications | ✅ VALID | Same `n_backup_cells`, `backup_offset()`, `cell_copy()` |
| Replay integration in `post_decode()` | ✅ VALID | Same insertion point (`server-context.cpp:4263-4267`) |
| Fallback to checkpoint rollback | ✅ VALID | Same try-catch pattern, same failure conditions |
| GDN replay via `ggml_gated_delta_net()` | ✅ VALID | Same core operation |

### D.2 What Is INVALID in Previous Blueprint

| Component | Previous Design | Why Invalid | Revised Design |
|-----------|----------------|-------------|---------------|
| **Tape storage location** | CPU buffer (`std::vector<float>`) | Based on wrong ~6.5 GiB estimate. Actual tape is ~85-117 MB GPU — no need for CPU storage. | GPU tape with device-aware placement |
| **Tape tensor dimensions** | Full S-state (v/g/b with S_v=786,432) | Task 5 captured wrong tensors. Old v0.3.2 captured rank-factored intermediates. | Rank-factored GDN intermediates (k=[128,16,25], v=[128,48,25], etc.) **CORRECTED** |
| **Tape capture location** | `delta-net-base.cpp` after `cb()` calls | Captures post-processing tensors with wrong dimensions. | `qwen35.cpp` after tensor computation, same point as old v0.3.2 |
| **PCIe transfer overhead** | ~26 ms/cycle (GPU→CPU + CPU→GPU) | Eliminated — tape is GPU-native. | 0 ms/cycle |
| **CPU RAM requirement** | ~6.5 GB | Eliminated — tape on GPU. | 0 MB |
| **Custom CUDA kernel** | `dflash_replay.cu` (~200 lines) | Not needed — existing GDN kernel handles replay. | No new CUDA code |
| **Tape buffer struct** | `std::vector<float>` arrays | Replaced with GPU tensors. | `ggml_tensor*` per layer |

### D.3 What Is Modified

| Component | Previous | Revised | Reason |
|-----------|----------|---------|--------|
| `server_dflash_custom_state` | CPU tape buffers + metadata | GPU tape pointer + metadata | Tape moved from CPU to GPU |
| VRAM savings estimate | ~3.6 GB saved | **~4.0-4.1 GB saved** | **CORRECTED** — tape + backup cell corrections |
| Performance estimate | ~26 ms overhead/cycle | **~1.6 ms overhead/cycle** | **CORRECTED** — backup cells reduced from 8 to 4 |
| Code size estimate | ~900 lines (670 new + 236 modified) | ~546 lines (330 new + 216 modified) | No CUDA kernel, simpler tape struct |
| Tape capture point | `delta-net-base.cpp` | `qwen35.cpp` | Match old v0.3.2 capture point |
| Tensor names captured | `k_in`, `v_in`, `g_in`, `b_in` | `k_conv_predelta`, `v_conv_predelta`, `gate`, `beta_sigmoid`, `linear_attn_qkv_mixed` | Match actual graph node names |

### D.4 What Is Removed

| Component | Previous Lines | Reason |
|-----------|---------------|--------|
| `ggml/src/ggml-cuda/dflash_replay.cu` | ~200 | Not needed — GDN kernel handles replay |
| CPU tape buffer allocation code | ~80 | Tape is GPU-native now |
| GPU→CPU transfer code (`ggml_backend_tensor_get`) | ~40 | No PCIe transfers |
| CPU→GPU transfer code (`ggml_backend_tensor_set`) | ~40 | No PCIe transfers |
| F16 quantization option | ~60 | Not needed with small GPU tape |
| Hybrid capture options | ~50 | Single GPU capture path |

### D.5 What Is Newly Introduced

| Component | Lines | Description |
|-----------|-------|-------------|
| GPU tape allocation (`dflash_custom_tape_alloc`) | ~80 | Device-aware tape tensor allocation matching old `allocate_tape_gpu()` |
| Graph-embedded capture in `qwen35.cpp` | ~35 | `ggml_cpy` operations after tensor computation, matching old v0.3.2 |
| Tape name map | ~20 | Maps model layer index to tape layer index with correct tensor names |
| qkv_mixed capture | ~10 | New tensor capture (not in previous blueprint, but in old v0.3.2) |

### D.6 Summary of Changes

| Metric | Previous Blueprint | Revised Blueprint | Change |
|--------|-------------------|-------------------|--------|
| Total code lines | ~900 | ~546 | -39% |
| New files | 3 | 2 | -1 (no CUDA kernel) |
| GPU VRAM overhead | ~2,598 MB | **~2,109 MB** | **-489 MB** (tape + backup cell corrections) |
| CPU RAM overhead | ~6,552 MB | ~0 MB | -100% |
| PCIe transfers/cycle | ~13 GB | 0 GB | -100% |
| Per-cycle overhead | ~26 ms | **~1.6 ms** | **-94%** (backup cells 8→4) |
| VRAM saved vs current | ~3.6 GB | **~4.0-4.1 GB** | **+11%** (better than previous estimate) |
| New CUDA code | ~200 lines | 0 lines | -100% |
| Fork drift risk | Medium | Low | Improved |

**CORRECTION (2026-08-08):** All values in this table have been updated with corrected tape size (~85-117 MB) and backup cell budget (~623 MB for 4 cells).

---

*End of Part 2 (Parts C-D). Continue with Part 3 for implementation subtasks, fallback, and testing.*
