# Task 6R DFlash Custom Mode — Adversarial Code-Level Audit

**Date:** 2026-08-11
**Auditor:** Roo (Code Mode)
**Scope:** Complete implementation review for correctness, lifecycle, state-management, device-placement, fallback, integration, and performance issues.
**Method:** Independent code review. No assumptions about previous subtask correctness.

---

## Table of Contents

1. [Audit Methodology](#1-audit-methodology)
2. [Finding Registry](#2-finding-registry)
3. [Area 1: CUDA Conv Rebuild Kernel Correctness](#3-area-1-cuda-conv-rebuild-kernel-correctness)
4. [Area 2: Partial CUDA Failure / Fallback Behavior](#4-area-2-partial-cuda-fallback-behavior)
5. [Area 3: CUDA Error Handling](#5-area-3-cuda-error-handling)
6. [Area 4: Tape Tensor Lifetime](#6-area-4-tape-tensor-lifetime)
7. [Area 5: Backup/Restore with n_rs_seq=0](#7-area-5-backuprestore-with-n_rs_seq0)
8. [Area 6: S-State Backup View in Replay](#8-area-6-s-state-backup-view-in-replay)
9. [Area 7: GDN Output State Writeback](#9-area-7-gdn-output-state-writeback)
10. [Area 8: Zero Acceptance Path](#10-area-8-zero-acceptance-path)
11. [Area 9: fail_count Lifecycle](#11-area-9-fail_count-lifecycle)
12. [Area 10: CUDA Device Index Extraction](#12-area-10-cuda-device-index-extraction)
13. [Area 11: Non-CUDA Build](#13-area-11-non-cuda-build)
14. [Area 12: Multi-GPU / Mixed Backend](#14-area-12-multi-gpu--mixed-backend)
15. [Area 13: Repeated Cycle Behavior](#15-area-13-repeated-cycle-behavior)
16. [Area 14: Additional Findings](#16-area-14-additional-findings)
17. [Summary Answers](#17-summary-answers)

---

## 1. Audit Methodology

Each finding follows this format:
- **Severity:** P0 (correctness bug — will cause runtime failure), P1 (performance or latent correctness risk), P2 (hygiene or edge-case concern), INFO (informational)
- **Location:** Exact file and line range
- **Description:** What the issue is
- **Impact:** What could go wrong at runtime
- **Verdict:** Fix now, accept risk, or defer

Source files examined:
- [`common/server-dflash-custom.h`](common/server-dflash-custom.h) (250 lines)
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) (854 lines)
- [`ggml/src/ggml-cuda/dflash-custom-conv.cuh`](ggml/src/ggml-cuda/dflash-custom-conv.cuh) (78 lines)
- [`ggml/src/ggml-cuda/dflash-custom-conv.cu`](ggml/src/ggml-cuda/dflash-custom-conv.cu) (175 lines)
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) (lines 3300-3363, 4300-4419)
- [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) (lines 1-120)
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) (lines 1-160)
- [`src/models/qwen35.cpp`](src/models/qwen35.cpp) (lines 450-549)

---

## 2. Finding Registry

| # | Severity | Area | Title | Verdict |
|---|----------|------|-------|---------|
| F1 | P1 | CUDA Error Handling | `cudaStreamSynchronize` return value unchecked | Fix recommended |
| F2 | P1 | CUDA Error Handling | `ggml_cuda_set_device` return value unchecked | Fix recommended |
| F3 | P2 | fail_count | `replay_failed` never reset — permanent disable | Fix recommended |
| F4 | P2 | CUDA Device Index | `atoi()` on device name fragile for non-standard names | Accept risk |
| F5 | P2 | Multi-sequence | GDN replay only handles cell 0 — multi-parallel limitation | Document |
| F6 | INFO | CUDA In-place | In-place conv rebuild kernel is mathematically safe | No action |
| F7 | INFO | CUDA Stream | `cudaStreamPerThread` usage is safe but blocks server thread | No action |
| F8 | INFO | Partial CUDA Failure | Silent CUDA corruption would skip CPU fallback | Accept risk (theoretical) |
| F9 | INFO | Tape Lifetime | Tape tensor lifecycle is correct | No action |
| F10 | INFO | Backup/Restore | `n_rs_seq=0` row layout is correct | No action |
| F11 | INFO | S-State Backup | Backup view reads correct pre-draft state | No action |
| F12 | INFO | GDN Writeback | State writeback overwrites correctly | No action |
| F13 | INFO | Zero Acceptance | Checkpoint rollback path is correct | No action |
| F14 | INFO | Non-CUDA Build | `#ifdef GGML_CUDA` guards are correct | No action |
| F15 | INFO | Multi-GPU | Per-layer device validation is correct | No action |
| F16 | INFO | Cycle Behavior | No state accumulation detected | No action |

---

## 3. Area 1: CUDA Conv Rebuild Kernel Correctness

### 3.1 Indexing Equivalence: CPU vs CUDA

**CPU algorithm** ([`server-dflash-custom.cpp:801-815`](common/server-dflash-custom.cpp:801)):
```cpp
for (uint32_t w = 0; w < conv_window; ++w) {
    int src_pos = n_accepted + (int)w;
    for (uint32_t ch = 0; ch < conv_channels; ++ch) {
        float val;
        if (src_pos < (int)conv_window) {
            val = old_conv[ch * conv_window + (uint32_t)src_pos];
        } else {
            val = qkv_tape[(size_t)(src_pos - (int)conv_window) * (size_t)conv_channels + ch];
        }
        new_conv[ch * conv_window + w] = val;
    }
}
```

**CUDA templated kernel** ([`dflash-custom-conv.cu:31-46`](ggml/src/ggml-cuda/dflash-custom-conv.cu:31)):
```cpp
int ch = blockIdx.x * blockDim.x + threadIdx.x;
const int base = ch * CONV_WINDOW;
for (int w = 0; w < CONV_WINDOW; w++) {
    int src_pos = n_accepted + w;
    float val;
    if (src_pos < CONV_WINDOW) {
        val = old_conv[base + src_pos];  // = old_conv[ch * CONV_WINDOW + src_pos]
    } else {
        int tape_token = src_pos - CONV_WINDOW;
        val = qkv_tape[tape_token * conv_channels + ch];
    }
    dst_conv[base + w] = val;
}
```

**Comparison:**
| Operation | CPU | CUDA | Match? |
|-----------|-----|------|--------|
| `old_conv` read | `ch * conv_window + src_pos` | `base + src_pos = ch * CONV_WINDOW + src_pos` | ✅ |
| `qkv_tape` read | `(src_pos - conv_window) * conv_channels + ch` | `tape_token * conv_channels + ch` | ✅ |
| `new_conv` write | `ch * conv_window + w` | `base + w = ch * CONV_WINDOW + w` | ✅ |

**Verdict:** Indexing is mathematically equivalent. No issue.

### 3.2 In-Place Operation Safety

At [`server-dflash-custom.cpp:727-729`](common/server-dflash-custom.cpp:727):
```cpp
const float * old_conv_ptr = static_cast<const float *>(r_tensor->data);
const float * qkv_ptr = static_cast<const float *>(tl.qkv->data);
float * dst_ptr = static_cast<float *>(r_tensor->data);
```

`old_conv_ptr` and `dst_ptr` point to the SAME memory (`r_tensor->data`). The kernel writes to position `w` and reads from position `src_pos = n_accepted + w`. Since `n_accepted > 0` (validated at [`server-dflash-custom.cpp:368`](common/server-dflash-custom.cpp:368)), `src_pos > w` always. For the same channel, the read position is always strictly ahead of the write position.

Each channel operates independently (different `base` offsets). No cross-channel interference.

**Finding F6 — INFO:** In-place conv rebuild kernel is mathematically safe. No action needed.

### 3.3 Synchronization

[`dflash-custom-conv.cu:173`](ggml/src/ggml-cuda/dflash-custom-conv.cu:173):
```cpp
cudaStreamSynchronize(cudaStreamPerThread);
```

This ensures the kernel completes before `ggml_cuda_dflash_conv_rebuild_host()` returns. The caller (`server-dflash-custom.cpp:732-735`) then marks the layer as rebuilt. Sufficient for correctness.

**Finding F7 — INFO:** `cudaStreamPerThread` is the per-thread default stream. When called from the server thread that also runs the main computation graph, this synchronously blocks the server thread until the kernel completes. This is correct but could be optimized with a dedicated async stream in the future. No action needed.

---

## 4. Area 2: Partial CUDA Fallback Behavior

### 4.1 Flow Analysis

At [`server-dflash-custom.cpp:673-741`](common/server-dflash-custom.cpp:673):
1. CUDA loop processes layers sequentially.
2. On success: `cuda_rebuilt_layers[ti] = true` at line 740 (AFTER kernel launch + synchronize).
3. On failure (null qkv, non-CUDA device): `cuda_rebuilt = false; break;` at lines 688-705.
4. CPU fallback at line 746: skips layers where `cuda_rebuilt_layers[ti] == true`.

### 4.2 Silent Corruption Scenario

If the CUDA kernel completes (synchronize returns) but produces INCORRECT results due to silent GPU corruption (e.g., ECC error, cosmic ray bit-flip in SRAM):
- `cuda_rebuilt_layers[ti] = true` is set.
- CPU fallback skips this layer.
- Incorrect conv state persists.

**Finding F8 — INFO:** This is a theoretical concern. CUDA kernel correctness is deterministic for this simple algorithm. Silent GPU corruption is not a realistic threat for production testing. Accept risk.

### 4.3 Partial Failure Recovery

If CUDA processes layers 0-23 successfully, then layer 24 fails:
- `cuda_rebuilt = false` at line 688-704; loop breaks.
- CPU fallback executes at line 746.
- CPU fallback skips layers 0-23 (already rebuilt by CUDA).
- CPU fallback processes layers 24-N.

Layers 0-23 have CUDA-rebuilt conv state. Layers 24-N have CPU-rebuilt conv state. Both are correct. The mix is acceptable because each layer's conv state is independent.

**Verdict:** Partial failure recovery is correct. No issue.

---

## 5. Area 3: CUDA Error Handling

### 5.1 `cudaStreamSynchronize` Return Value Unchecked

[`dflash-custom-conv.cu:173`](ggml/src/ggml-cuda/dflash-custom-conv.cu:173):
```cpp
cudaStreamSynchronize(cudaStreamPerThread);
// Return value NOT checked.
```

`cudaStreamSynchronize` returns `cudaError_t`. If the kernel encounters a runtime error (out of memory, invalid configuration, launch failure), the error is silently ignored. The caller assumes success.

**Finding F1 — P1:** The return value of `cudaStreamSynchronize` should be checked. If it fails:
- The conv state for that layer is undefined.
- `cuda_rebuilt_layers[ti]` should NOT be set.
- The CPU fallback should process this layer.

**Recommendation:** Check the return value. On error, log the CUDA error and return from `ggml_cuda_dflash_conv_rebuild_host()` without marking success. The caller in `server-dflash-custom.cpp` should detect the failure and fall back to CPU.

**Impact:** Without this check, a CUDA kernel failure (OOM, invalid config) produces undefined conv state silently. The model continues with corrupted state.

### 5.2 `ggml_cuda_set_device` Return Value Unchecked

[`dflash-custom-conv.cu:160`](ggml/src/ggml-cuda/dflash-custom-conv.cu:160):
```cpp
ggml_cuda_set_device(cuda_device);
// Return value NOT checked.
```

If device selection fails (e.g., GPU removed, driver error), the kernel runs on whatever device is currently active. This could be the wrong GPU in a multi-GPU setup.

**Finding F2 — P1:** The return value of `ggml_cuda_set_device` should be checked. On failure, the host function should return an error indicator.

**Impact:** In multi-GPU setups, running the kernel on the wrong device could read/write the wrong GPU's memory, causing data corruption or a GPU page fault.

### 5.3 Kernel Launch Errors

CUDA kernel launch errors are asynchronous. They are only detectable after stream synchronization. Since synchronization IS performed at line 173, kernel launch errors WOULD be detectable IF the return value was checked (see F1).

**Verdict:** Covered by F1. No separate finding.

---

## 6. Area 4: Tape Tensor Lifetime

### 6.1 Allocation and Deallocation

- **Allocated:** `dflash_custom_init()` at [`server-dflash-custom.cpp:209`](common/server-dflash-custom.cpp:209).
- **Freed:** `dflash_custom_free()` at [`server-dflash-custom.cpp:224-243`](common/server-dflash-custom.cpp:224).
- **Tape pointer:** Stored in `server_dflash_custom_state::tape`.
- **State ownership:** `server_slot::dflash_custom` is a `std::unique_ptr<server_dflash_custom_state>`.

### 6.2 Lifecycle During Speculative Cycle

1. Pre-draft: Tape activated via `dflash_custom_set_tape_gpu(ctx_tgt, tape)` at [`server-context.cpp:3331`](tools/server/server-context.cpp:3331).
2. Draft forward pass: Tape capture operations execute.
3. Post-draft: Tape deactivated via `dflash_custom_set_tape_gpu(ctx_tgt, nullptr)` at [`server-context.cpp:3356`](tools/server/server-context.cpp:3356).
4. Replay: Tape read (not modified) by `dflash_custom_replay()`.

The tape tensor data is overwritten each cycle during capture (`ggml_cpy` operations in qwen35.cpp). The tape pointers are never freed during the slot's lifetime.

### 6.3 Access After Free?

The tape is freed when `server_slot` is destroyed. The `server_slot` lifetime spans the entire server session for that slot. No scenario exists where the tape is accessed after the slot is destroyed, because:
- The slot owns the `dflash_custom` state.
- The `dflash_custom` state owns the tape.
- All access paths go through the slot.

**Finding F9 — INFO:** Tape tensor lifecycle is correct. No use-after-free scenario identified.

---

## 7. Area 5: Backup/Restore with n_rs_seq=0

### 7.1 Row Layout Verification

With `n_rs_seq = 0` and `n_backup_cells = n_parallel` (e.g., 4):

From [`llama-memory-recurrent.h:93-95`](src/llama-memory-recurrent.h:93):
```cpp
uint32_t backup_offset() const {
    return mem_size * (1 + n_rs_seq);  // = mem_size * 1 = mem_size
}
```

From [`llama-memory-recurrent.cpp:105`](src/llama-memory-recurrent.cpp:105):
```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells;
// = mem_size + n_backup_cells
```

| Row range | Purpose | Example (mem_size=4, n_backup=4) |
|-----------|---------|-------------------------------------|
| `[0, mem_size-1]` | Active rows | `[0, 3]` |
| `[mem_size, mem_size+n_backup_cells-1]` | Backup rows | `[4, 7]` |
| Total `n_rows` | — | `8` |

### 7.2 `mem_size` Source

[`llama-memory-recurrent.cpp:33-39`](src/llama-memory-recurrent.cpp:33):
```cpp
size = mem_size;
this->n_backup_cells = n_backup_cells;
this->mem_size = mem_size;
```

`mem_size` is passed from the constructor caller (`llama-model.cpp`). It equals `n_seq_max` which equals `n_parallel` for the target context. The value is consistent.

**Finding F10 — INFO:** `n_rs_seq=0` row layout is correct. No issue.

---

## 8. Area 6: S-State Backup View in Replay

### 8.1 Data Flow

1. **Backup** (before draft): `dflash_custom_backup()` copies active → backup.
   - Backup row now has pre-draft state.
   - Active row still has pre-draft state (source of copy).

2. **Draft forward pass**: Modifies active row on DRAFT context (not target context).
   - Target context's active row unchanged.
   - Target context's backup row unchanged.

3. **Verification**: Runs on target context. Modifies target context's R/S state.
   - Active row now has verification state (post-draft, post-verify).
   - Backup row still has pre-draft state.

4. **Restore** (before replay): `dflash_custom_restore()` copies backup → active.
   - Active row now has pre-draft state (restored from backup).
   - Backup row UNCHANGED — still has pre-draft state.

5. **GDN replay**: Reads from backup row at `backup_offset()`.
   - [`server-dflash-custom.cpp:546-552`](common/server-dflash-custom.cpp:546):
     ```cpp
     ggml_tensor * s_backup_2d = ggml_view_2d(replay_ctx, mem->s_l[il],
         (int64_t)hp.n_embd_s(), 1,
         mem->s_l[il]->nb[1],
         mem->backup_offset() * mem->s_l[il]->nb[1]);
     ```
   - Reads from backup row = pre-draft state. ✅

**Finding F11 — INFO:** Backup view reads correct pre-draft state. `dflash_custom_restore()` copies backup → active (backup is SOURCE, not DESTINATION). The backup row is NOT modified by restore. The GDN replay reads from the unchanged backup row. Correct.

---

## 9. Area 7: GDN Output State Writeback

### 9.1 Data Flow

1. `dflash_custom_restore()` at line 423: backup → active. Active row has pre-draft state.
2. GDN replay graph: reads S from backup row, computes updated S.
3. GDN output writeback at [`server-dflash-custom.cpp:585-614`](common/server-dflash-custom.cpp:585):
   - Extracts state portion from GDN output.
   - Copies to active S row (row 0, offset 0).
   - This OVERWRITES the pre-draft S state with the post-accepted-token S state.

The active row starts with pre-draft S state (from restore). The GDN replay overwrites it with the updated S state. The backup row is NOT modified.

**Finding F12 — INFO:** State writeback is correct. The GDN replay produces the post-accepted-token S state, overwriting the pre-draft state restored by `dflash_custom_restore()`.

---

## 10. Area 8: Zero Acceptance Path

### 10.1 Flow

[`server-context.cpp:4322-4323`](tools/server/server-context.cpp:4322):
```cpp
if (n_accepted == 0) {
    dflash_fallback_reason = "zero_acceptance";
}
```

When `n_accepted == 0`:
- Replay is SKIPPED (nothing to replay).
- `dflash_fallback_reason = "zero_acceptance"`.
- `fail_count` is NOT incremented (zero acceptance is not a failure).
- Falls through to checkpoint rollback at line 4355-4394.

### 10.2 State After Zero Acceptance

- Draft forward pass ran on DRAFT context. Target context R/S unchanged by draft.
- Verification ran on target context. Target context R/S modified by verification.
- Checkpoint rollback restores pre-draft state from checkpoint.

The checkpoint rollback is the correct behavior. The backup cells contain the pre-draft state (from `dflash_custom_backup()`), but since replay is skipped, the backup cells are not used. The checkpoint restore handles the rollback.

**Finding F13 — INFO:** Zero acceptance path is correct. Checkpoint rollback handles the state restoration.

---

## 11. Area 9: fail_count Lifecycle

### 11.1 Current Behavior

| Event | `fail_count` | `replay_failed` |
|-------|-------------|-----------------|
| Replay success | Reset to 0 | Unchanged (false) |
| Replay failure | Increment | Set to true at 3 |
| Replay exception | Increment | Set to true at 3 |
| Zero acceptance | Unchanged | Unchanged |
| Replay disabled | N/A | Already true |

### 11.2 The Problem

Once `replay_failed = true` (after 3 consecutive failures), it is NEVER reset. The custom mode is permanently disabled for that slot for the remainder of the server session.

There is no mechanism to:
- Reset `replay_failed` after a successful cycle.
- Reset `replay_failed` after a timeout.
- Reset `replay_failed` on model reload.

### 11.3 Impact

If a server encounters 3 consecutive replay failures (which could be transient — OOM spike, CUDA driver hiccup, tape capture glitch), the VRAM-efficient custom mode is permanently disabled. The server falls back to checkpoint rollback for every subsequent cycle, incurring the full ~5.4 GB VRAM overhead.

**Finding F3 — P2:** `replay_failed` should be resettable. Options:
1. Reset on model reload (slot reinitialization).
2. Reset after N successful cycles following the failure threshold.
3. Add a CLI flag to force-reset.

**Recommendation:** At minimum, reset `replay_failed` when the slot is reinitialized (new model load). This is a simple fix in the `server_slot` constructor or reset function.

---

## 12. Area 10: CUDA Device Index Extraction

### 12.1 Current Code

[`server-dflash-custom.cpp:717-723`](common/server-dflash-custom.cpp:717):
```cpp
int layer_cuda_device = 0;
{
    const char * layer_dev_name = ggml_backend_dev_name(tl.dev);
    if (layer_dev_name && strlen(layer_dev_name) > 4) {
        layer_cuda_device = atoi(layer_dev_name + 4);
    }
}
```

### 12.2 Edge Cases

| Device name | `strlen > 4` | `atoi(name + 4)` | Result |
|-------------|-------------|-------------------|--------|
| `"CUDA0"` | true (5) | `atoi("0")` = 0 | ✅ |
| `"CUDA1"` | true (5) | `atoi("1")` = 1 | ✅ |
| `"CUDA10"` | true (6) | `atoi("10")` = 10 | ✅ |
| `"CUDA"` | false (4) | skipped | defaults to 0 |
| `"CUDA "` | true (5) | `atoi(" ")` = 0 | defaults to 0 |
| `"ROCM0"` | false (prefix check fails earlier) | N/A | falls to CPU |

For `"CUDA"` without a number, `strlen("CUDA") = 4`, which is NOT `> 4`. The device index defaults to 0. This is correct for single-GPU CUDA setups.

**Finding F4 — P2:** The `atoi()` approach is fragile but functional for current CUDA naming conventions. If ggml backend naming changes, this could silently select the wrong device. Consider using `ggml_backend_cuda_device()` or a dedicated API if available. For now, accept risk.

---

## 13. Area 11: Non-CUDA Build

### 13.1 Guard Analysis

[`server-dflash-custom.cpp:656-744`](common/server-dflash-custom.cpp:656):
```cpp
#ifdef GGML_CUDA
    // CUDA conv rebuild code
#endif

if (!cuda_rebuilt) {
    // CPU fallback
}
```

- `cuda_rebuilt` defaults to `false` at line 653.
- Without `GGML_CUDA`, the `#ifdef` block is entirely skipped.
- `cuda_rebuilt` remains `false`.
- CPU fallback executes at line 746.

[`server-dflash-custom.cpp:14-16`](common/server-dflash-custom.cpp:14):
```cpp
#ifdef GGML_CUDA
#include "ggml/src/ggml-cuda/dflash-custom-conv.cuh"
#endif
```

The CUDA header is only included when `GGML_CUDA` is defined. Without CUDA, the header is skipped and the code compiles cleanly.

**Finding: No issue.** Non-CUDA build path is correct. CPU fallback is the default behavior.

---

## 14. Area 12: Multi-GPU / Mixed Backend

### 14.1 Entry Check

[`server-dflash-custom.cpp:662-664`](common/server-dflash-custom.cpp:662):
```cpp
if (!tape_layers.empty() && tape_layers[0].dev) {
    const char * dev_name = ggml_backend_dev_name(tape_layers[0].dev);
    if (dev_name && strncmp(dev_name, "CUDA", 4) == 0) {
```

The CUDA path enters ONLY if layer 0 is CUDA. If layer 0 is CPU or Vulkan, the CUDA path is skipped entirely.

### 14.2 Per-Layer Validation

[`server-dflash-custom.cpp:697-714`](common/server-dflash-custom.cpp:697):
```cpp
if (tl.dev) {
    const char * layer_dev_name = ggml_backend_dev_name(tl.dev);
    if (!layer_dev_name || strncmp(layer_dev_name, "CUDA", 4) != 0) {
        cuda_rebuilt = false;
        break;
    }
}
```

If any layer in the CUDA loop is NOT CUDA, the loop breaks and falls through to CPU fallback. Layers already processed by CUDA are tracked in `cuda_rebuilt_layers`.

### 14.3 Mixed-GPU Scenario

Layer 0 = CUDA0, Layer 1 = CUDA1:
1. CUDA path enters (layer 0 is CUDA).
2. Layer 0 processed on CUDA0 via `ggml_cuda_set_device(0)`.
3. Layer 1 processed on CUDA1 via `ggml_cuda_set_device(1)`.
4. All layers CUDA — `cuda_rebuilt = true`.
5. CPU fallback skipped.

Layer 0 = CUDA0, Layer 1 = CPU:
1. CUDA path enters (layer 0 is CUDA).
2. Layer 0 processed on CUDA0.
3. Layer 1 detected as non-CUDA — break.
4. `cuda_rebuilt = false`.
5. CPU fallback executes.
6. CPU fallback skips layer 0 (already rebuilt by CUDA).
7. CPU fallback processes layer 1 and remaining layers.

**Finding: No issue.** Multi-GPU and mixed-backend paths are correct.

---

## 15. Area 13: Repeated Cycle Behavior

### 15.1 State That Persists Across Cycles

| State | Reset? | Location |
|-------|--------|----------|
| `replay_ctx` | Freed at end of replay | [`server-dflash-custom.cpp:826-827`](common/server-dflash-custom.cpp:826) |
| `tokens_captured` | Updated after each draft | [`server-context.cpp:3359`](tools/server/server-context.cpp:3359) |
| `fail_count` | Reset on success, incremented on failure | [`server-context.cpp:4328-4343`](tools/server/server-context.cpp:4328) |
| `replay_failed` | NEVER reset | See F3 |
| Tape tensor data | Overwritten each cycle by capture | [`qwen35.cpp:523-527`](src/models/qwen35.cpp:523) |
| Backup cell data | Overwritten each cycle by backup | [`server-dflash-custom.cpp:313-317`](common/server-dflash-custom.cpp:313) |

### 15.2 Accumulation Risk

No persistent state accumulates numerical errors across cycles. The tape and backup cells are overwritten (not appended to) each cycle. The replay context is freed and recreated. The only concern is `replay_failed` which, once set, stays set (see F3).

**Finding: No issue** (except F3 which is already tracked).

---

## 16. Area 14: Additional Findings

### 16.1 GDN Replay Only Handles Cell 0

**Severity:** P2
**Location:** [`server-dflash-custom.cpp:546-552`](common/server-dflash-custom.cpp:546) and [`server-dflash-custom.cpp:608-611`](common/server-dflash-custom.cpp:608)

The GDN replay graph creates views into the backup state at `backup_offset()` (the FIRST backup row, i.e., cell 0). The GDN output is written to active row 0 (offset 0).

For `n_parallel > 1`:
- `dflash_custom_restore()` restores ALL `n_cells` rows (backup → active for cells 0..n_cells-1).
- GDN replay ONLY updates cell 0.
- Cells 1..n_cells-1 retain the pre-draft state (from restore) but are NOT advanced by the GDN replay.

**Impact:** If `n_parallel > 1` and multiple sequences are simultaneously active in the same target context, the non-cell-0 sequences will have stale S-state after replay. The conv rebuild (step 7) also only writes to row 0.

**Mitigation:** The behavioral invariant states "Single sequence — `n_seqs = 1` for per-slot serving." The server integration at [`server-context.cpp:3322`](tools/server/server-context.cpp:3322) breaks after the first backup, assuming shared recurrent memory. The design assumes only cell 0 is the active sequence for the main serving path.

**Verdict:** Document this limitation. The code is correct for `n_parallel = 1` (the common serving case). For `n_parallel > 1` with multiple active sequences, the GDN replay should be extended to handle all cells. This is a P3 enhancement, not a correctness bug for the intended use case.

### 16.2 Conv Rebuild Also Only Handles Row 0

**Location:** [`server-dflash-custom.cpp:727`](common/server-dflash-custom.cpp:727) and [`server-dflash-custom.cpp:781-782`](common/server-dflash-custom.cpp:781)

Both CUDA and CPU conv rebuild paths operate on `r_tensor->data` at offset 0 (row 0). This is consistent with the GDN replay limitation (F5 above).

**Verdict:** Same as F5. Document, do not fix.

### 16.3 `tokens_captured` vs Actual Tape Data

**Location:** [`server-context.cpp:3359`](tools/server/server-context.cpp:3359)

```cpp
slot->dflash_custom->tokens_captured = slot->spec_draft.size();
```

`tokens_captured` is set to `spec_draft.size()` AFTER the draft completes. The tape capture during the draft forward pass captures exactly `n_seq_tokens` tokens, which equals `spec_draft.size()`. These should always match.

The replay guard at [`server-dflash-custom.cpp:381-387`](common/server-dflash-custom.cpp:381):
```cpp
if (n_accepted > (int)state->tokens_captured) {
    return false;
}
```

This prevents replay from reading beyond the captured tape data. If `tokens_captured` were ever LESS than actual tape data, the replay would be more conservative (safe). If `tokens_captured` were MORE than actual tape data, the replay could read uninitialized tape memory.

Since `tokens_captured = spec_draft.size()` and the tape captures `n_seq_tokens = spec_draft.size()`, these are consistent.

**Verdict:** No issue. The values are consistent by design.

### 16.4 Backup Shared Across Slots

**Location:** [`server-context.cpp:3322`](tools/server/server-context.cpp:3322)

```cpp
dflash_custom_backup(mem, mem->n_backup_cells);
break; // Backup is shared across slots (same recurrent memory)
```

Only ONE backup is performed even if multiple drafting slots have custom DFlash enabled. The comment states "Backup is shared across slots (same recurrent memory)." This is correct because all slots sharing the same `llama_context` share the same recurrent memory tensors. One backup suffices.

**Verdict:** No issue. Correct behavior.

---

## 17. Summary Answers

### Q1: Are there any known code-level correctness issues remaining?

**No P0 correctness issues identified.** The implementation is structurally sound:
- CUDA kernel indexing matches the CPU algorithm exactly.
- In-place conv rebuild is mathematically safe.
- Backup/restore row layout is correct with `n_rs_seq = 0`.
- S-state backup view reads the correct pre-draft state.
- GDN output writeback correctly overwrites the restored state.
- Zero acceptance falls through to checkpoint rollback correctly.
- Partial CUDA failure recovery is correct.

The two P1 findings (F1: unchecked `cudaStreamSynchronize` return, F2: unchecked `ggml_cuda_set_device` return) could cause silent failure in edge cases but are unlikely to manifest during normal runtime testing.

### Q2: Are there any questionable interactions between subtasks?

**One design limitation (F5/F16.1):** The GDN replay and conv rebuild only handle cell 0 (the first parallel sequence). For `n_parallel > 1` with multiple simultaneously active sequences, non-cell-0 sequences would have stale state after replay. This is by design (per-slot serving assumes `n_seqs = 1`) but is not validated at runtime. If the server is configured with `n_parallel > 1` and multiple sequences are active, this could cause subtle output divergence.

**No other problematic interactions identified.** The backup/restore, tape capture, replay graph, and conv rebuild all operate on well-defined boundaries.

### Q3: Are there any runtime-test results that could currently be ambiguous because of an unresolved implementation concern?

**Two areas could produce ambiguous results:**

1. **CUDA error handling (F1/F2):** If a CUDA kernel fails silently (OOM, invalid config on certain hardware), the conv state would be corrupted but the server would not log an error. The model would produce incorrect output that appears plausible. This could be misinterpreted as a model quality issue rather than a runtime bug.

2. **Multi-parallel sequences (F5):** If testing with `n_parallel > 1` and multiple active sequences, output could diverge from the baseline for non-cell-0 sequences. This divergence would not be attributed to the replay limitation without specific diagnostic logging.

### Q4: Is there anything you would change before runtime testing?

**Recommended fixes (in priority order):**

1. **F1 (P1):** Check `cudaStreamSynchronize` return value in `ggml_cuda_dflash_conv_rebuild_host()`. On error, return a failure indicator so the caller can fall back to CPU.
   ```cpp
   cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
   if (err != cudaSuccess) {
       fprintf(stderr, "[dflash-custom] CUDA conv rebuild failed: %s\n",
               cudaGetErrorString(err));
       // Signal failure to caller
   }
   ```

2. **F2 (P1):** Check `ggml_cuda_set_device` return value. On failure, skip the CUDA path.

3. **F3 (P2):** Reset `replay_failed` on slot reinitialization (model reload). Add one line in the slot reset or constructor path.

4. **F5 (P2):** Add a runtime assertion or warning when `n_parallel > 1` with custom DFlash active, noting that only cell 0 is replayed.

### Q5: If not, exactly why you believe the implementation is ready for runtime validation.

**The implementation IS ready for runtime validation** with the following qualifications:

1. **All P0 issues have been resolved.** The `n_backup_cells` fix, conv state rebuild, and cell_copy extraction are all implemented and verified.

2. **The core algorithm is correct.** CUDA kernel indexing, in-place safety, backup/restore row layout, and GDN replay data flow have been verified against the CPU reference implementation.

3. **The P1 findings (F1/F2) are edge cases.** Unchecked CUDA return values could cause silent corruption only if the CUDA kernel encounters a runtime error (OOM, invalid config). For standard testing on compatible hardware with sufficient VRAM, these errors should not occur. The P1 fixes are recommended but not blocking for initial runtime validation.

4. **The P2 findings (F3/F4/F5/F16.1) are design limitations.** These affect edge cases (permanent disable after 3 failures, fragile device name parsing, multi-parallel sequence support) that are unlikely to manifest during initial runtime testing with `n_parallel = 1`.

5. **All INFO findings are confirmed correct.** No action needed.

**Bottom line:** The implementation will likely pass runtime testing on standard hardware with `n_parallel = 1`. The P1 fixes should be applied before production deployment. The P2 items should be addressed before multi-parallel or long-running production use.

---

*End of adversarial audit.*
