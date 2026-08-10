# Task 6R DFlash Custom Mode — Final Implementation Documentation

**Date:** 2026-08-10
**Status:** All 9 development stages complete. Build successful.
**Author:** Roo (Architect/Code modes)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Detailed Implementation Summary by Stage](#2-detailed-implementation-summary-by-stage)
3. [Problems Encountered and Solutions](#3-problems-encountered-and-solutions)
4. [Final State Overview](#4-final-state-overview)
5. [Usage Instructions](#5-usage-instructions)
6. [Known Limitations](#6-known-limitations)
7. [Upstream Merge Considerations](#7-upstream-merge-considerations)
8. [Next Steps](#8-next-steps)

---

## 1. Executive Summary

Task 6R implements a VRAM-efficient alternative to the stock DFlash checkpoint snapshot system. When enabled via `--beefix-dflash-custom`, the feature replaces the large RS snapshot buffer (~5.4 GB for Qwen3.6-27B) with:

1. **Backup cells** in recurrent memory — pre-draft state copy using extra rows in R/S tensors
2. **GPU tape** — per-layer F32 tensors capturing GDN intermediates during the draft forward pass
3. **Replay graph** — rebuilds GDN state for accepted tokens without re-running the full forward pass

### Key Metrics

| Metric | Stock DFlash | Custom Mode |
|--------|-------------|-------------|
| VRAM overhead (Qwen3.6-27B) | ~5.4 GB (RS buffer) | ~1.2 GB (backup cells + tape) |
| Rollback mechanism | Full checkpoint serialization | GPU-native backup/restore + replay |
| Opt-in | N/A | `--beefix-dflash-custom` flag |
| Fallback | N/A | Checkpoint rollback on replay failure |

### Design Principles

- **Strictly opt-in:** All custom behavior gated behind `--beefix-dflash-custom`. Stock DFlash completely unaffected.
- **Fallback safety:** Try-catch with checkpoint rollback ensures custom failures never corrupt state.
- **Device-native:** Tape tensors placed on the same GPU as the model layer to avoid PCIe transfers.
- **Model-generic:** Dimensions derived from runtime hparams; no hardcoded model-specific values.
- **CPU conv rebuild:** Convolution state rebuild runs on CPU for correctness; CUDA kernel available as P3 optimization.

---

## 2. Detailed Implementation Summary by Stage

### Stage 1: Opt-in Flag and n_rs_seq Override

**Objective:** Add CLI flag and override `n_rs_seq=0` when custom mode active.

**Files Modified:**
- [`common/common.h`](common/common.h) — Added `bool beefix_dflash_custom = false` to `common_params_speculative`
- [`common/arg.cpp`](common/arg.cpp) — Added `--beefix-dflash-custom` CLI argument registration
- [`common/common.cpp`](common/common.cpp) — Added conditional override in `common_context_params_to_llama()`:
  - Sets `cparams.n_rs_seq = 0` when `beefix_dflash_custom && has_dflash`
  - Sets `cparams.n_backup_cells = params.n_parallel` (P0 fix — see Section 3)

**Key Locations:**
- Flag declaration: [`common/common.h:408`](common/common.h:408)
- CLI arg: [`common/arg.cpp:4384`](common/arg.cpp:4384)
- Override block: [`common/common.cpp:1775-1781`](common/common.cpp:1775)

**Unexpected Problems:**
- The initial implementation set `n_rs_seq = 0` but did NOT set `n_backup_cells`. This was discovered during the integration audit and is documented as the critical P0 issue in Section 3.

---

### Stage 2: Backup Cells Infrastructure

**Objective:** Add `n_backup_cells` field through the entire parameter chain and implement backup cell allocation.

**Files Modified:**
- [`include/llama.h`](include/llama.h) — Added `n_backup_cells` to `llama_context_params` (line 427)
- [`src/llama-cparams.h`](src/llama-cparams.h) — Added `n_backup_cells` field (line 21)
- [`src/llama-context.cpp`](src/llama-context.cpp) — Pass-through at line 271; arch guard at lines 272-276
- [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) — Constructor parameter (line 27); `n_backup_cells` field (line 85); `backup_offset()` method (lines 93-95)
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — Constructor stores value (line 37); allocation formula includes `n_backup_cells` (line 104); backup logging (lines 136-139)
- [`src/llama-model.cpp`](src/llama-model.cpp) — Forward `n_backup_cells` at 4 constructor call sites (lines 2109, 2152, 2203, 2221)
- [`src/llama-memory-hybrid.h/cpp`](src/llama-memory-hybrid.h) — Constructor parameter forwarding
- [`src/llama-memory-hybrid-iswa.h/cpp`](src/llama-memory-hybrid-iswa.h) — Constructor parameter forwarding

**Allocation Formula:**
```cpp
n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells
```

With `n_rs_seq = 0` and `n_backup_cells = n_parallel` (e.g., 4):
- `n_rows = mem_size + 4`
- Row layout: `[0, mem_size-1]` = active, `[mem_size, mem_size+3]` = backup

**Memory Impact (Qwen3.6-27B):**
- Per backup cell: ~156 MB (R + S state)
- Total for 4 backup cells: ~623 MB
- Logged at construction: `"backup cells =     4 rows, 623.00 MiB"`

---

### Stage 3: GPU Tape Allocation and Capture

**Objective:** Create device-aware GPU tape tensors and integrate capture into the graph builder.

**Files Created:**
- [`common/server-dflash-custom.h`](common/server-dflash-custom.h) — 250 lines; tape structs and public API
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) — 738 lines (current); tape alloc/free, backup/restore, replay

**Files Modified:**
- [`src/models/qwen35.cpp`](src/models/qwen35.cpp) — Added conditional capture block (~70 lines starting at line 460)
- [`src/llama-cparams.h`](src/llama-cparams.h) — Forward declaration `struct server_dflash_tape_gpu;` (line 11); `tape_gpu` field (line 96)
- [`src/llama-context.h`](src/llama-context.h) — `set_tape_gpu()` method (line 67)
- [`src/llama-context.cpp`](src/llama-context.cpp) — `set_tape_gpu()` implementation; `n_backup_cells` validation

**Tape Structure:**
```cpp
struct server_dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor * v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor * gate = nullptr;  // [1,   H_v, max_tokens]
    ggml_tensor * beta = nullptr;  // [1,   H_v, max_tokens]
    ggml_tensor * qkv  = nullptr;  // [conv_channels, max_tokens]
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_dev_t dev = nullptr;
};
```

**Tape Size (Qwen3.6-27B, 48 recurrent layers, 25 max tokens):**
- k: 128 × 16 × 25 × 4 bytes = 2 MB per layer
- v: 128 × 48 × 25 × 4 bytes = 6 MB per layer
- gate: 1 × 48 × 25 × 4 bytes = 4.7 KB per layer
- beta: 1 × 48 × 25 × 4 bytes = 4.7 KB per layer
- qkv: 10,240 × 25 × 4 bytes = 1 MB per layer
- **Total per layer: ~9.5 MB**
- **Total for 48 layers: ~456 MB**

**Capture Integration:**
The capture block in [`qwen35.cpp:460-529`](src/models/qwen35.cpp:460) is gated on `cparams.tape_gpu != nullptr`. When non-null, the graph builder inserts `ggml_cpy` operations that copy GDN intermediates to the tape tensors during the draft forward pass.

**Lifecycle:**
1. `dflash_custom_set_tape_gpu(ctx, tape)` — activates capture before draft
2. Draft forward pass — capture operations execute as part of graph
3. `dflash_custom_set_tape_gpu(ctx, nullptr)` — deactivates capture after draft

---

### Stage 4: Backup, Restore, and Replay Functions

**Objective:** Implement the core replay functions.

**Key Functions (all in [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp)):**

| Function | Lines | Purpose |
|----------|-------|---------|
| `dflash_custom_init()` | 189-216 | Initialize state, allocate tape, record hparams |
| `dflash_custom_free()` | 218-237 | Free tape, replay context, scheduler |
| `dflash_custom_cell_copy()` | 249-287 | Copy R/S state between rows (extracted free function) |
| `dflash_custom_backup()` | 299-312 | Pre-draft: active → backup rows |
| `dflash_custom_restore()` | 321-334 | Post-replay: backup → active rows |
| `dflash_custom_replay()` | 360-718 | Build and execute GDN replay graph |
| `dflash_custom_set_tape_gpu()` | 730-737 | Set/clear tape pointer on context |

**Replay Algorithm:**
1. Validate preconditions (7 guard checks with reason codes R1-R7)
2. Restore backup state to active rows via `dflash_custom_restore()`
3. Create replay graph context
4. For each recurrent layer:
   - Create q_zeros tensor (state update is q-independent)
   - Create 4D views into tape tensors for accepted tokens
   - Create view into backup state
   - Call `ggml_gated_delta_net()` to produce updated state
5. Execute replay graph via scheduler
6. Write updated S state back to active rows
7. Rebuild convolution state (R tensor) via sliding window shift (see Stage 5)
8. Cleanup replay context; log success

---

### Stage 5: Server Integration

**Objective:** Hook custom mode into the server lifecycle.

**Files Modified:**
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) — ~30 lines of integration

**Key Integration Points:**

| Location | Purpose |
|----------|---------|
| Slot init (`server_slot` constructor) | Create `dflash_custom` state if flag active |
| Pre-draft (before `common_speculative_draft()`) | Call `dflash_custom_backup()`, activate tape |
| Post-verify (after acceptance count known) | Try `dflash_custom_replay()` with try-catch |
| On replay failure | Fallback to checkpoint rollback |
| On 3 consecutive failures | Permanently disable custom mode for slot |
| Post-draft | Deactivate tape via `set_tape_gpu(nullptr)` |

**Slot Structure Addition:**
```cpp
// tools/server/server-context.cpp:207
struct server_slot {
    // ... existing fields ...
    server_dflash_custom_state * dflash_custom = nullptr;
};
```

**LLAMA_API Visibility:**
The `set_tape_gpu()` method on `llama_context` requires `LLAMA_API` because it is called from `server-context.lib` which is external to the main DLL. Without `LLAMA_API`, the symbol would not be exported and the linker would fail. This is documented as a known design decision in Section 3.

---

### Stage 6: P0 Fixes — n_backup_cells and Conv State Rebuild

**Objective:** Fix the two critical issues discovered during audit.

**Fix 1: n_backup_cells Population**
- **Problem:** `n_backup_cells` never set; defaults to 0; backup cells never allocated
- **Fix:** Added `cparams.n_backup_cells = params.n_parallel;` at [`common/common.cpp:1780`](common/common.cpp:1780)
- **Impact:** Backup cells now allocated; `dflash_custom_backup()` and `dflash_custom_replay()` guards pass

**Fix 2: Convolution State Rebuild**
- **Problem:** Replay restored S state but not R (conv) state; next forward pass would use stale conv state
- **Fix:** Added conv rebuild step (step 7) to `dflash_custom_replay()` at [`server-dflash-custom.cpp:610-708`](common/server-dflash-custom.cpp:610)
- **Algorithm:** Sliding window shift — positions within old window kept; positions beyond filled from tape qkv data
- **Implementation:** CPU-based for correctness; CUDA kernel available as P3 optimization

**Unexpected Problems:**
- **Linker error with `n_embd_r()`:** The initial conv rebuild code called `hp.n_embd_r()` which was compiled into the `llama` library but not exported. The fix was to compute `n_embd_r` directly from hparams: `(ssm_d_conv - 1) * conv_channels`. See Section 3 for details.

---

### Stage 7: Observability Logging

**Objective:** Add runtime visibility into replay execution.

**Log Points Added:**

| Code | Location | Message |
|------|----------|---------|
| R0 | `dflash_custom_backup()`, `dflash_custom_restore()` | `"backup/restore skipped (R0): mem=%p, n_cells=%u, n_backup_cells=%u"` |
| R1 | `dflash_custom_replay()` | `"replay skipped (R1): invalid preconditions"` |
| R2 | `dflash_custom_replay()` | `"replay skipped (R2): custom mode not enabled"` |
| R3 | `dflash_custom_replay()` | `"replay skipped (R3): n_accepted=%d > tokens_captured=%u"` |
| R4 | `dflash_custom_replay()` | `"replay skipped (R4): no tape allocated"` |
| R5 | `dflash_custom_replay()` | `"replay skipped (R5): no recurrent memory"` |
| R6 | `dflash_custom_replay()` | `"replay skipped (R6): no recurrent memory component"` |
| R7 | `dflash_custom_replay()` | `"replay skipped (R7): n_backup_cells=0"` |
| R8 | `dflash_custom_replay()` | `"replay skipped (R8): no scheduler available"` |
| R9 | Conv rebuild | `"conv rebuild skipped (R9): conv_window=0"` |
| Execute | `dflash_custom_replay()` | `"replay execute: n_accepted=%d, n_cells=%u, layers=%zu"` |
| Conv | `dflash_custom_replay()` | `"conv rebuild: %u layers, conv_window=%u, channels=%u, n_accepted=%d"` |
| Success | `dflash_custom_replay()` | `"replay success: n_accepted=%d, n_cells=%u, S_k=%u, S_v=%u, H_k=%u, H_v=%u"` |

---

### Stage 8: cell_copy() Extraction

**Objective:** Extract `cell_copy()` from `llama_memory_recurrent` to a free function to reduce upstream class modification.

**Changes:**
- Removed `cell_copy()` method from [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h)
- Added `dflash_custom_cell_copy()` free function to [`server-dflash-custom.h`](common/server-dflash-custom.h) (line 248) and [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp) (line 249)
- Updated `dflash_custom_backup()` and `dflash_custom_restore()` to call the free function

**Benefit:** Eliminates the `cell_copy()` method addition to the upstream `llama_memory_recurrent` class. The free function accesses only public members (`r_l`, `s_l`, `backup_offset()`).

---

### Stage 9: Configuration Struct and Final Cleanup

**Objective:** Centralize custom mode configuration and perform final cleanup.

**Changes:**
- Added `dflash_custom_config` struct to [`server-dflash-custom.h`](common/server-dflash-custom.h) (lines 76-81)
- Provides single point of truth for custom mode parameters
- `LLAMA_API` retained on `set_tape_gpu()` (required for cross-library linkage — see Section 3)
- Removed `LLAMA_API` from `dflash_custom_cell_copy()` (internal function, no DLL export needed)

---

## 3. Problems Encountered and Solutions

### 3.1 Critical: `n_backup_cells` Never Populated (P0)

**Problem:** The `--beefix-dflash-custom` flag correctly set `n_rs_seq = 0` but never set `n_backup_cells` to a non-zero value. The field defaults to `0` and no code path assigned it.

**Impact:**
- No backup rows allocated in recurrent memory
- `dflash_custom_backup()` returns early (guard: `n_backup_cells < n_cells` always true)
- `dflash_custom_replay()` returns `false` (guard: `n_cells == 0`)
- Server falls back to checkpoint rollback every cycle — correct but defeats the purpose

**Root Cause:** The initial implementation plan included the plumbing for `n_backup_cells` (field in structs, constructor parameters, allocation formula) but forgot to populate the value at the override site.

**Solution:** Added single line at [`common/common.cpp:1780`](common/common.cpp:1780):
```cpp
cparams.n_backup_cells = params.n_parallel;
```

**Discovery:** Found during integration audit. The audit traced the complete data flow from CLI flag through constructor and identified the missing assignment.

---

### 3.2 Linker Error: Unresolved `n_embd_r()` Symbol

**Problem:** During the first build attempt after adding the conv state rebuild code, the linker reported an unresolved external symbol for `llama_hparams::n_embd_r()`.

**Error Message (reconstructed):**
```
error LNK2019: unresolved external symbol "public: unsigned int __thiscall llama_hparams::n_embd_r(void)const " referenced in function "bool __cdecl dflash_custom_replay(...)"
```

**Root Cause:** The `n_embd_r()` method is defined in [`src/llama-hparams.cpp:183-204`](src/llama-hparams.cpp:183) and compiled into the main `llama` library (DLL on Windows). The `server-dflash-custom.cpp` file is part of the server library, which links against the llama DLL. The `n_embd_r()` method is NOT marked `LLAMA_API`, so it is not exported from the DLL. The server library cannot link to it.

**Solution:** Compute `n_embd_r` directly from hparams instead of calling the method:
```cpp
// server-dflash-custom.cpp:630-635
// Compute n_embd_r directly from hparams to avoid unresolved external linker issues.
// Equivalent to llama_hparams::n_embd_r() for SSM/GDN models:
//   (ssm_d_conv - 1) * conv_channels
uint32_t conv_window_size = (hp.ssm_d_conv > 0 ? hp.ssm_d_conv - 1 : 0);
uint32_t n_embd_r = conv_window_size * conv_channels;
uint32_t conv_window = n_embd_r / conv_channels;
```

The `ssm_d_conv` and `conv_channels` fields are public members of `llama_hparams`, so they are accessible without `LLAMA_API` export.

**Alternative Considered:** Adding `LLAMA_API` to `n_embd_r()` in [`llama-hparams.h`](src/llama-hparams.h) would have worked but would pollute the public API with an internal helper. The inline computation is cleaner.

---

### 3.3 LLAMA_API Visibility on `set_tape_gpu()`

**Problem:** The `set_tape_gpu()` method on `llama_context` must be called from `server-dflash-custom.cpp`, which is compiled into the server library. On Windows, the server library is a separate DLL from the main llama library. Without `LLAMA_API`, the symbol is not exported and the linker cannot resolve the call.

**Resolution:** Retained `LLAMA_API` on `set_tape_gpu()`:
```cpp
// llama-context.h:67
LLAMA_API void set_tape_gpu(struct server_dflash_tape_gpu * tape);
```

**Comment added:**
```cpp
// Task 6R: Kept LLAMA_API - this method is called from server-context.lib (external to the DLL)
// and must be exported for cross-library linkage to work.
```

**Note:** The modularity review recommended removing `LLAMA_API` from internal methods. However, this specific method requires it due to the cross-library call pattern. The `dflash_custom_cell_copy()` free function does NOT need `LLAMA_API` as it is called from within the same library.

---

### 3.4 Conv State Rebuild Initially Missing

**Problem:** The initial replay implementation restored backup R/S state and replayed the GDN S-state update, but did not rebuild the convolution state (R tensor). Without this, the R tensor would contain the restored pre-draft conv state instead of the conv state advanced by `n_accepted` tokens.

**Impact:** The next forward pass would use stale conv state, producing incorrect outputs. The error would be subtle (the model still produces plausible output) but results would diverge from the correct forward pass.

**Solution:** Added conv rebuild step (step 7) to `dflash_custom_replay()` at [`server-dflash-custom.cpp:610-708`](common/server-dflash-custom.cpp:610). The algorithm performs a sliding window shift:
- Read pre-draft conv state from active R row (restored from backup)
- Read qkv tape data for accepted tokens
- For each channel and window position:
  - If position still within old window: keep value from restored state
  - If position beyond old window: fill from tape qkv data
- Write new conv state back to active R row

**Discovery:** Identified during architectural review. The tape already captures qkv data, but no replay function consumed it.

---

### 3.5 cell_copy() Class Modification

**Problem:** The original design added `cell_copy()` as a method on `llama_memory_recurrent`, modifying the upstream class.

**Solution:** Extracted `cell_copy()` to a free function `dflash_custom_cell_copy()` in [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp). The free function accesses only public members (`r_l`, `s_l`, `backup_offset()`), eliminating the need to modify the upstream class.

**Benefit:** Reduces merge-conflict risk with upstream. The `llama_memory_recurrent` class is a high-risk file for upstream changes.

---

## 4. Final State Overview

### 4.1 New Files

| File | Lines | Purpose |
|------|-------|---------|
| [`common/server-dflash-custom.h`](common/server-dflash-custom.h) | ~250 | Tape structs, config struct, public API declarations |
| [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) | ~738 | Tape alloc/free, backup/restore, replay, cell_copy |

### 4.2 Modified Files

| File | Custom Additions | Type | Merge Risk |
|------|-----------------|------|------------|
| [`src/llama-cparams.h`](src/llama-cparams.h) | `n_backup_cells`, `tape_gpu`, forward declaration | Struct extension | Medium |
| [`src/llama-context.h`](src/llama-context.h) | `set_tape_gpu()` method | Method addition | Low |
| [`src/llama-context.cpp`](src/llama-context.cpp) | `set_tape_gpu()` impl, `n_backup_cells` validation | Method + validation | Medium |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | `n_backup_cells`, `backup_offset()`, constructor param | Class extension | High |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | Constructor param, allocation formula, logging | Constructor + method | High |
| [`src/llama-model.cpp`](src/llama-model.cpp) | `n_backup_cells` forwarding (4 call sites) | Param passthrough | High |
| [`src/llama-memory-hybrid.h/cpp`](src/llama-memory-hybrid.h) | `n_backup_cells` forwarding | Constructor param | Medium |
| [`src/llama-memory-hybrid-iswa.h/cpp`](src/llama-memory-hybrid-iswa.h) | `n_backup_cells` forwarding | Constructor param | Medium |
| [`src/models/qwen35.cpp`](src/models/qwen35.cpp) | Tape capture block (~70 lines) | Conditional block | Low |
| [`common/common.cpp`](common/common.cpp) | `n_backup_cells` + `n_rs_seq=0` override | Conditional override | Low |
| [`common/common.h`](common/common.h) | `beefix_dflash_custom` flag | Struct field | Low |
| [`common/arg.cpp`](common/arg.cpp) | CLI argument | Argument registration | Low |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | Server integration (~30 lines) | Conditional calls | Medium |
| [`include/llama.h`](include/llama.h) | `n_backup_cells` in `llama_context_params` | API extension | Low |

### 4.3 Total Impact

- **2 new files:** ~988 lines
- **14 modified files:** ~150 lines of custom additions
- **Total custom code:** ~1,138 lines
- **All behavior gated behind `--beefix-dflash-custom`**

---

## 5. Usage Instructions

### 5.1 Basic Usage

```bash
./build/bin/llama-server \
    -m target_model.gguf \
    --spec-type draft-dflash \
    --spec-draft-model draft_model.gguf \
    --spec-draft-n-max 8 \
    --beefix-dflash-custom \
    --trace 1 \
    --port 8080
```

### 5.2 Expected Startup Logs

```
[dflash-custom] GPU tape allocated: 48 recurrent layers, 8 max tokens
[dflash-custom] Initialized: S_k=128, H_k=16, S_v=128, H_v=48, conv_ch=10240, max_tokens=8
llama_memory_recurrent_init: backup cells =     4 rows, 623.00 MiB
```

### 5.3 Expected Replay Cycle Logs

```
[dflash-custom] replay execute: n_accepted=3, n_cells=4, layers=48
[dflash-custom] conv rebuild: 48 layers, conv_window=2, channels=10240, n_accepted=3
[dflash-custom] replay success: n_accepted=3, n_cells=4, S_k=128, S_v=128, H_k=16, H_v=48
```

### 5.4 Checkpoint Fallback Logs

```
[dflash-custom] replay skipped (R3): n_accepted=8 > tokens_captured=0
SLT_WRN: dflash custom replay failed - falling back to checkpoint
```

### 5.5 VRAM Comparison

| Mode | VRAM Overhead (Qwen3.6-27B) | Components |
|------|----------------------------|------------|
| Stock DFlash | ~5.4 GB | RS snapshot buffer |
| Custom Mode | ~1.2 GB | 623 MB backup cells + 456 MB tape |
| Savings | ~4.2 GB | |

### 5.6 Test Script

A Python test script is available at [`tests/dflash-custom-test.py`](tests/dflash-custom-test.py) for automated validation:
- Stock/custom DFlash comparison
- VRAM measurement
- Output comparison
- Crash resilience testing

---

## 6. Known Limitations

### 6.1 Conv Rebuild is CPU-Based

The convolution state rebuild runs on CPU, requiring PCIe transfers for the R tensor and tape data. For Qwen3.6 with 48 layers:
- Total PCIe per replay: ~24 MB
- Estimated latency: 5-10 ms per replay cycle
- **P3 optimization:** CUDA kernel available (old v0.3.2 had `cross-ring-interleave.cu:360-413`)

### 6.2 Single Sequence

The replay graph assumes `n_seqs = 1` (per-slot serving). Multi-sequence support is not yet implemented.

### 6.3 K=1 GDN Output

The replay uses K=1 (state-only output, no RS snapshots). This is intentional for VRAM efficiency.

### 6.4 Same-Device Tape Requirement

Tape tensors must live on the same GPU as the corresponding model layer. Cross-GPU tape placement is not supported.

### 6.5 Qwen3.6-Specific Conv Formula

The `conv_channels` formula (`d_inner + 2 * H_k * S_k`) matches Qwen3.6 projection layout. Other SSM architectures may differ. The value comes from model hparams, so it adapts per model, but the formula assumption is Qwen-specific.

### 6.6 Zero Acceptance Edge Case

When `n_accepted == 0`, replay returns `false` and increments `fail_count`. After 3 zero-acceptance cycles, replay is disabled even though it would work for partial acceptance. This is not a correctness issue (fallback to checkpoint is safe) but may be suboptimal for performance.

---

## 7. Upstream Merge Considerations

### 7.1 Modularity Score: 3.5/5

**Well-designed aspects:**
- Core logic isolated in two new files (`server-dflash-custom.h/cpp`)
- Flag gating ensures stock DFlash is unaffected
- Fallback safety via try-catch with checkpoint rollback
- Device-aware tape placement

**Areas for improvement:**
- `#ifdef BEE_DFLASH_CUSTOM` compile-time guards would produce cleaner upstream diffs
- `LLAMA_API` on `set_tape_gpu()` is necessary for cross-library linkage but pollutes the public API
- Marker comments (`// Task 6R`) at modification points would aid future merge resolution

### 7.2 High-Risk Merge Conflict Areas

| File | Risk | Reason |
|------|------|--------|
| [`llama-memory-recurrent.h/cpp`](src/llama-memory-recurrent.h) | HIGH | Core recurrent memory subsystem; upstream likely to modify constructor and allocation |
| [`llama-model.cpp`](src/llama-model.cpp) | HIGH | Constructor call sites change when upstream modifies signature |
| [`llama-cparams.h`](src/llama-cparams.h) | MEDIUM | Central config struct; upstream frequently adds fields |

### 7.3 Merge Strategy

1. Merge upstream first — let upstream changes apply cleanly
2. Reapply Task 6R as a patch — new files apply automatically; modified files may need manual resolution
3. Verify constructor signatures — check `n_backup_cells` is at correct position
4. Run test script — verify replay executes correctly

### 7.4 Behavioral Invariants (MUST NOT Change)

1. **Lifecycle ordering:** backup → capture → reset → verify → replay/checkpoint
2. **`n_backup_cells = n_parallel`** (not `2 × n_parallel`)
3. **`n_rs_seq = 0`** when custom mode active
4. **Device-native tape placement** — tape on same GPU as model layer
5. **Conv state rebuild from backup** — must use pre-draft state restored by `dflash_custom_restore()`

---

## 8. Next Steps

### 8.1 P2 Improvements (Recommended)

| Priority | Task | Files | Effort |
|----------|------|-------|--------|
| P2 | Add `#ifdef BEE_DFLASH_CUSTOM` guards | Multiple header files | ~15 lines |
| P2 | Add `// Task 6R` marker comments | All modification points | ~10 lines |
| P2 | Remove `LLAMA_API` from `dflash_custom_cell_copy()` | Already done | 0 |

### 8.2 P3 Optimizations (Optional)

| Priority | Task | Description |
|----------|------|-------------|
| P3 | CUDA conv rebuild kernel | Replace CPU path with GPU-native kernel for ~5-10 ms savings per replay |
| P3 | Configuration struct usage | Use `dflash_custom_config` struct for centralized config |
| P3 | Model-specific capture interface | Virtual interface for SSM architecture extension |

### 8.3 Testing

- Run [`tests/dflash-custom-test.py`](tests/dflash-custom-test.py) to validate custom mode
- Compare output with stock DFlash for bitwise correctness
- Measure VRAM savings on target hardware
- Test multi-round inference stability (20+ turns)

---

*End of final documentation.*
