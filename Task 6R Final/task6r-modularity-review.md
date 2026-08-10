# Task 6R DFlash Custom Mode — Modularity and Upstream-Merge-Hygiene Review

**Date:** 2026-08-10
**Reviewer:** Roo (Architect Mode)
**Based on:** [`task6r-audit-findings.md`](task6r-audit-findings.md), [`task6r-architectural-recommendation.md`](task6r-architectural-recommendation.md)
**Status:** Review complete. Recommendations documented.

---

## Executive Summary

The Task 6R implementation scores **3.5/5 on modularity** (per the architectural recommendation). The core logic is isolated in two new files (`server-dflash-custom.h/cpp`), and the invasive changes to upstream files are necessary by architectural constraints. However, several areas could be improved to reduce merge-conflict risk and tighten the custom/upstream boundary.

**Key findings:**
- **7 upstream files modified** — all modifications are architecturally justified
- **2 new files added** — cleanly isolated, no upstream overlap
- **3 high-risk merge-conflict areas** identified
- **4 modularity improvements** proposed (none require behavioral changes)
- **5 behavioral invariants** that MUST NOT be changed

---

## 2. Implementation vs. Upstream Code Boundaries

### 2.1 New Files (Clean Isolation)

| File | Lines | Upstream Conflict Risk | Notes |
|------|-------|----------------------|-------|
| [`common/server-dflash-custom.h`](common/server-dflash-custom.h) | ~218 | None | Pure fork file. Defines tape structs and public API. |
| [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) | ~687 | None | Pure fork file. Implements tape alloc/free, backup/restore, replay. |

**Verdict:** These files are cleanly isolated. They will never conflict with upstream merges. They depend on upstream internal headers but do not modify them.

### 2.2 Modified Upstream Files

| File | Custom Additions | Type of Change | Merge Risk |
|------|-----------------|----------------|------------|
| [`src/llama-cparams.h`](src/llama-cparams.h) | `n_backup_cells`, `tape_gpu`, forward declaration | Struct extension | **Medium** |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | `n_backup_cells`, `backup_offset()`, `cell_copy()`, `mem_size` | Class extension | **High** |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | Constructor param, allocation formula, `cell_copy()` impl | Constructor + method | **High** |
| [`src/llama-context.h`](src/llama-context.h) | `set_tape_gpu()` | Method addition | **Low** |
| [`src/llama-context.cpp`](src/llama-context.cpp) | `set_tape_gpu()` impl, `n_backup_cells` validation | Method + validation | **Medium** |
| [`src/llama-model.cpp`](src/llama-model.cpp) | `n_backup_cells` forwarding (4 call sites) | Param passthrough | **High** |
| [`src/llama-memory-hybrid.h/cpp`](src/llama-memory-hybrid.h) | `n_backup_cells` forwarding | Constructor param | **Medium** |
| [`src/llama-memory-hybrid-iswa.h/cpp`](src/llama-memory-hybrid-iswa.h) | `n_backup_cells` forwarding | Constructor param | **Medium** |
| [`src/models/qwen35.cpp`](src/models/qwen35.cpp) | Tape capture block (~70 lines) | Conditional block | **Low** |
| [`common/common.cpp`](common/common.cpp) | `n_backup_cells` + `n_rs_seq=0` override | Conditional override | **Low** |
| [`common/common.h`](common/common.h) | `beefix_dflash_custom` flag | Struct field | **Low** |
| [`common/arg.cpp`](common/arg.cpp) | CLI argument | Argument registration | **Low** |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | Server integration (~30 lines) | Conditional calls | **Medium** |

### 2.3 Boundary Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     CUSTOM FILES (No Conflict)                  │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │ server-dflash-custom.h  │  │ server-dflash-custom.cpp     │  │
│  │ - tape structs          │  │ - tape alloc/free            │  │
│  │ - API declarations      │  │ - backup/restore/replay      │  │
│  └──────────┬──────────────┘  └──────────┬───────────────────┘  │
│             │                            │                      │
├─────────────┼────────────────────────────┼──────────────────────┤
│             │ DEPENDS ON                 │ DEPENDS ON           │
├─────────────┼────────────────────────────┼──────────────────────┤
│             ▼                            ▼                      │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │ llama-memory-recurrent  │  │ llama-cparams                │  │
│  │ + n_backup_cells        │  │ + tape_gpu, n_backup_cells   │  │
│  │ + backup_offset()       │  │                              │  │
│  │ + cell_copy()           │  │ llama-context                │  │
│  │                         │  │ + set_tape_gpu()             │  │
│  │ llama-model.cpp         │  │ llama-memory-hybrid          │  │
│  │   (param forwarding)    │  │   (param forwarding)         │  │
│  └─────────────────────────┘  └──────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ qwen35.cpp  (conditional capture block, flag-gated)      │   │
│  │ common.cpp  (conditional override, flag-gated)           │   │
│  │ server-context.cpp  (conditional calls, flag-gated)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. High-Risk Merge Conflict Areas

### 3.1 Risk Level: HIGH — [`llama-memory-recurrent.h/cpp`](src/llama-memory-recurrent.h)

**Why:** This file is the core of the recurrent memory subsystem. Upstream is likely to modify constructor signatures, add new memory features, and change allocation formulas.

**Current custom additions:**
- `n_backup_cells` field (line 85)
- `backup_offset()` method (line 93)
- `cell_copy()` method (line 99)
- `mem_size` field (line 145)
- Constructor parameter `n_backup_cells` (line 27)
- Allocation formula: `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` (line 104)

**Upstream change risk:**
- Upstream may add new constructor parameters (tail rollback tokens, visibility window, etc.)
- Upstream may modify the allocation formula for new memory features
- Upstream may refactor `cell_copy`-like operations into the base class

**Mitigation:**
1. Add `// Task 6R` marker comments around all custom additions (P2 recommendation from audit).
2. Consider wrapping custom members in `#ifdef BEE_DFLASH_CUSTOM` guards to produce cleaner diffs.
3. The `cell_copy()` method could be extracted to [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp) as a free function that takes `(ggml_tensor *r, ggml_tensor *s, uint32_t src_row, uint32_t dst_row, ...)` — this would eliminate the method addition to `llama_memory_recurrent` entirely.

### 3.2 Risk Level: HIGH — [`llama-model.cpp`](src/llama-model.cpp)

**Why:** This file contains all `llama_memory_recurrent` constructor call sites. Every time upstream changes the constructor signature, all 4 call sites must be updated.

**Current custom additions:**
- `cparams.n_backup_cells` forwarded at lines 2109, 2152, 2203, 2221

**Upstream change risk:**
- Constructor parameter order changes
- New parameters added between existing ones
- Constructor refactored to use builder pattern or parameter struct

**Mitigation:**
1. Add `// Task 6R` marker comments at each call site.
2. If upstream adopts a builder pattern for `llama_context_params`, the `n_backup_cells` field would naturally be part of that struct, reducing call-site coupling.

### 3.3 Risk Level: HIGH — [`llama-cparams.h`](src/llama-cparams.h)

**Why:** The `llama_cparams` struct is the central configuration struct. Upstream frequently adds new fields.

**Current custom additions:**
- Forward declaration `struct server_dflash_tape_gpu;` (line 11)
- `n_backup_cells` field (line 20)
- `tape_gpu` field (line 95)

**Upstream change risk:**
- Upstream may add fields near our insertion points
- Upstream may refactor the struct (grouping, renaming)

**Mitigation:**
1. The `tape_gpu` field is already at the end of the struct (line 95), which is the safest position.
2. The `n_backup_cells` field is next to `n_rs_seq` (line 20), which is architecturally correct but may conflict if upstream adds related fields.
3. Consider `#ifdef BEE_DFLASH_CUSTOM` guards around both fields.

---

## 4. Refactoring Opportunities

### 4.1 Opportunity A: Extract `cell_copy()` from `llama_memory_recurrent`

**Current state:** `cell_copy()` is a method on `llama_memory_recurrent` ([`llama-memory-recurrent.h:99`](src/llama-memory-recurrent.h:99)). This requires modifying the upstream class.

**Proposed refactoring:**
```cpp
// In server-dflash-custom.h (new free function)
void dflash_custom_cell_copy(
    const llama_memory_recurrent * mem,
    uint32_t src_row, uint32_t dst_row);

// In server-dflash-custom.cpp
void dflash_custom_cell_copy(
    const llama_memory_recurrent * mem,
    uint32_t src_row, uint32_t dst_row) {
    // Access mem->r_l, mem->s_l, mem->backup_offset() through public interface
    // Create temporary ggml context for views
    // Use ggml_backend_tensor_copy for device-native copy
}
```

**Benefit:** Eliminates the `cell_copy()` method addition to `llama_memory_recurrent`. The free function accesses existing public members (`r_l`, `s_l`, `backup_offset()`).

**Cost:** The function would need to duplicate the ggml context creation logic. However, this logic is already self-contained (~20 lines) and uses only public members.

**Risk:** LOW — `r_l` and `s_l` are public members. `backup_offset()` is already a public method. No private member access needed.

### 4.2 Opportunity B: Remove `LLAMA_API` from Internal Methods

**Current state:** `set_tape_gpu()` and `cell_copy()` are marked `LLAMA_API` (DLL export).

**From audit P2 recommendation:**
> Remove `LLAMA_API` from `set_tape_gpu()` / `cell_copy()`. Use direct member access instead. Eliminates DLL export concerns.

**Proposed:**
```cpp
// llama-context.h:65 - change from:
LLAMA_API void set_tape_gpu(struct server_dflash_tape_gpu * tape);
// to:
void set_tape_gpu(struct server_dflash_tape_gpu * tape);

// llama-memory-recurrent.h:99 - change from:
LLAMA_API void cell_copy(uint32_t src_row, uint32_t dst_row) const;
// to:
void cell_copy(uint32_t src_row, uint32_t dst_row) const;
```

**Benefit:** These methods are internal server-only helpers. They should not be exported as part of the public API. Removing `LLAMA_API` eliminates potential ABI issues on Windows and reduces the exported symbol surface.

**Risk:** LOW — These methods are only called from [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp) and [`server-context.cpp`](tools/server/server-context.cpp), both of which link against the same library.

### 4.3 Opportunity C: `#ifdef BEE_DFLASH_CUSTOM` Guards

**Current state:** Custom code is guarded at runtime by `--beefix-dflash-custom` flag and `cparams.tape_gpu != nullptr` checks. No compile-time guards exist.

**Proposed:**
```cpp
// In llama-cparams.h
#ifdef BEE_DFLASH_CUSTOM
    // Task 6R: Forward declaration for GPU tape struct
    struct server_dflash_tape_gpu;
#endif

struct llama_cparams {
    // ... existing fields ...
    uint32_t n_backup_cells;  // Task 6R: extra recurrent state rows (0 = disabled)

#ifdef BEE_DFLASH_CUSTOM
    // Task 6R: Active tape pointer for graph-embedded capture
    struct server_dflash_tape_gpu * tape_gpu = nullptr;
#endif
};
```

```cpp
// In llama-context.h
#ifdef BEE_DFLASH_CUSTOM
    void set_tape_gpu(struct server_dflash_tape_gpu * tape);
#endif
```

**Benefit:** When `BEE_DFLASH_CUSTOM` is not defined, the custom code is completely absent from the compiled binary. This produces a clean upstream-compatible build and makes diffs easier to review.

**Risk:** MEDIUM — Requires adding the `BEE_DFLASH_CUSTOM` define to CMakeLists.txt. All files that reference `tape_gpu` or `set_tape_gpu()` must be consistently guarded. The [`qwen35.cpp`](src/models/qwen35.cpp) capture block must also be guarded.

### 4.4 Opportunity D: Configuration Struct for Custom Mode

**Current state:** Custom mode configuration is scattered across:
- `common_params_speculative.beefix_dflash_custom` (flag)
- `llama_cparams.n_backup_cells` (backup cell count)
- `llama_cparams.tape_gpu` (tape pointer)
- `server_dflash_custom_state` (runtime state)

**Proposed:**
```cpp
// In server-dflash-custom.h
struct server_dflash_custom_config {
    bool enabled = false;
    uint32_t n_backup_cells = 0;
    uint32_t max_tokens = 0;
    // Future: convolution rebuild options, logging level, etc.
};

// Usage in common.cpp:
if (params.speculative.beefix_dflash_custom && has_dflash) {
    cparams.n_rs_seq = 0;
    cparams.n_backup_cells = params.n_parallel;
    // Store config in a well-known location for server to retrieve
}
```

**Benefit:** Centralizes custom mode configuration. Makes it easier to add new options without scattering them across multiple structs. Provides a single point of truth for what "custom mode" means.

**Risk:** LOW — This is an additive change. Existing behavior is preserved.

### 4.5 Opportunity E: Interface for Model-Specific Capture

**Current state:** Capture is hardcoded in [`qwen35.cpp:460-529`](src/models/qwen35.cpp:460). New models would need to add similar capture blocks in their own graph builders.

**Proposed:**
```cpp
// In server-dflash-custom.h
struct dflash_capture_interface {
    virtual ~dflash_capture_interface() = default;
    // Called by the graph builder to capture GDN intermediates
    virtual void capture(
        ggml_context * ctx0,
        ggml_cgraph * gf,
        ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * gate, ggml_tensor * beta,
        ggml_tensor * qkv_mixed,
        int layer_idx,
        int n_seq_tokens,
        server_dflash_tape_gpu * tape) = 0;
};

// Default implementation (Qwen3.6 pattern)
struct dflash_capture_qwen36 : dflash_capture_interface {
    void capture(...) override { /* current qwen35.cpp logic */ }
};
```

**Benefit:** Provides a clean extension point for new SSM architectures. Each model can implement its own capture strategy without modifying [`qwen35.cpp`](src/models/qwen35.cpp).

**Risk:** MEDIUM — Adds virtual dispatch overhead to the capture path. The current approach (direct code in graph builder) has zero overhead. This is a P3 optimization, not urgent.

---

## 5. Behavioral Invariants (MUST NOT Change)

These invariants are critical for correct DFlash custom mode operation. Any refactoring must preserve them.

### 5.1 Backup Before Draft, Capture During Draft, Reset After Draft

```
backup() → set_tape_gpu(tape) → draft_decode() → set_tape_gpu(nullptr) → verify() → replay() OR checkpoint
```

**Why:** The tape must be active ONLY during the draft forward pass. If capture is active during replay, the tape would be overwritten with replay data. If backup happens after tape activation, the backup would include tape state.

### 5.2 `n_backup_cells = n_parallel` (Not `2 × n_parallel`)

**Current:** [`common/common.cpp:1780`](common/common.cpp:1780): `cparams.n_backup_cells = params.n_parallel;`

**Why:** Each parallel slot needs exactly one backup row. The old v0.3.2 implementation used `n_parallel_user` cells, proven sufficient. Doubling would waste ~612 MB for Qwen3.6-27B.

### 5.3 `n_rs_seq = 0` When Custom Mode Active

**Current:** [`common/common.cpp:1779`](common/common.cpp:1779): `cparams.n_rs_seq = 0;`

**Why:** The RS snapshot buffer is ~5.4 GB for Qwen3.6-27B. Custom mode replaces it with backup cells (~12 MB) + tape (~1.1 GB). Having both `n_rs_seq > 0` AND `n_backup_cells > 0` would waste the RS buffer that custom mode is designed to eliminate.

### 5.4 Device-Native Tape Placement

**Current:** [`server-dflash-custom.cpp:111`](common/server-dflash-custom.cpp:111): `ggml_backend_dev_t dev = model_dev_layer(model, (int)il);`

**Why:** Tape tensors must live on the same GPU as the model layer they capture. If tape is on CPU while the layer is on CUDA0, `ggml_cpy` would force a PCIe transfer, negating the performance benefit of replay.

### 5.5 Conv State Rebuild Must Use Pre-Draft State from Backup

**Current:** [`server-dflash-custom.cpp:612-616`](common/server-dflash-custom.cpp:612): Reads conv state from active R row after `dflash_custom_restore()`.

**Why:** The conv state rebuild algorithm shifts the sliding window forward by `n_accepted` tokens. It needs the pre-draft conv state as the starting point. If the active row still contains post-draft conv state (i.e., `dflash_custom_restore()` was not called), the shift would produce incorrect results.

---

## 6. Tight Coupling Analysis

### 6.1 Acceptable Coupling (Architecturally Necessary)

| Coupling | Location | Why Necessary |
|----------|----------|---------------|
| `qwen35.cpp` → `server-dflash-custom.h` | [`qwen35.cpp:6`](src/models/qwen35.cpp:6) | Capture MUST be in model's graph builder. No other location can access GDN intermediates. |
| `server-dflash-custom.cpp` → `llama-memory-recurrent.h` | [`server-dflash-custom.cpp:241`](common/server-dflash-custom.cpp:241) | Backup/restore requires direct access to R/S tensor rows. |
| `server-dflash-custom.cpp` → `llama-context.h` | [`server-dflash-custom.cpp:313`](common/server-dflash-custom.cpp:313) | Replay needs access to `llama_get_memory()` and `ctx->get_sched()`. |
| `server-dflash-custom.cpp` → `llama-model.h` | [`server-dflash-custom.cpp:6`](common/server-dflash-custom.cpp:6) | Tape allocation needs model hparams and device mapping. |

### 6.2 Avoidable Coupling (Refactoring Targets)

| Coupling | Location | Why Avoidable | Refactoring Target |
|----------|----------|---------------|-------------------|
| `cell_copy()` method on `llama_memory_recurrent` | [`llama-memory-recurrent.h:99`](src/llama-memory-recurrent.h:99) | Free function can access the same public members. | Opportunity A |
| `set_tape_gpu()` method on `llama_context` | [`llama-context.h:65`](src/llama-context.h:65) | Could be a free function that sets `cparams.tape_gpu` through a getter. | Opportunity B |
| `tape_gpu` in `llama_cparams` | [`llama-cparams.h:95`](src/llama-cparams.h:95) | Could be stored in a sidecar config struct referenced by `llama_context`. | Opportunity D |
| `n_backup_cells` in `llama_cparams` | [`llama-cparams.h:20`](src/llama-cparams.h:20) | Could be part of a `server_dflash_custom_config` struct. | Opportunity D |

### 6.3 Accidental Coupling (Low Risk)

| Coupling | Location | Risk | Notes |
|----------|----------|------|-------|
| `conv_channels = d_inner + 2 * H_k * S_k` | [`server-dflash-custom.cpp:65`](common/server-dflash-custom.cpp:65) | LOW | Qwen3.6-specific projection layout. Other SSM architectures may differ. Document as model-specific. |
| Internal header includes (`llama-model.h`, `llama-context.h`) | [`server-dflash-custom.cpp:4-7`](common/server-dflash-custom.cpp:4) | LOW | Uses internal headers for hparams/device access. Upstream may refactor these headers. |

---

## 7. Recommended Refactoring Priority

| Priority | Refactoring | Files Affected | Effort | Merge Risk Reduction |
|----------|-------------|----------------|--------|---------------------|
| **P2** | Remove `LLAMA_API` from `set_tape_gpu()` / `cell_copy()` | [`llama-context.h`](src/llama-context.h), [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | ~4 lines | Low (ABI cleanup) |
| **P2** | Add `#ifdef BEE_DFLASH_CUSTOM` guards | [`llama-cparams.h`](src/llama-cparams.h), [`llama-context.h`](src/llama-context.h), [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h), [`qwen35.cpp`](src/models/qwen35.cpp) | ~15 lines | High (clean diffs) |
| **P2** | Add marker comments in `qwen35.cpp` | [`qwen35.cpp:454-529`](src/models/qwen35.cpp:454) | ~4 lines | Medium (merge resolution aid) |
| **P3** | Extract `cell_copy()` to free function | [`server-dflash-custom.cpp`](common/server-dflash-custom.cpp), [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | ~25 lines | High (eliminates class modification) |
| **P3** | Configuration struct for custom mode | [`server-dflash-custom.h`](common/server-dflash-custom.h), [`common.cpp`](common/common.cpp) | ~15 lines | Low (better organization) |
| **P3** | Model-specific capture interface | [`server-dflash-custom.h`](common/server-dflash-custom.h), [`qwen35.cpp`](src/models/qwen35.cpp) | ~50 lines | Low (extension point only) |

---

## 8. Upstream Merge Strategy Implications

### 8.1 Current State

The Task 6R implementation produces the following diff against upstream:

| Category | Files | Lines | Merge Strategy |
|----------|-------|-------|---------------|
| New files | 2 | ~905 | No conflict — always clean |
| Struct extensions | 2 | ~5 fields | Low risk — additive |
| Constructor param forwarding | 6 | ~12 lines | Medium risk — parameter order changes |
| Method additions | 2 | ~25 lines | Medium risk — upstream may add similar methods |
| Conditional blocks | 3 | ~100 lines | Low risk — flag-gated, non-overlapping |

### 8.2 Merge Conflict Prediction

**Likely conflicts:**
1. [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h) — if upstream modifies constructor or adds backup-like methods
2. [`llama-model.cpp`](src/llama-model.cpp) — if upstream changes constructor signature
3. [`llama-cparams.h`](src/llama-cparams.h) — if upstream adds fields near `n_backup_cells`

**Unlikely conflicts:**
- [`qwen35.cpp`](src/models/qwen35.cpp) — capture block is in a unique location within the GDN builder
- [`common/common.cpp`](common/common.cpp) — override is in a unique location within `common_context_params_to_llama()`
- [`server-context.cpp`](tools/server/server-context.cpp) — integration points are in unique locations within speculative scheduling

### 8.3 Recommended Merge Approach

When upstream llama.cpp evolves:

1. **Merge upstream first** — let upstream changes apply cleanly
2. **Reapply Task 6R as a patch** — the new files apply automatically; the modified files may need manual resolution
3. **Verify constructor signatures** — check that `n_backup_cells` is at the correct position in each constructor call
4. **Run the test script** — verify replay still executes correctly after merge

---

## 9. Summary

### What Is Well-Designed

- **Core isolation:** The ~900 lines of custom logic in `server-dflash-custom.h/cpp` are cleanly separated from upstream.
- **Flag gating:** All custom behavior is behind `--beefix-dflash-custom`. Stock DFlash is completely unaffected.
- **Fallback safety:** Try-catch with checkpoint rollback guarantee means custom failures never corrupt state.
- **Device awareness:** Tape placement matches model layer devices, ensuring GPU-native performance.

### What Could Be Improved

- **`cell_copy()` extraction:** Moving this to a free function would eliminate one class modification.
- **`LLAMA_API` removal:** Internal methods should not be DLL-exported.
- **Compile-time guards:** `#ifdef BEE_DFLASH_CUSTOM` would produce cleaner upstream diffs.
- **Marker comments:** Adding `// Task 6R` markers at every modification point would aid future merge resolution.

### What Must Not Change

- **Lifecycle ordering:** backup → capture → reset → verify → replay/checkpoint.
- **`n_backup_cells = n_parallel`:** Not `2 × n_parallel`.
- **`n_rs_seq = 0` with custom mode:** The RS buffer must be eliminated.
- **Device-native tape placement:** Tape must live on the same GPU as the model layer.
- **Conv state rebuild from backup:** Must use pre-draft state restored by `dflash_custom_restore()`.

---

*End of modularity review document.*
