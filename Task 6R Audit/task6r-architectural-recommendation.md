# Task 6R DFlash Custom Mode — Architectural Recommendation

**Date:** 2026-08-10
**Based on:** Subtask A (old implementation deep dive), Subtask B (constraint analysis), Subtask C (modularity assessment)
**Status:** Investigation complete. Recommendation: **Improve in place with targeted fixes.**

---

## Executive Summary

**Primary Question:** Are we unnecessarily constraining the custom implementation by trying to fit it into the upstream architecture?

**Answer:** **Partially yes.** The current Task 6R implementation is structurally sound and follows the upstream architecture well, but it has two areas where it accepted compromises that materially affect functionality:

1. **Missing `n_backup_cells` population** (CRITICAL, already identified in audit) — replay can never execute
2. **Missing conv state rebuild in replay** (CRITICAL, newly identified) — replay produces incorrect conv state even if it executes

These two issues mean the custom replay path **cannot currently function correctly**. The server falls back to checkpoint rollback every cycle, which is correct but eliminates the performance benefit of custom replay.

Beyond these fixes, the implementation is **not unnecessarily constrained**. The ggml graph approach for replay is portable and correct. The row-based backup cells are simpler than the old deferred expansion. The device-aware tape placement matches the old implementation. The try-catch fallback to checkpoint is a safety improvement over the old implementation.

**Recommendation:** Improve the current implementation in place with P0 fixes. Do not restart, do not substantially redesign.

---

## Detailed Analysis

### What the Old Implementation Did Better

| Capability | Old 0.3.2 | Task 6R Current | Can Task 6R Match? |
|-----------|-----------|-----------------|-------------------|
| `n_rs_seq=0` for DFlash | Excluded from `need_n_rs_seq()` | Override in `common_context_params_to_llama()` | ✅ Already equivalent |
| Backup cells | Extra sequences, deferred expansion | Extended R/S tensor rows, static allocation | ✅ Already equivalent (static is simpler) |
| Tape capture | Graph-embedded `ggml_cpy` | Graph-embedded `ggml_cpy` | ✅ Identical |
| Tape replay | Direct CUDA kernel + ggml fallback | ggml graph only | ⚠️ ggml is portable but slower (~1-2ms/replay) |
| Conv state replay | Separate `tape_replay_conv()` | **NOT IMPLEMENTED** | ❌ **Must add** |
| Multi-GPU support | Per-layer device, mixed-device detection | Per-layer device via `model_dev_layer()` | ✅ Already equivalent |
| Async replay | Launch/sync split | Synchronous | ✅ Already equivalent (minimal benefit) |
| Fallback safety | Path selection (no try-catch) | Try-catch → checkpoint | ✅ Task 6R is safer |
| CPU replay fallback | Full CPU implementation | Falls back to checkpoint | ✅ Checkpoint fallback is adequate |

### What Task 6R Does Better

| Capability | Old 0.3.2 | Task 6R |
|-----------|-----------|---------|
| Fallback safety | Path selection, no try-catch | Try-catch with checkpoint rollback guarantee |
| Backup model | Dynamic sequence expansion/shrink | Simple row-based within same tensor |
| Code footprint | ~3,000+ lines across 12+ files | ~800 lines in 2 new files + ~200 lines modifications |
| Upstream compatibility | Deep fork divergence | Minimal invasive changes |
| Maintainability | High fork drift burden | Isolated core, clear conditional boundaries |

---

## The Four Options Evaluated

### Option 1: Improve the current implementation in place

**Scope:** Fix the two P0 issues (n_backup_cells + conv state rebuild), plus minor improvements.

**Pros:**
- Minimal disruption — the existing architecture is sound
- Fixes correctness issues without changing design
- Preserves all existing safety guarantees (try-catch fallback, triple-gated opt-in)
- ~100 lines of additional code

**Cons:**
- Does not add the direct CUDA kernel (replay remains ggml graph-bound)
- Does not add deferred backup expansion (static allocation)

**Estimated effort:** ~100 lines of code changes

### Option 2: Substantially redesign/refactor the custom path in place

**Scope:** Add direct CUDA kernel, deferred backup expansion, async replay, multi-path fallback chain.

**Pros:**
- Would match old implementation performance more closely
- More robust replay fallback chain

**Cons:**
- ~500+ lines of additional CUDA-specific code
- Deferred expansion requires invasive tensor resize changes
- Significantly increases fork drift from upstream
- Most changes provide marginal benefits (async replay saves ~2-5ms, deferred expansion saves ~612 MB only during non-speculative workloads)

**Estimated effort:** ~500-800 lines of code changes

### Option 3: Revert to pre-Task-6R and implement again

**Scope:** Start from scratch using everything learned.

**Pros:**
- Could theoretically produce a cleaner design

**Cons:**
- Would lose all the work already done (6 completed subtasks, successful builds)
- Would need to rediscover and re-verify the same interfaces
- The current implementation is already architecturally sound
- No material design insight has been discovered that would justify a complete restart

**Estimated effort:** Full reimplement (~800 lines), high risk of reintroducing already-fixed bugs

### Option 4: Leave the implementation as-is

**Scope:** No changes beyond the P0 `n_backup_cells` fix.

**Pros:**
- Minimal effort

**Cons:**
- Conv state rebuild is a correctness issue — without it, replay produces incorrect conv state
- Replay would execute (after n_backup_cells fix) but produce wrong results for the conv window
- This would go undetected by the test script (output comparison passes because checkpoint fallback produces correct output)

---

## Recommendation: Option 1 with P0 Fixes

**Improve the current implementation in place.** The architecture is sound. The two P0 issues are straightforward fixes that complete the implementation as designed.

### Priority Action List

| Priority | Action | Files | Lines | Rationale |
|----------|--------|-------|-------|-----------|
| **P0** | Fix `n_backup_cells` population | [`common/common.cpp:1775-1781`](common/common.cpp:1775) | ~3 | Without this, replay never executes. Already identified in audit. |
| **P0** | Add conv state rebuild to replay | [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) | ~50-80 | Without this, replay produces incorrect conv state. Correctness fix. |
| **P1** | Add replay-path logging | [`tools/server/server-context.cpp:4303-4327`](tools/server/server-context.cpp:4303) | ~5 | Test script needs to distinguish replay success from checkpoint fallback. |
| **P2** | Remove `LLAMA_API` from `set_tape_gpu()` / `cell_copy()` | [`src/llama-context.h:65`](src/llama-context.h:65), [`src/llama-memory-recurrent.h:99`](src/llama-memory-recurrent.h:99) | ~4 | Use direct member access instead. Eliminates DLL export concerns. |
| **P2** | Add `#ifdef BEE_DFLASH_CUSTOM` guards | [`src/llama-cparams.h`](src/llama-cparams.h), [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | ~10 | Compile-time isolation for cleaner upstream diffs. |
| **P2** | Add marker comments in `qwen35.cpp` | [`src/models/qwen35.cpp:454-529`](src/models/qwen35.cpp:454) | ~4 | Aid future merge resolution. |
| **P3** | Add direct CUDA kernel fast path | [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp), CUDA backend | ~300 | Performance optimization. Can be deferred. |
| **P3** | Document `conv_channels` formula | [`common/server-dflash-custom.cpp:65`](common/server-dflash-custom.cpp:65) | ~2 | Note Qwen-specific assumption for future reference. |

### What NOT to Change

| Item | Reason |
|------|--------|
| Deferred backup expansion | Requires invasive tensor resize changes. ~612 MB savings only during non-speculative workloads. |
| Async replay | ggml graph compute is inherently synchronous. ~2-5ms overlap opportunity is minimal in serving context. |
| Multi-path fallback chain | Try-catch → checkpoint is simpler and safer than the old path-selection approach. |
| CPU replay fallback | Checkpoint fallback is correct and adequate. CPU replay adds complexity for a path that should rarely trigger. |
| Tree mode (DDTree) support | Separate feature, not required for flat DFlash. |

---

## Why Option 1 is the Right Choice

1. **The architecture is already sound.** All cross-subtask interfaces are structurally correct (verified by audit). The runtime lifecycle, opt-in safety, fallback guarantees, and device placement are all properly implemented.

2. **The P0 issues are straightforward fixes.** `n_backup_cells` population is a 3-line change. Conv state rebuild is ~50-80 lines of sliding window logic using tape data already captured.

3. **The old implementation's advantages are largely matched.** The old implementation had 8 advantages. Task 6R already matches 5 of them (n_rs_seq=0, tape capture, multi-GPU, async, and has a safety improvement over the old fallback). The remaining 3 (direct CUDA kernel, deferred expansion, CPU replay) are either nice-to-have optimizations or not worth the complexity.

4. **No material design insight justifies a restart.** The investigation did not discover that the current architecture is fundamentally wrong. The issues found are implementation gaps (missing lines of code), not architectural flaws.

5. **Modularity is adequate.** The implementation scores 3.5/5 on modularity. The core logic is isolated in 2 new files. The invasive changes to upstream files are necessary by architectural constraints (capture MUST be in model graph builder, backup cells MUST extend recurrent memory).

---

## Impact on Upstream Compatibility

The recommended changes do not affect upstream compatibility:

- `n_backup_cells` fix is in `common/common.cpp` behind the `beefix_dflash_custom` flag
- Conv state rebuild is in `server-dflash-custom.cpp` (new file, behind flag)
- P2 improvements (LLAMA_API removal, #ifdef guards, marker comments) are isolated changes
- P3 direct CUDA kernel would be behind `ggml_backend_reg_get_proc_address()` (extensible, upstream-compatible)

---

## Timeline

| Phase | Actions | Status |
|-------|---------|--------|
| Phase 1 | Fix P0 issues (n_backup_cells + conv state rebuild) | Not started |
| Phase 2 | Add P1 improvements (replay logging, test validation) | Not started |
| Phase 3 | Add P2 improvements (LLAMA_API, #ifdef, comments) | Not started |
| Phase 4 (optional) | Add P3 optimizations (direct CUDA kernel) | Not started |

---

*End of architectural recommendation.*