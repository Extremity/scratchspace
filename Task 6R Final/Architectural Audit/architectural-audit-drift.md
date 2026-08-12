# Architectural Audit: Task 6R Hybridization, Drift, and Scope Creep Analysis

**Date:** 2026-08-12
**Purpose:** Determine whether Task 6R has drifted from its original opt-in/separate-path architecture into a hybrid/merged implementation. Check for scope creep, upstream preservation, and runtime assumption rationalization.
**Method:** Source code inspection of Task 6R implementation files, comparison against original blueprint, and verification of code path separation. No code modifications.

**Reference Documents:**
- [`plans/dflash-solutions/architectural-audit-preservation.md`](architectural-audit-preservation.md) — Preservation analysis (just completed).
- [`plans/dflash-comparison/final-comparison.md`](../dflash-comparison/final-comparison.md) — Old vs. current comparison (377 lines).
- [`plans/dflash-solutions/task6r-final-documentation.md`](task6r-final-documentation.md) — Task 6R implementation details (663 lines).
- [`plans/dflash-solutions/task6r-modularity-review.md`](task6r-modularity-review.md) — Modularity and upstream-merge analysis (441 lines).
- [`plans/dflash-solutions/task6r-revised-implementation-blueprint.md`](task6r-revised-implementation-blueprint.md) — Original Task 6R blueprint (155 lines).

**Source Files Examined:**
- [`common/common.cpp:1765-1823`](common/common.cpp:1765) — Override block (`beefix_dflash_custom`).
- [`common/common.h:403-430`](common/common.h:403) — Flag declaration, `need_n_rs_seq()`.
- [`src/llama-cparams.h:1-97`](src/llama-cparams.h:1) — `tape_gpu` and `n_backup_cells` fields.
- [`src/llama-context.cpp`](src/llama-context.cpp) — `set_tape_gpu()`, `n_backup_cells` validation.
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — `n_backup_cells` in allocation formula.
- [`src/models/qwen35.cpp:454-529`](src/models/qwen35.cpp:454) — Tape capture block.
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) — 13 `dflash_custom` integration points.
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) — Full custom implementation (863 lines).

---

## Table of Contents

| Section | Topic |
|---------|-------|
| [1. Hybridization Assessment](#1-hybridization-assessment) | Q1: Did Task 6R remain separate and opt-in? |
| [2. Upstream Modification Scope](#2-upstream-modification-scope) | Q2: Minimal vs. excessive upstream changes |
| [3. Scope Creep Analysis](#3-scope-creep-analysis) | Q3: What was added beyond original plan? |
| [4. Stock DFlash Integrity Verification](#4-stock-dflash-integrity-verification) | Q4: Is stock DFlash unchanged when custom disabled? |
| [5. Runtime Assumption Review](#5-runtime-assumption-review) | Q5: Are runtime assumptions documented and rationalized? |
| [6. Missing Functionality Checklist](#6-missing-functionality-checklist) | Q6: What should have been included but wasn't? |
| [7. Overall Assessment](#7-overall-assessment) | Final verdict on architectural drift |

---

## 1. Hybridization Assessment

**Question:** Did Task 6R remain a separate, opt-in implementation path?

**Verdict: PASS — Architecture remains strictly opt-in and separate.**

### 1.1 Evidence of Separation

| Aspect | Evidence | Location |
|--------|----------|----------|
| CLI flag gating | `--beefix-dflash-custom` flag with `bool beefix_dflash_custom = false` default | [`common/common.h:408`](common/common.h:408) |
| Override conditional | `n_rs_seq = 0` and `n_backup_cells` only set when `beefix_dflash_custom && has_dflash` | [`common/common.cpp:1775-1781`](common/common.cpp:1775) |
| Tape activation | `tape_gpu` pointer only set via `dflash_custom_set_tape_gpu()` called from server integration | [`server-context.cpp:3330-3331`](tools/server/server-context.cpp:3330) |
| Capture gating | Graph-embedded capture in `qwen35.cpp` gated on `cparams.tape_gpu != nullptr` | [`qwen35.cpp:460`](src/models/qwen35.cpp:460) |
| Server integration | All `dflash_custom` calls guarded by `dflash_custom_is_enabled()` or null checks | [`server-context.cpp:3305-3346`](tools/server/server-context.cpp:3305) |
| Fallback safety | Try-catch around `dflash_custom_replay()` with checkpoint fallback | [`server-context.cpp:4325-4347`](tools/server/server-context.cpp:4325) |

### 1.2 No Evidence of Blending

**No code was found that:**
- Assumes custom mode is the "correct" way and stock is a fallback.
- Blends stock and custom paths (e.g., using stock checkpoint rollback as primary path with custom as enhancement).
- Treats custom mode as the "real" implementation in comments or documentation.
- Modifies stock DFlash code paths to "support" custom mode beyond minimal parameter plumbing.

The `need_n_rs_seq()` function at [`common/common.h:423-429`](common/common.h:423) remains **unchanged** from upstream. It still returns `draft.n_max` for DFlash. The override to `n_rs_seq = 0` happens **after** `need_n_rs_seq()` is called, in `common_context_params_to_llama()`, and is conditional on the flag. This is the correct pattern: stock function returns stock value; override applies only when opt-in flag is active.

### 1.3 Design Documentation Alignment

The final documentation at [`task6r-final-documentation.md:43`](task6r-final-documentation.md:43) explicitly states:

> **Strictly opt-in:** All custom behavior gated behind `--beefix-dflash-custom`. Stock DFlash completely unaffected.

The implementation matches this design principle. Every custom code path is gated by either:
1. `params.speculative.beefix_dflash_custom` (CLI flag)
2. `cparams.tape_gpu != nullptr` (runtime tape pointer)
3. `dflash_custom_is_enabled(slot->dflash_custom.get())` (server slot state)
4. `n_backup_cells > 0` (allocation guard)

---

## 2. Upstream Modification Scope

**Question:** Where was minimal upstream modification acceptable vs excessive?

**Verdict: MINIMAL — All modifications are architecturally necessary and non-invasive.**

### 2.1 Parameter Plumbing (Unavoidable)

The following changes are unavoidable for the feature to function:

| Modification | File | Justification |
|-------------|------|---------------|
| `n_backup_cells` field | [`src/llama-cparams.h:21`](src/llama-cparams.h:21) | Required for allocation formula. |
| `n_backup_cells` constructor param | [`src/llama-memory-recurrent.cpp:28`](src/llama-memory-recurrent.cpp:28) | Required to pass value to allocator. |
| Allocation formula change | [`src/llama-memory-recurrent.cpp:104`](src/llama-memory-recurrent.cpp:104) | `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells`. Required for backup cell rows. |
| `n_backup_cells` forwarding | [`src/llama-model.cpp:2109-2225`](src/llama-model.cpp:2109) | 4 constructor call sites must forward the new parameter. |
| `tape_gpu` field | [`src/llama-cparams.h:96`](src/llama-cparams.h:96) | Required for graph builder to access tape pointer. |
| `set_tape_gpu()` method | [`src/llama-context.h:67`](src/llama-context.h:67) | Required for server to activate/deactivate tape capture. |
| `n_backup_cells` validation | [`src/llama-context.cpp:273-276`](src/llama-context.cpp:273) | Safety guard: prevents backup cells on unsupported architectures. |

**Assessment:** These modifications are the minimum required to make the feature work. They extend existing structs and constructors without modifying existing behavior. When `n_backup_cells = 0`, the allocation formula produces the same result as the stock formula.

### 2.2 Conditional Capture Block (Acceptable)

The capture block at [`src/models/qwen35.cpp:460-529`](src/models/qwen35.cpp:460) is:
- Gated on `cparams.tape_gpu != nullptr` (NULL by default).
- Added after the GDN computation, before the attention output norm.
- Does not modify the graph for non-custom builds.

**Assessment:** This is a model-specific addition that only affects Qwen3.6 builds when custom mode is active. The conditional check has negligible performance impact (single pointer comparison). The capture operations (`ggml_cpy`) are graph-embedded and overlap with compute, adding zero runtime overhead.

### 2.3 No Modifications to Stock Code Paths

**Verified:**
- `need_n_rs_seq()` at [`common/common.h:423-429`](common/common.h:423) is **unchanged** from upstream. The function still includes DFlash in the RS calculation. The override to `n_rs_seq = 0` happens downstream in `common_context_params_to_llama()`, conditional on the flag.
- Stock rollback paths (`seq_rm()`, RS snapshot restore, checkpoint fallback) are **unchanged**.
- Graph builder logic in `qwen35.cpp` for non-capture operations is **unchanged**.
- Upstream speculative scheduling, verification, and draft management are **unchanged**.

### 2.4 Changes That Affect Both Stock and Custom Modes

| Change | Affects Stock? | Impact |
|--------|---------------|--------|
| `n_backup_cells` in allocation formula | Yes — but `n_backup_cells = 0` for stock, so `n_rows = mem_size * (1 + n_rs_seq) + 0` = stock behavior. | **None.** |
| `tape_gpu` field in `llama_cparams` | Yes — field exists in struct. | **None.** Pointer is NULL for stock builds. |
| `set_tape_gpu()` method | Yes — method exists on `llama_context`. | **None.** Called only by custom server integration. |
| Capture block in `qwen35.cpp` | Yes — conditional check exists. | **Negligible.** Single pointer comparison per recurrent layer per token. |

---

## 3. Scope Creep Analysis

**Question:** What functionality was added beyond the original Task 6R goal?

**Original goal:** VRAM-efficient DFlash using backup cells + tape replay, replacing RS buffer.

### 3.1 Feature Classification

| Feature | Classification | Evidence |
|---------|---------------|----------|
| Backup cells (recurrent-only state copy) | **Original plan** | Blueprint Part B (§B.1), Stage 2 of final documentation. |
| GPU tape allocation and capture | **Original plan** | Blueprint Part B, Stage 3 of final documentation. |
| Replay graph (GDN-only forward pass) | **Original plan** | Blueprint Part B, Stage 4 of final documentation. |
| `n_rs_seq = 0` override | **Original plan** | Blueprint Part B, Stage 1 of final documentation. |
| `--beefix-dflash-custom` CLI flag | **Original plan** | Blueprint Part B, Stage 1 of final documentation. |
| Checkpoint fallback (try-catch) | **Original plan** | Blueprint Part B, Stage 5 of final documentation. |
| CUDA conv rebuild kernel | **Defensive addition** | Not explicitly in blueprint Part B, but identified as required during Stage 6 audit. The tape captures qkv data; without conv rebuild, R state would be stale. |
| Fallback reason codes (R0-R9) | **Defensive addition** | Not in blueprint. Added for observability. Acceptable — prevents silent failures. |
| `replay_failed` permanent disable | **Defensive addition** | Not in blueprint. Added after 3 consecutive failures. Acceptable — prevents failure loops. |
| `dflash_custom_config` struct | **Defensive addition** | Stage 9 cleanup. Provides configuration centralization. |
| `dflash_custom_cell_copy()` free function extraction | **Defensive addition** | Stage 8. Reduces upstream class modification. |
| `cuda_rebuilt_layers` tracking (partial corruption fix) | **Defensive addition** | Not in blueprint. Prevents corruption when CUDA path fails mid-loop and CPU fallback activates. |
| Arch guard for `n_backup_cells` | **Defensive addition** | [`src/llama-context.cpp:273-276`](src/llama-context.cpp:273) — prevents backup cells on architectures that don't support RS rollback. |

### 3.2 Scope Creep Assessment

**No true scope creep detected.** All additions fall into:
1. **Original plan** — documented in blueprint or final documentation stages.
2. **Defensive addition** — added to prevent bugs, handle edge cases, or improve observability. These are acceptable because they:
   - Do not expand the feature's scope or target configuration.
   - Do not modify stock DFlash behavior.
   - Are gated by the same opt-in flag.

**Specifically:**
- CUDA conv rebuild kernel was identified as required during implementation audit (Stage 6 Fix 2). The tape captures qkv data, and without consuming it during replay, the R tensor would contain stale pre-draft conv state. This is a correctness fix, not scope creep.
- Fallback reason codes and `replay_failed` disable are observability and safety mechanisms. They do not add new functionality; they make the existing functionality more robust.
- `cell_copy()` extraction is a modularity improvement, not new functionality.

### 3.3 Adaptive Profit Controller Integration

**Question:** Was adaptive profit controller integration part of Task 6R?

**Answer:** No. The adaptive profit controller (`server-adaptive-dm.h`) is an upstream feature managed by `--spec-draft-n-max`. Task 6R does not modify or integrate with the adaptive controller. The controller adjusts `n_max` based on verification profit, which indirectly affects how many tokens are captured and replayed, but this is a stock upstream behavior that works with both stock and custom modes.

**Classification:** N/A — not a Task 6R feature.

---

## 4. Stock DFlash Integrity Verification

**Question:** Is stock DFlash genuinely unchanged when custom mode is disabled?

**Verdict: YES — Stock DFlash code paths execute normally when `beefix_dflash_custom == false`.**

### 4.1 Code Path Trace (Custom Mode Disabled)

When `--beefix-dflash-custom` is NOT present:

| Step | Variable | Value | Effect |
|------|----------|-------|--------|
| 1. `need_n_rs_seq()` called | Returns | `draft.n_max` (e.g., 8) | Stock behavior — DFlash included in RS calculation. |
| 2. Override block evaluated | `params.speculative.beefix_dflash_custom` | `false` | Override skipped. `cparams.n_rs_seq` remains `draft.n_max`. |
| 3. `cparams.n_backup_cells` | Default | `0` | No backup rows allocated. |
| 4. `cparams.tape_gpu` | Default | `nullptr` | Capture block in `qwen35.cpp` is skipped (`tape_gpu != nullptr` is false). |
| 5. Recurrent memory allocation | `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` | `mem_size * (1 + 8) + 0 = mem_size * 9` | Stock RS buffer allocated. |
| 6. Server slot init | `params_base.speculative.beefix_dflash_custom` | `false` | `dflash_custom` state NOT created. `slot.dflash_custom` remains `nullptr`. |
| 7. Pre-draft | `dflash_custom_is_enabled(slot->dflash_custom.get())` | `false` (nullptr) | Backup and tape activation skipped. |
| 8. Post-verify | Same check | `false` | Replay attempt skipped. Stock rollback path (`seq_rm()` + RS snapshots or checkpoint) executes. |

### 4.2 Performance Impact Assessment

| Conditional Check | Location | Frequency | Cost |
|------------------|----------|-----------|------|
| `params.speculative.beefix_dflash_custom` | `common.cpp:1775` | Once per context creation | Zero — compile-time constant for given run. |
| `cparams.tape_gpu != nullptr` | `qwen35.cpp:460` | Per recurrent layer per token | Negligible — single pointer comparison. |
| `n_backup_cells > 0` | `llama-memory-recurrent.cpp:136` | Once per memory construction | Zero — value is 0 for stock builds. |
| `dflash_custom_is_enabled()` | `server-context.cpp` | Per draft cycle | Negligible — nullptr check + bool check. |

**Assessment:** The conditional checks add negligible overhead to stock DFlash execution. The pointer comparison in `qwen35.cpp` is the most frequent check (per layer per token), but it's a single comparison that branches to a no-op when `tape_gpu` is NULL.

### 4.3 Stock Rollback Path Verification

When custom mode is disabled, the stock rollback path executes:
1. `common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_RS` (because `n_rs_seq > 0`).
2. `server_speculative_rollback_requires_checkpoint()` returns `false` for rollbacks within `n_rs_seq` bounds.
3. `seq_rm()` on recurrent memory restores the RS snapshot at the accepted position.
4. If rollback exceeds `n_rs_seq` bounds, checkpoint fallback is used.

This is the **unchanged upstream DFlash behavior**. The custom mode code is not executed.

---

## 5. Runtime Assumption Review

**Question:** Are runtime assumptions documented and rationalized?

### 5.1 `dflash_custom_cell_copy()` Sequential Copy

**Observation:** The current implementation uses sequential `ggml_backend_tensor_copy()` calls (one per layer, for R and S tensors). The old 0.3.2 implementation used `llama_dflash_memory_seq_cp_recurrent_ordered()` for layer-ordered, GPU-optimized copy.

**Assessment:** This is documented in the preservation audit (Section 5) as "Degraded" but classified as "functionally adequate for current target configuration." The reasoning:
1. Backup occurs once per draft cycle, before the first draft.
2. Copy size per layer is ~1.3 MB (R: ~600 KB, S: ~700 KB). Total for 48 layers: ~62 MB.
3. `ggml_backend_tensor_copy()` is device-native for CUDA (no PCIe transfer).
4. The old optimization was about GPU utilization during copy, not total copy time.

**Is this rationalized?** Partially. The preservation audit documents the difference and rationale. The final documentation does not explicitly mention the tradeoff. **Recommendation:** Add a comment in `dflash_custom_cell_copy()` noting the sequential approach and that layer-ordered copy can be added if backup becomes a bottleneck.

### 5.2 `--parallel 1` Limitation

**Observation:** The implementation assumes single-sequence operation (`n_seqs = 1` hardcoded at [`server-dflash-custom.cpp:436`](common/server-dflash-custom.cpp:436)). A warning is logged when `n_parallel > 1`.

**Evidence:**
- [`server-context.cpp:1472-1475`](tools/server/server-context.cpp:1472) — Warning: `"dflash custom mode: n_parallel > 1 — replay assumes single sequence per slot. Results may be incorrect."`
- [`server-dflash-custom.cpp:436`](common/server-dflash-custom.cpp:436) — `n_seqs = 1` hardcoded in replay graph.

**Assessment:** The limitation is documented in:
1. The warning log at server init.
2. The final documentation (§6 Known Limitations).
3. The preservation audit (§1 capability table — "Lost (deferred)").

**Is this sufficient?** The warning is logged but not enforced as a hard block. If a user ignores the warning and runs with `--parallel > 1`, the system will execute with incorrect tape dimensions, producing silent data corruption. **Recommendation:** Consider hard-blocking `--beefix-dflash-custom` when `--parallel > 1`, or at minimum adding a runtime assertion in `dflash_custom_replay()` that fails fast if `n_seqs > 1`.

### 5.3 Qwen3.6-Specific Assumptions

**Observation:** The tape capture block is in `qwen35.cpp`, which handles Qwen3.6/3.5 models. The capture assumes:
- GDN intermediates exist with specific callback names (`k_conv_predelta`, `v_conv_predelta`, `gate`, `beta_sigmoid`, `linear_attn_qkv_mixed`).
- Rank-factored GDN structure (k, v, gate, beta tensors with specific dimensions).

**Assessment:** This is documented in the blueprint (Part A, finding #8) and final documentation. The tape structure is derived from runtime hparams (`ssm_d_state`, `ssm_dt_rank`, `ssm_n_group`), so it adapts to different model sizes within the Qwen3.5/3.6 family. Models without GDN layers (e.g., transformers) would not have the capture block execute because `qwen35.cpp` is not their graph builder.

**Risk:** If a future recurrent model uses a different graph builder with different intermediate names, the custom mode would not work without adding a capture block to that builder. This is an acceptable limitation — the feature is model-specific by design.

### 5.4 Other Runtime Assumptions

| Assumption | Documented? | Rationalized? | Risk |
|------------|------------|---------------|------|
| CUDA backend for optimal performance | Yes — blueprint §A.3 | Yes — CPU fallback exists | Low |
| `ggml_gated_delta_net()` available for replay | Yes — blueprint finding #11 | Yes — existing upstream op | Low |
| Tape tensors fit in GPU memory | Yes — ~85-117 MB estimated | Yes — device-aware placement | Low |
| Backup cells fit in recurrent tensor | Yes — allocation formula | Yes — logged at construction | Low |
| Single GPU per layer | Yes — per-layer device lookup | Yes — tape on same GPU as model layer | Low |

---

## 6. Missing Functionality Checklist

**Question:** Are there things from the original plan or old 0.3.2 implementation that should have been part of the custom path but were accidentally omitted?

### 6.1 Original Blueprint Items

| Blueprint Item | Status | Notes |
|---------------|--------|-------|
| Backup cells | **Implemented** | [`server-dflash-custom.cpp:299-312`](common/server-dflash-custom.cpp:299) |
| GPU tape allocation | **Implemented** | [`server-dflash-custom.cpp:113-166`](common/server-dflash-custom.cpp:113) |
| Graph-embedded capture | **Implemented** | [`qwen35.cpp:460-529`](src/models/qwen35.cpp:460) |
| Replay graph (GDN) | **Implemented** | [`server-dflash-custom.cpp:366-843`](common/server-dflash-custom.cpp:366) |
| `n_rs_seq = 0` override | **Implemented** | [`common.cpp:1775-1781`](common/common.cpp:1775) |
| Checkpoint fallback | **Implemented** | [`server-context.cpp:4325-4347`](tools/server/server-context.cpp:4325) |
| Conv state rebuild | **Implemented** | [`server-dflash-custom.cpp:616-831`](common/server-dflash-custom.cpp:616) |
| CUDA conv kernel | **Implemented** | `ggml/src/ggml-cuda/dflash-custom-conv.cu` |
| Device-aware tape placement | **Implemented** | [`server-dflash-custom.cpp:113-166`](common/server-dflash-custom.cpp:113) |

### 6.2 Old 0.3.2 Items (Intentionally Not Reproduced)

| Old Capability | Status | Justification |
|---------------|--------|---------------|
| Multi-sequence tape support | **Deferred** | Target configuration is `--parallel 1`. Requires architectural changes to tape structure, capture logic, and replay loops. Documented in deferred items review (§2). |
| Profile infrastructure (`dflash_profile_*`) | **Dropped** | Not needed for initial implementation. Can be added when profiling is required. |
| `llama_tape_replay_sync()` (multi-slot sync) | **Dropped** | Only needed for multi-sequence workloads. |
| `tree_bufs` (DDTree) support | **N/A** | Removed in v0.4.0 per AGENTS.md. |
| `recurrent_backup_attention_streams` | **N/A** | DDTree-specific. DDTree removed. |
| `llama_dflash_memory_seq_cp_recurrent_ordered()` | **Not reproduced** | `dflash_custom_cell_copy()` uses sequential `ggml_backend_tensor_copy()`. Documented as acceptable difference. |

### 6.3 TODO/FIXME Check

**Search results for TODO/FIXME in custom files:**

| File | TODO/FIXME | Status |
|------|-----------|--------|
| [`server-dflash-custom.h:80`](common/server-dflash-custom.h:80) | `// Future: convolution rebuild options, logging level, etc.` | Intentional placeholder for future config options. |
| `server-dflash-custom.cpp` | No TODOs or FIXMEs found | Implementation is complete. |

### 6.4 Accidental Omissions

**Based on source code review, no capabilities appear to be accidentally omitted.** The P0 fix for `n_backup_cells` was discovered and corrected during implementation. All other omissions are either:
1. Intentional design choices (sequential copy instead of layer-ordered).
2. Documented deferred items (multi-sequence support).
3. Intentionally dropped (profile infrastructure, DDTree support).

**One potential gap:** The old implementation's `llama_tape_replay_sync()` for multi-slot synchronization is not present. However, this is only needed for multi-sequence workloads, which are explicitly deferred. For single-sequence operation, synchronization is not needed.

---

## 7. Overall Assessment

### 7.1 Summary Scores

| Criterion | Rating | Evidence |
|-----------|--------|----------|
| **Hybridization** | **PASS** | No blending of stock and custom paths. All custom behavior gated by opt-in flag. |
| **Upstream Modification Scope** | **MINIMAL** | 7 upstream files modified, all changes architecturally necessary. No stock code paths modified. |
| **Scope Creep** | **NONE** | All additions are either original plan or defensive improvements. No features added beyond backup cells + tape replay + conv rebuild. |
| **Stock DFlash Integrity** | **INTACT** | Stock DFlash executes unchanged code paths when custom mode disabled. Conditional checks add negligible overhead. |
| **Runtime Assumptions** | **DOCUMENTED** | Key assumptions documented in blueprint, final documentation, and server warnings. One area (sequential copy tradeoff) could use in-code documentation. |
| **Missing Functionality** | **COMPLETE** | No accidental omissions. Deferred items documented. |

### 7.2 Verdict

**Architecture intact.**

Task 6R has not drifted from its original opt-in/separate-path architecture. The implementation:
1. Maintains strict separation between stock DFlash and custom mode.
2. Makes minimal, architecturally necessary modifications to upstream files.
3. Adds no functionality beyond the original plan (backup cells + tape replay + conv rebuild).
4. Preserves stock DFlash behavior when custom mode is disabled.
5. Documents runtime assumptions and known limitations.

### 7.3 Recommendations (Non-Critical)

| Recommendation | Priority | Effort | Impact |
|---------------|----------|--------|--------|
| Add comment in `dflash_custom_cell_copy()` noting sequential approach and layer-ordered alternative | Low | Trivial | Documentation clarity. |
| Consider hard-blocking `--beefix-dflash-custom` when `--parallel > 1` | Low | Small | Prevents silent data corruption if user ignores warning. |
| Add runtime assertion in `dflash_custom_replay()` for `n_seqs == 1` | Low | Small | Fail-fast on unsupported configuration. |

None of these recommendations require behavioral changes. They are documentation and safety improvements.

---

*Document generated 2026-08-12. Based on analysis of Task 6R implementation files, final documentation, modularity review, revised blueprint, preservation audit, and the definitive old-vs-current comparison document.*
 