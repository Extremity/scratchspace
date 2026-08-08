# DFlash VRAM Research — Complete Summary for Context Recovery

**Last Updated:** 2026-08-07

This file is a condensed summary of all research completed to date. If context is condensed, read this file first to recover the full picture, then reference the detailed documents listed at the bottom.

---

## The Problem

DFlash speculative decoding adds **5.4GB VRAM overhead** on RTX 3090 (24GB with DFlash vs 18.6GB without), making it impractical on consumer hardware.

**Root cause:** `need_n_rs_seq()` at `common/common.h:417` includes `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`, causing the recurrent memory layer to allocate `mem_size * (1 + n_rs_seq)` rows for RS (recurrent state) snapshots. For Qwen3.6-27B with `n_rs_seq=8`, that's 5,386 MiB.

**VRAM math (verified from test logs):**
- Qwen3.6: 48 recurrent layers, n_embd_r=30720, n_embd_s=786432
- n_rows = mem_size * (1 + n_rs_seq) = 4 * 9 = 36
- R: 30720 * 36 * 4B * 48 layers = 202.5 MiB
- S: 786432 * 36 * 4B * 48 layers = 5,184 MiB
- Total: 5,386.5 MiB (matches log output exactly)

---

## Research Phase 1: Side-by-Side Comparison (COMPLETE)

**Documents:** `plans/dflash-comparison/` (3 files)

### Key Findings

| Aspect | Old BeeLlama v0.3.2 | Current Upstream v0.4.0+ |
|--------|---------------------|-------------------------|
| `need_n_rs_seq()` | DFlash **excluded** (only MTP) | DFlash **included** (MTP, EAGLE3, DFlash) |
| Primary rollback | Backup cells (recurrent-only copy to backup sequences) | RS snapshots (inline per-token snapshots) |
| Fallback rollback | Checkpoints | Checkpoints (when rollback > n_rs_seq) |
| `llama_dflash_rollback()` | Existed (3-phase: KV cleanup + recurrent restore + tape replay) | **Removed** |
| `dflash_backup_recurrent_state()` | Existed | **Removed** |
| `tape_replay()` | Existed (DeltaNet replay for accepted tokens) | **Removed** |
| DFlash `accept()` | No-op | Still no-op |
| VRAM overhead | ~0 (backup cells deferred, ~150MB/slot) | ~5.4GB (RS buffer) |

### Critical Question Answer

**Does n_rs_seq=0 reproduce old behavior? NO.**

With n_rs_seq=0, current DFlash falls back to `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` (checkpoint-only rollback). Old used backup cells as PRIMARY mechanism. The backup cell code was permanently removed, not bypassed. Tape replay was removed without equivalent.

### Git History

- `d1b34251b` (June 28 2026) — Upstream added DFlash to `need_n_rs_seq()`
- `c9e746733` (July 10 2026) — BeeLlama v0.4.0 rebase removed ALL custom DFlash code in one commit
- Removed symbols: `llama_dflash_rollback`, `tape_replay`, `dflash_backup_recurrent_state`, `server_dflash_recurrent_rollback_plan`, `has_recurrent_only_backup`, `seq_backup`, `tree_bufs`

---

## Research Phase 2: Two Solutions Investigated (COMPLETE)

**Documents:** `plans/dflash-solutions/` (4 files)

### Solution 1: Re-implement Old Custom DFlash

- **Scope:** ~3,376 lines across 12+ files
- **Reuse:** Only 2 of 10 components directly reusable; 6 must be rewritten
- **Hardest part:** GPU tape replay (~1,100 lines of CUDA-specific DeltaNet replay)
- **VRAM savings:** ~5.2GB (150MB/slot backup cells)
- **Rollback speed:** Medium (copy + tape replay)
- **Risk:** High (untested GPU code, CUDA API changes, memory layer modifications)
- **Maintainability:** High burden (fork drift from upstream)

### Solution 2: Modify Upstream DFlash

- **Scope:** 1 line (remove DFlash from `need_n_rs_seq()`) or ~30 lines (extended with reduced n_rs_seq)
- **Downstream:** All 46 code paths handle n_rs_seq=0 correctly (conditional logic, no assertions)
- **Rollback:** Falls back to checkpoint serialize/restore (COMMON_CONTEXT_SEQ_RM_TYPE_FULL)
- **VRAM savings:** ~4.8GB (0 extra RS buffer, ~598MB active state only)
- **Rollback speed:** Slow (checkpoint vs RS pointer swap)
- **Risk:** Low (existing upstream paths, verified across 46 code paths)
- **Maintainability:** Minimal (no fork-specific code)

### Recommendation

**Solution 2 recommended.** Achieves primary goal (eliminating 5.4GB VRAM overhead) with a single-line change, guaranteed correctness, and zero maintainability burden. Solution 1 only warranted if benchmarking shows checkpoint rollback is prohibitively slow.

---

## All Documents

| File | Purpose |
|------|---------|
| `plans/dflash-vram-investigation-context.md` | Original problem statement, VRAM math, origin story |
| `plans/dflash-comparison/old-dflash-trace.md` | Old DFlash source trace (need_n_rs_seq, accept, rollback, backup cells) |
| `plans/dflash-comparison/current-dflash-trace.md` | Current DFlash source trace (RS snapshots, unified rollback) |
| `plans/dflash-comparison/final-comparison.md` | Side-by-side comparison (Parts 3-7) |
| `plans/dflash-solutions/investigation-status.md` | Solutions tracking doc with summary |
| `plans/dflash-solutions/solution1-old-dflash-restore.md` | Solution 1 detailed feasibility study |
| `plans/dflash-solutions/solution2-upstream-modify.md` | Solution 2 detailed feasibility study |
| `plans/dflash-solutions/final-solution-comparison.md` | 7-criteria comparison with recommendation |
| `plans/dflash-solutions/research-summary.md` | This file — condensed summary for context recovery |

---

## Research Phase 3: Hybrid Approach Investigation (COMPLETE)

**Documents:** `plans/dflash-solutions/` (5 additional files: task3-*)

### Question

Is there a **middle-ground hybrid approach** that preserves the modern upstream DFlash while selectively reproducing only the important parts of the old VRAM-efficient AND performant rollback strategy?

### Key Findings

**What made the old implementation both VRAM-efficient AND performant:**
1. **Excluded DFlash from `need_n_rs_seq()`** — eliminated 5.4GB RS buffer (KEY insight, directly reusable)
2. **Backup cells** — ~150MB/slot recurrent-only copy (NOT reusable — requires ~550 lines to re-add)
3. **Tape replay** — DeltaNet forward pass replay after rollback (NOT reusable — requires ~1,800+ lines of infrastructure)

**Most promising hybrid: Combination A — reduce `n_rs_seq` to 1-2 for DFlash**
- ~10 lines in 1 file ([`common/common.h:417`](common/common.h:417))
- 67-78% VRAM savings (RS buffer: 5.4GB → 1.2-1.8GB)
- 80-90% of cycles get fast RS rollback (small rollbacks are common for DFlash)
- 10-20% of cycles use checkpoint fallback (slow but rare)
- Zero new infrastructure — existing [`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20) handles the decision

### VRAM Comparison

| Approach | VRAM Overhead | Savings vs Current |
|----------|--------------|-------------------|
| Current (n_rs_seq=8) | ~5.4 GB | — |
| Hybrid (n_rs_seq=2) | ~1.8 GB | 67% |
| Hybrid (n_rs_seq=1) | ~1.2 GB | 78% |
| n_rs_seq=0 (Solution 2) | ~0.6 GB | 89% |
| Old implementation | ~0.25-0.35 GB | 94% |

### Performance Tradeoff

The hybrid approach trades checkpoint rollback speed (rare, 10-20% of cycles) for 67-78% VRAM savings. DFlash typically accepts 10-14 of 15 draft tokens, meaning small rollbacks (≤2 tokens) hit the fast RS path most of the time.

### Recommendation

**Strongly recommended.** The hybrid approach achieves the primary goal (making DFlash practical on consumer hardware like RTX 3090) with minimal code change (~10 lines), zero new infrastructure, and upstream-compatible architecture. Benchmark checkpoint rollback overhead to validate the performance tradeoff before shipping.

### All Task 3 Documents

| File | Purpose |
|------|---------|
| `plans/dflash-solutions/task3-hybrid-investigation.md` | Task 3 tracking document |
| `plans/dflash-solutions/task3-part1-backup-cell-analysis.md` | Old backup cell mechanism analysis |
| `plans/dflash-solutions/task3-part2-current-rollback-analysis.md` | Current rollback extension points |
| `plans/dflash-solutions/task3-part3-tape-replay-analysis.md` | Tape replay adaptability (NOT adaptable) |
| `plans/dflash-solutions/task3-part4-hybrid-combinations.md` | 4 hybrid combinations evaluated |
| `plans/dflash-solutions/task3-final-assessment.md` | Final assessment with recommendation |

---

## All Documents (Complete List)

| File | Purpose |
|------|---------|
| `plans/dflash-vram-investigation-context.md` | Original problem statement, VRAM math, origin story |
| `plans/dflash-comparison/old-dflash-trace.md` | Old DFlash source trace (need_n_rs_seq, accept, rollback, backup cells) |
| `plans/dflash-comparison/current-dflash-trace.md` | Current DFlash source trace (RS snapshots, unified rollback) |
| `plans/dflash-comparison/final-comparison.md` | Side-by-side comparison (Parts 3-7) |
| `plans/dflash-solutions/investigation-status.md` | Solutions tracking doc with summary |
| `plans/dflash-solutions/solution1-old-dflash-restore.md` | Solution 1 detailed feasibility study |
| `plans/dflash-solutions/solution2-upstream-modify.md` | Solution 2 detailed feasibility study |
| `plans/dflash-solutions/final-solution-comparison.md` | 7-criteria comparison with recommendation |
| `plans/dflash-solutions/task3-hybrid-investigation.md` | Task 3 tracking document |
| `plans/dflash-solutions/task3-part1-backup-cell-analysis.md` | Old backup cell mechanism analysis |
| `plans/dflash-solutions/task3-part2-current-rollback-analysis.md` | Current rollback extension points |
| `plans/dflash-solutions/task3-part3-tape-replay-analysis.md` | Tape replay adaptability (NOT adaptable) |
| `plans/dflash-solutions/task3-part4-hybrid-combinations.md` | 4 hybrid combinations evaluated |
| `plans/dflash-solutions/task3-final-assessment.md` | Task 3 final assessment with recommendation |
| `plans/dflash-solutions/research-summary.md` | This file — condensed summary for context recovery |

---

## Research Phase 4: Recurrent Backup/Restore Feasibility (COMPLETE)

**Documents:** `plans/dflash-solutions/` (6 additional files: task4-*)

### Question

> Can we take current upstream DFlash, eliminate its enormous RS snapshot allocation by setting DFlash `n_rs_seq=0`, give each DFlash slot a separate recurrent-only backup cell, and restore the accepted recurrent state after verification using the smallest amount of existing or newly reintroduced machinery possible — while preserving most of DFlash's speculative-decoding performance?

### Key Findings (12 subtask answers A-J)

| Q | Answer | Detail |
|---|--------|--------|
| A. Can DFlash use n_rs_seq=0 without checkpoint serialization? | **YES** | Server-level override prevents routine checkpoint |
| B. Can recurrent memory provide backup cell without dynamic expansion? | **YES** | Static `n_parallel × 2` cells at construction |
| C. Can recurrent-only backup use existing APIs? | **NO** | New `cell_copy()` API required (existing `seq_cp()` copies metadata only) |
| D. How is accepted recurrent state reconstructed? | **Re-decode** | K accepted tokens batched with next draft cycle |
| E. Is old tape replay required? | **NO** | Not required for linear DFlash |
| F. Minimum tape subset? | **N/A** | Tape replay not required |
| G. Linear DFlash VRAM-efficient without DDTree? | **YES** | DDTree separate concern |
| H. Smallest realistic patch? | **~120 lines, 5 files** | MUST CHANGE: `common.h`, `llama-memory-recurrent.cpp/h`, `server-context.cpp`, `llama-model.cpp` |
| I. VRAM savings? | **~4.7 GB (87%)** | 5.4GB → ~0.7GB (backup cells ~100MB + base RS) |
| J. Performance penalty? | **K tokens re-verified/cycle** | Batched with new draft, minimal overhead |

### Old vs New mechanism comparison

Only **2 of 10** old mechanisms survive in the current approach:
- **Survives:** Extra backup cells (static allocation), recurrent-only backup copy (new API)
- **Dropped:** Deferred expansion, async copy, `dflash_rollback()`, tape replay, DDTree tape buffers, `dflash_prepare_branch()`, full KV backup

### Recommendation

**Recommended for implementation.** The patch is small (~120 lines), achieves 87% VRAM savings, and preserves DFlash performance through batched re-verify. The main risk is that K-token re-verify adds latency per speculative cycle — benchmark to validate.

### All Task 4 Documents

| File | Purpose |
|------|---------|
| `plans/dflash-solutions/task4-backup-restore-feasibility.md` | Task 4 tracking document |
| `plans/dflash-solutions/task4-part1-lifecycle-and-apis.md` | Current DFlash lifecycle + API analysis |
| `plans/dflash-solutions/task4-part2-allocation-and-rollback.md` | Backup cell allocation + rollback extension |
| `plans/dflash-solutions/task4-part3-accepted-state-recovery.md` | Accepted state recovery (critical question) |
| `plans/dflash-solutions/task4-part4-checkpoint-and-assumptions.md` | Checkpoint machinery + n_rs_seq=0 audit |
| `plans/dflash-solutions/task4-part5-final-verdict.md` | Final verdict (Sections 10-13) |

---

## All Documents (Complete List)

| File | Purpose |
|------|---------|
| `plans/dflash-vram-investigation-context.md` | Original problem statement, VRAM math, origin story |
| `plans/dflash-comparison/old-dflash-trace.md` | Old DFlash source trace |
| `plans/dflash-comparison/current-dflash-trace.md` | Current DFlash source trace |
| `plans/dflash-comparison/final-comparison.md` | Side-by-side comparison (Parts 3-7) |
| `plans/dflash-solutions/investigation-status.md` | Solutions tracking doc |
| `plans/dflash-solutions/solution1-old-dflash-restore.md` | Solution 1 feasibility |
| `plans/dflash-solutions/solution2-upstream-modify.md` | Solution 2 feasibility |
| `plans/dflash-solutions/final-solution-comparison.md` | 7-criteria comparison |
| `plans/dflash-solutions/task3-hybrid-investigation.md` | Task 3 tracking |
| `plans/dflash-solutions/task3-part1-backup-cell-analysis.md` | Old backup cell analysis |
| `plans/dflash-solutions/task3-part2-current-rollback-analysis.md` | Current rollback extension points |
| `plans/dflash-solutions/task3-part3-tape-replay-analysis.md` | Tape replay adaptability |
| `plans/dflash-solutions/task3-part4-hybrid-combinations.md` | 4 hybrid combinations |
| `plans/dflash-solutions/task3-final-assessment.md` | Task 3 final assessment |
| `plans/dflash-solutions/task4-backup-restore-feasibility.md` | Task 4 tracking |
| `plans/dflash-solutions/task4-part1-lifecycle-and-apis.md` | Current lifecycle + APIs |
| `plans/dflash-solutions/task4-part2-allocation-and-rollback.md` | Backup cell allocation |
| `plans/dflash-solutions/task4-part3-accepted-state-recovery.md` | Accepted state recovery |
| `plans/dflash-solutions/task4-part4-checkpoint-and-assumptions.md` | Checkpoint + n_rs_seq=0 audit |
| `plans/dflash-solutions/task4-part5-final-verdict.md` | Task 4 final verdict |
| `plans/dflash-solutions/research-summary.md` | This file — condensed summary |

---

## Research Phase 5: Minimal Linear-DFlash Recurrent-State Replay / Tape-Replay Extraction (COMPLETE)

**Documents:** `plans/dflash-solutions/` (8 additional files: task5-*)

### Question

> Can the performance-critical mathematical operation from old tape replay be extracted and implemented in a minimal, current-upstream-compatible form for ordinary linear DFlash — combining Task 4's low-VRAM backup cells with near-old-DFlash rollback performance?

### Desired End State

>```
current upstream DFlash
      +
Task 4-style low-VRAM backup cells
      +
minimal recurrent-state replay
      =
low VRAM AND near-old-DFlash rollback performance
```

### Key Findings

**What old tape replay actually did (Part 1):**
- Captured 5 intermediates per recurrent layer per token: K, V, gate, beta, qkv_mixed
- Replay was recurrent-only (GDN update), NOT full forward pass
- GDN math: `S_new = g*S + k⊗delta` (rank-1 matrix update)
- Exactly `n_accepted` sequential state transitions per layer
- Partial acceptance: rollback + replay K tokens. Full acceptance: skip replay entirely
- Core replay is fundamentally linear; tree support (DDTree, branching) was orthogonal

**Mathematical minimum (Parts 2-3):**
- Only 4 inputs needed: k, v, gate, beta (Q not needed — passed as zeros in old code)
- All 4 exist as named ggml graph nodes: `k_in-{il}`, `v_in-{il}`, `g_in-{il}`, `b_in-{il}`
- The gap is CAPTURE, not COMPUTATION — values exist during graph execution but are not persisted
- GDN math is IDENTICAL between old and current upstream
- Tape capture overhead: ~19 MB for 8 draft tokens (48 layers × 8 tokens × 12.4 KB/token/layer)

**Capture points (Parts 4-5):**
- Intermediates produced at [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) and [`src/models/qwen35.cpp:338-450`](src/models/qwen35.cpp:338)
- Beta is post-sigmoid in current graph (applied at [`qwen35.cpp:365`](src/models/qwen35.cpp:365))
- Linear-only replay is feasible — current upstream DFlash generates linear chains (no tree speculation)
- 52% of old tape code (~800 lines) is tree-specific and can be discarded

**ggml primitives and integration (Parts 6-7):**
- **All replay operations use existing ggml primitives** — no new CUDA kernel, no new ggml op
- Core replay: single `ggml_gated_delta_net()` call with captured tape data and backup state
- GDN kernel at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) natively processes tokens sequentially
- Integrates cleanly with backup-cell design: restore backup → replay K tokens → remove rejected KV
- Recurrent and attention KV managed independently through hybrid memory

**Performance comparison (Part 8):**

| Metric | Task 4 (re-decode) | Minimal Replay | Old DFlash |
|--------|--------------------|----------------|------------|
| Full model evaluations | 2 (verify + re-decode) | 1 (verify only) | 1 (verify only) |
| Recurrent GDN evaluations | 0 | 48 (one per layer) | 48 (one per layer) |
| Memory traffic (K=8) | ~540 GB | ~320 MB | ~322 MB |
| Relative speed | 1.0× (slowest) | ~50-200× faster | ~60-250× faster |

**VRAM cost (Part 9):**

| Approach | Additional Overhead | Savings vs Current |
|----------|-------------------|--------------------|
| Current upstream (n_rs_seq=8) | ~4.8 GB | — |
| Task 4 (backup + re-decode) | ~150 MB | 97% |
| Minimal replay (proposed) | ~186 MB | 96% |
| Old DFlash (0.3.2) | ~186 MB | 96% |

### Old Tape Reduction (Part 10)

- 28 old components classified: 7 REQUIRED (direct) + 5 REQUIRED (simplified) + 8 REPLACEABLE + 13 UNNECESSARY
- 52% of old tape code is tree-specific (DDTree, branching, deferred expansion) — discarded
- Remaining ~750 lines of required functionality maps to ~650 lines of new code

### Smallest Realistic Implementation (Part 11)

- ~650 lines across 8 files (2 new + 6 modified)
- No new CUDA kernels, no new ggml operations, no new backend code
- Opt-in via `--dflash-replay` flag with safe fallback to Task 4 re-decode

### Final Verdict: Classification A — Strongly Viable (Part 12)

> **Yes, there is a genuine middle ground** between reverting to old 0.3.2 DFlash and accepting Task 4 re-decode.

> **If you were maintaining this fork and wanted the old 0.3.2 DFlash's combination of low VRAM usage and good performance while retaining current upstream llama.cpp, would you implement the proposed minimal replay design?**

> **YES.** It captures the essential insight of old DFlash (backup cells + GDN-only replay) while discarding unnecessary infrastructure (DDTree, direct kernel calls, old scheduler), resulting in a small (~650 lines), maintainable, upstream-compatible feature that achieves 96% VRAM savings and 90-95% of old DFlash performance.

### All Task 5 Documents

| File | Purpose |
|------|---------|
| `plans/dflash-solutions/task5-replay-extraction.md` | Task 5 tracking document |
| `plans/dflash-solutions/task5-part1-old-tape-mechanics.md` | Old tape mechanics (Part 1) |
| `plans/dflash-solutions/task5-part2-current-delta-replay.md` | Mathematical minimum + DeltaNet comparison (Parts 2-3) |
| `plans/dflash-solutions/task5-part4-capture-integration.md` | Capture points in current upstream (Part 4) |
| `plans/dflash-solutions/task5-part5-linear-only-feasibility.md` | Linear-only feasibility (Part 5) |
| `plans/dflash-solutions/task5-part6-ggml-and-integration.md` | ggml primitives + Task 4 integration (Parts 6-7) |
| `plans/dflash-solutions/task5-part7-cost-and-performance.md` | Performance + VRAM cost (Parts 8-9) |
| `plans/dflash-solutions/task5-part8-final-verdict.md` | Tape reduction, implementation, verdict (Parts 10-12) |

---

## All Documents (Complete List)

| File | Purpose |
|------|---------|
| `plans/dflash-vram-investigation-context.md` | Original problem statement, VRAM math, origin story |
| `plans/dflash-comparison/old-dflash-trace.md` | Old DFlash source trace |
| `plans/dflash-comparison/current-dflash-trace.md` | Current DFlash source trace |
| `plans/dflash-comparison/final-comparison.md` | Side-by-side comparison (Parts 3-7) |
| `plans/dflash-solutions/investigation-status.md` | Solutions tracking doc |
| `plans/dflash-solutions/solution1-old-dflash-restore.md` | Solution 1 feasibility |
| `plans/dflash-solutions/solution2-upstream-modify.md` | Solution 2 feasibility |
| `plans/dflash-solutions/final-solution-comparison.md` | 7-criteria comparison |
| `plans/dflash-solutions/task3-hybrid-investigation.md` | Task 3 tracking |
| `plans/dflash-solutions/task3-part1-backup-cell-analysis.md` | Old backup cell analysis |
| `plans/dflash-solutions/task3-part2-current-rollback-analysis.md` | Current rollback extension points |
| `plans/dflash-solutions/task3-part3-tape-replay-analysis.md` | Tape replay adaptability |
| `plans/dflash-solutions/task3-part4-hybrid-combinations.md` | 4 hybrid combinations |
| `plans/dflash-solutions/task3-final-assessment.md` | Task 3 final assessment |
| `plans/dflash-solutions/task4-backup-restore-feasibility.md` | Task 4 tracking |
| `plans/dflash-solutions/task4-part1-lifecycle-and-apis.md` | Current lifecycle + APIs |
| `plans/dflash-solutions/task4-part2-allocation-and-rollback.md` | Backup cell allocation |
| `plans/dflash-solutions/task4-part3-accepted-state-recovery.md` | Accepted state recovery |
| `plans/dflash-solutions/task4-part4-checkpoint-and-assumptions.md` | Checkpoint + n_rs_seq=0 audit |
| `plans/dflash-solutions/task4-part5-final-verdict.md` | Task 4 final verdict |
| `plans/dflash-solutions/task5-replay-extraction.md` | Task 5 tracking |
| `plans/dflash-solutions/task5-part1-old-tape-mechanics.md` | Old tape mechanics |
| `plans/dflash-solutions/task5-part2-current-delta-replay.md` | Mathematical minimum + compatibility |
| `plans/dflash-solutions/task5-part4-capture-integration.md` | Capture points |
| `plans/dflash-solutions/task5-part5-linear-only-feasibility.md` | Linear-only feasibility |
| `plans/dflash-solutions/task5-part6-ggml-and-integration.md` | ggml primitives + integration |
| `plans/dflash-solutions/task5-part7-cost-and-performance.md` | Performance + VRAM cost |
| `plans/dflash-solutions/task5-part8-final-verdict.md` | Task 5 final verdict |
| `plans/dflash-solutions/research-summary.md` | This file — condensed summary |

---

## What Comes Next

All five research phases are complete. The user may provide additional research tasks or proceed to implementation.