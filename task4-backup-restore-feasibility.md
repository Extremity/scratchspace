# Research Task 4 — Current DFlash Recurrent-State Backup/Restore Feasibility

**Started:** 2026-08-07
**Completed:** 2026-08-07

## Objective

Determine the **smallest viable implementation** that would allow current DFlash to operate without DFlash RS snapshot rows (`n_rs_seq=0`), using a separate recurrent-only backup cell for rollback.

## Primary Question

> Can we take current upstream DFlash, eliminate its enormous RS snapshot allocation by setting DFlash `n_rs_seq=0`, give each DFlash slot a separate recurrent-only backup cell, and restore the accepted recurrent state after verification using the smallest amount of existing or newly reintroduced machinery possible — while preserving most of DFlash's speculative-decoding performance?

## Subtasks

| ID | Sections | Status | Key Findings |
|----|----------|--------|--------------|
| 4.1 | 1-2: Current lifecycle + existing recurrent backup APIs | **Complete** | Recurrent-only backup NOT possible with existing APIs. `seq_cp()` is reference-only, `state_write/read()` includes attention KV. New `cell_copy()` API required. |
| 4.2 | 3-4: Backup cell allocation + rollback extension point | **Complete** | Static extra cells viable — double `mem_size` at construction. Rollback extension at `server-context.cpp:4225`. No dynamic expansion needed. |
| 4.3 | 5-7: Accepted recurrent state recovery (critical question) | **Complete** | Option A-refined recommended: use existing `n_rs_seq` snapshots to select SK. Re-decode (Option B) is current behavior. Tape replay (Option C) not required for linear DFlash. |
| 4.4 | 8-9: Checkpoint machinery + n_rs_seq=0 assumptions | **Complete** | NO DFlash code assumes `n_rs_seq > 0`. Server-level override recommended to prevent routine checkpoint serialization. Arch whitelist may clamp `n_rs_seq` to 0. |
| 4.5 | 10-12: Minimal patch architecture + final verdict | **Complete** | Five files, ~120 lines. `n_rs_seq=0` + static backup cells + `cell_copy()` + server override. ~87% VRAM savings (~4.7 GB). Re-decode overhead accepted. |

## Background from Task 3

- Old implementation: `n_rs_seq=0` + backup cells (~150MB/slot) + tape replay = ~250-350MB total
- Current implementation: `n_rs_seq=8` = 5.4GB RS buffer
- Task 3 recommended: reduce `n_rs_seq` to 1-2 (67-78% savings, checkpoint fallback)
- Task 4 goes further: `n_rs_seq=0` + recurrent backup cell (closer to old behavior)
- Key concern: with `n_rs_seq=0`, `draft.size() > n_rs_seq` triggers checkpoint every cycle
- Desired: avoid routine checkpoint serialization while achieving ~5GB savings

## Deliverables

- Complete trace of current DFlash recurrent-state lifecycle
- Feasibility assessment for recurrent-only backup using existing APIs
- Backup cell allocation strategy (static vs dynamic)
- Rollback extension point identification
- Accepted recurrent state recovery options (tape replay vs re-decode vs existing mechanism)
- Minimal patch architecture with MUST/LIKELY/OPTIONAL/NOT NEEDED categorization
- Final verdict on all 12 questions (A through J)

## Output Documents

| Document | Sections | Description |
|----------|----------|-------------|
| [`task4-part1-lifecycle-and-apis.md`](task4-part1-lifecycle-and-apis.md) | 1-2 | Current DFlash recurrent-state lifecycle trace + backup/restore API inventory |
| [`task4-part2-allocation-and-rollback.md`](task4-part2-allocation-and-rollback.md) | 3-4 | Backup cell allocation strategy + rollback extension points |
| [`task4-part3-accepted-state-recovery.md`](task4-part3-accepted-state-recovery.md) | 5-7 | Accepted recurrent state recovery options + DDTree analysis |
| [`task4-part4-checkpoint-and-assumptions.md`](task4-part4-checkpoint-and-assumptions.md) | 8-9 | Checkpoint machinery interaction + n_rs_seq=0 assumption audit |
| [`task4-part5-final-verdict.md`](task4-part5-final-verdict.md) | 10-13 | Minimal patch architecture + old implementation comparison + final verdict + risk assessment |

## Executive Summary

**Answer to primary question: YES, with trade-offs.**

The minimal patch requires:
- **5 files modified**, ~120 lines of new code
- New `cell_copy()` API for recurrent-only backup/restore
- Static backup cell allocation (`n_parallel * 2` cells at construction)
- Server-level override to prevent routine checkpoint serialization
- Re-decode of accepted tokens in next cycle (compute-for-VRAM trade-off)

**VRAM impact:**
- Current: ~5.4 GB R/S tensor (`n_rs_seq = 14`)
- Patched: ~0.7 GB (`n_rs_seq = 0` + 2× backup cells)
- **Net savings: ~4.7 GB (87% reduction)**

**Performance impact:**
- Re-decode overhead: K tokens re-verified per speculative cycle
- For typical acceptance rates (70-80%), 10-12 tokens re-verified per cycle
- Batched with new draft verification, so some compute overlap
- Benchmark required to determine if overhead is acceptable
