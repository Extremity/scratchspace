# Research Task 3 — Hybrid DFlash Rollback Investigation

**Started:** 2026-08-07

## Objective

Investigate whether a **middle-ground hybrid approach** exists that:
- Preserves the modern upstream DFlash implementation
- Selectively reproduces the mechanisms that made the old implementation **both VRAM-efficient AND performant**
- Avoids both the full ~3,376-line restoration AND the performance cost of pure checkpoint rollback

## Critical Framing

The goal is NOT just "what saves VRAM." The goal is "what let the old implementation work well WITHOUT excessive VRAM." We need to identify mechanisms that simultaneously:
1. Reduced VRAM overhead (vs 5.4GB RS buffer)
2. Contributed to performant rollback (vs slow checkpoint serialize/restore)

A mechanism that saves VRAM but destroys performance is not a good hybrid candidate. A mechanism that preserves performance but adds no VRAM savings is also not useful. We want the intersection.

## Subtasks

| ID | Topic | Status | Key Findings |
|----|-------|--------|--------------|
| 3.1 | Old backup cell mechanism analysis | Complete | n_rs_seq exclusion + backup cells = 5.4GB savings |
| 3.2 | Current checkpoint/RS rollback extension points | Complete | 5 extension points, Extension A (reduce n_rs_seq) recommended |
| 3.3 | Tape replay adaptability to current architecture | Complete | Not adaptable without rebuilding old DFlash (~1,800+ lines). Recommend n_rs_seq reduction instead. |
| 3.4 | Hybrid combinations (checkpoint + specialized backup) | Complete | Combination A (reduced n_rs_seq + checkpoint fallback) recommended. 67-78% VRAM savings, ~10 lines code. |
| 3.5 | Final assessment | Complete | Combination A strongly recommended. See [`task3-final-assessment.md`](task3-final-assessment.md) for full details. |

## Key Questions to Answer

1. What did the old implementation actually do to avoid the enormous `n_rs_seq`/RS buffer allocation?
2. Which parts of the old implementation were fundamentally responsible for low VRAM usage?
3. Which parts were supporting infrastructure that current upstream already replaces or makes unnecessary?
4. Which parts of the old implementation contributed to PERFORMANCE (fast rollback, high acceptance throughput)?
5. Could backup-cell, tape replay, or other mechanisms be adapted to current upstream in a simpler form?
6. Could some combination of checkpoint + smaller specialized backup provide most benefits?
7. What other middle-ground approaches emerge from the code comparison?

## Deliverables

- [x] Per-mechanism analysis (what it did, why it existed, do we need it, simpler alternative?)
- [x] Most promising middle-ground approach
- [x] Approximate implementation scope and affected files
- [x] Expected VRAM characteristics vs current (n_rs_seq=8), n_rs_seq=0, and old
- [x] Expected rollback/performance characteristics
- [x] Major technical risks or unknowns
- [x] Recommendation: is this worth pursuing?

## Final Output

- [`task3-final-assessment.md`](task3-final-assessment.md) — Complete final assessment covering all 7 required items.

## Final Recommendation Summary

**Combination A (reduced n_rs_seq + checkpoint fallback) is strongly recommended.** Reduce `n_rs_seq` for DFlash from 8 to 2 in [`need_n_rs_seq()`](common/common.h:417). This achieves 67% VRAM savings (5.4 GB → 1.8 GB) with ~10 lines of code change, preserves fast RS rollback for 80-90% of cycles, and falls back to checkpoint for the remaining 10-20% of cycles. No new infrastructure, APIs, or fork-specific code required.
