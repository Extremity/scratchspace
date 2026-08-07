# Research Task 5 — Minimal Linear-DFlash Recurrent-State Replay / Tape-Replay Extraction

**Started:** 2026-08-07
**Completed:** 2026-08-07
**Status:** COMPLETE

## Objective

Extract the performance-critical mathematical operation from old tape replay and implement it in a minimal, current-upstream-compatible form for ordinary linear DFlash — combining Task 4's low-VRAM backup cells with near-old-DFlash rollback performance.

**Desired end state:**
```
current upstream DFlash
      +
Task 4-style low-VRAM backup cells
      +
minimal recurrent-state replay
      =
low VRAM AND near-old-DFlash rollback performance
```

## 12-Part Investigation Structure

| Part | Topic | Status | Output Document |
|------|-------|--------|-----------------|
| 1 | Reconstruct exactly what old tape replay did | COMPLETE | task5-part1-old-tape-mechanics.md |
| 2 | Mathematical minimum required for replay | COMPLETE | task5-part2-current-delta-replay.md |
| 3 | Compare old vs current DeltaNet/GDN | COMPLETE | task5-part2-current-delta-replay.md |
| 4 | Where current upstream captures intermediates | COMPLETE | task5-part4-capture-integration.md |
| 5 | Can replay be linear-only? | COMPLETE | task5-part5-linear-only-feasibility.md |
| 6 | Are current ggml primitives sufficient? | COMPLETE | task5-part6-ggml-and-integration.md |
| 7 | Integration with Task 4 backup-cell design | COMPLETE | task5-part6-ggml-and-integration.md |
| 8 | Theoretical performance comparison | COMPLETE | task5-part7-cost-and-performance.md |
| 9 | VRAM cost of minimal replay | COMPLETE | task5-part7-cost-and-performance.md |
| 10 | Reduce old tape to small subset | COMPLETE | task5-part8-final-verdict.md |
| 11 | Smallest realistic implementation | COMPLETE | task5-part8-final-verdict.md |
| 12 | Final verdict — worth implementing? | COMPLETE | task5-part8-final-verdict.md |

## Subtask Breakdown

| Subtask | Parts Covered | Mode | Status |
|---------|--------------|------|--------|
| 5.1 | Part 1 — Old tape mechanics (source inspection) | architect | COMPLETE |
| 5.2 | Parts 2-3 — Mathematical minimum + DeltaNet/GDN comparison | architect | COMPLETE |
| 5.3 | Parts 4-5 — Capture points + linear-only feasibility | architect | COMPLETE |
| 5.4 | Parts 6-7 — ggml primitives + Task 4 integration | architect | COMPLETE |
| 5.5 | Parts 8-9 — Performance + VRAM cost analysis | architect | COMPLETE |
| 5.6 | Parts 10-12 — Tape reduction, implementation design, final verdict | architect | COMPLETE |

## Required Output Documents

| Document | Parts | Status |
|----------|-------|--------|
| task5-replay-extraction.md | This file | Created |
| task5-part1-old-tape-mechanics.md | Part 1 | Created |
| task5-part2-current-delta-replay.md | Parts 2-3 | Created |
| task5-part4-capture-integration.md | Part 4 | Created |
| task5-part5-linear-only-feasibility.md | Part 5 | Created |
| task5-part7-cost-and-performance.md | Parts 8-9 | Created |
| task5-part8-final-verdict.md | Parts 10-12 | Created |

## Key Source References

- Old 0.3.2 code: `old-versions/beellama.cpp-preview-v0.3.2/`
- Current upstream: main workspace
- Task 3 findings: `plans/dflash-solutions/task3-*.md`
- Task 4 findings: `plans/dflash-solutions/task4-*.md`

## Final Question to Answer

> **If you were maintaining this fork and wanted the old 0.3.2 DFlash's combination of low VRAM usage and good performance while retaining current upstream llama.cpp, would you implement the proposed minimal replay design? Why or why not?**
