# Task 6R: `n_backup_cells` P0 Fix — Implementation Guidance

**Date:** 2026-08-10
**Priority:** P0 (CRITICAL — replay cannot function without this fix)
**Target:** Architect/Code mode implementation

---

## Problem Statement

When `--beefix-dflash-custom` is enabled, `n_backup_cells` is never set to a non-zero value. The field defaults to `0` in [`llama_context_default_params()`](src/llama-context.cpp:3941) and no code path assigns it a non-zero value. As a result:

1. **No backup rows are allocated** — the recurrent memory allocation formula `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` produces `n_rows = mem_size` with zero backup rows.
2. **`dflash_custom_backup()` returns early** — guard at [`server-dflash-custom.cpp:254`](common/server-dflash-custom.cpp:254): `mem->n_backup_cells < n_cells` is always true.
3. **`dflash_custom_replay()` returns `false`** — guard at [`server-dflash-custom.cpp:345`](common/server-dflash-custom.cpp:345): `n_cells == 0` check fails.
4. **Server falls back to checkpoint rollback** — correct but slower, defeating the purpose of custom DFlash.

---

## Complete Data Flow Trace

```
--beefix-dflash-custom (CLI flag)
      │
      ▼
[common/arg.cpp:4384] params.speculative.beefix_dflash_custom = true
      │
      ▼
[common/common.cpp:1770] cparams.n_rs_seq = params.speculative.need_n_rs_seq()
                          // Sets to draft.n_max (e.g., 8) for DFlash
      │
      ▼
[common/common.cpp:1775-1781] if (beefix_dflash_custom && has_dflash) {
                                  cparams.n_rs_seq = 0;  // ← EXISTS
                                  // ← n_backup_cells NOT SET HERE (THE BUG)
                                }
      │
      ▼ (n_backup_cells remains at default 0)
      │
      ▼
[src/llama-context.cpp:271] cparams.n_backup_cells = params.n_backup_cells;
                          // Passes through 0
      │
      ▼
[src/llama-model.cpp:2109] new llama_memory_recurrent(..., cparams.n_backup_cells, ...)
                          // Passes 0 to constructor
      │
      ▼
[src/llama-memory-recurrent.cpp:37] this->n_backup_cells = n_backup_cells;
                          // Stores 0
      │
      ▼
[src/llama-memory-recurrent.cpp:104] n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells
                          // n_rows = mem_size + 0 = mem_size (no backup rows)
```

---

## Fix Location

### Primary Change: [`common/common.cpp:1775-1781`](common/common.cpp:1775)

**Current code:**
```cpp
// Lines 1775-1781 (current)
if (params.speculative.beefix_dflash_custom) {
    bool has_dflash = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH; });
    if (has_dflash) {
        cparams.n_rs_seq = 0;
    }
}
```

**Required change:**

Add `cparams.n_backup_cells = params.n_parallel;` inside the `if (has_dflash)` block, immediately after `cparams.n_rs_seq = 0;`:

```cpp
if (params.speculative.beefix_dflash_custom) {
    bool has_dflash = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH; });
    if (has_dflash) {
        cparams.n_rs_seq = 0;
        cparams.n_backup_cells = params.n_parallel;
    }
}
```

**Rationale for `params.n_parallel`:**

- `params.n_parallel` is defined at [`common/common.h:518`](common/common.h:518) as "number of parallel sequences to decode" with a default of `1`.
- Each parallel slot needs one backup cell for pre-draft backup and post-rollback restore.
- Old v0.3.2 used `n_parallel_user` backup cells (one per slot), proven sufficient. See [`old-versions/.../server-context.cpp:5108`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108): `seq_backup = slot.id + n_parallel_user`.
- The audit correction at [`task6r-correction-part2-backup-cells.md`](plans/dflash-solutions/task6r-correction-part2-backup-cells.md) confirmed `n_parallel` cells (not `2 × n_parallel`) is the correct value.

---

## Verification Points

After the fix, verify the following chain executes correctly:

### 1. Value Propagation

| Step | File:Line | Check |
|------|-----------|-------|----------|
| CLI sets flag | [`common/arg.cpp:4384`](common/arg.cpp:4384) | `beefix_dflash_custom = true` | ✅ Already correct |
| Value assigned | [`common/common.cpp:1780`](common/common.cpp:1780) | `cparams.n_backup_cells = params.n_parallel` | **NEW — add this line** |
| Value passed through | [`src/llama-context.cpp:271`](src/llama-context.cpp:271) | `cparams.n_backup_cells = params.n_backup_cells` | ✅ Already correct (passes non-zero through) |
| Arch guard | [`src/llama-context.cpp:272-276`](src/llama-context.cpp:272) | Only disables if arch doesn't support RS rollback | ✅ Qwen3.5 supports RS rollback |
| Constructor receives | [`src/llama-model.cpp:2109`](src/llama-model.cpp:2109) | `cparams.n_backup_cells` passed to `llama_memory_recurrent` | ✅ Already correct |
| Constructor stores | [`src/llama-memory-recurrent.cpp:37`](src/llama-memory-recurrent.cpp:37) | `this->n_backup_cells = n_backup_cells` | ✅ Already correct |

### 2. Allocation Impact

With `n_parallel = 4` (default), `n_rs_seq = 0`, and `mem_size = 4` (equals `n_seq_max = n_parallel`):

| Before Fix | After Fix |
|------------|-----------|
| `n_rows = 4 * (1 + 0) + 0 = 4` | `n_rows = 4 * (1 + 0) + 4 = 8` |
| No backup rows | 4 backup rows |

For Qwen3.6-27B specifically:
- S per row: `n_embd_s × n_layers = 786432 × 48 = 37,748,736` elements ≈ 149.9 MB F32
- R per row: `n_embd_r × n_layers = 30720 × 48 = 1,474,560` elements ≈ 5.9 MB F32
- **Per backup cell: ~156 MB**
- **Total for 4 backup cells: ~623 MB**

This is logged at [`src/llama-memory-recurrent.cpp:134-138`](src/llama-memory-recurrent.cpp:134):
```cpp
if (n_backup_cells > 0) {
    const size_t backup_size = n_backup_cells * n_layer * (hparams.n_embd_r() + hparams.n_embd_s()) * sizeof(float);
    LLAMA_LOG_INFO("%s: backup cells = %6u rows, %7.2f MiB\n", __func__, n_backup_cells, backup_size / (1024.0f * 1024.0f));
}
```

### 3. Runtime Behavior

After the fix, these guards should pass:

| Guard | File:Line | Condition | Before Fix | After Fix |
|-------|-----------|-----------|------------|-----------|
| Backup guard | [`server-dflash-custom.cpp:254`](common/server-dflash-custom.cpp:254) | `mem->n_backup_cells < n_cells` | Always true (0 < n) — early return | False (4 >= 4) — backup executes |
| Replay guard | [`server-dflash-custom.cpp:345`](common/server-dflash-custom.cpp:345) | `n_cells == 0` | Always true — returns false | False (4 != 0) — replay proceeds |
| Server backup call | [`server-context.cpp:3314`](tools/server/server-context.cpp:3314) | `mem->n_backup_cells > 0` | Always false — backup skipped | True — backup executed |

---

## Existing Plumbing (No Changes Needed)

The following infrastructure already exists and works correctly. The fix only needs to populate `n_backup_cells` to activate it:

| Component | File:Lines | Status |
|-----------|-----------|--------|
| Field declaration | [`include/llama.h:427`](include/llama.h:427) | ✅ Exists |
| Internal struct field | [`src/llama-cparams.h:20`](src/llama-cparams.h:20) | ✅ Exists |
| Default value | [`src/llama-context.cpp:3941`](src/llama-context.cpp:3941) | ✅ Defaults to 0 |
| Pass-through | [`src/llama-context.cpp:271`](src/llama-context.cpp:271) | ✅ Passes through |
| Arch guard | [`src/llama-context.cpp:272-276`](src/llama-context.cpp:272) | ✅ Disables if unsupported |
| Constructor parameter | [`src/llama-memory-recurrent.h:27`](src/llama-memory-recurrent.h:27) | ✅ Accepted |
| Constructor storage | [`src/llama-memory-recurrent.cpp:37`](src/llama-memory-recurrent.cpp:37) | ✅ Stored |
| Allocation formula | [`src/llama-memory-recurrent.cpp:104`](src/llama-memory-recurrent.cpp:104) | ✅ Includes `n_backup_cells` |
| Backup logging | [`src/llama-memory-recurrent.cpp:134-138`](src/llama-memory-recurrent.cpp:134) | ✅ Logs when > 0 |
| `backup_offset()` | [`src/llama-memory-recurrent.h:93-95`](src/llama-memory-recurrent.h:93) | ✅ Returns `mem_size * (1 + n_rs_seq)` |
| `cell_copy()` | [`src/llama-memory-recurrent.cpp:141-170`](src/llama-memory-recurrent.cpp:141) | ✅ Device-native copy |
| `dflash_custom_backup()` | [`server-dflash-custom.cpp:253-263`](common/server-dflash-custom.cpp:253) | ✅ Uses `n_backup_cells` guard |
| `dflash_custom_restore()` | [`server-dflash-custom.cpp:271-281`](common/server-dflash-custom.cpp:271) | ✅ Uses `n_backup_cells` guard |
| `dflash_custom_replay()` | [`server-dflash-custom.cpp:343-348`](common/server-dflash-custom.cpp:343) | ✅ Reads `n_backup_cells` from mem |
| Server backup call | [`server-context.cpp:3314-3316`](tools/server/server-context.cpp:3314) | ✅ Guards on `n_backup_cells > 0` |

---

## Old v0.3.2 Reference

The old implementation used a different mechanism (sequence-based backup) but the same principle:

- **Backup cell count:** `n_parallel_user` cells (one per slot). See [`old-versions/.../server-context.cpp:5108`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108): `seq_backup = slot.id + n_parallel_user`.
- **Backup operation:** [`dflash_backup_recurrent_state()`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788) copied R/S state from active sequence to backup sequence.
- **Restore operation:** [`dflash_rollback()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218) restored from backup sequence before tape replay.

The current Task 6R implementation uses a simpler row-based approach within the same tensor, but the cell count (`n_parallel`) is the same.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `n_parallel` not available at call site | NONE | `params` is the function parameter; `params.n_parallel` is directly accessible |
| Arch doesn't support RS rollback | LOW | Qwen3.5 returns true for `llm_arch_supports_rs_rollback()`. Guard at line 272 will warn and disable if unsupported |
| Memory pressure from backup cells | LOW | ~623 MB for Qwen3.6 with default `n_parallel=4`. This is the intended design cost |
| Stock DFlash affected | NONE | Change is inside `if (beefix_dflash_custom && has_dflash)` — only triggers with both conditions |

---

## Summary

**Single-line fix** at [`common/common.cpp:1780`](common/common.cpp:1780):

```cpp
cparams.n_backup_cells = params.n_parallel;
```

Add this line inside the existing `if (has_dflash)` block, immediately after `cparams.n_rs_seq = 0;`. This activates all the existing plumbing (backup cells, allocation, backup/restore functions, replay) that was already implemented but never triggered because `n_backup_cells` remained at its default value of 0.

---

*End of implementation guidance.*
