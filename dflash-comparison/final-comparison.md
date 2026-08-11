# DFlash Implementation Comparison: BeeLlama v0.3.2 vs Current Upstream

**Date:** 2026-08-07
**Purpose:** Comprehensive side-by-side analysis of BeeLlama's custom DFlash (v0.3.2 preview) versus the current upstream DFlash implementation (post-v0.4.0 merge).

**Reference Documents:**
- [`old-dflash-trace.md`](old-dflash-trace.md) — Old BeeLlama DFlash implementation trace
- [`current-dflash-trace.md`](current-dflash-trace.md) — Current DFlash implementation trace
- [`dflash-vram-investigation-context.md`](../dflash-vram-investigation-context.md) — VRAM math and problem statement

---

## Table of Contents

| Section | Topic |
|---------|-------|
| [Part 1](#part-1--executive-summary) | Executive Summary |
| [Part 2](#part-2--git-history-timeline) | Git History Timeline |
| [Part 3](#part-3--side-by-side-comparison-table) | Side-by-Side Comparison Table |
| [Part 4](#part-4--git-history-analysis) | Git History Analysis |
| [Part 5](#part-5--critical-question) | Critical Question: Does n_rs_seq=0 Reproduce Old Behavior? |
| [Part 6](#part-6--state-type-mapping) | State Type to Mechanism Mapping |
| [Part 7](#part-7--other-differences) | Other Differences |

---

## Part 1 — Executive Summary

BeeLlama v0.3.2 used a **custom DFlash implementation** with backup cells (recurrent-only backup sequences) and a dedicated `llama_dflash_rollback()` function. The current codebase uses **upstream DFlash** with RS (recurrent state) snapshots as the primary rollback mechanism.

The fundamental architectural change:

| Aspect | BeeLlama v0.3.2 (Old) | Current (Upstream) |
|--------|----------------------|-------------------|
| Primary rollback | Backup cells (recurrent state copied to backup sequences) | RS snapshots (inline per-token snapshots bounded by `n_rs_seq`) |
| Fallback rollback | Checkpoints (full context serialize/restore) | Checkpoints (same, when rollback exceeds RS bounds) |
| `need_n_rs_seq()` | Excluded DFlash (only MTP) | **Includes DFlash** (MTP, EAGLE3, DFlash) |
| `llama_dflash_rollback()` | Existed | **Removed** |
| `dflash_backup_recurrent_state()` | Existed | **Removed** |
| DFlash `accept()` | No-op | Still no-op |
| VRAM overhead | ~0 extra (backup cells deferred) | **~5.4GB** (RS buffer for Qwen3.6) |
| Rollback unified? | No (DFlash had separate path) | **Yes** (all speculative types share upstream rollback) |

---

## Part 2 — Git History Timeline

| Date | Commit | Message | Significance |
|------|--------|---------|--------------|
| 2026-05-23 | [`80bb3c794`](80bb3c794) | `core: preserve BeeLlama speculative decoding support` | Introduced `llama_dflash_rollback()`, `tape_replay()`, backup cells. Preserved custom DFlash on cleaned history. |
| 2026-06-06 | [`84d4f1df6`](84d4f1df6) | `Fix flat DFlash recurrent rollback allocation` | Fixed backup cell sizing for flat DFlash. Last substantive fix to custom DFlash. |
| 2026-06-28 | [`d1b34251b`](d1b34251b) | `spec : add DFlash support (#22105)` | **Upstream** added `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` to `need_n_rs_seq()`. Added upstream DFlash (`src/models/dflash.cpp`). |
| 2026-07-10 | [`c9e746733`](c9e746733) | `Complete BeeLlama v0.4.0 upstream rebase` | **THE MERGE.** Removed custom DFlash (`llama_dflash_rollback`, `tape_replay`, `dflash_backup_recurrent_state`, `server_dflash_recurrent_rollback_plan`, `has_recurrent_only_backup`, `tree_bufs`, `src/models/dflash_draft.cpp`). Adopted upstream DFlash with RS buffer allocation. |
| 2026-08-04 | [`b572baab6`](b572baab6) | `Fix test runner: add --spec-draft-model for DFlash tests` | Post-merge test fix for upstream DFlash. |

---

## Part 3 — Side-by-Side Comparison Table

| Behavior | BeeLlama v0.3.2 Custom DFlash | Current Upstream DFlash |
|----------|--------------------------------|------------------------|
| **DFlash initialization** | [`common_speculative_impl_dflash`](old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp:2080) — custom class with block-diffusion draft | [`common_speculative_impl_draft_dflash`](common/speculative.cpp:906) — upstream class, same block-diffusion approach |
| **Draft context** | Separate draft model context (`ctx_dft`) loaded via `llama_load_model_from_file` | Same — draft model loaded upstream, managed by `common_speculative_load_dflash` |
| **Speculative length** | `draft.n_max` controlled by `--spec-draft-n-max`, default tied to `block_size - 1` | Same — `draft.n_max` from `--spec-draft-n-max` |
| **Target SSM state handling** | Backed up to `seq_backup` via `dflash_backup_recurrent_state()` before draft | Inline per-token snapshots in recurrent buffer (`mem_size * (1 + n_rs_seq)` rows) |
| **RS snapshots** | **NOT used.** DFlash excluded from `need_n_rs_seq()`. | **Primary mechanism.** DFlash included in [`need_n_rs_seq()`](common/common.h:417). |
| **n_rs_seq** | Returns `0` for DFlash. Only MTP triggers RS allocation. | Returns `draft.n_max` for DFlash, MTP, and EAGLE3. |
| **Checkpoint state** | Used when `seq_rm_type == FULL` or rollback exceeds RS bounds. Created via `ckpt.update_tgt()` before draft. | Same checkpoint mechanism. Created at [`server-context.cpp:3305`](tools/server/server-context.cpp:3305) when `use_ckpt_tgt == true`. |
| **Partial rejection** | [`llama_dflash_rollback()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218) — restore from `seq_backup` + tape replay | Unified path at [`server-context.cpp:4219`](tools/server/server-context.cpp:4219) — RS snapshot rollback via `seq_rm()` |
| **Full rejection** | Same `llama_dflash_rollback()` path with `n_accepted = 0` | Same unified path, `n_rollback = draft.size() + 1` |
| **Draft rollback** | Attention KV: `seq_rm()` rejected range, keep accepted (flat) or restore all (tree). Recurrent: copy from backup. | `common_sampler_sample_and_accept_n()` handles acceptance. Memory `seq_rm()` uses RS snapshots. |
| **Target rollback** | `mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id)` then `tape_replay()` | `seq_rm()` on recurrent memory restores snapshot at accepted position |
| **State restoration** | Three-phase: (1) attention KV cleanup, (2) recurrent restore from backup, (3) tape replay for accepted tokens | Two-phase: (1) RS snapshot restore via `seq_rm()`, (2) checkpoint fallback if rollback > n_rs_seq |
| **Accept() method** | [`common/speculative.cpp:3145`](old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp:3145) — no-op | [`common/speculative.cpp:1195`](common/speculative.cpp:1195) — still no-op |
| **Server-specific code** | Extensive: `server_dflash_recurrent_rollback_plan`, `dflash_backup_recurrent_state()`, `recurrent_backup_sequences`, `has_recurrent_only_backup`, `seq_backup` | Minimal: DFlash integrated into upstream unified speculative path. No DFlash-specific server code. |
| **VRAM overhead** | ~0 extra for RS buffer. Backup cells were deferred (copied on-demand before first draft). | **~5.4GB** for Qwen3.6 with `n_rs_seq=8`. Formula: `mem_size * (1 + n_rs_seq) * layers * (n_embd_r + n_embd_s) * 4`. |

---

## Part 4 — Git History Analysis

### Commits that Removed Old Code

All removal happened in a **single merge commit**:

| Commit | Date | Action |
|--------|------|--------|
| [`c9e746733`](c9e746733) | 2026-07-10 | **Complete BeeLlama v0.4.0 upstream rebase.** Removed ALL custom DFlash code in one commit: |

Specific removals verified by `git log -S` searches:

| Symbol | Last Commit Containing It | Commit That Removed It |
|--------|--------------------------|----------------------|
| `llama_dflash_rollback` | `80bb3c794` (2026-05-23, added) | `c9e746733` (2026-07-10, removed) |
| `tape_replay` | `80bb3c794` (2026-05-23, added) | `c9e746733` (2026-07-10, removed) |
| `has_recurrent_only_backup` | `ee1d1a308` (fix commit) | `c9e746733` (2026-07-10, removed) |
| `server_dflash_recurrent_rollback_plan` | `ee1d1a308` (fix commit) | `c9e746733` (2026-07-10, removed) |
| `recurrent_backup_sequences` | `84d4f1df6` (2026-06-06, fix) | `c9e746733` (2026-07-10, removed) |
| `dflash_backup_recurrent_state` | `afaffd0e9` / `fe54d13a0` | `c9e746733` (2026-07-10, removed) |

### Commits that Introduced Upstream Code

| Commit | Date | Action |
|--------|------|--------|
| [`d1b34251b`](d1b34251b) | 2026-06-28 | Upstream added `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` to `need_n_rs_seq()`. Added `src/models/dflash.cpp`. |
| [`c9e746733`](c9e746733) | 2026-07-10 | BeeLlama merge adopted upstream DFlash. Removed custom code, brought in upstream `need_n_rs_seq()` including DFlash. |

### Single Merge vs Series of Changes

This was a **single merge** for the DFlash transition. Commit `c9e746733` was a massive rebase commit (217 files changed, +7042 / -16267 lines) that:
1. Removed ALL custom BeeLlama DFlash code (backup cells, `llama_dflash_rollback`, `tape_replay`, etc.)
2. Adopted upstream DFlash with RS buffer allocation
3. Rebased KVarN and low-bit cache types onto upstream APIs

The upstream DFlash support was added separately by upstream in commit `d1b34251b` (June 28), before BeeLlama's rebase (July 10).

### Key Commit IDs

| Event | Commit ID |
|-------|-----------|
| DFlash added to `need_n_rs_seq()` | `d1b34251b` (upstream, June 28) |
| `llama_dflash_rollback()` removed | `c9e746733` (BeeLlama rebase, July 10) |
| Backup cells removed | `c9e746733` (BeeLlama rebase, July 10) |
| `tape_replay()` removed | `c9e746733` (BeeLlama rebase, July 10) |
| `has_recurrent_only_backup` removed | `c9e746733` (BeeLlama rebase, July 10) |

---

## Part 5 — Critical Question

**Question:** If we take CURRENT upstream DFlash and prevent DFlash from contributing to `need_n_rs_seq()`, resulting in `n_rs_seq = 0`, does the resulting execution path actually reproduce the OLD BeeLlama behavior?

**Answer: NO — the functional differences are significant.**

### Detailed Analysis

#### 1. With n_rs_seq=0, does current DFlash fall back to checkpoint-based rollback?

**YES — but with important caveats.**

When `n_rs_seq = 0`:

1. [`llama_memory_recurrent::get_seq_rm_capability()`](src/llama-memory-recurrent.cpp:781) returns `suffix_rollback_tokens = 0`.
2. [`common_context_can_seq_rm()`](common/common.cpp:1666) checks `suffix_rollback_tokens > 0` (line 1677). With value 0, this fails.
3. The function then checks `full_clear` (line 1681), which is `true` for recurrent memory.
4. Result: `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` is returned.

With `type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL`:

- [`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20) returns `true` for ANY proposed rollback.
- The server creates checkpoints before draft and restores from checkpoint on rejection.

This matches the OLD behavior's checkpoint fallback path, BUT the old implementation had a **third path** (backup cells) that was NOT checkpoint-based and NOT RS-based. The old implementation used backup cells as its PRIMARY mechanism and checkpoints only as a fallback. The current implementation with `n_rs_seq=0` would use checkpoints as its PRIMARY (and only) mechanism.

#### 2. With n_rs_seq=0, are backup cells restored or permanently removed?

**Permanently removed.** The backup cell code was deleted in commit `c9e746733`. There is no code path to restore it without re-implementing:

- [`dflash_backup_recurrent_state()`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788) — the backup lambda
- [`server_dflash_recurrent_rollback_plan`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.h:18) — the planning struct
- [`llama_dflash_rollback()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218) — the rollback function
- [`tape_replay()`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp) — the DeltaNet replay function
- `seq_backup` computation, `has_recurrent_only_backup` flag, `recurrent_backup_sequences` flag

Setting `n_rs_seq=0` does NOT restore this code. It simply removes the RS buffer allocation.

#### 3. With n_rs_seq=0, does `common_context_can_seq_rm()` return a different type?

**YES.** As analyzed above:

| `n_rs_seq` | `suffix_rollback_tokens` | `common_context_can_seq_rm()` returns |
|-----------|-------------------------|--------------------------------------|
| > 0 | > 0 | `COMMON_CONTEXT_SEQ_RM_TYPE_RS` |
| 0 | 0 | `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` |

The old implementation with DFlash excluded would have returned `COMMON_CONTEXT_SEQ_RM_TYPE_PART` for standard KV (because `arbitrary_ranges = true` for attention memory) or `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` for recurrent-only models. The current implementation returns `FULL` for recurrent memory when `n_rs_seq=0`.

#### 4. With n_rs_seq=0, does `server_speculative_rollback_requires_checkpoint()` behave differently?

**YES.** With `type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL`:

```cpp
return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
       (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
```

The first condition is `true`, so the function returns `true` for ALL proposed rollbacks. This means checkpoints are ALWAYS used for rollback when `n_rs_seq=0`.

#### 5. Are there other code paths that depend on n_rs_seq > 0 for DFlash?

**YES.** Several code paths check `n_rs_seq`:

- [`src/llama-context.cpp:264`](src/llama-context.cpp:264) — `cparams.n_rs_seq` validation against model arch support.
- [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) — RS tensor allocation: `n_rows = mem_size * (1 + n_rs_seq)`.
- [`src/llama-memory-recurrent.cpp:214`](src/llama-memory-recurrent.cpp:214) — Partial rollback via snapshots only works when `rollback <= n_rs_seq`.
- [`src/llama-memory-recurrent.cpp:477`](src/llama-memory-recurrent.cpp:477) — Snapshot index clamping: `rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx`.

With `n_rs_seq=0`, all snapshot-based operations become no-ops or fail, falling through to checkpoint-based paths.

#### 6. Was `llama_dflash_rollback()` simply removed, or replaced by something functionally equivalent?

**Removed without functional equivalent.** The replacement is the unified upstream rollback path at [`server-context.cpp:4219`](tools/server/server-context.cpp:4219), which uses:
- RS snapshots (when `n_rs_seq > 0` and rollback within bounds)
- Checkpoints (when `n_rs_seq = 0` or rollback exceeds bounds)

Neither of these is functionally equivalent to the old `llama_dflash_rollback()`, which had three unique phases:
1. Attention KV cleanup (flat vs tree mode distinction)
2. Recurrent state restore from backup sequence (not checkpoint)
3. Tape replay for accepted tokens (DeltaNet state replay)

The checkpoint path is a full context serialize/restore, which is fundamentally different from the old backup cell approach that only copied recurrent state.

#### 7. Was the backup cell code simply bypassed by RS snapshots, or was it actually removed?

**Actually removed.** The code was deleted, not bypassed. Search results confirm zero matches for `recurrent_backup_sequences`, `dflash_backup_recurrent_state`, `has_recurrent_only_backup`, `seq_backup`, or `llama_dflash_rollback` in the current codebase.

### Summary: Why the Answer is NO

Setting `n_rs_seq=0` in the current codebase produces:

| Aspect | Old BeeLlama (v0.3.2) | Current with n_rs_seq=0 |
|--------|----------------------|------------------------|
| Primary rollback | Backup cells (recurrent-only copy) | Checkpoints (full context serialize/restore) |
| Rollback speed | Medium (copy + tape replay) | Slow (full memory copy) |
| VRAM overhead | Low (~150MB per slot for backup) | Zero extra (no RS buffer) |
| `seq_rm_type` | `PART` (for attention) or `FULL` (for recurrent) | `FULL` (recurrent with no RS) |
| `accept()` | No-op | No-op (same) |
| Tape replay | YES — replayed DeltaNet for accepted tokens | NO — tape_replay() was removed |
| Backup sequence | `slot.id + n_parallel_user` | Does not exist |
| Checkpoint usage | Fallback only | Always used for rollback |

The critical missing piece is **tape replay**. The old implementation replayed DeltaNet state updates for accepted tokens after restoring the backup. The current implementation with `n_rs_seq=0` would use checkpoints, which restore the full context state (including SSM state) from before the draft. This is correct but fundamentally different:

- **Old approach:** Restore pre-draft recurrent state from backup, then replay accepted tokens' DeltaNet updates.
- **Current with n_rs_seq=0:** Restore entire context from checkpoint (no replay needed because checkpoint captures pre-draft state).

Both produce correct results, but the performance characteristics are different. The old approach was faster for partial acceptance (only copy recurrent state + replay a few tokens), while checkpoints are slower (full context copy) but simpler.

---

## Part 6 — State Type Mapping

| State Type | Old Save Mechanism | Old Restore Mechanism | Current Save Mechanism | Current Restore Mechanism |
|------------|-------------------|----------------------|-----------------------|--------------------------|
| **DFlash draft-model state** | Draft context `ctx_dft` with its own KV cache. Checkpoint via `ckpt.update_dft()` when `use_ckpt_dft == true`. | `ckpt.load_dft()` on rejection. `seq_rm()` on acceptance. | Same upstream checkpoint mechanism. | Same. |
| **Target-model KV cache** | Attention memory `mem_attn`. Flat mode: keep accepted KV inline. Tree mode: full restore from `seq_backup` stream. | `mem_attn->seq_rm()` + `mem_attn->seq_cp()` from backup (tree) or partial `seq_rm()` (flat). | Standard attention memory with `seq_rm()`. | `seq_rm()` removes rejected tokens. RS snapshots handle recurrent state. |
| **Target-model recurrent/SSM state** | **Backup cells.** `dflash_backup_recurrent_state(slot.id, seq_backup)` copied recurrent state to backup sequence before draft. | `mem_recr->seq_rm(seq_id)` + `mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id)` + `tape_replay(seq_id, n_accepted)`. | **RS snapshots.** Inline per-token snapshots in recurrent buffer (`mem_size * (1 + n_rs_seq)` rows). | `seq_rm()` on recurrent memory restores snapshot at accepted position. |
| **RS snapshots** | Not used for DFlash. Used for MTP only. | Not applicable for DFlash. | Primary mechanism. Maintained during forward pass. Indexed by rollback depth (1 to n_rs_seq). | `set_rs_idx()` selects snapshot. Cell position truncated to accepted position. |
| **Checkpoints** | `slot.spec_ckpt` with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`. Created when `seq_rm_type == FULL` or rollback exceeds bounds. | `ckpt.load_tgt()` + `ckpt.load_dft()` + `seq_rm()` post-restore. | Same. Created when `use_ckpt_tgt == true` (rollback > n_rs_seq or `seq_rm_type == FULL`). | Same. |
| **Speculative acceptance** | `common_speculative_accept()` called after `llama_dflash_rollback()` completed. DFlash `accept()` was no-op. | Acceptance tracked by `common_speculative_accept()` statistics. | `common_speculative_accept()` called in unified path. DFlash `accept()` still no-op. | Same. |
| **Rollback depth** | Computed as `n_rollback = draft.size() + 1 - accepted.size()`. Compared against `llama_n_rs_seq()` for checkpoint fallback. | Same computation. | Same computation at [`server-context.cpp:4219`](tools/server/server-context.cpp:4219). | Same. |

---

## Part 7 — Other Differences

### 1. Was `llama_dflash_rollback()` replaced by seq_rm() with RS snapshots, or was tape replay functionality lost?

**The tape replay functionality was lost and replaced by a fundamentally different mechanism.**

The old `llama_dflash_rollback()` had three phases:
1. Attention KV cleanup (remove rejected, keep accepted)
2. Recurrent state restore from backup sequence
3. **Tape replay** — replay DeltaNet state updates for accepted tokens

The current implementation replaces all three phases with:
- `seq_rm()` on recurrent memory (uses RS snapshots to restore to accepted position)
- OR checkpoint restore (full context restore from before draft)

The tape replay was a compute-time operation that replayed DeltaNet forward passes for accepted tokens. This was necessary because the backup cell only held the pre-draft state, and accepted tokens needed their DeltaNet updates applied.

With RS snapshots, each token position has a snapshot of the recurrent state, so rollback to the accepted position is a pointer swap (set_rs_idx). No replay is needed because the snapshot at the accepted position already contains the correct state.

With checkpoints, the full context is restored from before the draft, so no replay is needed because the checkpoint contains the pre-draft state.

**The tape replay functionality was not lost — it was made unnecessary by RS snapshots and checkpoints.**

### 2. Were backup cells functionally equivalent to RS snapshots, or fundamentally different?

**Fundamentally different.**

| Aspect | Backup Cells | RS Snapshots |
|--------|-------------|--------------|
| Storage model | Separate sequence (`seq_backup`) holding full recurrent state copy | Inline per-token snapshots in recurrent buffer |
| Allocation timing | Deferred — copied before first draft via `dflash_backup_recurrent_state()` | Pre-allocated at context creation (`mem_size * (1 + n_rs_seq)` rows) |
| VRAM cost | ~150MB per slot (one copy of recurrent state) | `n_rs_seq` copies per cell (5.4GB for Qwen3.6 with n_rs_seq=8) |
| Rollback granularity | Full context (backup holds all state) | Per-token (each snapshot is one token back) |
| Rollback mechanism | Copy from backup + tape replay | Pointer swap (set_rs_idx) |
| Scaling | O(1) per slot | O(n_rs_seq) per cell |
| GPU optimization | `llama_dflash_memory_seq_cp_recurrent_ordered()` for layer-ordered copy | Native tensor operations on pre-allocated buffer |

Backup cells were a **copy-once, replay-once** model. RS snapshots are a **pre-allocate, snapshot-per-token** model.

### 3. Did the old implementation's `tape_replay()` have any VRAM cost?

**No — tape replay was purely a compute-time operation.**

The tape replay function at `old-versions/.../src/llama-context.cpp` replayed DeltaNet forward passes for accepted tokens. It did not allocate additional VRAM. The VRAM cost of the old implementation was:
- Backup sequence: one copy of recurrent state per slot (~150MB for Qwen3.6)
- No RS buffer (DFlash excluded from `need_n_rs_seq()`)
- No additional compute buffers beyond what the forward pass already used

The tape replay added compute time (replaying N accepted tokens through DeltaNet layers) but no VRAM overhead.

### 4. Were there other BeeLlama-specific DFlash optimizations removed?

**Yes, several:**

| Optimization | Old Location | Status |
|-------------|-------------|--------|
| `llama_dflash_memory_seq_cp_recurrent_ordered()` | [`server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788) | **Removed.** GPU-optimized, layer-ordered recurrent copy for backup cells. |
| `dflash_profile_start()` / `dflash_profile_end()` | `src/dflash-profile.h` | **Removed.** DFlash-specific profiling infrastructure. |
| `tree_bufs` (DDTree) | Various | **Removed.** Tree-based draft support with branch tokens. |
| `dflash_recurrent_profile_reset()` | `src/dflash-profile.h` | **Removed.** Profile reset for backup operations. |
| `llama_tape_replay_sync()` | `src/llama-context.cpp` | **Removed.** Multi-slot tape synchronization. |
| `recurrent_backup_attention_streams` | `server-context.cpp:2048` | **Removed.** Full attention backup for DDTree branch rollback. |
| `branch_budget` parameter | `common/common.h:439` | **Removed.** Controlled DDTree branch depth. |

### 5. Did the old implementation use `llama_dflash_memory_seq_cp_recurrent_ordered()` for GPU-optimized recurrent copies? Does the current implementation have an equivalent?

**Yes, the old implementation used it. The current implementation does NOT have an equivalent.**

Old code at [`server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788):

```cpp
auto dflash_backup_recurrent_state = [&](llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    auto * mem = llama_get_memory(ctx_tgt);
    dflash_recurrent_profile_reset(mem);
    const int64_t t_backup_start = dflash_profile_start();
    const bool ordered = llama_dflash_memory_seq_cp_recurrent_ordered(ctx_tgt, seq_id_src, seq_id_dst, -1, -1);
    if (!ordered) {
        llama_memory_seq_cp_recurrent(mem, seq_id_src, seq_id_dst, -1, -1);
    }
    // ... profiling ...
};
```

This function:
1. First tried `llama_dflash_memory_seq_cp_recurrent_ordered()` — a layer-ordered copy optimized for GPU execution (copying layers in dependency order to maximize GPU utilization).
2. Fell back to `llama_memory_seq_cp_recurrent()` if the ordered path failed.

The current implementation has no equivalent because RS snapshots eliminate the need for copy-before-draft. The snapshots are maintained during the forward pass, so no explicit copy operation is needed before drafting.

---

## Appendix A: Resolution Options

Based on this analysis, here are the options for reducing DFlash VRAM overhead:

| Option | VRAM Savings | Speed Impact | Complexity | Notes |
|--------|-------------|--------------|------------|-------|
| **Keep upstream design** | 0 | Baseline | Low | Current behavior. 5.4GB overhead. |
| **Remove DFlash from `need_n_rs_seq()`** | ~4.8GB (5386 - 598) | Slower rollback (checkpoint-only) | Low | One-line change. Falls back to `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`. |
| **Reduce `n_rs_seq` for DFlash** (e.g., 2-4) | 1.8-3GB | Partial fast rollback | Medium | Would require DFlash-specific `n_rs_seq` parameter. |
| **Port BeeLlama backup cells** | ~4.8GB (backup = ~150MB vs RS = 5.4GB) | Medium rollback speed | High | Requires re-implementing `dflash_backup_recurrent_state()`, `llama_dflash_rollback()`, `tape_replay()`, and associated server code. |

---

## Appendix B: Key Source References

| Component | Old Location | Current Location |
|-----------|-------------|-----------------|
| `need_n_rs_seq()` | [`common/common.h:503`](old-versions/beellama.cpp-preview-v0.3.2/common/common.h:503) | [`common/common.h:417`](common/common.h:417) |
| DFlash `accept()` | [`common/speculative.cpp:3145`](old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp:3145) | [`common/speculative.cpp:1195`](common/speculative.cpp:1195) |
| `llama_dflash_rollback()` | [`src/llama-context.cpp:4218`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218) | **Removed** |
| `dflash_backup_recurrent_state()` | [`tools/server/server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788) | **Removed** |
| `server_dflash_recurrent_rollback_plan` | [`tools/server/server-context.h:18`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.h:18) | **Removed** |
| `common_context_can_seq_rm()` | [`common/common.cpp:1448`](old-versions/beellama.cpp-preview-v0.3.2/common/common.cpp:1448) | [`common/common.cpp:1666`](common/common.cpp:1666) |
| `server_speculative_rollback_requires_checkpoint()` | N/A (old used direct checks) | [`tools/server/server-task.h:20`](tools/server/server-task.h:20) |
| RS buffer allocation | N/A | [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) |
| Server rollback path | [`tools/server/server-context.cpp:7494`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:7494) | [`tools/server/server-context.cpp:4219`](tools/server/server-context.cpp:4219) |
| `get_seq_rm_capability()` | N/A (old used `llama_n_rs_seq()` directly) | [`src/llama-memory-recurrent.cpp:782`](src/llama-memory-recurrent.cpp:782) |

---

*Document generated 2026-08-07. Based on analysis of BeeLlama v0.3.2 preview (`old-versions/beellama.cpp-preview-v0.3.2/`) and current codebase (post-v0.4.0 merge with upstream llama.cpp 0.4.1).*
