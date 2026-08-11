# CURRENT BeeLlama DFlash Implementation Trace

**Source:** Current project root (post-v0.4.0 merge with upstream llama.cpp 0.4.1)

This document traces the DFlash implementation in the current codebase and documents how it differs from the old v0.3.2 preview implementation traced in [`old-dflash-trace.md`](old-dflash-trace.md).

---

## Table of Contents

| Section | Topic |
|---------|-------|
| 1 | Executive Summary: Key Architectural Change |
| 2 | `need_n_rs_seq()` - DFlash NOW Included in RS |
| 3 | DFlash `accept()` Method |
| 4 | `llama_dflash_rollback()` - REMOVED |
| 5 | Backup Cell System - REMOVED |
| 6 | RS Buffer Allocation in `llama-memory-recurrent.cpp` |
| 7 | `server_speculative_rollback_requires_checkpoint()` |
| 8 | Server Rollback Path for DFlash |
| 9 | `common_context_can_seq_rm()` - RS Detection |
| 10 | Summary: What Changed from Old to Current |
| 11 | VRAM Cost of RS Buffer Allocation |

---

## 1. Executive Summary: Key Architectural Change

**The current DFlash implementation uses RS (recurrent state) snapshots as its PRIMARY rollback mechanism, completely replacing the old backup cell system.**

This is a fundamental architectural shift:

| Aspect | Old (v0.3.2) | Current (v0.4.0+) |
|----------|:------------:|:-----------------:|
| Primary rollback | Backup cells (recurrent state copied to backup sequences) | RS snapshots (inline per-token snapshots bounded by `n_rs_seq`) |
| Fallback rollback | Checkpoints (full context serialize/restore) | Checkpoints (same, when rollback exceeds RS bounds) |
| `need_n_rs_seq()` | Excluded DFlash (only MTP) | **Includes DFlash** (MTP, EAGLE3, DFlash) |
| `llama_dflash_rollback()` | Existed | **Removed** |
| `dflash_backup_recurrent_state()` | Existed | **Removed** |
| `has_recurrent_only_backup` | Existed | **Removed** |
| `seq_backup` computation | `slot.id + n_parallel_user` | **Removed** |
| `server_dflash_recurrent_rollback_plan` | Existed | **Removed** |
| DFlash `accept()` | No-op | Still no-op |
| Rollback unified? | No (DFlash had separate path) | **Yes** (all speculative types share upstream rollback) |

The current implementation integrates DFlash into upstream's unified speculative decoding rollback system. DFlash now relies on the same RS snapshot mechanism as MTP and EAGLE3, with checkpoint fallback when the rollback depth exceeds `n_rs_seq`.

---

## 2. `need_n_rs_seq()` - DFlash NOW Included in RS

**Location:** [`common/common.h:417`](common/common.h:417)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });

    return needs_rs_seq ? draft.n_max : 0u;
}
```

### Findings

- **DFlash is NOW included** in `need_n_rs_seq()` alongside MTP and EAGLE3.
- When DFlash is the active speculative type, this function returns `draft.n_max` (the maximum draft horizon).
- The returned value flows into `cparams.n_rs_seq` which controls RS buffer allocation.

### Old vs Current

| Version | DFlash in `need_n_rs_seq()` | Return for DFlash |
|---------|----------------------------|-------------------|
| Old (v0.3.2) | **NO** - only checked `DRAFT_MTP` | 0 |
| Current (v0.4.0+) | **YES** - checks MTP, EAGLE3, DFlash | `draft.n_max` |

### Flow

```
need_n_rs_seq() -> params.n_rs_seq -> cparams.n_rs_seq -> llama_memory_recurrent constructor
```

At [`src/llama-context.cpp:264`](src/llama-context.cpp:264):
```cpp
cparams.n_rs_seq = params.n_rs_seq;
if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
    LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                    __func__, cparams.n_rs_seq);
    cparams.n_rs_seq = 0;
}
```

---

## 3. DFlash `accept()` Method

**Location:** [`common/speculative.cpp:1195`](common/speculative.cpp:1195)

```cpp
void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
}
```

### Findings

- **Still a complete no-op.** No change from the old implementation.
- The DFlash `accept()` method does nothing with `seq_id`, `n_accepted`, or `is_other`.
- This is consistent with DFlash's block-based commit model where per-token acceptance tracking at the speculative implementation level is unnecessary.

### Why this is still correct

The rollback work is now handled by:
1. **RS snapshots** - The recurrent memory layer maintains per-token snapshots inline. When `seq_rm()` is called during verification, the memory layer uses these snapshots to restore state to the accepted position.
2. **Checkpoint fallback** - When rollback exceeds RS bounds, the server serializes/restores full context state.
3. The `common_speculative_accept()` function at [`common/speculative.cpp:2633`](common/speculative.cpp:2633) calls `impl->accept()` but primarily tracks acceptance statistics (`n_acc_tokens_per_pos`, `n_acc_drafts`, etc.).

---

## 4. `llama_dflash_rollback()` - REMOVED

### Search Results

- **0 results** for `llama_dflash_rollback` in `.cpp` files across the entire project.
- **0 results** for `llama_dflash_rollback` in `.h` files across the entire project.

### Old Implementation (for reference)

The old `llama_dflash_rollback()` at `old-versions/.../src/llama-context.cpp:4218` performed:
1. Attention KV cleanup (remove rejected, keep accepted)
2. Recurrent state restore from backup sequence
3. Tape replay for accepted tokens

### Current State

**This function no longer exists.** DFlash rollback is now handled by the upstream unified speculative decoding system:
- When `use_ckpt_tgt == false`: RS-native rollback through `seq_rm()` with snapshot restore.
- When `use_ckpt_tgt == true`: Checkpoint-based rollback (serialize/restore).

The server's rejection path at [`tools/server/server-context.cpp:4225`](tools/server/server-context.cpp:4225) is the single unified path for all speculative types.

---

## 5. Backup Cell System - REMOVED

### Search Results

| Search Term | Results |
|-------------|---------|
| `backup.*cell` in `tools/server/*.cpp` | **0** |
| `recurrent_backup` in `tools/server/*.cpp` | **0** |
| `has_recurrent_only_backup` in `tools/server/*.cpp` | **0** |
| `dflash_backup_recurrent` in `tools/server/*.cpp` | **0** |
| `seq_backup` in `tools/server/*.cpp` | **0** |
| `dflash_rollback` in `tools/server/*.cpp` | **0** |
| `server_dflash_recurrent_rollback_plan` in `tools/server/*.cpp` | **0** |

### Old Components (all removed)

| Old Component | Purpose | Status |
|--------------|---------|--------|
| `dflash_backup_recurrent_state()` | Copy recurrent state to backup sequence | **Removed** |
| `recurrent_backup_sequences` flag | Enable backup sequences for DFlash | **Removed** |
| `recurrent_backup_attention_streams` flag | Enable full backup for DDTree | **Removed** |
| `has_recurrent_only_backup` flag | Distinguish flat vs tree backup | **Removed** |
| `seq_backup = slot.id + n_parallel_user` | Compute backup sequence ID | **Removed** |
| `server_dflash_recurrent_rollback_plan` struct | Plan backup requirements | **Removed** |
| `server_context_dflash_recurrent_rollback_plan()` function | Determine backup needs | **Removed** |
| `speculative_recurrent_rollback_plan()` method | Context-level plan | **Removed** |

### Architectural Implication

The backup cell system has been **completely replaced** by the RS snapshot system. The recurrent memory now maintains per-token snapshots inline (allocated as `mem_size * (1 + n_rs_seq)` rows), eliminating the need for separate backup sequences and the associated copy-before-draft overhead.

---

## 6. RS Buffer Allocation in `llama-memory-recurrent.cpp`

**Location:** [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99)

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

### How it works

- Each recurrent layer allocates `(1 + n_rs_seq)` rows per cell.
- The `1` row is the current state.
- The `n_rs_seq` rows are per-token snapshots for rollback.
- When `n_rs_seq = draft.n_max` (e.g., 8 for DFlash with block_size=16), each cell has 9 rows: 1 active + 8 snapshots.

### Log output

**Location:** [`src/llama-memory-recurrent.cpp:123`](src/llama-memory-recurrent.cpp:123)

```cpp
LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB\n", __func__,
        (float)(memory_size_r + memory_size_s) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
        ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
        ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f));
```

### Snapshot index management

**Location:** [`src/llama-memory-recurrent.cpp:477`](src/llama-memory-recurrent.cpp:477)

```cpp
rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
```

The snapshot index is clamped to `n_rs_seq`. This ensures the rollback window never exceeds the allocated buffer.

### Partial rollback via snapshots

**Location:** [`src/llama-memory-recurrent.cpp:214`](src/llama-memory-recurrent.cpp:214)

```cpp
// partial rollback via per-token snapshot index (bounded by n_rs_seq)
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
        return true;
    }
    return false;
}
```

When `seq_rm()` requests a rollback to position `p0`, the memory layer:
1. Computes `rollback = cell.pos - (p0 - 1)` (number of tokens to rollback).
2. If `rollback <= n_rs_seq`, restores the snapshot and truncates the cell position.
3. If `rollback > n_rs_seq`, returns `false` (indicating RS cannot handle this rollback depth).

### Capability reporting

**Location:** [`src/llama-memory-recurrent.cpp:782`](src/llama-memory-recurrent.cpp:782)

```cpp
llama_memory_i::seq_rm_capability llama_memory_recurrent::get_seq_rm_capability() const {
    return {
        /* .full_clear = */ true,
        /* .arbitrary_ranges = */ false,
        /* .suffix_rollback_tokens = */ n_rs_seq,
    };
}
```

- `arbitrary_ranges = false` - cannot do arbitrary range removal.
- `suffix_rollback_tokens = n_rs_seq` - can rollback up to `n_rs_seq` tokens via snapshots.

---

## 7. `server_speculative_rollback_requires_checkpoint()`

**Location:** [`tools/server/server-task.h:20`](tools/server/server-task.h:20)

```cpp
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
          (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

### Behavior when DFlash is active with `n_rs_seq > 0`

When DFlash is active and `n_rs_seq > 0`, the target context reports `COMMON_CONTEXT_SEQ_RM_TYPE_RS`. The function then:

| `proposed_rollback` vs `max_rollback` | Returns | Meaning |
|--------------------------------------|---------|---------|
| `proposed_rollback <= max_rollback` | `false` | RS snapshots handle the rollback. No checkpoint needed. |
| `proposed_rollback > max_rollback` | `true` | Rollback exceeds RS bounds. Checkpoint required. |

### `max_rollback` source

**Location:** [`common/common.cpp:1688`](common/common.cpp:1688)

```cpp
uint32_t common_context_seq_rm_max_rollback(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (!mem) {
        return 0;
    }
    const auto capability = llama_memory_get_seq_rm_capability(mem);
    return capability.arbitrary_ranges ? UINT32_MAX : capability.suffix_rollback_tokens;
}
```

For recurrent memory with `n_rs_seq > 0`:
- `arbitrary_ranges = false`
- `suffix_rollback_tokens = n_rs_seq`
- So `max_rollback = n_rs_seq`

This means `proposed_rollback > n_rs_seq` triggers checkpoint fallback.

---

## 8. Server Rollback Path for DFlash

### `ctx_tgt_seq_rm_type` determination

**Location:** [`tools/server/server-context.cpp:1389`](tools/server/server-context.cpp:1389)

```cpp
ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
```

For DFlash with `n_rs_seq > 0` on a recurrent model, `common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_RS`.

### Unified rollback flow

**Location:** [`tools/server/server-context.cpp:4219`](tools/server/server-context.cpp:4219)

```cpp
const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);

// check for partial draft acceptance
if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        // ... checkpoint restore path ...
        return;
    }
}

// RS-native rollback path (when use_ckpt_tgt == false)
if (trace > 0) {
    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
}

common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
slot.spec_draft = std::move(accepted);
```

### Two rollback paths

| Path | Condition | Mechanism |
|------|-----------|-----------|
| **RS-native rollback** | `use_ckpt_tgt == false` (rollback <= n_rs_seq) | `common_sampler_sample_and_accept_n()` handles verification. The memory layer's `seq_rm()` uses RS snapshots to restore to the accepted position. |
| **Checkpoint rollback** | `use_ckpt_tgt == true` (rollback > n_rs_seq) | Server serializes context before draft, then restores from checkpoint on rejection. |

### Checkpoint creation (before draft)

**Location:** [`tools/server/server-context.cpp:3305`](tools/server/server-context.cpp:3305)

```cpp
if (!draft.empty()) {
    const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
            ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());

    if (use_ckpt_tgt) {
        ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
}
```

Checkpoints are only created when `use_ckpt_tgt == true`. For DFlash with `n_rs_seq >= draft.size()`, no checkpoint is created, and rollback uses RS snapshots exclusively.

### Checkpoint creation decision for save/restore

**Location:** [`tools/server/server-context.cpp:3784`](tools/server/server-context.cpp:3784)

```cpp
do_checkpoint = do_checkpoint && (
        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
        n_swa > 0);
```

When `ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS`, checkpoints may be used for slot save/restore operations (not speculative rollback).

---

## 9. `common_context_can_seq_rm()` - RS Detection

**Location:** [`common/common.cpp:1666`](common/common.cpp:1666)

```cpp
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    const auto capability = llama_memory_get_seq_rm_capability(mem);
    if (capability.arbitrary_ranges) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    }
    if (capability.suffix_rollback_tokens > 0) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    }
    if (capability.full_clear) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    }
    return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
}
```

### Detection order

1. `arbitrary_ranges == true` → `PART` (standard KV cache with full flexibility)
2. `suffix_rollback_tokens > 0` → `RS` (recurrent memory with snapshot-based rollback)
3. `full_clear == true` → `FULL` (can only clear entire sequences)
4. Otherwise → `NO` (no sequence removal supported)

For DFlash with recurrent memory and `n_rs_seq > 0`, step 2 triggers, returning `COMMON_CONTEXT_SEQ_RM_TYPE_RS`.

---

## 10. Summary: What Changed from Old to Current

### Removed Components

| Component | Old Location | Status |
|-----------|-------------|--------|
| `llama_dflash_rollback()` | `src/llama-context.cpp:4218` | **Removed** |
| `dflash_backup_recurrent_state()` | `tools/server/server-context.cpp:4788` | **Removed** |
| `server_dflash_recurrent_rollback_plan` | `tools/server/server-context.h:18` | **Removed** |
| `server_context_dflash_recurrent_rollback_plan()` | `tools/server/server-context.cpp:54` | **Removed** |
| `speculative_recurrent_rollback_plan()` | `tools/server/server-context.cpp:2163` | **Removed** |
| `recurrent_backup_sequences` flag | `tools/server/server-context.cpp:2048` | **Removed** |
| `recurrent_backup_attention_streams` flag | `tools/server/server-context.cpp:2048` | **Removed** |
| `has_recurrent_only_backup` flag | `tools/server/server-context.cpp:5123` | **Removed** |
| `seq_backup` computation | `tools/server/server-context.cpp:5108` | **Removed** |
| `tape_replay()` | `src/llama-context.cpp` | **Removed** |
| `tree_bufs` (DDTree) | Various | **Removed** |

### Changed Components

| Component | Old Behavior | Current Behavior |
|-----------|-------------|------------------|
| `need_n_rs_seq()` | Only MTP | MTP + EAGLE3 + **DFlash** |
| DFlash rollback | Dedicated `llama_dflash_rollback()` | Unified upstream path via RS snapshots |
| State backup | Proactive copy to backup sequence | Inline per-token snapshots in recurrent buffer |
| Rollback on rejection | Restore from `seq_backup` + tape replay | `seq_rm()` with RS snapshot restore |
| Checkpoint usage | Fallback when backup unavailable | Fallback when rollback > n_rs_seq |
| Server DFlash-specific code | Extensive (backup, plan, rollback) | Minimal (integrated into upstream) |

### Unchanged Components

| Component | Status |
|-----------|--------|
| DFlash `accept()` | Still no-op |
| DFlash `draft()` | Same block-diffusion approach |
| DFlash `process()` | Same feature extraction + KV injection |
| `common_speculative_impl_draft_dflash` structure | Same core structure |

---

## 11. VRAM Cost of RS Buffer Allocation for DFlash

### Formula

For each recurrent layer:
```
n_rows = mem_size * (1 + n_rs_seq)
```

Where:
- `mem_size` = number of cells (typically = context size / ubatch size)
- `n_rs_seq` = `draft.n_max` (for DFlash, typically `block_size - 1`, e.g., 8 for block_size=16)

The VRAM overhead for RS snapshots is:
```
RS_overhead = n_rs_seq / (1 + n_rs_seq) of total recurrent buffer
```

For `n_rs_seq = 8`:
- Total rows per cell: 9 (1 active + 8 snapshots)
- Snapshot overhead: 8/9 = 88.9% of recurrent buffer is snapshots
- The active state uses only 1/9 = 11.1% of the buffer

### Comparison with old backup cells

| Metric | Old (backup cells) | Current (RS snapshots) |
|--------|-------------------|----------------------|
| VRAM per token | Full recurrent state per backup sequence | Per-token snapshot rows inline |
| Copy overhead | Copy-before-draft (`dflash_backup_recurrent_state()`) | No copy needed (snapshots maintained during forward pass) |
| Scalability | O(1) per slot (one backup per slot) | O(n_rs_seq) per cell |
| Max rollback | Full context (backup sequence holds all state) | Bounded by `n_rs_seq` (checkpoint fallback beyond that) |

### Exact VRAM formula

For a recurrent model with DFlash:
```
Total RS VRAM = n_layers * mem_size * (1 + n_rs_seq) * (sizeof(type_r) * n_embd_r + sizeof(type_s) * n_embd_s)
```

Where `n_rs_seq = draft.n_max` comes from `need_n_rs_seq()`.

---

## Appendix: File Index

| File | Relevant Lines | Content |
|------|---------------|---------|
| `common/common.h` | 417-423 | `need_n_rs_seq()` - DFlash included |
| `common/speculative.cpp` | 906-1202 | `common_speculative_impl_draft_dflash` class |
| `common/speculative.cpp` | 1195-1197 | DFlash `accept()` - no-op |
| `common/speculative.cpp` | 2633-2656 | `common_speculative_accept()` |
| `common/common.cpp` | 1666-1686 | `common_context_can_seq_rm()` |
| `common/common.cpp` | 1688-1695 | `common_context_seq_rm_max_rollback()` |
| `src/llama-context.cpp` | 264-270 | `cparams.n_rs_seq` validation |
| `src/llama-context.cpp` | 4147-4149 | `llama_n_rs_seq()` |
| `src/llama-context.cpp` | 4385-4395 | `llama_memory_get_seq_rm_capability()` |
| `src/llama-memory-recurrent.cpp` | 99-106 | RS tensor allocation with `n_rs_seq` |
| `src/llama-memory-recurrent.cpp` | 123-125 | RS buffer log output |
| `src/llama-memory-recurrent.cpp` | 214-222 | Partial rollback via snapshots |
| `src/llama-memory-recurrent.cpp` | 477 | `set_rs_idx()` clamping |
| `src/llama-memory-recurrent.cpp` | 782-786 | `get_seq_rm_capability()` |
| `tools/server/server-task.h` | 20-26 | `server_speculative_rollback_requires_checkpoint()` |
| `tools/server/server-context.cpp` | 1389-1394 | `ctx_tgt_seq_rm_type` detection |
| `tools/server/server-context.cpp` | 3305-3329 | Pre-draft checkpoint decision |
| `tools/server/server-context.cpp` | 3784-3787 | Checkpoint save/restore decision |
| `tools/server/server-context.cpp` | 4219-4273 | Unified rollback flow |
