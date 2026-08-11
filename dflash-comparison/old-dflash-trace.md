# OLD BeeLlama DFlash Implementation Trace

**Source:** `old-versions/beellama.cpp-preview-v0.3.2/` (READ-ONLY - not modified)

This document traces the DFlash implementation in the v0.3.2 preview release to understand its rollback mechanism, backup cell system, and how it differs from upstream MTP/EAGLE speculative decoding.

---

## Table of Contents

| Section | Topic |
|---------|-------|
| 1 | `need_n_rs_seq()` - DFlash Exclusion from RS |
| 2 | DFlash `accept()` Method |
| 3 | `llama_dflash_rollback()` - Rollback Mechanism |
| 4 | Backup Cell Mechanism |
| 5 | `server_dflash_recurrent_rollback_plan` |
| 6 | Checkpoint Usage (`COMMON_CONTEXT_SEQ_RM_TYPE`) |
| 7 | Summary: RS Snapshots vs Backup Cells |

---

## 1. `need_n_rs_seq()` - DFlash Exclusion from RS

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/common/common.h:503`](old-versions/beellama.cpp-preview-v0.3.2/common/common.h:503)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
    });

    if (!needs_rs_seq) {
        return 0u;
    }

    return draft.n_max > 0 ? (uint32_t) draft.n_max : 0u;
}
```

### Findings

- **DFlash is NOT included in `need_n_rs_seq()`.** The function only checks for `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.
- DFlash, EAGLE, and n-gram modes return `0` from this function, meaning they do NOT require RS (recurrent state) sequences.
- Only MTP (multi-token prediction) draft types need RS sequences.
- This means DFlash does NOT use the RS snapshot mechanism for rollback. It uses a different approach (backup cells, described below).

### Implication

DFlash was intentionally designed to NOT use RS sequences. The rollback mechanism for DFlash is completely separate from the MTP RS snapshot system.

---

## 2. DFlash `accept()` Method

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp:3145`](old-versions/beellama.cpp-preview-v0.3.2/common/speculative.cpp:3145)

```cpp
void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool /*is_other*/) override {
    GGML_UNUSED(n_accepted);
}
```

### Findings

- The DFlash `accept()` method is a **complete no-op**. It does nothing with the acceptance result.
- The parameters `seq_id`, `n_accepted`, and `is_other` are all unused (explicitly commented out).
- This is in stark contrast to MTP/EAGLE implementations, which typically use `accept()` to update RS snapshots or draft tree state.

### Why this matters

DFlash does not need per-token acceptance tracking at the speculative implementation level because:

1. DFlash uses a **block-based** commit model (entire blocks of tokens are verified together).
2. The actual rollback work happens at the **server level** through `llama_dflash_rollback()`, not through the speculative implementation's `accept()` method.
3. State backup is done **before drafting begins** (proactive backup), not during acceptance (reactive snapshots).

---

## 3. `llama_dflash_rollback()` - Rollback Mechanism

### Public API

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:8993`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:8993)

```cpp
void llama_dflash_rollback(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    ctx->dflash_rollback(seq_id, seq_backup, n_past_before, n_accepted);
}
```

### Internal Implementation

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218)

```cpp
void llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_rollback requires hybrid memory\n", __func__);
        return;
    }

    // ... profiling setup ...

    auto * mem_attn = mem_hybrid->get_mem_attn();
    auto * mem_recr = mem_hybrid->get_mem_recr();

    if (tree_bufs.n_tokens > 0) {
        // Tree mode: branch tokens may have polluted KV at accepted positions.
        // Remove ALL entries from n_past_before onwards and restore from backup.
        mem_attn->seq_rm(seq_id, n_past_before, -1);
        mem_attn->seq_cp(seq_backup, seq_id, n_past_before, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    } else {
        // Flat mode: no duplicate entries at same position, safe to keep accepted KV
        int kv_keep_pos = n_past_before + n_accepted;
        mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
    }

    // Recurrent state: restore from backup, then tape replay
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);
    mem_recr->seq_rm(seq_backup, -1, -1);

    // Replay DeltaNet state updates for accepted tokens
    tape_replay(seq_id, n_accepted);
}
```

### Key Observations

The rollback has **three distinct phases**:

| Phase | Action | Target |
|-------|--------|--------|
| **1. Attention KV cleanup** | Remove rejected KV, keep accepted KV (flat) or restore all from backup (tree) | `mem_attn` |
| **2. Recurrent state restore** | Delete current recurrent state, copy from backup sequence, delete backup | `mem_recr` |
| **3. Tape replay** | Replay DeltaNet state updates for accepted tokens | Hybrid memory tape |

### Flat Mode vs Tree Mode

- **Flat mode** (`tree_bufs.n_tokens == 0`): Only removes rejected KV after the accepted tokens. The accepted KV is kept because "no duplicate entries at same position."
- **Tree mode** (`tree_bufs.n_tokens > 0`): Removes ALL KV from `n_past_before` onwards and restores everything from backup. This is because "branch tokens may have polluted KV at accepted positions."

### Recurrent State Handling

The recurrent state is ALWAYS restored from backup (both flat and tree modes):
1. `mem_recr->seq_rm(seq_id, -1, -1)` - delete current recurrent state
2. `mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1)` - copy from backup
3. `mem_recr->seq_rm(seq_backup, -1, -1)` - delete backup
4. `tape_replay(seq_id, n_accepted)` - replay DeltaNet for accepted tokens

---

## 4. Backup Cell Mechanism

### Overview

DFlash uses **backup sequences** (also called "backup cells") for recurrent state rollback. These are pre-allocated sequence IDs that hold copies of recurrent state before drafting begins.

### Key Variables

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2048`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2048)

```cpp
bool recurrent_expanded = true;           // false = backup cells deferred, expand before first draft
bool recurrent_backup_sequences = false;  // explicit backup seqs for speculative recurrent rollback
bool recurrent_backup_attention_streams = false; // backup seqs also have attention KV streams
```

### `dflash_backup_recurrent_state()`

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788)

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

### What it copies

- `dflash_backup_recurrent_state()` copies **recurrent state only** (not attention KV).
- It first tries `llama_dflash_memory_seq_cp_recurrent_ordered()` (layer-ordered copy for GPU optimization).
- Falls back to `llama_memory_seq_cp_recurrent()` if the ordered path fails.
- The copy is from `seq_id_src` (the main slot sequence) to `seq_id_dst` (the backup sequence).

### `seq_backup` Computation

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108)

```cpp
const llama_seq_id seq_backup = slot.id + n_parallel_user;
```

The backup sequence ID is computed as `slot.id + n_parallel_user`. This means:

- If you have 4 parallel users (slots 0-3), the backup sequences are 4-7.
- Backup sequences are allocated from the upper range of the sequence pool.
- Each slot gets one backup sequence.

### `has_recurrent_only_backup` vs Full Backup

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5123`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5123) and [`5278`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5278)

```cpp
// For DFlash (flat mode, no branches):
dflash_backup_recurrent_state(slot.id, seq_backup);
slot.has_recurrent_only_backup = true;

// For DDTree (branch mode):
llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);  // full copy including attention
slot.has_recurrent_only_backup = false;
```

| Flag | Value | Meaning |
|------|-------|---------|
| `has_recurrent_only_backup = true` | DFlash flat mode | Backup contains recurrent state only. Attention KV is NOT backed up (kept inline). |
| `has_recurrent_only_backup = false` | DDTree / full backup | Backup contains full state (attention + recurrent). Used when branches could pollute accepted KV. |

### How Recurrent State is Restored After Rejection

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:7494`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:7494)

```cpp
if (all_accepted_flat) {
    // All tokens accepted - clean up backup
    llama_clear_tree_parent_ids(ctx_tgt);
    auto * mem = llama_get_memory(ctx_tgt);
    if (slot.has_recurrent_only_backup) {
        llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1);
    } else {
        llama_memory_seq_rm(mem, seq_backup, -1, -1);
    }
    llama_memory_seq_rm(mem, slot.id, slot.prompt.tokens.pos_next(), -1);
} else {
    // Partial or no acceptance - rollback
    llama_clear_tree_parent_ids(ctx_tgt);
    llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep);
    if (n_slots_drafted > 1) {
        llama_tape_replay_sync(ctx_tgt);
    }
}
```

The rejection path:
1. If **all tokens accepted**: delete backup (recurrent-only or full depending on flag), clean up post-prompt KV.
2. If **partial/no acceptance**: call `llama_dflash_rollback()` which restores recurrent state from backup and replays the tape for accepted tokens.
3. For multi-slot scenarios, synchronize the tape replay before the next slot mutates state.

---

## 5. `server_dflash_recurrent_rollback_plan`

### Struct Definition

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.h:18`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.h:18)

```cpp
struct server_dflash_recurrent_rollback_plan {
    bool needs_backup_sequences = false;
    bool needs_attention_backup_streams = false;
};
```

### Free-standing Function

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:54`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:54)

```cpp
server_dflash_recurrent_rollback_plan server_context_dflash_recurrent_rollback_plan(
        const common_params_speculative & speculative,
        bool target_recurrent_or_hybrid) {
    server_dflash_recurrent_rollback_plan plan;

    if (!speculative.dflash_selected_or_pending() || !target_recurrent_or_hybrid) {
        return plan;
    }

    plan.needs_backup_sequences = true;
    plan.needs_attention_backup_streams = speculative.branch_budget > 0;
    return plan;
}
```

### Method Version (in `server_context`)

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2163`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2163)

```cpp
server_dflash_recurrent_rollback_plan speculative_recurrent_rollback_plan() const {
    server_dflash_recurrent_rollback_plan plan;

    if (!params_base.speculative.dflash_selected_or_pending()) {
        return plan;
    }

    const server_model_arch_probe arch = server_probe_model_arch(params_base.model.path);
    if (!arch.known) {
        SRV_WRN("%s", "could not determine target architecture before DFlash context allocation; "
                      "reserving recurrent-only backup cells conservatively\n");
        plan.needs_backup_sequences = true;
        return plan;
    }

    plan = server_context_dflash_recurrent_rollback_plan(
            params_base.speculative,
            arch.recurrent_or_hybrid);

    if (!arch.recurrent_or_hybrid) {
        SRV_INF("DFlash target architecture %s does not need recurrent rollback; keeping n_parallel=%d\n",
                arch.name.c_str(), params_base.n_parallel);
    } else if (plan.needs_attention_backup_streams) {
        SRV_WRN("DFlash DDTree target architecture %s needs attention backup streams for branch rollback\n",
                arch.name.c_str());
    } else if (plan.needs_backup_sequences) {
        SRV_INF("flat DFlash target architecture %s will use recurrent-only backup cells; "
                "keeping attention streams at n_parallel=%d\n",
                arch.name.c_str(), params_base.n_parallel);
    }

    return plan;
}
```

### What Determines `needs_backup_sequences` vs `needs_attention_backup_streams`

| Condition | `needs_backup_sequences` | `needs_attention_backup_streams` |
|-----------|-------------------------|----------------------------------|
| DFlash NOT selected | `false` | `false` |
| Target is NOT recurrent/hybrid | `false` | `false` |
| DFlash + recurrent target + flat mode (`branch_budget == 0`) | `true` | `false` |
| DFlash + recurrent target + tree mode (`branch_budget > 0`) | `true` | `true` |
| Architecture unknown | `true` (conservative) | `false` |

**Key insight:**
- `needs_backup_sequences` = true when the target model has recurrent state (Mamba, Hyena, etc.) that needs to be backed up before drafting.
- `needs_attention_backup_streams` = true ONLY when using DDTree (tree-based drafting with `branch_budget > 0`), because branch tokens can pollute attention KV at accepted positions, requiring full attention backup.

---

## 6. Checkpoint Usage (`COMMON_CONTEXT_SEQ_RM_TYPE`)

### Enum Definition

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/common/common.h:1078`](old-versions/beellama.cpp-preview-v0.3.2/common/common.h:1078)

```cpp
enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO   = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS   = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};
```

### Detection Logic

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/common/common.cpp:1448`](old-versions/beellama.cpp-preview-v0.3.2/common/common.cpp:1448)

```cpp
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    // ... null checks ...
    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);
    // eval 2 tokens to check compatibility ...

    if (llama_n_rs_seq(ctx) > 0) {
        res = COMMON_CONTEXT_SEQ_RM_TYPE_RS;  // RS bounded
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;  // full context only
        goto done;
    }

    return res;  // PART (default if partial removal succeeded)
}
```

### DFlash Checkpoint Usage in Server

**Location:** [`old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5258`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5258)

```cpp
// MTP uses the checkpoint-based accept path (which cleans ctx_dft via seq_rm)
if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
    if (!recurrent_backup_sequences) {
        GGML_ABORT("speculative recurrent rollback requires backup sequences when bounded snapshots are unavailable\n");
    }
    // ... backup setup ...
}
```

### Summary of Checkpoint Types for DFlash

| `ctx_tgt_seq_rm_type` | DFlash Behavior |
|----------------------|-----------------|
| `NO` | Speculative decoding disabled entirely. |
| `PART` | DFlash uses backup cells for rollback (no checkpoints needed). |
| `FULL` | DFlash uses backup cells. Target checkpoint (`use_ckpt_tgt`) may be used for full context restore. |
| `RS` | DFlash may use RS snapshots if `n_rollback > llama_n_rs_seq(ctx_tgt)`. Otherwise uses backup cells. |

### Key Code Paths

**Target checkpoint usage:**
```cpp
const bool use_ckpt_tgt =
    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));
```

**Draft checkpoint usage:**
```cpp
const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
```

**Rejection checkpoint fallback:**
```cpp
const bool use_ckpt_tgt =
    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));
```

---

## 7. Summary: RS Snapshots vs Backup Cells for DFlash

### Primary Rollback Mechanism: **Backup Cells** (NOT RS Snapshots)

DFlash uses **backup sequences** (backup cells) as its PRIMARY rollback mechanism. This is confirmed by:

1. **`need_n_rs_seq()` excludes DFlash** - DFlash returns 0 from `need_n_rs_seq()`, meaning it does NOT use RS sequences. Only MTP uses RS sequences.

2. **`accept()` is a no-op** - DFlash's `accept()` method does nothing. RS-based rollback would require tracking accepted tokens through `accept()`.

3. **`dflash_backup_recurrent_state()` copies recurrent state to backup sequences** - Before drafting begins, recurrent state is copied to a backup sequence (`seq_backup = slot.id + n_parallel_user`).

4. **`llama_dflash_rollback()` restores from backup sequences** - On rejection, recurrent state is restored by copying from `seq_backup` to `seq_id`, then replaying the tape for accepted tokens.

5. **`server_dflash_recurrent_rollback_plan` plans backup sequences** - The server explicitly plans for `needs_backup_sequences` when DFlash is selected with a recurrent target.

### When Checkpoints Are Used

Checkpoints (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`) are a SECONDARY mechanism used when:

- The target context cannot do partial sequence removal (`seq_rm_type == FULL`).
- The rollback depth exceeds RS bounds (`n_rollback > llama_n_rs_seq`).
- In these cases, the server falls back to full context checkpoints instead of fine-grained backup cells.

### Architecture Decision Table

| Scenario | Rollback Mechanism | Backup Type |
|----------|-------------------|-------------|
| DFlash + flat mode + recurrent target | Backup cells | Recurrent state only |
| DFlash + tree mode (DDTree) + recurrent target | Backup cells | Full (attention + recurrent) |
| DFlash + non-recurrent target | No backup needed | None |
| MTP + any target | RS snapshots | RS sequences |
| Fallback (seq_rm_type == FULL) | Checkpoints | Full context |

### Memory Flow Diagram

```
Before Draft:
  slot.id (main)     -> [attention KV | recurrent state]
  seq_backup         -> [empty]

dflash_backup_recurrent_state(slot.id, seq_backup):
  slot.id (main)     -> [attention KV | recurrent state]
  seq_backup         -> [recurrent state copy]

After Draft (partial acceptance, n_accepted < n_draft):
  slot.id (main)     -> [attention KV (dirty) | recurrent state (dirty)]
  seq_backup         -> [recurrent state (clean pre-draft copy)]

llama_dflash_rollback():
  1. Attention: remove rejected KV (flat) or restore from backup (tree)
  2. Recurrent: rm(slot.id) -> cp(seq_backup, slot.id) -> rm(seq_backup)
  3. Tape: replay DeltaNet for n_accepted tokens

After Rollback:
  slot.id (main)     -> [attention KV (clean) | recurrent state (restored + replayed)]
  seq_backup         -> [empty]
```

---

## Appendix: File Index

| File | Key Content |
|------|-------------|
| `common/common.h:503` | `need_n_rs_seq()` - confirms DFlash excluded |
| `common/common.h:1078` | `common_context_seq_rm_type` enum |
| `common/common.h:439` | `dflash_selected_or_pending()` |
| `common/speculative.cpp:2080` | `common_speculative_impl_dflash` class |
| `common/speculative.cpp:3145` | DFlash `accept()` - no-op |
| `src/llama-context.cpp:4218` | `llama_context::dflash_rollback()` |
| `src/llama-context.cpp:8993` | Public `llama_dflash_rollback()` |
| `tools/server/server-context.h:18` | `server_dflash_recurrent_rollback_plan` struct |
| `tools/server/server-context.cpp:54` | `server_context_dflash_recurrent_rollback_plan()` |
| `tools/server/server-context.cpp:2048` | `recurrent_backup_sequences` flag |
| `tools/server/server-context.cpp:2163` | `speculative_recurrent_rollback_plan()` method |
| `tools/server/server-context.cpp:4788` | `dflash_backup_recurrent_state()` lambda |
| `tools/server/server-context.cpp:5108` | `seq_backup` computation |
| `tools/server/server-context.cpp:5258` | DFlash checkpoint check |
| `tools/server/server-context.cpp:7494` | Rejection/rollback decision |
