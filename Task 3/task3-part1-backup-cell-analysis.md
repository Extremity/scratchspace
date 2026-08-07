# Old DFlash Backup Cell Mechanism Analysis

**Task:** 3.1 of Hybrid Investigation — Understand what made old DFlash VRAM-efficient.
**Source:** `old-versions/beellama.cpp-preview-v0.3.2/`
**Goal:** Identify the MINIMAL set of mechanisms responsible for VRAM savings, and determine if current architecture can support them without sacrificing speculative decoding performance.

---

## Executive Summary

The old DFlash achieved VRAM efficiency through **one fundamental design decision**: excluding DFlash from `need_n_rs_seq()`, which eliminated the RS buffer rows allocated for draft sequences. This single change saved approximately **5.4GB** of VRAM per slot (for typical Qwen3.6 serving parameters). The backup cell mechanism existed to compensate for the absence of those RS buffer rows, providing a minimal-footprint way to rollback recurrent state after verification.

The **minimal set of mechanisms** responsible for VRAM savings is:

1. **Exclude DFlash from `need_n_rs_seq()`** — the root cause of all VRAM savings.
2. **Backup cell allocation** — `n_seq_max_full = n_parallel_user * 2` instead of RS buffer rows.
3. **Recurrent-only backup copy** — `dflash_backup_recurrent_state()` copies only recurrent tensors (not attention KV), keeping backup cost at ~150MB/slot.
4. **3-phase rollback** — `dflash_rollback()` restores from backup cell instead of using RS buffer rows.

Everything else (tape replay, deferred expansion, tree rollback) is optimization or correctness support built on top of these four pillars.

---

## 2. The Root Cause: `need_n_rs_seq()` Exclusion

### Current Code (6 common/common.h:417)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;  // <-- DFlash INCLUDED
    });
    return needs_rs_seq ? draft.n_max : 0u;
}
```

### Old Code (old-versions/.../common/common.h:503)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;  // <-- DFlash EXCLUDED
    });
    if (!needs_rs_seq) {
        return 0u;
    }
    return draft.n_max > 0 ? (uint32_t) draft.n_max : 0u;
}
```

### What This Means

The RS buffer in [`llama_memory_recurrent`](src/llama-memory-recurrent.cpp:99) allocates:

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
```

| Spec Type | `n_rs_seq` | `n_rows` (per cell) | Effect |
|-----------|-----------|---------------------|--------|
| MTP, EAGLE3 | `draft.n_max` (e.g. 8) | `9 * mem_size` | Full RS buffer for all draft states |
| **Old DFlash** | **0** | **`1 * mem_size`** | **Only base cell — no draft rows** |
| **Current DFlash** | `draft.n_max` (e.g. 8) | `9 * mem_size` | Same as MTP — 5.4GB overhead |

### Why This Was the Key Insight

DFlash's draft model produces tokens **sequentially** (not in parallel like MTP/EAGLE3). The draft states are transient — they exist only during the draft phase and are discarded after verification. Storing them in the RS buffer was wasteful because:

- MTP/EAGLE3 need all draft states simultaneously for parallel verification scoring.
- DFlash processes drafts one-by-one and only needs the **final recurrent state at the point before drafting** for rollback.

The old code recognized this and used a single backup cell (recurrent state snapshot) instead of `n_max` RS buffer rows.

---

## 3. Backup Cell Allocation

### What It Did

Instead of pre-allocating `n_rs_seq` RS buffer rows, the old code allocated **one extra recurrent cell per slot** as a backup. The total cell count was:

```cpp
// old-versions/.../tools/server/server-context.cpp:2598-2600
if (recurrent_plan.needs_backup_sequences) {
    n_seq_max_full = n_parallel_user * 2;  // user slots + backup slots
    recurrent_expanded = false;              // defer expansion until needed
    recurrent_backup_sequences = true;
}
```

### Deferred Expansion

The backup cells were NOT allocated at startup. They were deferred:

```cpp
// old-versions/.../tools/server/server-context.cpp:2642-2644
SRV_INF("shrunk recurrent state to %d cells before draft load (deferred %d backup cells)\n",
        n_parallel_user, n_seq_max_full - n_parallel_user);
```

Expansion happened lazily at the first speculative draft:

```cpp
// old-versions/.../tools/server/server-context.cpp:5098-5106
if (!recurrent_expanded) {
    if (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)) {
            SRV_INF("expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
        } else {
            GGML_ABORT("failed to expand recurrent state for speculative backup");
        }
        recurrent_expanded = true;
    }
}
```

Shrink-back was also supported when prompt cache made speculation unnecessary:

```cpp
// old-versions/.../tools/server/server-context.cpp:2257-2284
bool recurrent_shrink_for_prompt_cache(const char * reason) {
    if (!recurrent_backup_sequences || !recurrent_expanded || !needs_reeval) {
            return true;
        }
        // ... shrink recurrent state back to n_parallel_user cells
        SRV_INF("shrunk recurrent state to %d cells for prompt cache (%s, removed %d backup cells)\n",
                n_parallel_user, reason, n_seq_max_full - n_parallel_user);
}
```

### Why It Existed

- **VRAM efficiency:** Backup cells are only allocated when DFlash speculation is actually used.
- **Prompt cache compatibility:** When a request is fully served from KV cache, no backup cells are needed.
- **Sized correctly:** `n_parallel_user * 2` cells is exactly what's needed (one working + one backup per slot).

### VRAM Cost

| Component | Old Backup Cell | Current RS Buffer |
|-----------|----------------|-------------------|
| Recurrent cells | `n_parallel * 2` cells | `n_parallel` cells (base) |
| RS buffer rows per cell | `1` (just the cell) | `1 + n_max` rows |
| Per-layer R buffer | `n_embd_r * cell * type_size` | `n_embd_r * cell * (1+n_max) * type_size` |
| Per-layer S buffer | `n_embd_s * cell * type_size` | `n_embd_s * cell * (1+n_max) * type_size` |

For Qwen3.6-235B-A22B on typical hardware (n_parallel=1, n_max=8):

- **Old:** ~150MB/slot backup cell (recurrent state only)
- **Current:** ~5.4GB RS buffer (`mem_size * 9` rows)

---

## 4. Recurrent State Backup Copy

### What It Did

Before verification, the server copied the recurrent state from the user's sequence to the backup sequence. The copy was **selective** — it only copied recurrent state tensors, not attention KV.

```cpp
// old-versions/.../tools/server/server-context.cpp:4788-4813
auto dflash_backup_recurrent_state = [&](llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    auto * mem = llama_get_memory(ctx_tgt);
    dflash_recurrent_profile_reset(mem);
    const bool ordered = llama_dflash_memory_seq_cp_recurrent_ordered(ctx_tgt, seq_id_src, seq_id_dst, -1, -1);
    if (!ordered) {
        llama_memory_seq_cp_recurrent(mem, seq_id_src, seq_id_dst, -1, -1);
    }
    // ... profiling
};
```

The backup was created at line 5108-5125:

```cpp
const llama_seq_id seq_backup = slot.id + n_parallel_user;  // backup slot is user slot + offset
auto * mem = llama_get_memory(ctx_tgt);
llama_memory_seq_rm(mem, seq_backup, -1, -1);

int n_branches = 0;
for (size_t i = 1; i < tree.parents.size(); ++i) {
    if (tree.parents[i] != -1 && tree.parents[i] != (int32_t)(i - 1)) {
        n_branches++;
    }
}

if (n_branches > 0) {
    // DDTree: full KV copy (branches pollute KV at accepted positions)
    llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);
} else {
    // Linear draft: recurrent-only copy (no KV pollution)
    dflash_backup_recurrent_state(slot.id, seq_backup);
}

slot.has_draft_backup = true;
slot.has_recurrent_only_backup = (n_branches == 0);
slot.seq_id_backup = seq_backup;
```

### Why It Existed

- **Linear drafts (no branches):** Only recurrent state needs backup. Attention KV is clean because no branch tokens wrote to positions that could be accepted. The recurrent-only copy is ~150MB.
- **Tree drafts (branches exist):** Branch tokens may have written to positions that overlap with accepted tokens, polluting the KV cache. Full KV copy needed.

### The Ordered Copy Optimization

[`dflash_memory_seq_cp_recurrent_ordered()`](old-versions/.../src/llama-context.cpp:2732) attempted an optimized CUDA D2D copy:

```cpp
bool llama_context::dflash_memory_seq_cp_recurrent_ordered(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0, llama_pos p1) {
    // ... try ordered CUDA D2D enqueue (async, no sync)
    mem_recr->seq_cp_recurrent_no_sync(seq_id_src, seq_id_dst, p0, p1);
    // ... return true if ordered path succeeded
}
```

The `_no_sync` variant queued CUDA D2D copies without synchronizing, allowing the backup to overlap with subsequent operations. Fallback to the standard `seq_cp_recurrent` path if the ordered path failed.

### VRAM Cost

| Path | Data Copied | Approx. Size |
|------|-------------|-------------|
| Recurrent-only backup | R+S tensors for recurrent layers only | ~150MB/slot |
| Full KV backup (DDTree branches) | Full attention + recurrent for the sequence | Varies with context |

The recurrent-only path is the common case for linear DFlash drafts.

---

## 5. Three-Phase Rollback

### What It Did

After verification determined how many tokens to accept, `dflash_rollback()` restored state from the backup cell in three phases:

```cpp
// old-versions/.../src/llama-context.cpp:4218-4292
void llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup,
                                     int n_past_before, int n_accepted) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    // ...

    auto * mem_attn = mem_hybrid->get_mem_attn();
    auto * mem_recr = mem_hybrid->get_mem_recr();

    // Phase 1: KV cleanup
    if (tree_bufs.n_tokens > 0) {
        // Tree mode: full restore from backup
        mem_attn->seq_rm(seq_id, n_past_before, -1);
        mem_attn->seq_cp(seq_backup, seq_id, n_past_before, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    } else {
        // Flat mode: keep accepted KV, remove rejected
        int kv_keep_pos = n_past_before + n_accepted;
        mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
    }

    // Phase 2: Recurrent state restore from backup
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);
    mem_recr->seq_rm(seq_backup, -1, -1);

    // Phase 3: Tape replay for accepted tokens
    tape_replay(seq_id, n_accepted);
}
```

### Phase Details

| Phase | Operation | Purpose | VRAM Impact |
|-------|-----------|---------|-------------|
| 1 | KV cleanup | Remove rejected draft KV; keep accepted KV | No new allocation |
| 2 | Recurrent restore | Copy backup cell → working cell | Uses existing backup cell |
| 3 | Tape replay | Update DeltaNet state for accepted tokens | Uses tape buffers (~small) |

### Why Three Phases

- **Phase 1** handles the attention KV cache. In flat mode (no tree), accepted KV is valid and can be kept. Only rejected tail needs removal.
- **Phase 2** restores the recurrent state to the pre-draft point. The backup cell contains the recurrent state before any draft tokens were processed.
- **Phase 3** advances the recurrent state forward by `n_accepted` tokens using tape replay. This is necessary because the backup contains the state *before* drafting, and after accepting some tokens, the recurrent state needs to reflect processing those accepted tokens. Without tape replay, you'd need to re-decode the accepted tokens (expensive).

---

## 6. Tape Replay

### What It Did

Tape replay advanced the recurrent state for accepted tokens without re-running the full model forward pass. During normal decoding, the model records ("tapes") the intermediate state transformations for recurrent layers. After rollback restores the pre-draft state, tape replay applies those transformations for the accepted tokens.

```cpp
// old-versions/.../src/llama-context.cpp:2898-3255
void llama_context::tape_replay(llama_seq_id seq_id, int n_accepted) {
    // Sync previous async replay
    tape_replay_sync();

    // Get GPU tape if available
    dflash_tape_gpu * const gpu_tape = dflash_capture->active_tape();
    const bool use_gpu_tape = (gpu_tape != nullptr && ...);

    // Find recurrent memory and target cell
    auto * mem_recurrent = ...;
    int32_t cell_idx = mem_recurrent->cells[seq_id].tail;

    // GPU path: direct GPU replay (fastest)
    if (use_gpu_tape && tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)) {
        dflash_capture->replay_pending = true;
        return;
    }

    // Fallback: CPU replay
    tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
    tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
}
```

### Why It Existed

Without tape replay, after restoring the pre-draft recurrent state, you'd need to re-decode the accepted tokens to advance the state. Tape replay avoids this by applying the already-computed state transformations. This is critical for performance because:

- Re-decoding accepted tokens would negate the performance benefit of speculative decoding.
- Tape replay is essentially a series of tensor operations on pre-recorded intermediates — much cheaper than full forward pass.

### VRAM Cost

Tape buffers (`tree_bufs`) are allocated per-recurrent-layer and sized to `max_tree_tokens`:

```cpp
// old-versions/.../src/llama-context.cpp:5405-5510
void llama_context::allocate_tree_buffers(int max_tree_tokens) {
    // parent_ids_gpu: [max_tree_tokens] i32
    // ssm_intermediates: per recurrent layer: [S_v*S_v*H*max_tokens] f16
}
```

For typical DFlash with `n_max=8`, tape buffers are small (~few MB). The GPU tape path records intermediates during the verify decode and replays them asynchronously.

---

## 7. DFlash Accept() No-Op

### What It Did

```cpp
// old-versions/.../common/speculative.cpp:3145-3147
void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool /*is_other*/) override {
    GGML_UNUSED(n_accepted);
}
```

The `accept()` method in the DFlash speculative handler is intentionally empty.

### Why

MTP/EAGLE3 use `accept()` to update their internal state (which draft states to keep for the next round). DFlash doesn't need this because:

- DFlash draft states are transient and discarded after each verify cycle.
- The backup cell mechanism handles state restoration.
- The committed length and cross-attention context track what's been accepted.

---

## 8. `dflash_prepare_branch()` for DDTree

### What It Did

For DDTree branch exploration, before processing each branch, restore recurrent state from backup and replay to the branch point:

```cpp
// old-versions/.../src/llama-context.cpp:4294-4309
void llama_context::dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * mem_recr = mem_hybrid->get_mem_recr();

    // Restore recurrent state from backup
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);

    // Tape replay to advance to branch depth
    tape_replay(seq_id, depth);
}
```

### Why

DDTree explores multiple branches from different depths. Each branch needs the recurrent state at the point where that branch diverges. The backup cell provides the pre-draft state, and tape replay advances to the specific branch depth.

---

## 9. VRAM Cost Summary

### Comparison Table

| Component | Old DFlash (Backup Cell) | Current DFlash (RS Buffer) |
|-----------|--------------------------|---------------------------|
| `n_rs_seq` | 0 | `draft.n_max` (e.g. 8) |
| RS buffer rows | `mem_size * 1` | `mem_size * (1 + n_max)` |
| Backup cells | `n_parallel` extra cells | None (RS buffer used instead) |
| Recurrent state backup | ~150MB/slot (on-demand copy) | Built into RS buffer (pre-allocated) |
| Tape buffers | ~few MB (tree_bufs) | Same (upstream inherited) |
| **Total extra VRAM** | **~150MB/slot** | **~5.4GB/slot** |

### The VRAM Savings Breakdown

The ~5.4GB RS buffer in the current implementation breaks down as:

```
n_rows = mem_size * (1 + n_rs_seq)
       = n_ctx * n_seq_max * (1 + n_max)
       = n_ctx * n_seq_max * 9  (for n_max=8)
```

The old implementation used:
```
n_rows = mem_size * (1 + 0)
       = n_ctx * n_seq_max * 1
```

Plus `n_parallel` extra backup cells (same cell size as base, no extra rows):
```
backup_cells = n_parallel * n_ctx_per_cell * n_layer * (n_embd_r + n_embd_s) * type_size
```

For Qwen3.6-235B-A22B with typical params:
- Base RS buffer (old): ~600MB
- Backup cells (old): ~150MB/slot
- **Old total: ~750MB**
- **Current total: ~6.1GB** (base + 8x RS buffer rows)

---

## 10. Minimal Set of Mechanisms for VRAM Savings

Based on the analysis, the VRAM savings come from exactly **two mechanisms**:

### Mechanism 1: Exclude DFlash from `need_n_rs_seq()`

**File:** [`common/common.h:503`](old-versions/beellama.cpp-preview-v0.3.2/common/common.h:503)

**Change:** Remove `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` from the `needs_rs_seq` check.

**Impact:** Eliminates `mem_size * n_max` RS buffer rows. This is the **primary** VRAM saving (~5.4GB).

**Performance impact:** None. DFlash doesn't need RS buffer rows because it doesn't store draft states — it uses backup cells for rollback.

### Mechanism 2: Backup Cell Rollback Path

**Files:**
- [`tools/server/server-context.cpp:2598`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2598) — `n_seq_max_full = n_parallel_user * 2`
- [`tools/server/server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788) — `dflash_backup_recurrent_state()`
- [`tools/server/server-context.cpp:5108`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108) — `seq_backup` computation
- [`src/llama-context.cpp:4218`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218) — `dflash_rollback()`

**Change:** Allocate `n_parallel * 2` recurrent cells. Before verification, copy recurrent state to backup cell. After verification, restore from backup cell and tape replay.

**Impact:** Adds ~150MB/slot for backup cells. Dramatically less than RS buffer.

**Performance impact:** Minimal. Backup copy is async CUDA D2D. Tape replay is cheaper than re-decode.

### Everything Else Is Optimization

| Component | Role | Necessary for VRAM? |
|-----------|------|---------------------|
| Deferred expansion | Don't allocate backup cells until needed | No — optimization |
| Shrink for prompt cache | Release backup cells when not needed | No — optimization |
| Ordered copy | Async CUDA D2D for backup | No — performance optimization |
| Tape replay | Avoid re-decode of accepted tokens | No — performance optimization |
| `dflash_prepare_branch()` | DDTree branch support | No — feature support |
| `tree_bufs` | Tree attention intermediates | No — feature support |
| `accept()` no-op | DFlash doesn't need accept state | No — correctness detail |

---

## 11. Current Architecture Compatibility Assessment

### What Exists Today

| Component | Old Source | Current Status |
|-----------|-----------|---------------|
| `need_n_rs_seq()` | Excluded DFlash | **Includes DFlash** — needs change |
| `llama_memory_recurrent` | Same basic structure | Exists — supports `n_rs_seq=0` |
| `seq_cp_recurrent()` | Recurrent copy API | Exists in current [`llama-memory.h:149`](src/llama-memory.h:149) |
| `resize_recurrent_memory()` | Expand/shrink cells | **Removed** in current upstream — needs re-add |
| `dflash_rollback()` | 3-phase rollback | **Removed** — needs re-add |
| `tape_replay()` | DeltaNet state replay | **Removed** — needs re-add |
| `tree_bufs` | Tree intermediates | **Removed** — needs re-add |
| Server backup cell logic | `server_context` | **Removed** — needs re-add |
| `llama_context_recurrent_expand()` | Public API | **Removed** — needs re-add |

### What Would Need to Change

#### Easy (Single-Line Changes)

1. **Exclude DFlash from `need_n_rs_seq()`** — [`common/common.h:419`](common/common.h:419)
   - Remove `t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` from the check.
   - One-line change. No API impact.

#### Moderate (Re-add Removed Functions)

2. **`dflash_rollback()`** — Need to re-add the 3-phase rollback to `llama-context.cpp`.
   - Depends on `llama_memory_hybrid` having `get_mem_attn()` and `get_mem_recr()`.
   - Current upstream may have split this differently.

3. **`tape_replay()`** — Need to re-add tape recording during verify and replay during rollback.
   - This is the most complex piece. Requires graph builder changes to record tape intermediates.
   - Current upstream DFlash may not have tape recording in the graph.

4. **`resize_recurrent_memory()`** — Need to re-add expand/shrink API.
   - Current upstream may have fixed-size recurrent allocation.

#### Hard (Server Integration)

5. **Backup cell allocation in server** — Need to set `n_seq_max = n_parallel * 2` for DFlash targets.
   - Current upstream server may not have `n_seq_max_full` / `recurrent_expanded` concepts.

6. **Backup copy timing** — Need to call `dflash_backup_recurrent_state()` before verification.
   - Requires hooking into the upstream speculative verification pipeline.

7. **Rollback integration** — Need to call `dflash_rollback()` after verification instead of the standard MTP/EAGLE3 rollback.
   - Current upstream uses RS buffer rows for rollback. DFlash needs backup cell rollback.

### Key Question: Can Tape Replay Work Without Graph Changes?

The tape replay mechanism requires that during the verify forward pass, the model records intermediate recurrent state transformations. This is typically done by embedding `ggml_copy` operations in the computation graph that store state deltas to a tape buffer.

If the current upstream DFlash graph doesn't include these tape recording operations, tape replay cannot work. In that case, the fallback would be:

- **Option A:** Re-decode accepted tokens after rollback (performance cost but correct).
- **Option B:** Re-add tape recording to the graph builder (more work but preserves performance).
- **Option C:** Use a simpler rollback that doesn't need tape — restore full state from backup including accepted tokens (requires keeping backup intact through accept).

---

## 12. Performance Considerations

### Why Backup Cells Don't Sacrifice Performance

The old approach maintained performance despite not using RS buffer rows because:

1. **Backup copy is async:** `seq_cp_recurrent_no_sync()` queues CUDA D2D copies that overlap with verification decode. The backup completes before rollback needs it, with no CPU stall.

2. **Tape replay is cheaper than re-decode:** Applying recorded state transformations (~tensor ops) is orders of magnitude cheaper than running the full model forward pass for accepted tokens.

3. **No extra graph complexity:** The verify graph is the same as current upstream DFlash. The tape recording is embedded in the same graph. No extra forward passes.

4. **Rollback is sequential but fast:** The 3-phase rollback (KV cleanup → recurrent restore → tape replay) is all GPU-native operations with minimal host synchronization.

### What Could Impact Performance

- **If tape replay is not available:** Falling back to re-decode accepted tokens would add `n_accepted * model_latency` overhead per verification cycle, potentially negating speculative speedup.
- **If backup copy blocks:** Synchronous backup copy before verification would add latency to each draft cycle.
- **If deferred expansion fails:** Pre-allocating all backup cells at startup (instead of deferred) adds VRAM overhead but simplifies the code path.

---

## 13. Conclusion

The old DFlash VRAM efficiency came from recognizing that DFlash doesn't need `n_max` RS buffer rows (unlike MTP/EAGLE3) and replacing them with a single backup cell per slot. The backup cell mechanism — copy before verify, restore after verify, tape replay accepted — provided the same rollback capability at ~150MB/slot instead of ~5.4GB/slot.

The **minimal implementation** to reproduce VRAM savings:

1. Exclude DFlash from `need_n_rs_seq()` — **one-line change, primary VRAM saving**.
2. Add backup cell allocation (`n_seq_max = n_parallel * 2`) — **moderate server change**.
3. Add backup copy before verification — **moderate server change**.
4. Add `dflash_rollback()` with 3-phase restore — **moderate src change**.
5. Add tape replay (or accept re-decode fallback) — **hard graph change or performance tradeoff**.

Without tape replay, the approach still saves VRAM but may lose some speculative throughput. With tape replay, both VRAM and performance are preserved.
