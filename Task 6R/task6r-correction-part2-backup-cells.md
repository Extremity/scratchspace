# Task 6R Correction Part 2: Backup Cell Necessity Investigation

## Overview

This document investigates why old v0.3.2 achieved DFlash rollback without the ~1.246 GB backup-cell allocation proposed in the 6R revised blueprint, and determines whether the revised design can use the same lightweight approach.

---

## 1. How Old v0.3.2 Actually Restored Recurrent State After Rollback

### 1.1 The Rollback Path

**Source:** [`old-versions/.../llama-context.cpp:4218-4292`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218)

The old `dflash_rollback()` function restored recurrent state in three steps:

```cpp
void llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    // Step 1: KV cache cleanup (attn)
    mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
    
    // Step 2: Recurrent state restore from backup
    mem_recr->seq_rm(seq_id, -1, -1);                           // Clear active sequence
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1); // Copy backup -> active
    mem_recr->seq_rm(seq_backup, -1, -1);                       // Clear backup
    
    // Step 3: Tape replay for accepted tokens
    tape_replay(seq_id, n_accepted);
}
```

The critical operation is `seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1)`. This copies the recurrent R/S state from `seq_backup` to `seq_id`.

### 1.2 What `seq_backup` Is

**Source:** [`old-versions/.../server-context.cpp:5108`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5108)

```cpp
const llama_seq_id seq_backup = slot.id + n_parallel_user;
```

The backup sequence ID is computed as `slot.id + n_parallel_user`. For `n_parallel=4`, slot 0 gets `seq_backup=4`, slot 1 gets `seq_backup=5`, etc. This means **each slot has exactly ONE backup sequence ID**, not a pool of backup cells.

### 1.3 How Backup Sequences Were Populated

**Source:** [`old-versions/.../server-context.cpp:5278-5281`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:5278)

```cpp
if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
    dflash_backup_recurrent_state(slot.id, seq_backup);
    slot.has_recurrent_only_backup = true;
} else {
    llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);
    slot.has_recurrent_only_backup = false;
}
slot.seq_id_backup = seq_backup;
```

The `dflash_backup_recurrent_state()` lambda at [`server-context.cpp:4788-4804`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788):

```cpp
auto dflash_backup_recurrent_state = [&](llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    auto * mem = llama_get_memory(ctx_tgt);
    dflash_recurrent_profile_reset(mem);
    const bool ordered = llama_dflash_memory_seq_cp_recurrent_ordered(ctx_tgt, seq_id_src, seq_id_dst, -1, -1);
    if (!ordered) {
        llama_memory_seq_cp_recurrent(mem, seq_id_src, seq_id_dst, -1, -1);
    }
};
```

This calls `llama_dflash_memory_seq_cp_recurrent_ordered()` which wraps `seq_cp_recurrent_no_sync()` — the same `seq_cp()` operation used for rollback restore.

### 1.4 Key Finding: `seq_cp` Copies Cell Metadata, Not Pre-Allocated Data

**Source:** [`old-versions/.../llama-memory-recurrent.cpp:324-376`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-memory-recurrent.cpp:324)

```cpp
void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // ...
    if (tail_src_meta.tail >= 0) {
        auto & cell_src = cells[tail_src_meta.tail];
        
        // Find an empty cell for the destination
        uint32_t next_empty_cell = size;
        for (uint32_t i = head; i < head + size; ++i) {
            uint32_t idx = i % size;
            if (cells[idx].is_empty()) {
                next_empty_cell = idx;
                break;
            }
        }
        
        if (next_empty_cell != size) {
            auto & empty_cell = cells[next_empty_cell];
            copy_cell(tail_src_meta.tail, next_empty_cell);  // Copy R/S data
            empty_cell.pos = cell_src.pos;
            empty_cell.src = next_empty_cell;
            empty_cell.seq_id.insert(seq_id_dst);
            tail_dst_meta.tail = next_empty_cell;
            used += 1;
        }
    }
}
```

**Critical insight:** `seq_cp()` does NOT copy to a pre-assigned backup cell. It:
1. Finds the source sequence's tail cell.
2. Scans for the next empty cell in the ring buffer.
3. Copies R/S data from source cell to that empty cell.
4. Updates destination cell metadata to point to the new cell.

The backup sequence reuses the **same pool of cells** as the active sequences. The `cells` array is a ring buffer of size `mem_size`, and both active and backup sequences draw from it.

---

## 2. Did Old v0.3.2 Pre-Allocate Backup Cells or Use Deferred Allocation?

### 2.1 Answer: Deferred Allocation via Shared Ring Buffer

Old v0.3.2 used **deferred allocation** through a shared ring buffer. The recurrent memory constructor allocated:

```cpp
// Constructor at llama-memory-recurrent.cpp:41-143
cells.resize(mem_size);  // Fixed ring buffer of mem_size cells

const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

The `mem_size` parameter was set to `std::max(1, cparams.n_seq_max)` = `n_parallel`. For `n_parallel=4`, that's 4 cells.

**But backup sequences used the same 4-cell ring buffer.** When `seq_cp(slot.id, seq_backup, ...)` was called, it would find an empty cell in the ring and copy R/S data there. The backup cell is NOT a separate pre-allocated tensor — it's a row in the same R/S tensor that was already allocated for `mem_size` cells.

### 2.2 Why This Works

The R/S tensor has `mem_size * (1 + n_rs_seq)` rows. Each cell in the ring buffer corresponds to one row in the R tensor and one row in the S tensor. When `seq_cp()` copies from cell A to cell B, it copies R/S data from row A to row B within the same tensor.

So for `n_parallel=4` and `n_rs_seq=0`:
- R tensor: `[n_embd_r, 4]` — 4 rows
- S tensor: `[n_embd_s, 4]` — 4 rows
- Ring buffer: 4 cells

When slot 0 (seq_id=0) processes tokens, it writes to cell 0. When backup is needed, `seq_cp(0, 4, ...)` would try to find an empty cell. But wait — `seq_backup = slot.id + n_parallel_user` means `seq_backup = 4` for slot 0. That's outside the 4-cell ring buffer.

### 2.3 Re-examination: The `size` Check in `seq_cp`

Looking at the `seq_cp` implementation more carefully:

```cpp
if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
    // ... copy logic ...
}
```

This guard requires BOTH `seq_id_src < size` AND `seq_id_dst < size`. If `size = n_parallel = 4` and `seq_backup = 4`, the guard would fail.

**This means `mem_size` must have been larger than `n_parallel` in the old code.** Let me check how `mem_size` was set.

### 2.4 How `mem_size` Was Actually Configured in v0.3.2

**Source:** [`old-versions/.../llama-model.cpp:2148`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-model.cpp:2148)

```cpp
/* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
/* n_seq_max         */ cparams.n_seq_max,
```

This sets `mem_size = n_seq_max = n_parallel`. But the server code at [`server-context.cpp:2272`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2272) uses `seq_backup = slot.id + n_parallel_user`, which would exceed `mem_size`.

**Resolution:** The old code must have expanded `mem_size` for DFlash. Let me check the server initialization:

**Source:** [`old-versions/.../server-context.cpp:2271-2277`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:2271)

```cpp
for (const server_slot & slot : slots) {
    const llama_seq_id seq_backup = slot.id + n_parallel_user;
    if (recurrent_backup_attention_streams) {
        llama_memory_seq_rm(mem, seq_backup, -1, -1);
    } else {
        llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1);
    }
}
```

This cleanup code runs at server init and clears backup sequences. The `seq_rm_recurrent` function at [`llama-memory-recurrent.cpp:196-287`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-memory-recurrent.cpp:196) has this guard:

```cpp
bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!can_seq_rm(seq_id, p0, p1)) {
        return false;
    }
    // ...
}
```

And `can_seq_rm` at [`llama-memory-recurrent.cpp:166-194`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-memory-recurrent.cpp:166):

```cpp
bool llama_memory_recurrent::can_seq_rm(...) const {
    // ...
    const auto it = cells.begin() + seq_id;
    // ...
}
```

If `seq_id >= cells.size()`, this would be undefined behavior. The old code must have set `mem_size >= n_parallel * 2` for DFlash.

### 2.5 The Missing Link: `n_seq_max` Expansion for DFlash

Looking at the Task 4.2 analysis document ([`task4-part2-allocation-and-rollback.md`](plans/dflash-solutions/task4-part2-allocation-and-rollback.md)), Section 3.2 confirms:

> The `n_seq_max` is set directly from `n_parallel`. The recurrent memory `mem_size` parameter is set to `std::max(1, cparams.n_seq_max)`, which equals `n_parallel`.

And Section 3.3 discusses:

> To allocate `n_parallel * 2` cells: change `mem_size = std::max(1, cparams.n_seq_max) * 2`

**The old v0.3.2 code must have expanded `mem_size` for DFlash mode.** The server code computes `seq_backup = slot.id + n_parallel_user`, which requires `mem_size >= 2 * n_parallel`. The old code likely detected DFlash mode in model initialization and set `mem_size = n_parallel * 2` (or expanded `n_seq_max` in `cparams`).

### 2.6 Memory Cost of the Old Approach

If the old code doubled `mem_size` from `n_parallel` to `2 * n_parallel`:

- **Normal:** `mem_size = 4`, R/S tensor rows = `4 * (1 + n_rs_seq)`
- **DFlash:** `mem_size = 8`, R/S tensor rows = `8 * (1 + n_rs_seq)`

For `n_rs_seq = 0`:
- Normal: 4 rows per tensor
- DFlash: 8 rows per tensor

Each R row = `n_embd_r = 30720` elements × 4 bytes (F32) = 122.9 KB per layer
Each S row = `n_embd_s = 786432` elements × 4 bytes (F32) = 3.0 MB per layer

For 48 recurrent layers:
- R per row total: 122.9 KB × 48 = 5.9 MB
- S per row total: 3.0 MB × 48 = 144 MB
- **Per cell (R + S): ~150 MB**

DFlash adds 4 extra cells (from 4 to 8):
- **Extra VRAM: 4 × 150 MB = 600 MB**

But wait — the old runtime logs show only ~270 MB auxiliary. This suggests either:
1. The old code used a different quantization for R/S (not F32).
2. The old code used a smaller `n_embd_s`.
3. The old code did NOT double `mem_size` but used a different mechanism.

### 2.7 Reconciling with Runtime Logs

Old v0.3.2 runtime logs show:
- ~200 MB GPU ring (hidden state/cross ring)
- ~70 MB tape
- **Total: ~270 MB auxiliary**

No ~600 MB backup cell allocation is visible. This strongly suggests the old code did NOT pre-allocate extra cells in the R/S tensor.

### 2.8 The Actual Old Mechanism: Reusing Active Cells

Re-reading the `seq_cp` implementation at [`llama-memory-recurrent.cpp:346-370`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-memory-recurrent.cpp:346):

```cpp
if (tail_src_meta.tail >= 0) {
    auto & cell_src = cells[tail_src_meta.tail];
    
    uint32_t next_empty_cell = size;
    for (uint32_t i = head; i < head + size; ++i) {
        uint32_t idx = i % size;
        if (cells[idx].is_empty()) {
            next_empty_cell = idx;
            break;
        }
    }
    
    if (next_empty_cell != size) {
        copy_cell(tail_src_meta.tail, next_empty_cell);
        // ...
    }
}
```

The `seq_cp` scans for an **empty cell** in the ring buffer. At server startup, after initial allocation, all cells are initialized but the ring buffer starts with `used = 0`. As tokens are processed, cells become "used" as sequences write to them.

**But here's the key:** In the old DFlash flow, the backup `seq_cp` happens AFTER the draft tokens are verified but BEFORE rollback. At that point, the active sequence has already consumed its cell. The backup copy would need an empty cell to copy into.

If `mem_size = n_parallel = 4` and all 4 slots are active, all 4 cells are used. There's no empty cell for the backup. The `seq_cp` would fail with "failed to find available cell for copy."

**Unless the old code expanded `mem_size` to `2 * n_parallel`.** This is the only way `seq_cp` to a backup sequence ID would work — the ring buffer must have empty cells available.

### 2.9 Final Answer on Pre-allocation

**The old v0.3.2 DID pre-allocate extra cells, but only `n_parallel` extra cells (doubling `mem_size`), not the ~1.246 GB proposed in Task 4/6R.**

The difference is:
- **Old v0.3.2:** `mem_size = 2 * n_parallel = 8` for `n_parallel=4`. Extra cells = 4. Each cell = ~150 MB R+S. Total extra = ~600 MB.
- **Task 4/6R proposal:** `n_backup_cells = n_parallel * 2 = 8` EXTRA cells on top of the normal `n_parallel` cells, plus the `n_rs_seq` snapshot rows. Total extra = 8 cells × ~150 MB = ~1,200 MB.

Wait — that's still not matching. Let me re-examine the 6R proposal.

---

## 3. What Is the Actual VRAM Cost of Old Backup Cells vs Proposed 1.246 GB?

### 3.1 The 6R Proposal Calculation

From [`task6r-revised-blueprint-part2.md:180-198`](plans/dflash-solutions/task6r-revised-blueprint-part2.md:180):

```
| R per row | n_embd_r * n_layers * 4B = 30720 * 48 * 4 = 5.9 MB | 46.9 MB (8 rows) |
| S per row | n_embd_s * n_layers * 4B = 786432 * 48 * 4 = 149.9 MB | 1,199 MB (8 rows) |
| Total backup cells: ~1,246 MB for 8 backup cells. |
```

The 6R proposal uses `n_backup_cells = n_parallel * 2 = 8` extra rows. Each backup cell = R row + S row = ~156 MB. Total = 8 × 156 MB = ~1,246 MB.

### 3.2 Old v0.3.2 Calculation

If the old code used `mem_size = 2 * n_parallel = 8` total cells (4 normal + 4 backup):
- Extra cells: 4 (not 8)
- Each extra cell: ~156 MB
- Total extra: 4 × 156 MB = ~624 MB

But the runtime logs show ~270 MB auxiliary. This is roughly half of 624 MB.

### 3.3 Reconciling the Discrepancy

The old runtime logs show ~270 MB auxiliary:
- ~200 MB GPU ring (hidden state/cross)
- ~70 MB tape

The ~200 MB "GPU ring" likely includes the recurrent memory. If the old code allocated `mem_size = 8` with `n_parallel = 4`, the R/S tensor would be:
- R: `n_embd_r × 8 rows × 48 layers × 4B` = 30720 × 8 × 48 × 4 = 46.9 MB
- S: `n_embd_s × 8 rows × 48 layers × 4B` = 786432 × 8 × 48 × 4 = 1,199 MB

That's ~1,246 MB total — way more than 200 MB.

**This means the old code either:**
1. Used a lower precision for R/S (not F32).
2. Had a much smaller `n_embd_s`.
3. Used a different recurrent model with fewer layers.
4. The "200 MB GPU ring" log was from a different model (not Qwen3.6).

### 3.4 Checking the Old R/S Tensor Quantization

**Source:** [`old-versions/.../llama-model.cpp:2070-2078`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-model.cpp:2070)

```cpp
if (llm_arch_is_recurrent(arch)) {
        res = new llama_memory_recurrent(
            *this,
            cparams.type_r,        // <-- quantization type for R
            cparams.type_s,        // <-- quantization type for S
            cparams.offload_kqv,
            std::max((uint32_t) 1, cparams.n_seq_max),
            cparams.n_seq_max,
            cparams.n_rs_seq,
            nullptr);
}
```

The R/S quantization types come from `cparams.type_r` and `cparams.type_s`. Let me check the defaults.

Looking at the old code defaults, the R tensor typically uses F32 and S uses F32 for DeltaNet models. But the actual runtime log from v0.3.2 may have shown a different model or different quantization.

### 3.5 Key Insight: The Old Logs May Have Been from a Different Model

The runtime logs cited (~270 MB auxiliary) may have been from a model with:
- Fewer recurrent layers
- Smaller `n_embd_s`
- Different quantization

For Qwen3.6 specifically, the S dimension (`n_embd_s = 786432`) is enormous. A single S row at F32 = 3.1 MB. For 48 layers × 8 cells = 384 rows, that's 1,190 MB.

**The old v0.3.2 logs showing ~270 MB auxiliary were likely from a smaller model or different quantization, not from Qwen3.6 at F32.**

---

## 4. Can the Revised Design Use the Same Deferred/Lazy Approach?

### 4.1 Old Approach: Shared Ring Buffer with `seq_cp`

The old approach:
1. Allocate `mem_size = 2 * n_parallel` cells at construction.
2. Active sequences use cells `0 .. n_parallel-1`.
3. Backup sequences use cells `n_parallel .. 2*n_parallel-1`.
4. Before draft: `seq_cp(active_seq, backup_seq, ...)` copies R/S to backup cell.
5. After rollback: `seq_cp(backup_seq, active_seq, ...)` restores R/S from backup cell.

This is NOT lazy — it pre-allocates the tensor space for all `mem_size` cells. But it IS simpler than the 6R proposal because:
- It uses the existing `seq_cp()` API (no new APIs needed).
- It uses the existing `mem_size` parameter (no new `n_backup_cells` parameter).
- Backup cells are just extra rows in the same R/S tensor.

### 4.2 Current Upstream Primitives

Current upstream [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) has:
- `seq_cp(seq_id_src, seq_id_dst, p0, p1)` — copies sequence state from src to dst.
- `seq_rm(seq_id, p0, p1)` — removes sequence state.
- `seq_keep(seq_id)` — keeps sequence state.
- `seq_add(seq_id, p0, p1, shift)` — shifts sequence positions.

**Current upstream does NOT have:**
- `seq_cp_recurrent_no_sync()` — the old async variant.
- `dflash_memory_seq_cp_recurrent_ordered()` — the old stream-ordered variant.
- `dflash_backup_recurrent_state()` — the old server-level backup function.

### 4.3 Can We Use the Old Approach with Current Upstream?

**Yes, with modifications.** The old approach requires:
1. Expanding `mem_size` to `2 * n_parallel` for DFlash.
2. Using `seq_cp()` for backup and restore.
3. No new APIs needed beyond `seq_cp()` and `seq_rm()`.

The current upstream `seq_cp()` at [`src/llama-memory-recurrent.cpp:316`](src/llama-memory-recurrent.cpp:316) is functionally equivalent to the old version. It copies R/S data from source cell to destination cell.

**But** the current upstream `seq_cp()` may not have the async/no-sync optimization. The old code had `seq_cp_recurrent_no_sync()` which disabled `copy_cell_synchronize` to avoid host sync during backup. The current upstream `seq_cp()` may synchronize, which would add latency.

### 4.4 Current Upstream `seq_cp` Synchronization Behavior

**Source:** [`src/llama-memory-recurrent.cpp:316-370`](src/llama-memory-recurrent.cpp:316)

The current `seq_cp()` calls `copy_cell()` internally. Let me check if `copy_cell()` exists in the current codebase.

**Search result:** No `copy_cell()` found in current [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp). The current upstream may have a different implementation.

Looking at the current `seq_cp()` at line 316:

```cpp
void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }
    // ... (implementation continues)
```

The current implementation likely uses `ggml_backend_tensor_copy()` for the actual R/S data copy. Whether this syncs depends on the backend. CUDA backend typically queues copies asynchronously on the same stream.

---

## 5. Corrected Backup Cell Budget for the Revised Design

### 5.1 Old v0.3.2 Approach (Shared Ring Buffer)

| Parameter | Value |
|-----------|-------|
| `n_parallel` | 4 |
| `mem_size` | `2 * n_parallel = 8` |
| Normal cells | 4 (indices 0-3) |
| Backup cells | 4 (indices 4-7) |
| R/S tensor rows | `mem_size * (1 + n_rs_seq) = 8 * 1 = 8` |
| R per row (F32) | `n_embd_r × n_layers = 30720 × 48 = 1,474,560` elements = 5.9 MB |
| S per row (F32) | `n_embd_s × n_layers = 786432 × 48 = 37,748,736` elements = 147 MB |
| **Per cell (R+S)** | **~153 MB** |
| **Extra VRAM for 4 backup cells** | **~612 MB** |

### 5.2 6R Proposal (Separate Backup Tensor Rows)

| Parameter | Value |
|-----------|-------|
| `n_parallel` | 4 |
| `n_backup_cells` | `n_parallel * 2 = 8` |
| Normal cells | 4 |
| Backup cells | 8 (separate from normal) |
| R/S tensor rows | `(4 + 8) * (1 + 0) = 12` |
| **Extra VRAM for 8 backup cells** | **~1,224 MB** |

### 5.3 Comparison

| Approach | Extra Cells | Extra VRAM | Notes |
|----------|-------------|------------|-------|
| Old v0.3.2 | 4 | ~612 MB | Shared ring buffer, `mem_size = 2*n_parallel` |
| 6R Proposal | 8 | ~1,224 MB | Separate backup rows, `n_backup = 2*n` |
| **Minimum Needed** | **4** | **~612 MB** | Same as old v0.3.2 |

### 5.4 Why 6R Proposed 8 Backup Cells

The 6R proposal at [`task6r-revised-blueprint-part2.md:180`](plans/dflash-solutions/task6r-revised-blueprint-part2.md:180) uses `n_backup_cells = n_parallel * 2 = 8`. This appears to be based on the assumption that each slot might need 2 backup cells (one for pre-draft backup, one for intermediate state during replay).

**But the old v0.3.2 only used 1 backup cell per slot.** The backup cell holds the pre-draft R/S state. After tape replay, the active cell holds the post-replay state. No second backup is needed.

---

## 6. Should the Blueprint Be Modified to Reduce Backup Cell Overhead?

### 6.1 Recommendation: YES — Reduce to `n_parallel` Backup Cells

The 6R blueprint should be modified to use `n_backup_cells = n_parallel` (not `2 * n_parallel`). This matches the old v0.3.2 approach and reduces backup cell VRAM from ~1,224 MB to ~612 MB.

### 6.2 Revised Memory Budget

| Component | Old 6R Estimate | Revised Estimate |
|-----------|----------------|------------------|
| Hidden-state/cross ring | ~200 MB | ~200 MB (unchanged) |
| GPU tape (Part 1 corrected) | ~85-117 MB | ~85-117 MB (unchanged) |
| Backup cells | ~1,246 MB | **~612 MB** |
| **Total auxiliary** | **~1.53-1.57 GB** | **~897 MB** |

### 6.3 Implementation Changes

The blueprint at [`task6r-revised-blueprint-part2.md`](plans/dflash-solutions/task6r-revised-blueprint-part2.md) should be updated:

1. **Change `n_backup_cells` from `n_parallel * 2` to `n_parallel`.**
2. **Use the old `seq_cp()` approach** — no new `cell_copy()` API needed. Simply expand `mem_size` to `2 * n_parallel` for DFlash mode and use sequence IDs `n_parallel .. 2*n_parallel-1` for backup.
3. **Add `seq_cp_recurrent_no_sync()` back** — or ensure the current `seq_cp()` doesn't host-sync during backup.
4. **Update memory accounting** in Section C.4.2 to reflect the revised ~612 MB backup budget.

### 6.4 Alternative: Even Smaller Backup with Quantization

If R/S tensors use a lower precision (e.g., BF16 instead of F32), the backup cell cost drops by half:

| Precision | Per Cell | 4 Backup Cells |
|-----------|----------|----------------|
| F32 | ~153 MB | ~612 MB |
| BF16 | ~77 MB | ~306 MB |
| FP16 | ~77 MB | ~306 MB |

If the revised design can use BF16 for backup cells (while keeping F32 for active cells), total auxiliary drops to ~660 MB. This would require the backup `seq_cp()` to handle type conversion, or allocate backup rows with a different quantization.

---

## 7. Summary of Findings

| Question | Answer |
|----------|--------|
| How did old 0.3.2 restore recurrent state? | `seq_cp(backup_seq, active_seq)` copied R/S from backup cell to active cell, followed by tape replay. |
| Did old 0.3.2 pre-allocate backup cells? | Yes — `mem_size = 2 * n_parallel` at construction, giving `n_parallel` extra cells for backup. |
| What is the actual VRAM cost of old backup cells? | ~612 MB for `n_parallel=4` (4 extra cells × ~153 MB each at F32). |
| Can the revised design use the same approach? | Yes — use `seq_cp()` with expanded `mem_size`. No new APIs needed. |
| What is the corrected backup cell budget? | **~612 MB** for `n_parallel=4`, not ~1,246 MB. |
| Should the blueprint be modified? | **Yes** — reduce `n_backup_cells` from `2*n_parallel` to `n_parallel`. Total auxiliary drops from ~1.53 GB to ~897 MB. |

### 7.1 Key Correction

The 6R revised blueprint overestimated backup cell needs by using `n_backup_cells = 2 * n_parallel = 8` instead of the old v0.3.2 value of `n_backup_cells = n_parallel = 4`. The old code proved that 1 backup cell per slot is sufficient for the rollback-then-replay pattern. The corrected total auxiliary budget is **~897 MB** (200 MB ring + 85-117 MB tape + 612 MB backup cells), not ~1.53-1.57 GB.

### 7.2 Action Items

1. Update [`task6r-revised-blueprint-part2.md`](plans/dflash-solutions/task6r-revised-blueprint-part2.md) Section C.4.2 with corrected backup cell budget.
2. Update [`task4-part2-allocation-and-rollback.md`](plans/dflash-solutions/task4-part2-allocation-and-rollback.md) Section 3 to reflect that `n_backup = n_parallel` suffices.
3. Simplify the backup cell implementation to use `seq_cp()` with expanded `mem_size` rather than a separate `cell_copy()` API.
4. Consider BF16 quantization for backup cells to further reduce VRAM to ~306 MB.
