# Task 4.2: DFlash Backup Cell Allocation and Rollback Extension Points

## Overview

This document investigates **backup cell allocation strategy** (Section 3) and **rollback extension points** (Section 4) for the upstream DFlash implementation. It builds on Task 4.1 findings that draft/target contexts have separate recurrent memory and that recurrent-only backup/restore requires new APIs.

---

## SECTION 3: BACKUP CELL ALLOCATION STRATEGY

### 3.1 How Recurrent Cells Are Allocated

**File:** [`src/llama-memory-recurrent.cpp:20-128`](src/llama-memory-recurrent.cpp:20)

Recurrent cells are **fixed at construction time**. The constructor takes `mem_size` (total cell count) and `n_seq_max` (maximum sequence count):

```cpp
// Constructor signature (line 20-27)
llama_memory_recurrent(
    const llama_model & model,
            ggml_type   type_r,
            ggml_type   type_s,
                 bool   offload,
             uint32_t   mem_size,    // <-- total cells (FIXED)
             uint32_t   n_seq_max,   // <-- max sequences
             uint32_t   n_rs_seq,
    const layer_filter_cb & filter);
```

**Cell metadata allocation** (line 38-39):
```cpp
cells.clear();
cells.resize(mem_size);  // std::vector<mem_cell> — fixed size
```

**R/S tensor allocation** (line 99-105):
```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);  // <-- CRITICAL
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

The tensors are allocated with `mem_size * (1 + n_rs_seq)` rows. Each row is one cell's R or S state. The `(1 + n_rs_seq)` factor provides rollback snapshot groups.

**No dynamic resize exists.** There is no `llama_memory_recurrent::resize()` method in the current code. The `resize` method mentioned in the header (`bool resize(uint32_t new_mem_size);` at [`llama-memory-recurrent.h:130`](src/llama-memory-recurrent.h:130)) is declared private but the implementation was not found in the current source, suggesting it was either removed or never implemented.

### 3.2 What Is `n_seq_max` and How Is It Set

**File:** [`common/common.cpp:1769`](common/common.cpp:1769)

```cpp
cparams.n_seq_max = params.n_parallel;
```

The `n_seq_max` is set directly from `n_parallel` (the number of parallel user slots configured for the server).

**File:** [`src/llama-model.cpp:2148`](src/llama-model.cpp:2148) (hybrid-iswa path) and [`src/llama-model.cpp:2198`](src/llama-model.cpp:2198) (KVarN hybrid path) and [`src/llama-model.cpp:2215`](src/llama-model.cpp:2215) (standard hybrid path):

```cpp
/* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
/* n_seq_max         */ cparams.n_seq_max,
```

The recurrent memory `mem_size` parameter is set to `std::max(1, cparams.n_seq_max)`, which equals `n_parallel`. This means:

- **Cell count = n_parallel**
- **Sequence count = n_parallel**
- Each user slot gets exactly one cell

### 3.3 Can We Allocate `n_parallel * 2` Cells Staticly

**Yes, with modifications.** The cell count is determined by the `mem_size` parameter passed to `llama_memory_recurrent`. This flows through:

1. `common_params.n_parallel` →
2. `llama_context_params.n_seq_max` →
3. `llama_model::init()` creates `llama_memory_recurrent` with `mem_size = std::max(1, cparams.n_seq_max)`

To allocate `n_parallel * 2` cells:

**Option A — Simple multiplier in model.cpp:**
Change [`src/llama-model.cpp:2198`](src/llama-model.cpp:2198) from:
```cpp
std::max((uint32_t) 1, cparams.n_seq_max)
```
to:
```cpp
std::max((uint32_t) 1, cparams.n_seq_max) * 2  // double for backup cells
```

**Option B — New context parameter:**
Add `cparams.n_backup_seq` and compute `mem_size = n_seq_max + n_backup_seq`.

**Option C — Server-level override for DFlash:**
Detect DFlash mode in server initialization and pass a larger `n_seq_max` or new parameter.

**Critical constraint:** The `rs_idx` vector is sized to `n_seq_max` (line 36):
```cpp
rs_idx.assign(n_seq_max, 0);
```
If backup cells use sequence IDs `n_parallel .. 2*n_parallel-1`, then `rs_idx` must also be enlarged to `n_seq_max + n_backup_count`, or backup cells must not use `rs_idx` (they don't need rollback snapshots themselves).

Similarly, `n_seq_max` is used in [`llama-memory-recurrent.cpp:593-597`](src/llama-memory-recurrent.cpp:593) for seq_id validation in `find_slot()`:
```cpp
if (seq_id < 0 || (uint32_t) seq_id >= size) {
    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n",
        __func__, seq_id, n_seq_max);
    return false;
}
```

Note: This checks `seq_id >= size` (the cell count), NOT `seq_id >= n_seq_max`. The error message references `n_seq_max` but the guard uses `size`. This means:
- **Cell count (`size`) controls valid sequence IDs for cell access.**
- **`n_seq_max` controls `rs_idx` sizing.**
- If we increase `size` (cell count) but not `n_seq_max`, backup sequence IDs would pass the `size` check but fail the `rs_idx` bounds check at [`llama-memory-recurrent.cpp:474`](src/llama-memory-recurrent.cpp:474):
  ```cpp
  void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
      if (seq_id < 0 || (size_t) seq_id >= rs_idx.size()) {
          return;  // silent fail — would need to also enlarge rs_idx
      }
  }
  ```

### 3.4 Do Unused Cells Cost VRAM

**YES.** The R and S tensors are allocated with `mem_size * (1 + n_rs_seq)` rows regardless of whether cells are used:

```cpp
// Line 99-105
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

These tensors are immediately allocated to GPU memory:
```cpp
// Line 109-117
for (auto & [buft, ctx] : ctx_map) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
    ggml_backend_buffer_clear(buf.get(), 0);
    // ...
}
```

**VRAM impact of doubling cells:** If `n_parallel = 4` and each cell is ~256 MB (typical for Qwen3.6 with F32 R/S), doubling to 8 cells adds approximately 4 × 256 MB = 1 GB. For KVarN hybrid with `n_rs_seq > 0`, the formula is:

```
Total RS VRAM = mem_size * (1 + n_rs_seq) * (n_embd_r + n_embd_s) * sizeof(element) * n_layers_recurrent
```

Doubling `mem_size` doubles this VRAM.

### 3.5 Are Extra Cells Compatible With `n_rs_seq=0`

**Yes.** When `n_rs_seq = 0`:
- Tensor rows = `mem_size * (1 + 0)` = `mem_size` (no snapshot groups)
- The `rs_idx` vector is still allocated but unused for rollback
- The `can_seq_rm` path at [`llama-memory-recurrent.cpp:173-174`](src/llama-memory-recurrent.cpp:173) would reject rollback attempts beyond 0 tokens:
  ```cpp
  if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
      // this branch never executes when n_rs_seq == 0
  }
  ```

Extra cells would function as normal cells without rollback capability, which is fine for backup purposes since backup cells don't need their own rollback snapshots.

### 3.6 Can the Server Use Sequence IDs Outside Normal User-Slot Range

**Yes, with caveats.** The server assigns `slot.id` as the sequence ID for each user slot (range `0 .. n_parallel-1`). Backup cells at indices `n_parallel .. 2*n_parallel-1` would use sequence IDs outside the normal slot range.

**Validation points that must accommodate extended range:**

| Location | Check | Impact |
|----------|-------|--------|
| [`llama-memory-recurrent.cpp:593`](src/llama-memory-recurrent.cpp:593) | `seq_id >= size` | OK if `size` is doubled |
| [`llama-memory-recurrent.cpp:474`](src/llama-memory-recurrent.cpp:474) | `seq_id >= rs_idx.size()` | Need to enlarge `rs_idx` or skip `set_rs_idx` for backup |
| [`llama-memory-recurrent.cpp:158`](src/llama-memory-recurrent.cpp:158) | `seq_id >= size` in `can_seq_rm` | OK if `size` is doubled |
| [`llama-memory-recurrent.cpp:329`](src/llama-memory-recurrent.cpp:329) | `seq_id_dst < size` in `seq_cp` | OK if `size` is doubled |
| Server slot management | `slot.id` used as seq_id | Backup cells would use synthetic seq_ids not tied to slots |

**Key insight:** The `cells` array doubles as both cell storage AND sequence-metadata index. At index `seq_id`, `cells[seq_id].tail` points to the current working cell for that sequence. If backup cells use seq_ids `n_parallel .. 2*n_parallel-1`, the corresponding `cells[seq_id]` entries would serve as metadata for backup sequences.

### 3.7 Can Backup Cells Be Persistent and Reused

**Yes.** The proposed static approach:

- **Normal cells:** `0 .. n_parallel-1` — assigned to user slots at server startup
- **Backup cells:** `n_parallel .. 2*n_parallel-1` — persistent, never assigned to slots

The backup cells would be:
1. Pre-allocated at construction (no dynamic expansion)
2. Populated by `seq_cp(normal_seq_id, backup_seq_id, 0, -1)` before verification
3. Restored to normal via `seq_cp(backup_seq_id, normal_seq_id, 0, -1)` or R/S data copy after rollback

No dynamic expansion/shrinking machinery needed. The cells exist from startup and are reused every verification cycle.

### 3.8 Summary: Static Extra Cells Approach

| Question | Answer |
|----------|--------|
| Fixed or dynamic allocation? | Fixed at construction |
| What is `n_seq_max`? | Equals `n_parallel`; controls cell count and `rs_idx` size |
| Can we allocate `n_parallel * 2`? | Yes — modify `mem_size` in `llama-model.cpp` |
| Do unused cells cost VRAM? | Yes — tensors allocated for all `mem_size` rows |
| Compatible with `n_rs_seq=0`? | Yes |
| Server can use extended seq IDs? | Yes — if `size` and `rs_idx` are both enlarged |
| Persistent and reusable? | Yes — static allocation, no dynamic machinery |
| Additional changes needed? | Enlarge `rs_idx`, `samplers`, and `samplers_seq_config` arrays in `common.cpp` |

---

## SECTION 4: ROLLBACK EXTENSION POINT

### 4.1 Current Upstream DFlash Rollback Flow

The complete rollback flow spans three files:

**File 1: [`tools/server/server-context.cpp`](tools/server/server-context.cpp)** — Server-level verification and acceptance.

**File 2: [`common/speculative.cpp`](common/speculative.cpp)** — DFlash draft/accept implementation.

**File 3: [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp)** — Recurrent state manipulation.

### 4.2 The Exact Decision Point for "Rollback N Tokens"

**File:** [`tools/server/server-context.cpp:4219`](tools/server/server-context.cpp:4219)

```cpp
const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
```

This is computed immediately after `common_sampler_sample_and_accept_n()` returns the `accepted` token list. The `n_rollback` value determines how many draft tokens were rejected.

**Checkpoint decision** (line 4221-4222):
```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);
```

**File:** [`tools/server/server-task.h:20-26`](tools/server/server-task.h:20)
```cpp
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
          (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

### 4.3 Current Rollback Execution (Two Paths)

**Path A — Checkpoint-based rollback** (line 4226-4263):

When `use_ckpt_tgt` is true (rollback exceeds `n_rs_seq` or seq_rm is FULL-only):

```cpp
if (use_ckpt_tgt) {
    // Restore full state from checkpoint
    ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);

    if (slot.ctx_dft) {
        ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
    }

    // Restore sampler, loop guard, prompt, etc.
    slot.smpl = std::move(smpl_save);
    // ...
    return;  // <-- early return, skips Path B
}
```

This path restores **ALL state** (KV cache + recurrent + sampler) from the checkpoint. It is expensive.

**Path B — In-place rollback via `seq_rm`** (line 4271):

When `use_ckpt_tgt` is false (rollback within `n_rs_seq` bound):

```cpp
// Line 4271
common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
```

Followed by (line 4319):
```cpp
common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
```

This path relies on the recurrent memory's native rollback mechanism:
- `seq_rm(seq_id, p0, p1)` is called with `p0 = pos_next` (the position after the last accepted token)
- For recurrent memory, this triggers the rollback logic at [`llama-memory-recurrent.cpp:215-221`](src/llama-memory-recurrent.cpp:215):
  ```cpp
  if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
      const llama_pos rollback = cell.pos - (p0 - 1);
      if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
          set_rs_idx(seq_id, (uint32_t) rollback);
          cell.pos = p0 - 1;
          return true;
      }
  }
  ```
- The `rs_idx` is set to the rollback depth, and the graph builder uses this index to read from the correct snapshot group in the R/S tensors.

### 4.4 Where We Could Substitute Backup Cell Restore

The substitution point is **between the checkpoint decision and the in-place rollback**:

```
Current flow:
  n_rollback > 0
  ├── use_ckpt_tgt == true  → ckpt.load_tgt() (full state restore)
  └── use_ckpt_tgt == false → seq_rm() + rs_idx (native rollback)

Proposed flow:
  n_rollback > 0
  ├── use_ckpt_tgt == true  → ckpt.load_tgt() (unchanged — safety net)
  ├── use_backup_cell == true → seq_cp(backup → working) (NEW)
  └── fallback                → seq_rm() + rs_idx (current native rollback)
```

**The exact insertion point** is at [`tools/server/server-context.cpp:4225`](tools/server/server-context.cpp:4225), just before the `if (use_ckpt_tgt)` block:

```cpp
// Line 4225 (current):
if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        // checkpoint restore
    }
}

// Proposed modification:
if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        // checkpoint restore (unchanged)
    } else if (backup_cell_available) {
        // NEW: Restore recurrent state from backup cell
        common_context_seq_cp(
            slot.ctx_tgt, backup_seq_id, slot.id, 0, -1);
        // KV cache: remove rejected tokens (existing seq_rm handles this)
    }
    // else: fall through to existing seq_rm + rs_idx path
}
```

### 4.5 Can Rollback Be Changed to "KV: retain accepted, Recurrent: copy backup → working"

**Conceptually yes, but with implementation details:**

**KV cache side:** The current `common_context_seq_rm()` already handles "remove rejected tokens from position X onward." This is unchanged.

**Recurrent side:** The current mechanism uses `rs_idx` to select snapshot groups within the same cell's R/S tensor. The backup cell approach would instead:

1. **Before verification:** `seq_cp(working_seq_id, backup_seq_id, 0, -1)` — copies cell metadata AND triggers R/S data copy through the graph builder's src/src0 mechanism.

2. **After rejection:** `seq_cp(backup_seq_id, working_seq_id, 0, -1)` — restores working cell from backup.

**The challenge:** `seq_cp()` at [`llama-memory-recurrent.cpp:316-351`](src/llama-memory-recurrent.cpp:316) **only copies cell metadata** (tail pointer, seq_id set), NOT R/S tensor data:

```cpp
void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // Only updates cells[seq_id_dst].tail to point to cells[seq_id_src].tail
    // Does NOT copy R/S tensor data.
}
```

The R/S data "copy" happens implicitly through the graph builder's `src`/`src0` mechanism: when a cell's `src` field points to another cell index, the graph builder generates operations that copy from the source cell's tensor row to the destination cell's tensor row during the next decode graph evaluation.

This means `seq_cp(backup, working, 0, -1)` would make the working cell's `src` point to the backup cell, and the R/S data would be copied **during the next forward pass**, not immediately. For rollback, we need **immediate** data restoration.

**Two approaches:**

**Approach 1 — Use `state_write`/`state_read` for immediate copy:**
```cpp
// Pseudo-code for immediate backup → working restore:
{
    std::vector<uint8_t> buffer;
    // Write backup cell state to buffer
    llama_memory_recurrent * mem = get_recurrent(ctx_tgt);
    // ... use internal state_write to buffer, then state_read to working cell
}
```
This requires exposing internal `state_write`/`state_read` through a public API or adding a new `cell_copy` method.

**Approach 2 — Add `llama_memory_recurrent::cell_copy(src_cell, dst_cell)` method:**
```cpp
void llama_memory_recurrent::cell_copy(uint32_t src, uint32_t dst) {
    // Directly copy R/S tensor rows from src to dst using ggml_backend_tensor_set
    for (int il = 0; il < n_layer; il++) {
        if (r_l[il] && s_l[il]) {
            ggml_backend_tensor_set_row(r_l[il], dst, src);
            ggml_backend_tensor_set_row(s_l[il], dst, src);
        }
    }
    // Copy cell metadata
    cells[dst].pos = cells[src].pos;
    cells[dst].src = cells[src].src;
    cells[dst].seq_id = cells[src].seq_id;
}
```

### 4.6 Would This Require Touching DFlash Speculative Algorithm?

**No.** The DFlash speculative algorithm in [`common/speculative.cpp`](common/speculative.cpp) is concerned with:
- Draft token generation (`draft()` method)
- Post-verification statistics (`accept()` method — currently a no-op for DFlash)

The rollback mechanism is **orthogonal** to the DFlash algorithm. The server decides how to rollback after `common_sampler_sample_and_accept_n()` returns, and the DFlash implementation does not participate in that decision.

The only connection is that `common_speculative_accept()` at [`common/speculative.cpp:2633`](common/speculative.cpp:2633) is called after rollback, but it only updates statistics and calls `impl->accept()` which is a no-op for DFlash.

### 4.7 Server-Context Functions That Would Need Modification

| Function | File | Line | Change |
|----------|------|------|--------|
| `server_context_impl::update_generation()` | [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | ~4225 | Add backup-cell rollback path |
| `server_context_impl::process_speculative()` | [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | ~3305 | Add backup-cell copy before verification |
| New: `common_context_cell_copy()` | [`common/common.cpp`](common/common.cpp) | — | Public API for cell-to-cell R/S copy |
| New: `llama_memory_recurrent::cell_copy()` | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | — | Direct tensor row copy |
| `llama_memory_recurrent` constructor | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | 20-128 | Accept extended `mem_size` |
| `llama_model::init()` | [`src/llama-model.cpp`](src/llama-model.cpp) | ~2198 | Pass doubled `mem_size` for DFlash |
| `common_context_params_to_llama()` | [`common/common.cpp`](common/common.cpp) | ~1769 | Optionally expose backup cell count |

### 4.8 Complete Modified Rollback Flow

```
Before verification (server-context.cpp ~3305):
  ┌─────────────────────────────────────────────┐
  │ if (backup_cell_enabled && !draft.empty()) { │
  │     // Save working cell state to backup     │
  │     common_context_cell_copy(                │
  │         ctx_tgt, slot.id, backup_seq_id);    │
  │ }                                            │
  └─────────────────────────────────────────────┘
  ↓
  Verification (common_sampler_sample_and_accept_n)
  ↓
After verification (server-context.cpp ~4225):
  ┌─────────────────────────────────────────────┐
  │ if (n_rollback > 0) {                        │
  │     if (use_ckpt_tgt) {                      │
  │         // Existing checkpoint path          │
  │         ckpt.load_tgt(...);                  │
  │     } else if (backup_cell_enabled) {        │
  │         // NEW: Restore from backup cell     │
  │         common_context_cell_copy(            │
  │             ctx_tgt, backup_seq_id, slot.id);│
  │         // KV already handled by seq_rm      │
  │     }                                        │
  │     // else: native rs_idx rollback          │
  │ }                                            │
  └─────────────────────────────────────────────┘
```

---

*(Continued in part 2b — see next write for cost analysis and recommendation.)*
