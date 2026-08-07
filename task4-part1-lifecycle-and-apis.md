# Task 4.1: DFlash Recurrent-State Lifecycle and Backup/Restore APIs

## Overview

This document investigates the **current upstream DFlash recurrent-state lifecycle** and determines whether **recurrent-only backup/restore is already possible** with existing APIs.

---

## SECTION 1: CURRENT DFLASH RECURRENT-STATE LIFECYCLE TRACE

### 1.1 Architecture Context

DFlash uses upstream's `draft-dflash` speculative decoding type. The target model for DFlash (Qwen3.6, Gemma 4) has a **hybrid memory architecture** (`llama_memory_hybrid`) combining:

- **Attention KV cache** (`llama_kv_cache`) — for attention layers
- **Recurrent state** (`llama_memory_recurrent`) — for recurrent layers (R and S tensors)

The hybrid memory is created in [`llama_memory_hybrid.cpp:11`](src/llama-memory-hybrid.cpp:11) with separate `mem_attn` and `mem_recr` sub-components. Both share the same `n_rs_seq` (rollback snapshot count) parameter.

### 1.2 The Complete Lifecycle: Request → Draft → Verify → Accept → Next Cycle

#### Stage 1: Request Arrival and Slot Setup

**File:** [`tools/server/server-context.cpp`](tools/server/server-context.cpp)

When a request arrives for a DFlash-enabled model:

1. **Line ~1930-1939:** Server detects `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` in `task.params.speculative.types`
2. **Line ~1084-1088:** `common_speculative_resolve_dflash_draft_n_max()` resolves draft max from model metadata
3. **Line ~1408-1413:** `common_speculative_init()` creates the speculative decoder, which instantiates `common_speculative_impl_draft_dflash`

The DFlash draft model (`ctx_dft`) and target model (`ctx_tgt`) are loaded as separate `llama_context` instances. Each has its own memory module.

#### Stage 2: Draft Generation

**File:** [`common/speculative.cpp`](common/speculative.cpp) — `common_speculative_impl_draft_dflash`

**Key methods:**

- **`begin()` (line 994-1010):** Called after prefill. Warns if draft context pos_max is insufficient.
- **`process()` (line 1012-1107):** During prefill, extracts target-layer input embeddings from `ctx_tgt`, fuses them through DFlash encoder in `ctx_dft`, and injects K/V cache into draft context.
- **`draft()` (line 1109-1193):** Builds noise-token batch `[id_last, <mask> * (n_draft)]`, decodes all sequences in one batch via `llama_decode(ctx_dft, batch)`, and greedily samples predicted tokens.

**Critical: Draft decode modifies recurrent state in `ctx_dft`.**

The draft context (`ctx_dft`) runs its own forward pass on the noise tokens. This means:

- **Draft attention KV:** Modified in `ctx_dft`'s attention cache at positions `n_past .. n_past + n_draft`
- **Draft recurrent state:** Modified in `ctx_dft`'s recurrent memory at the cell assigned to `seq_id`

The draft context's recurrent state after drafting represents the model's recurrent state at the *drafted* positions, which is speculative and may be discarded.

#### Stage 3: Target Verification

**File:** [`tools/server/server-context.cpp:3294-3320`](tools/server/server-context.cpp:3294)

Before verification, the server may create a checkpoint:

```cpp
// Line 3306-3307
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());
const bool use_ckpt_dft = server_speculative_rollback_requires_checkpoint(
        ctx_dft_seq_rm_type, common_context_seq_rm_max_rollback(ctx_dft), draft.size());
```

The checkpoint decision depends on `common_context_seq_rm_type`:

| Type | Meaning | Checkpoint needed for rollback? |
|------|---------|--------------------------------|
| `NO` | No seq_rm support | Speculative decoding disabled |
| `PART` | Arbitrary partial seq_rm | No (can truncate in-place) |
| `RS` | Bounded partial_rm (n_rs_seq limit) | Only if rollback > n_rs_seq |
| `FULL` | Full seq_rm only | Yes (must restore from checkpoint) |

**For DFlash with hybrid memory + n_rs_seq > 0:** The target context returns `COMMON_CONTEXT_SEQ_RM_TYPE_RS` (bounded rollback). If `draft.size() <= n_rs_seq`, no checkpoint is needed — rollback happens via the recurrent rollback mechanism. If `draft.size() > n_rs_seq`, a full checkpoint is required.

**During verification** (line 4192-4214):

1. Sampler state is saved: `common_sampler_clone(slot.smpl.get())`
2. Loop guard state is saved
3. `common_sampler_sample_and_accept_n()` verifies draft tokens through `ctx_tgt`
4. Returns `accepted` vector (includes the last accepted token + any accepted draft tokens)

**Target recurrent state during verification:**

The target model processes ALL draft tokens through its forward pass. This means the target's recurrent state is updated for each draft token position. The R/S tensors advance through `n_draft` positions.

#### Stage 4: Accept or Reject

**File:** [`tools/server/server-context.cpp:4217-4319`](tools/server/server-context.cpp:4217)

```cpp
const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
```

**Case A: Full acceptance (n_rollback == 0)**

- All draft tokens accepted
- Target recurrent state is already correct (it was updated during verification for all accepted tokens)
- Draft recurrent state is discarded on next cycle (draft context re-uses cells)
- `common_speculative_accept()` is called — **for DFlash this is a NO-OP** (line 1195-1197)

**Case B: Partial acceptance with bounded rollback (n_rollback <= n_rs_seq)**

- `use_ckpt_tgt` is false (rollback within n_rs_seq bound)
- Server calls `common_context_seq_rm(ctx_tgt, slot.id, pos_max + 1, -1)` to truncate the rejected suffix
- **Recurrent state rollback:** The `seq_rm` call on hybrid memory calls `mem_recr->seq_rm()` which:
  - Computes `rollback = cell.pos - (p0 - 1)` 
  - Calls `set_rs_idx(seq_id, (uint32_t) rollback)` to set the rollback index
  - The next forward pass will read R/S from rollback snapshot plane `idx * mem->size + src0` (line 1347)
- **Attention KV rollback:** The attention cache truncates the rejected positions via normal `seq_rm`
- `common_speculative_accept()` called — still no-op for DFlash

**Case C: Partial acceptance requiring checkpoint (n_rollback > n_rs_seq)**

- `use_ckpt_tgt` is true
- Full checkpoint restore:
  ```cpp
  ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
  common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
  ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
  common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
  ```
- **Both attention KV AND recurrent state are restored from the full checkpoint**
- Sampler state restored from pre-verification save
- Loop guard state restored

### 1.3 Recurrent State During Draft vs Verification

#### Draft Phase (ctx_dft)

| Aspect | Behavior |
|--------|----------|
| **Context** | `ctx_dft` (draft model) |
| **Memory type** | Depends on draft model architecture |
| **R/S tensors** | Updated for noise-token positions `n_past .. n_past + n_draft` |
| **Cell assignment** | Draft context's own cell for `seq_id` |
| **Seq ID** | Same `seq_id` as target (typically `slot.id`) |
| **Fate on accept** | Discarded — draft context re-encodes next cycle |
| **Fate on reject** | Discarded — either truncated or restored from checkpoint |

#### Verification Phase (ctx_tgt)

| Aspect | Behavior |
|--------|----------|
| **Context** | `ctx_tgt` (target model) |
| **Memory type** | `llama_memory_hybrid` (attention + recurrent) |
| **R/S tensors** | Updated for ALL draft token positions during verify forward pass |
| **Cell assignment** | Target's cell for `seq_id` |
| **Seq ID** | Same `seq_id` (typically `slot.id`) |
| **Fate on full accept** | Kept — recurrent state is correct for accepted tokens |
| **Fate on partial accept (bounded)** | Rollback via `rs_idx` — next forward reads from snapshot plane |
| **Fate on partial accept (checkpoint)** | Full restore from checkpoint |

### 1.4 Do Draft and Target Share Recurrent Memory?

**No.** Draft (`ctx_dft`) and target (`ctx_tgt`) are separate `llama_context` instances with separate memory modules. They do NOT share recurrent state.

The DFlash draft model is a separate GGUF with `dflash` architecture. Its memory is independent. The only connection is:
1. Target's prefill embeddings are extracted and fed into the draft encoder
2. Draft's predictions are verified by the target

### 1.5 Seq ID / Cell Assignment for DFlash

- **Seq ID:** Both draft and target use the same `seq_id` (the slot ID, typically 0 for single-batch server)
- **Cell assignment:** Each context independently assigns cells to seq IDs
- **Draft cell:** Draft context's recurrent cell for `seq_id` — updated during draft decode
- **Target cell:** Target context's recurrent cell for `seq_id` — updated during verification

### 1.6 How Correct Recurrent State is Determined After Accepting N Tokens

After accepting N tokens from a draft of M tokens:

1. **If N == M (full accept):** Target recurrent state was updated during verification for all M tokens. The state is correct. No action needed.

2. **If N < M (partial accept, bounded):** 
   - `seq_rm(seq_id, pos_max + 1, -1)` is called
   - This sets `rs_idx[seq_id] = M - N` (the rollback distance)
   - The next forward pass will resolve the source row as `rs_idx * mem->size + src0` (line 1347 of [`llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:1347))
   - After the forward pass consumes the rollback snapshot, `rs_idx[seq_id]` is reset to 0 (line 1344)

3. **If N < M (partial accept, checkpoint):**
   - Full state restored from checkpoint (includes both attention KV and recurrent state)
   - `rs_idx` reset to 0 after `state_read` (line 926)

### 1.7 How Rejected Recurrent State is Discarded/Restored

The recurrent state for rejected tokens is handled by the **rollback snapshot mechanism**:

- **Tensor layout:** R and S tensors have shape `[mem_size * (1 + n_rs_seq), n_embd_r/s]` (line 99)
- **Plane 0** (`idx=0`): Current/committed state
- **Planes 1..n_rs_seq:** Rollback snapshots created during the forward pass
- **Rollback index (`rs_idx[seq_id]`):** Which plane to read from on the next forward pass

When `rs_idx[seq_id] = k > 0`, the next forward pass reads R/S from plane `k`, which represents the state `k` tokens ago. After consuming, `rs_idx` resets to 0.

This means **rejected recurrent state is not explicitly restored** — it's implicitly discarded by reading from an earlier snapshot plane. The forward pass then writes new state to plane 0 (the current plane).

### 1.8 What Does `common_speculative_accept()` Do for Current DFlash?

**For DFlash, `accept()` is a NO-OP.**

[`common/speculative.cpp:1195-1197`](common/speculative.cpp:1195):
```cpp
void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
}
```

The `common_speculative_accept()` wrapper ([`common/speculative.cpp:2633-2664`](common/speculative.cpp:2633)) calls `impl->accept()` and updates statistics, but for DFlash the implementation does nothing.

**Contrast with other speculative types:**
- **draft-simple:** `accept()` updates draft model's sampler state
- **draft-mtp:** `accept()` stores hidden states from verification for chain-of-thought prediction
- **draft-dflash:** No state needs to be carried forward — DFlash re-encodes from target embeddings each cycle

### 1.9 Key Unknowns Requiring Further Investigation

1. **Does the draft context also have `n_rs_seq > 0`?** If the draft model is also hybrid, its recurrent state may also support bounded rollback. Need to verify how draft context params are configured.

2. **Does `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` exclude recurrent state?** The checkpoint uses `PARTIAL_ONLY` flag. Need to verify whether this flag affects recurrent state serialization in `state_write`/`state_read`.

3. **Is there a per-sequence recurrent-only copy API?** `seq_cp` for recurrent memory only copies metadata (cell references), not R/S tensor data. The full state requires `state_write`/`state_read`.

---

*End of Section 1. Section 2 (Recurrent-only backup APIs) follows below.*

---

## SECTION 2: RECURRENT-ONLY BACKUP/RESTORE APIs

### 2.1 API Inventory

This section inspects all APIs relevant to recurrent-state backup and restore, evaluating whether **recurrent-only** backup/restore is possible with existing APIs.

#### 2.1.1 `llama_memory_recurrent::seq_cp()`

**Location:** [`src/llama-memory-recurrent.cpp:316-351`](src/llama-memory-recurrent.cpp:316)

**Signature:**
```cpp
void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
```

**Analysis:**

| Question | Answer |
|----------|--------|
| **Exists?** | Yes |
| **Copies R+S tensor data?** | **No** — only copies cell metadata (references) |
| **What it actually does** | Copies `tail` pointer from source seq to destination seq. The destination seq shares the same cell (and thus the same R/S data rows) as the source. |
| **Works with n_rs_seq=0?** | Yes — the `seq_cp` logic doesn't depend on `n_rs_seq` |
| **Needs pre-existing dest cell?** | No — creates dest reference if needed. Clears dest if it had its own cell. |
| **Async/GPU D2D?** | N/A — no tensor data copied |
| **Safe before verify?** | Yes — but only shares references, doesn't create independent backup |

**Critical limitation:** This is a **reference copy**, not a data copy. Both sequences share the same R/S rows. When the forward pass writes to the cell, both sequences see the new state. This is useful for branching but NOT for backup/restore where you need independent copies.

#### 2.1.2 `llama_memory_recurrent::state_write()` / `state_read()`

**Location:** [`src/llama-memory-recurrent.cpp:822-929`](src/llama-memory-recurrent.cpp:822)

**Signature:**
```cpp
void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const;
void state_read (llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
```

**Analysis:**

| Question | Answer |
|----------|--------|
| **Exists?** | Yes |
| **Copies R+S?** | **Yes** — serializes both R tensors and S tensors for all layers |
| **Works with n_rs_seq=0?** | Yes — when `n_rs_seq == 0`, the `rs_idx` is always 0 and data comes from plane 0 |
| **Needs pre-existing dest cell?** | For `state_read`: reads into `head` position sequentially. The dest cells must have been prepared (via `seq_add` or ubatch preparation) or the data writes to wrong positions. |
| **Async/GPU D2D?** | Uses `llama_io_write_i::write_tensor()` / `read_tensor()` — these support device-to-device transfers when the IO backend is device memory (`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`). Normal file/memory IO uses CPU paths. |
| **Safe before verify?** | Yes — can serialize R/S at any point |

**Serialization format:**
1. Cell count
2. Metadata (pos, seq_id count, seq_ids for each cell)
3. R tensor data: for each layer, writes type + row size + tensor rows
4. S tensor data: for each layer, writes type + row size + tensor rows

**With `n_rs_seq > 0`:** The `state_write` respects `rs_idx[seq_id]` and writes from the rollback snapshot plane if `rs_idx > 0` (line 858: `cell_id = rs_idx_cur * size + src0`). After `state_read`, `rs_idx` resets to 0 (line 926).

#### 2.1.3 `llama_memory_hybrid::state_write()` / `state_read()`

**Location:** [`src/llama-memory-hybrid.cpp:281-297`](src/llama-memory-hybrid.cpp:281)

**Analysis:**

| Flag | Attention included? | Recurrent included? |
|------|-------------------|-------------------|
| `LLAMA_STATE_SEQ_FLAGS_NONE` | Yes | Yes |
| `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` | Only if `requires_state_for_partial_restore()` returns true | **Always Yes** |

**Key insight:** With `PARTIAL_ONLY`, the recurrent state is **always serialized** (line 287), while attention may be skipped. For KVarN caches, `requires_state_for_partial_restore()` returns true (line 1840 of kvarn.cpp), so KVarN attention IS included even with PARTIAL_ONLY. For standard KV caches without SWA, attention would be excluded with PARTIAL_ONLY.

**This means `PARTIAL_ONLY` is NOT "recurrent-only"** — it includes recurrent PLUS whatever attention the implementation requires for partial restore. For KVarN, that means full attention state is also included.

#### 2.1.4 Public C API: `llama_state_seq_*`

**Location:** [`include/llama.h:1024-1094`](include/llama.h:1024)

| API | Description | Recurrent-only? |
|-----|-------------|----------------|
| `llama_state_seq_get_size()` | Size for seq_id 0, full state | No |
| `llama_state_seq_get_data()` | Copy seq_id 0 state to buffer | No |
| `llama_state_seq_set_data()` | Restore from buffer to seq_id | No |
| `llama_state_seq_save_file()` | Save seq_id 0 to file | No |
| `llama_state_seq_load_file()` | Load from file to seq_id | No |
| `llama_state_seq_get_size_ext()` | Size for any seq_id, with flags | Partial with PARTIAL_ONLY |
| `llama_state_seq_get_data_ext()` | Copy any seq_id, with flags | Partial with PARTIAL_ONLY |
| `llama_state_seq_set_data_ext()` | Restore to any seq_id, with flags | Partial with PARTIAL_ONLY |

**All public APIs serialize through `llama_memory_hybrid::state_write/read`, which always includes both attention AND recurrent state.**

#### 2.1.5 `llama_memory_seq_cp()`

**Location:** [`src/llama-context.cpp:4435-4446`](src/llama-context.cpp:4435)

**Analysis:** Calls both `mem_attn->seq_cp()` and `mem_recr->seq_cp()`. As established, `mem_recr->seq_cp()` is a reference copy, not a data copy.

#### 2.1.6 `llama_memory_recurrent::seq_cp()` — Deep Dive

**Location:** [`src/llama-memory-recurrent.cpp:316-351`](src/llama-memory-recurrent.cpp:316)

The implementation:
```cpp
void seq_cp(seq_id_src, seq_id_dst, p0, p1) {
    // Ignores p0/p1 entirely (no range support)
    // Copies tail pointer from src to dst
    // Dest now shares src's cell
}
```

**Key observations:**
1. Position range `(p0, p1)` is ignored — always copies the entire sequence tail
2. No R/S tensor data is copied — only the `tail` metadata pointer
3. After the copy, both seq IDs reference the same cell
4. When the next forward pass writes to this cell, both sequences see the update

### 2.2 Can Recurrent-Only Backup/Restore Be Done Today?

#### Direct Answer: **No, not with existing public APIs.**

The reasons:

1. **No recurrent-only serialization API exists.** All `state_write`/`state_read` paths go through `llama_memory_hybrid`, which always serializes both attention AND recurrent state. The `PARTIAL_ONLY` flag skips attention for some cache types but NOT for KVarN (which returns `requires_state_for_partial_restore() == true`).

2. **`seq_cp` is reference-only, not data copy.** You cannot create an independent backup of recurrent state using `seq_cp`. Both source and destination share the same R/S rows.

3. **No API to copy R/S data between cells within the same context.** The recurrent memory has no `cell_copy(src_cell, dst_cell)` operation that would copy R/S tensor rows from one cell to another.

4. **No API to access raw R/S tensor pointers for manual copying.** The `r_l[il]` and `s_l[il]` tensors are internal to `llama_memory_recurrent` and not exposed through the public API.

### 2.3 Workarounds and Partial Solutions

#### Option A: Full Checkpoint with PARTIAL_ONLY

Use `llama_state_seq_get_data_ext(ctx, ..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` to serialize the minimum state. For non-KVarN attention, this excludes attention and includes only recurrent. For KVarN, this still includes attention.

**Limitation:** Not truly recurrent-only for KVarN models. Still expensive for large attention caches.

#### Option B: Abuse `seq_cp` + Independent Seq ID

1. Call `seq_cp(src_seq, backup_seq, -1, -1)` to share the cell reference
2. The backup_seq now points to the same R/S data
3. After verification, the original seq's cell may have changed
4. But the backup_seq still references the same cell — so it also sees the changes

**This doesn't work** because both sequences share the same underlying data. The "backup" is not independent.

#### Option C: Full State Save/Restore to Different Seq ID

1. Save full state: `llama_state_seq_get_data_ext(ctx, buf, src_seq, flags)`
2. Prepare dest seq: ensure dest_seq has a cell allocated
3. Restore: `llama_state_seq_set_data_ext(ctx, buf, dst_seq, flags)`

**Limitation:** This saves AND restores both attention AND recurrent. It's the full checkpoint approach, not recurrent-only.

#### Option D: Internal API Access (Not Public)

If you have access to internal headers (`llama-memory-recurrent.h`), you could:
1. Get `llama_memory_recurrent *` from `llama_memory_hybrid::get_mem_recr()`
2. Access `r_l[il]` and `s_l[il]` tensors directly
3. Use `ggml_backend_tensor_get()` / `ggml_backend_tensor_set()` for manual R/S copies

**Limitation:** Requires internal API access, not portable across llama.cpp versions.

### 2.4 Summary Table

| API | Copies R+S Data? | Recurrent-Only? | Public? | GPU D2D? |
|-----|-----------------|-----------------|---------|----------|
| `mem_recr->seq_cp()` | No (reference only) | N/A | Internal | N/A |
| `mem_recr->state_write/read()` | Yes | Yes (internal) | Internal | Via IO backend |
| `mem_hybrid->state_write/read()` | Yes (R+S+attention) | No | Internal | Via IO backend |
| `llama_state_seq_*_ext()` | Yes (R+S+attention) | No | **Yes** | Via IO backend |
| `llama_memory_seq_cp()` | No (reference only) | N/A | **Yes** | N/A |
| Direct tensor access | Yes (manual) | Yes (manual) | No | Via ggml backend |

### 2.5 Conclusion

**Recurrent-only backup/restore is NOT currently possible through public APIs.** The existing mechanisms are:

1. **Full checkpoint** (attention + recurrent) via `llama_state_seq_*_ext()` — works but includes unnecessary attention data
2. **Reference copy** via `seq_cp()` — fast but doesn't create independent backup
3. **Bounded rollback** via `n_rs_seq` snapshots — works within the rollback window but doesn't support arbitrary backup/restore

To enable recurrent-only backup/restore, new APIs would be needed:
- A `llama_memory_recurrent::cell_copy(src_cell, dst_cell)` for in-context R/S data copying
- Or a public wrapper around `mem_recr->state_write/read()` that exposes recurrent-only serialization
- Or exposure of R/S tensor pointers through the C API for manual GPU D2D copies
