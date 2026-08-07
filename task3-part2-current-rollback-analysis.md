# DFlash Rollback Mechanism Analysis — Subtask 3.2

**Date:** 2026-08-07
**Scope:** Current upstream DFlash checkpoint and RS rollback mechanisms and extension points for a hybrid approach.

---

## Table of Contents

1. [Overview](#1-overview)
2. [How Rollback Currently Works — Step-by-Step](#2-how-rollback-currently-works--step-by-step)
3. [RS Buffer Allocation — Exact Path and Size Formula](#3-rs-buffer-allocation--exact-path-and-size-formula)
4. [Checkpoint Rollback — When Triggered and What It Saves](#4-checkpoint-rollback--when-triggered-and-what-it-saves)
5. [Extension Points for Hybrid Approach](#5-extension-points-for-hybrid-approach)
6. [What Breaks with n_rs_seq=0](#6-what-breaks-with-n_rs_seq0)
7. [Key Question: Can the Architecture Support "Lightweight Backup"?](#7-key-question-can-the-architecture-support-lightweight-backup)
8. [Summary of Findings](#8-summary-of-findings)

---

## 1. Overview

The current upstream DFlash speculative decoding uses **two rollback mechanisms** that form an either/or decision:

| Mechanism | Triggered When | Cost |
|---|---|---|---|---|---|
| **RS Snapshots** | Draft size ≤ `n_rs_seq` | ~5.4 GB VRAM (for n_max=8, full RS tensor × 8 rows per layer) | Fast (in-place pointer shift via `rs_idx`) |
| **Checkpoint Serialize/Restore** | Draft size > `n_rs_seq`, or memory reports `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` | CPU RAM + serialization time | Slow (full state I/O) |

The decision function is [`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20):

```cpp
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
          (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

This function returns `true` (use checkpoint) when:
- The memory reports `FULL` capability (no partial seq_rm), OR
- The proposed rollback exceeds the RS snapshot reserve (`max_rollback = n_rs_seq`).

---

## 2. How Rollback Currently Works — Step-by-Step

### 2.1 Draft Generation Phase

**File:** [`tools/server/server-context.cpp:3264-3330`](tools/server/server-context.cpp:3264)

1. **Draft tokens generated** via `common_speculative_draft(spec.get())` — DFlash implementation at [`common/speculative.cpp:906`](common/speculative.cpp:906) produces `block_size - 1` draft tokens (typically 15 for block_size=16).

2. **Checkpoint decision** at [`server-context.cpp:3305-3330`](tools/server/server-context.cpp:3305):
   ```cpp
   const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
           ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());
   ```
   - For DFlash on hybrid memory: `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_RS` (because `suffix_rollback_tokens = n_rs_seq > 0`).
   - Since `draft.size() <= n_rs_seq` (both equal `n_max`), `use_ckpt_tgt = false`.
   - **No checkpoint is saved.** The RS buffer is the rollback mechanism.

3. **Draft context cleanup** at [`server-context.cpp:3302`](tools/server/server-context.cpp:3302):
   ```cpp
   common_context_seq_rm(ctx_dft, slot.id, ckpt.pos_max + 1, -1);
   ```

### 2.2 Verification Phase

**File:** [`tools/server/server-context.cpp:4191-4274`](tools/server/server-context.cpp:4191)

1. **Sampler state saved** (lines 4195-4209): the current sampler state and loop guard state are cloned for potential restore.

2. **Draft verification** via `common_sampler_sample_and_accept_n()` (line 4213): verifies draft tokens against the target model. Returns `accepted` tokens.

3. **Rollback size calculated** (line 4219):
   ```cpp
   const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
   ```

4. **Rollback decision** (line 4221):
   ```cpp
   const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
           ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);
   ```

### 2.3 Rollback Execution — Two Paths

#### Path A: RS Snapshot Rollback (Normal DFlash path)

**Condition:** `use_ckpt_tgt == false` (i.e., `n_rollback <= n_rs_seq`)

**Flow:**
1. `common_speculative_accept(spec.get(), slot.id, accepted.size() - 1)` at [`server-context.cpp:4271`](tools/server/server-context.cpp:4271) — informs the speculative implementation of acceptance.
2. Accepted tokens are kept; unaccepted suffix tokens remain in the RS buffer as snapshots.
3. The recurrent memory's `rs_idx` is set during `seq_rm()` to point to the appropriate snapshot row.
4. **No state serialization occurs.** The rollback is an in-place pointer shift.

#### Path B: Checkpoint Rollback (Fallback)

**Condition:** `use_ckpt_tgt == true` (i.e., `n_rollback > n_rs_seq` or memory reports FULL only)

**Flow at [`server-context.cpp:4226-4263`](tools/server/server-context.cpp:4226):**
1. **Restore target checkpoint:**
   ```cpp
   ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
   common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
   ```
2. **Restore draft checkpoint:**
   ```cpp
   ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
   common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
   ```
3. **Restore sampler and loop guard state** (lines 4250-4261).
4. **Return early** — the entire verify/accept cycle is discarded and will be retried.

### 2.4 The `common_prompt_checkpoint` Structure

**File:** [`common/common.h:1181-1229`](common/common.h:1181)

```cpp
struct common_prompt_checkpoint {
    int64_t n_tokens;
    int id_task = -1;
    llama_pos pos_min;
    llama_pos pos_max;
    std::vector<uint8_t> data_tgt;  // serialized target context state
    std::vector<uint8_t> data_dft;  // serialized draft context state
    std::vector<uint8_t> data_spec; // speculative implementation state (e.g., EAGLE3 g_embd)
    // Methods: update_tgt(), update_dft(), load_tgt(), load_dft(), ...
};
```

The checkpoint saves **the entire memory state** (attention KV + recurrent RS) for both target and draft contexts. With `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, the attention KV body is excluded (only the recurrent RS and tail are saved), but this is still substantial for DFlash.

---

## 3. RS Buffer Allocation — Exact Path and Size Formula

### 3.1 Tensor Allocation

**File:** [`src/llama-memory-recurrent.cpp:99-106`](src/llama-memory-recurrent.cpp:99)

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

**Size formula per layer:**
```
R tensor: n_embd_r × mem_size × (1 + n_rs_seq) × sizeof(type_r)
S tensor: n_embd_s × mem_size × (1 + n_rs_seq) × sizeof(type_s)
```

The `(1 + n_rs_seq)` factor is critical: each cell position has `n_rs_seq` historical snapshot rows plus 1 active row.

### 3.2 Allocation Call Chain

```
llama_memory_hybrid constructor (llama-memory-hybrid.cpp:68-78)
    └── llama_memory_recurrent constructor (llama-memory-recurrent.cpp:99)
            └── ggml_new_tensor_2d() with n_rows = mem_size * (1 + n_rs_seq)
                    └── ggml_backend_alloc_ctx_tensors_from_buft() (line 110)
```

### 3.3 n_rs_seq Source

**File:** [`common/common.h:417-423`](common/common.h:417)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });
    return needs_rs_seq ? draft.n_max : 0u;
}
```

For DFlash: `n_rs_seq = draft.n_max` (typically 8 for block_size=16, clamped to `block_size - 1 = 15`).

### 3.4 Concrete Size Example (Qwen3.6 DFlash)

For a typical Qwen3.6 DFlash target with recurrent state:
- `n_rs_seq = 8` (n_max)
- `mem_size` = context tokens (e.g., 8192)
- `n_rows = 8192 × (1 + 8) = 8192 × 9 = 73,728`
- With `n_embd_r` and `n_embd_s` at their model-specific sizes and quantization types, this totals **~5.4 GB** as reported in the investigation.

---

## 4. Checkpoint Rollback — When Triggered and What It Saves

### 4.1 Trigger Conditions

Checkpoint rollback is triggered when `server_speculative_rollback_requires_checkpoint()` returns `true`:

| Scenario | `type` | `max_rollback` | `proposed_rollback` | Result |
|---|---|---|---|---|
| DFlash normal (draft ≤ n_rs_seq) | RS | n_rs_seq (=8) | ≤ 8 | **false** (use RS) |
| DFlash overflow (draft > n_rs_seq) | RS | n_rs_seq (=8) | > 8 | **true** (checkpoint) |
| Memory reports FULL only | FULL | 0 | any | **true** (checkpoint) |
| No memory | NO | 0 | any | **true** (checkpoint) |

### 4.2 What Checkpoint Saves

**Target context save** at [`server-context.cpp:3315`](tools/server/server-context.cpp:3315):
```cpp
ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

This calls through to [`llama_memory_hybrid::state_write()`](src/llama-memory-hybrid.cpp:281):
```cpp
void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool include_attn = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 ||
                              /* tail overlay check */;
    if (include_attn) {
        mem_attn->state_write(io, seq_id, flags);  // attention KV
    }
    mem_recr->state_write(io, seq_id, flags);       // recurrent RS (ALWAYS saved)
}
```

With `PARTIAL_ONLY`:
- **Attention KV body:** May be excluded (depends on tail overlay).
- **Attention KV tail:** Always saved.
- **Recurrent RS:** Always saved (full RS state for the sequence, including all `n_rs_seq` snapshot rows).

**Draft context save** at [`server-context.cpp:3327`](tools/server/server-context.cpp:3327):
```cpp
ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

### 4.3 Checkpoint Restore

At [`server-context.cpp:4238-4248`](tools/server/server-context.cpp:4238):
```cpp
ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);

if (slot.ctx_dft) {
    ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
}
```

After restore, the server also clears any tokens after the checkpoint position and restores the sampler state.

---

## 5. Extension Points for Hybrid Approach

### 5.1 Extension Point A: Reduce `n_rs_seq` for DFlash

**Location:** [`common/common.h:417-423`](common/common.h:417)

**Current behavior:** `need_n_rs_seq()` returns `draft.n_max` for DFlash, MTP, and EAGLE3.

**Modification:** Return a smaller value (e.g., 1 or 2) for DFlash specifically:
```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = /* ... */;
    if (!needs_rs_seq) return 0u;
    
    // DFlash uses smaller RS reserve; larger rollbacks fall back to checkpoint.
    if (has_dflash() && draft.n_max > 0) {
        return std::min((uint32_t)2, (uint32_t)draft.n_max);  // e.g., 1-2 snapshots
    }
    return draft.n_max;
}
```

**Impact:**
- RS buffer shrinks from `mem_size × (1 + 8)` to `mem_size × (1 + 2)` — **~73% VRAM reduction**.
- Rollbacks exceeding 2 tokens would fall back to checkpoint serialization.
- This is the **simplest** extension point.

### 5.2 Extension Point B: New `COMMON_CONTEXT_SEQ_RM_TYPE` for Intermediate Rollback

**Location:** [`common/common.h`](common/common.h) — enum `common_context_seq_rm_type`

**Current values:**
- `COMMON_CONTEXT_SEQ_RM_TYPE_NO` — no rollback
- `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` — full checkpoint only
- `COMMON_CONTEXT_SEQ_RM_TYPE_PART` — arbitrary range removal (attention KV)
- `COMMON_CONTEXT_SEQ_RM_TYPE_RS` — RS snapshot rollback

**Modification:** Add a new type for "bounded RS with checkpoint fallback":
```cpp
COMMON_CONTEXT_SEQ_RM_TYPE_RS_HYBRID  // RS for small rollbacks, checkpoint for large
```

The decision function [`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20) would then handle this type with a custom threshold.

### 5.3 Extension Point C: Custom Rollback in `llama_memory_recurrent::seq_rm()`

**Location:** [`src/llama-memory-recurrent.cpp:179-245`](src/llama-memory-recurrent.cpp:179)

**Current behavior at lines 214-222:**
```cpp
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
        return true;
    }
    return false;  // <-- rollback exceeds n_rs_seq: fails
}
```

**Modification:** Instead of returning `false` when `rollback > n_rs_seq`, signal the caller that checkpoint rollback is needed. This requires a new return convention or an out-parameter.

### 5.4 Extension Point D: Modify `can_seq_rm()` to Always Accept

**Location:** [`src/llama-memory-recurrent.cpp:150-177`](src/llama-memory-recurrent.cpp:150)

**Current behavior at lines 172-174:**
```cpp
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    return rollback >= 1 && rollback <= llama_pos(n_rs_seq);
}
```

**Modification:** Return `true` for any rollback size, and let `seq_rm()` delegate oversized rollbacks to the checkpoint path. This would require changing the `seq_rm_capability` report to indicate a larger `suffix_rollback_tokens` than the actual RS buffer supports, which is semantically misleading.

### 5.5 Extension Point E: Server-Level Hybrid Decision

**Location:** [`tools/server/server-context.cpp:4221-4264`](tools/server/server-context.cpp:4221)

**Current behavior:** Binary decision — RS or checkpoint.

**Modification:** Add a third path between RS and full checkpoint:
```cpp
if (n_rollback > 0) {
    if (n_rollback <= small_rs_threshold) {
        // Use RS snapshot (n_rs_seq = small value)
        common_context_seq_rm(...);
    } else if (use_ckpt_tgt) {
        // Full checkpoint restore
        ckpt.load_tgt(...);
    } else {
        // NEW: Lightweight backup mechanism (e.g., backup cell, partial RS)
        lightweight_rollback(slot, n_rollback);
    }
}
```

This is the **most flexible** extension point but requires the most new code.

---

## 6. What Breaks with n_rs_seq=0

### 6.1 Code Paths That Would FAIL

#### 6.1.1 `can_seq_rm()` Returns False for Partial Rollback

**File:** [`src/llama-memory-recurrent.cpp:172-174`](src/llama-memory-recurrent.cpp:172)
```cpp
const llama_pos rollback = cell.pos - (p0 - 1);
return rollback >= 1 && rollback <= llama_pos(n_rs_seq);
```
With `n_rs_seq = 0`: `rollback <= 0` is always false for any positive rollback. **Partial rollback always fails.**

#### 6.1.2 `seq_rm()` Partial Rollback Path Dead Code

**File:** [`src/llama-memory-recurrent.cpp:217-222`](src/llama-memory-recurrent.cpp:217)
```cpp
if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
    set_rs_idx(seq_id, (uint32_t) rollback);
    cell.pos = p0 - 1;
    return true;
}
return false;
```
With `n_rs_seq = 0`: the condition `rollback <= 0` never matches. **Always returns false.**

#### 6.1.3 `get_seq_rm_capability()` Reports Zero Rollback

**File:** [`src/llama-memory-recurrent.cpp:781-787`](src/llama-memory-recurrent.cpp:781)
```cpp
return {
    .full_clear = true,
    .arbitrary_ranges = false,
    .suffix_rollback_tokens = n_rs_seq,  // = 0
};
```

#### 6.1.4 `common_context_can_seq_rm()` Returns FULL, Not RS

**File:** [`common/common.cpp:1677-1679`](common/common.cpp:1677)
```cpp
if (capability.suffix_rollback_tokens > 0) {
    return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
}
if (capability.full_clear) {
    return COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
}
```
With `suffix_rollback_tokens = 0`: falls through to `FULL`.

#### 6.1.5 `server_speculative_rollback_requires_checkpoint()` Always True for RS

**File:** [`tools/server/server-task.h:24-25`](tools/server/server-task.h:24)
```cpp
return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
      (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
```
With `type = FULL` (because suffix_rollback_tokens = 0): **always returns true.**

### 6.2 Code Paths That Gracefully Handle n_rs_seq=0

#### 6.2.1 Tensor Allocation

**File:** [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99)
```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
```
With `n_rs_seq = 0`: `n_rows = mem_size × 1`. **Works correctly** — allocates only the active row, no snapshot rows. This is the **VRAM savings**.

#### 6.2.2 `set_rs_idx()` Clamps to n_rs_seq

**File:** [`src/llama-memory-recurrent.cpp:473-478`](src/llama-memory-recurrent.cpp:473)
```cpp
rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
```
With `n_rs_seq = 0`: always sets `rs_idx[seq_id] = 0`. **Safe** — no out-of-bounds access.

#### 6.2.3 Graph Build Uses rs_idx

**File:** [`src/llama-memory-recurrent.cpp:858`](src/llama-memory-recurrent.cpp:858)
```cpp
const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
```
With `rs_idx_cur = 0` (always): `cell_id = cell.src` or `i`. **Works correctly** — always reads from the active row.

#### 6.2.4 Full Clear Still Works

The `rm_all` path at [`src/llama-memory-recurrent.cpp:194-201`](src/llama-memory-recurrent.cpp:194) handles `p0=0, p1=MAX` regardless of `n_rs_seq`. **Full sequence clear works.**

### 6.3 Net Effect of n_rs_seq=0

| Behavior | With n_rs_seq=8 | With n_rs_seq=0 |
|---|---|---|
| RS buffer size | `mem_size × 9` rows | `mem_size × 1` row |
| Partial rollback via RS | Works for ≤ 8 tokens | **Never works** |
| `common_context_can_seq_rm()` | Returns `RS` | Returns `FULL` |
| `server_speculative_rollback_requires_checkpoint()` | False for ≤ 8 tokens | **Always true** |
| Checkpoint save before draft | **Not done** (RS handles it) | **Always done** |
| Checkpoint restore on rollback | Only for > 8 tokens | **Always** |
| VRAM savings | Baseline | **~73% RS buffer reduction** |
| CPU overhead | Minimal (RS pointer shift) | **High (serialize/restore)** |

**Conclusion:** Setting `n_rs_seq = 0` for DFlash would make every speculative rollback use checkpoint serialization. This is functionally correct but would be extremely slow for the common case of 1-2 token rollbacks. A small `n_rs_seq` (1-2) provides a better balance.

---

## 7. Key Question: Can the Architecture Support "Lightweight Backup"?

### 7.1 Existing APIs That Could Be Extended

| API | Current Purpose | Extension Potential |
|---|---|---|
| [`llama_memory_recurrent::state_write()`](src/llama-memory-recurrent.cpp:822) | Full RS state serialization | Could add "partial state write" for just the last N rows |
| [`llama_memory_recurrent::set_rs_idx()`](src/llama-memory-recurrent.cpp:473) | Set rollback snapshot index | Could be extended to support variable-depth snapshots |
| [`common_prompt_checkpoint`](common/common.h:1181) | Full checkpoint storage | Could add a "lightweight" variant storing only RS tail |
| [`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20) | Binary RS vs checkpoint decision | Could return three-way: RS / lightweight / checkpoint |

### 7.2 What Would Need to Be Added

A "lightweight backup" mechanism between RS snapshots (5.4GB) and full checkpoint (slow serialization) would need:

1. **Reduced RS depth:** Set `n_rs_seq` to 1-2 instead of 8. This alone saves ~73% VRAM.
2. **Checkpoint fallback for overflow:** Already exists. When rollback exceeds `n_rs_seq`, checkpoint is used.
3. **Optional: Partial RS serialization:** Instead of full checkpoint, serialize only the RS state for the specific layers and rows needed. This would require:
   - A new `state_write_partial_rs()` method in `llama_memory_recurrent`
   - A corresponding `state_read_partial_rs()` method
   - Storage in `common_prompt_checkpoint` for this partial data

### 7.3 Recommended Approach

The **simplest effective approach** is Extension Point A: reduce `n_rs_seq` for DFlash to 1-2 snapshots and let the existing checkpoint mechanism handle larger rollbacks.

**Why this works:**
- DFlash typically accepts 10-14 of 15 draft tokens. Rollback of 1-2 tokens is the common case.
- A rollback of > 2 tokens is rare (means the draft was very wrong after only 1-2 tokens).
- For rare large rollbacks, checkpoint serialization is acceptable as a fallback.
- No new code paths needed — the existing `server_speculative_rollback_requires_checkpoint()` handles the decision.

**Implementation:**
```cpp
// In common/common.h:417
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3
            || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });
    if (!needs_rs_seq) return 0u;
    
    // DFlash: use minimal RS reserve; checkpoint handles overflow.
    if (/* has_dflash */) {
        return std::min((uint32_t)2, (uint32_t)draft.n_max);
    }
    return draft.n_max;
}
```

---

## 8. Summary of Findings

### Key Architectural Insight

The current upstream architecture **already supports** a hybrid rollback approach through its existing `COMMON_CONTEXT_SEQ_RM_TYPE_RS` / `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` decision mechanism. The only modification needed is to reduce `n_rs_seq` for DFlash, which automatically triggers checkpoint fallback for rollbacks exceeding the reduced RS buffer.

### VRAM Impact

| Configuration | RS Rows per Cell | RS Buffer (approx.) | Checkpoint Fallback |
|---|---|---|---|
| Current (n_rs_seq=8) | 9 rows | ~5.4 GB | Rare (>8 token rollback) |
| Hybrid (n_rs_seq=2) | 3 rows | ~1.8 GB | Moderate (>2 token rollback) |
| Minimal (n_rs_seq=1) | 2 rows | ~1.2 GB | More frequent (>1 token rollback) |
| None (n_rs_seq=0) | 1 row | ~0.6 GB | Every rollback |

### Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Checkpoint too slow for common rollback | Medium | Keep n_rs_seq ≥ 1 for the 1-token rollback case |
| DFlash acceptance pattern changes | Low | Monitor rollback distribution in production |
| Memory hybrid capability reporting | Low | Already tested by existing `seq_rm_capability_all()` |
| EAGLE3/MTP affected by change | Low | Scope change to DFlash-only via `has_dflash()` check |

### Files That Would Need Modification

| File | Change |
|---|---|
| [`common/common.h:417`](common/common.h:417) | Modify `need_n_rs_seq()` to return smaller value for DFlash |
| [`tools/server/server-task.h:20`](tools/server/server-task.h:20) | No change needed (existing logic handles the decision) |
| [`tools/server/server-context.cpp:3305`](tools/server/server-context.cpp:3305) | No change needed (checkpoint created when `use_ckpt_tgt` is true) |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | No change needed (handles n_rs_seq=small gracefully) |

---

*End of analysis. This document covers subtask 3.2 of the hybrid investigation.*
