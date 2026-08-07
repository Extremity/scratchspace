# Task 4.3: Recovering Accepted Recurrent State After Backup Restore

**Author:** Architect mode investigation
**Date:** 2026-08-07
**Related:** task4-backup-restore-feasibility.md (Section 4), task4-part2b-cost-and-recommendation.md

---

## Section 5 — Does Current Upstream Already Have a Mechanism for This?

### 5.0 The Problem Restated

After DFlash drafts N tokens and verification accepts only K < N of them:

```
state_before_draft = S0
DFlash drafts tokens 1..N (e.g., N=15)
Target verifies: accepts tokens 1..K (e.g., K=12), rejects K+1..N
Backup contains S0
After restoring backup, recurrent state is S0
But the CORRECT post-acceptance state is SK (S12)
How do we get from S0 → SK?
```

### 5.1 Mechanism A: `n_rs_seq` Per-Token Recurrent Snapshots — EXISTS

**Location:** [`src/llama-memory-recurrent.h:79-85`](src/llama-memory-recurrent.h:79), [`src/llama-memory-recurrent.cpp:214-222`](src/llama-memory-recurrent.cpp:214)

The recurrent memory already records per-token snapshots. The tensors are widened to `(1 + n_rs_seq)` groups, where each group holds the R and S state at a different position offset.

**Configuration for DFlash:** [`common/common.h:417-423`](common/common.h:417)

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });
    return needs_rs_seq ? draft.n_max : 0u;
}
```

For DFlash with `draft.n_max = 14` (block_size 16 minus 1), `n_rs_seq = 14`. This means the recurrent memory captures up to 14 per-token snapshots.

**Rollback mechanism:** [`src/llama-memory-recurrent.cpp:214-222`](src/llama-memory-recurrent.cpp:214)

```cpp
// partial rollback via per-token snapshot index (bounded by n_rs_seq)
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
        return true;
    }
}
```

When `seq_rm(seq_id, p0, p1)` is called, it:
1. Computes `rollback = cell.pos - (p0 - 1)` (how many positions to go back)
2. Sets `rs_idx[seq_id] = rollback`
3. Updates `cell.pos` to the new position

The next decode reads from snapshot group `rs_idx` instead of group 0. See [`src/llama-memory-recurrent.cpp:1330-1348`](src/llama-memory-recurrent.cpp:1330):

```cpp
int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;
    if (mem->n_rs_seq == 0) {
        return src0;
    }
    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
```

**Verdict:** The snapshot infrastructure EXISTS and is CONFIGURED for DFlash. The snapshots S0, S1, ..., SN are captured during the verification forward pass.

### 5.2 Mechanism B: Checkpoint Save/Restore — EXISTS

**Location:** [`common/common.h:1181-1229`](common/common.h:1181), [`tools/server/server-context.cpp:3238-3319`](tools/server/server-context.cpp:3238)

The `common_prompt_checkpoint` struct saves full partial-state (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) for both target and draft contexts. The checkpoint is saved BEFORE drafting begins and restored AFTER partial acceptance.

**Checkpoint decision logic:** [`tools/server/server-task.h:20-26`](tools/server/server-task.h:20)

```cpp
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
           (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

- `COMMON_CONTEXT_SEQ_RM_TYPE_PART` — arbitrary seq_rm supported; no checkpoint needed
- `COMMON_CONTEXT_SEQ_RM_TYPE_RS` — bounded rollback; checkpoint only if `proposed_rollback > max_rollback`
- `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` — full clear only; checkpoint ALWAYS needed

**Verdict:** Checkpoint infrastructure EXISTS. It saves S0 (pre-draft state).

### 5.3 Mechanism C: `common_replay_last_token()` — EXISTS (Limited)

**Location:** [`common/common.cpp:2191-2199`](common/common.cpp:2191)

```cpp
bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    llama_batch batch = llama_batch_get_one(&last_token, 1);
    batch.pos = &pos;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}
```

Used only for session-state replay after loading from file ([`common/common.h:1053-1067`](common/common.h:1053)):

> "We save state before the last token so that we can replay it to ensure compatibility with all memory types. Recurrent/hybrid models cannot remove tokens from their memory, so we can't just remove the last token from the memory and replay the last token."

**Verdict:** EXISTS but only replays ONE token. Not directly applicable to replaying 10-14 accepted tokens.

### 5.4 Mechanism D: Eagle3 Deferred-Boundary g_embd Stash — EXISTS (Draft-Specific)

**Location:** [`common/speculative.cpp:857-898`](common/speculative.cpp:857)

```cpp
// we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
// their single-position checkpoints drop it on restore
bool need_boundary_stash() const {
    const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
    return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
}
```

EAGLE3 stashes the g_embd (feature embedding) row at the accepted boundary. This is draft-model-specific state, not target recurrent state.

**Verdict:** EXISTS for EAGLE3's draft-side state. Not applicable to target recurrent state recovery.

### 5.5 CRITICAL FINDING: The `n_rs_seq` Snapshots Are NOT Used for Speculative Rollback

Despite the `n_rs_seq` mechanism existing and being configured for DFlash, the current speculative verification code at [`tools/server/server-context.cpp:4219-4274`](tools/server/server-context.cpp:4219) does NOT use `seq_rm` with `rs_idx` to recover the accepted state:

```cpp
const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
    ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);

if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        // Restore full checkpoint (S0), then clean up
        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
        // ... restore sampler, loop guard, etc. ...
        return;  // EARLY RETURN
    }
}

// Falls through when use_ckpt_tgt == false or n_rollback == 0
// NO seq_rm call to set rs_idx for partial rollback
common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
slot.spec_draft = std::move(accepted);
```

When `use_ckpt_tgt` is true (the common case for recurrent/hybrid models), the system restores S0 and returns. The `n_rs_seq` snapshots S0..SN captured during verification are discarded with the checkpoint restore.

When `use_ckpt_tgt` is false, the code falls through without calling `seq_rm`, so `rs_idx` is never set to select SK.

**This means the current upstream restores S0 after partial acceptance and must re-decode all accepted tokens in the next cycle to arrive at SK.** This is the "re-decode" approach (Option B below), and it is the current behavior.

### 5.6 Summary of Existing Mechanisms

| Mechanism | Exists? | Used for Speculative Rollback? | Notes |
|-----------|---------|-------------------------------|-------|
| `n_rs_seq` snapshots | Yes | **No** | Configured but not invoked in verification path |
| Checkpoint save/restore | Yes | Yes | Restores S0, not SK |
| `common_replay_last_token()` | Yes | No | Single-token only |
| Eagle3 g_embd stash | Yes | Yes (draft only) | Not target recurrent state |
| `seq_rm` with `rs_idx` | Yes | **No** in spec path | Available but not called |

---

## Section 6 — Three Options for Recovering Accepted Recurrent State

### Option A: Use Existing `n_rs_seq` Snapshots (Minimal Code Change)

**Concept:** After verification accepts K tokens, call `seq_rm` with the right range to set `rs_idx` to select snapshot SK, instead of restoring the full checkpoint.

**How it would work:**

After verification at [`server-context.cpp:4219`](tools/server/server-context.cpp:4219):

```cpp
// Current: restore S0 checkpoint
// ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

// Proposed: use rs_idx to select SK directly
if (n_rollback > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
    llama_pos current_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
    llama_pos target_pos = current_pos - n_rollback;  // = position after K accepted tokens
    common_context_seq_rm(ctx_tgt, slot.id, target_pos + 1, current_pos + 1);
    // This sets rs_idx to select the snapshot at target_pos (state SK)
}
```

**Advantages:**
- Snapshots already exist — `n_rs_seq` is configured for DFlash
- No additional VRAM overhead for the tape itself
- No re-decode needed — state SK is directly available
- Minimal code change: add `seq_rm` call in verification path

**Disadvantages:**
- **Only works for recurrent state (R/S tensors), NOT for attention KV cache**
- For hybrid models (Qwen3.6, Gemma 4), the attention KV cache for rejected tokens still needs cleanup
- The `rs_idx` mechanism resets after one use (`rs_idx[seq_id] = 0` in `s_copy()`), so it's a one-shot rollback
- The checkpoint still needs to be saved for the draft context (ctx_dft), which may have different seq_rm capabilities

**Critical limitation:** This only recovers the *recurrent* state. For hybrid models, the attention KV cache for rejected tokens must still be cleaned up. The attention KV cache uses `seq_rm` for cleanup, which is a separate operation.

**VRAM cost:** Zero additional VRAM (snapshots already allocated).

**Speed impact:** Eliminates re-decode of accepted tokens. Instead of re-running target forward pass for K tokens, we directly read snapshot SK.

**Files to modify:**
- [`tools/server/server-context.cpp:4221-4264`](tools/server/server-context.cpp:4221) — add `seq_rm` path for `COMMON_CONTEXT_SEQ_RM_TYPE_RS`

---

### Option B — Re-decode Accepted Tokens (Current Behavior)

**Concept:** After restoring S0, decode the K accepted tokens through the target model to arrive at SK.

**How current upstream works:**

1. Restore checkpoint (S0) at [`server-context.cpp:4239`](tools/server/server-context.cpp:4239)
2. Set `slot.spec_draft = accepted` (the K accepted tokens + new sample)
3. Next iteration: accepted tokens are treated as "previous partial draft" at [`server-context.cpp:3230`](tools/server/server-context.cpp:3230)
4. New drafting extends from accepted tokens
5. Full verification batch includes accepted tokens + new draft tokens
6. Target forward pass re-processes accepted tokens to advance state S0 → SK → SN'

**Does this require a full target-model forward pass?**

Yes. The accepted tokens are re-included in the verification batch. The target model runs its full forward pass on all tokens (previously accepted + newly drafted).

**Can it batch the accepted tokens?**

Yes, they're batched with new draft tokens in a single verification call. The accepted tokens are at the beginning of the batch, followed by new draft tokens.

**Does it require another sampler operation?**

No. The sampler state was already saved (`smpl_save` at [`server-context.cpp:4195`](tools/server/server-context.cpp:4195)) and restored at [`server-context.cpp:4251`](tools/server/server-context.cpp:4251). The sampler is not re-run for accepted tokens; only the forward pass is re-executed.

**Does it interfere with speculative scheduling?**

Yes, partially. The re-decode of accepted tokens adds latency to the next speculative cycle. The accepted tokens are verified AGAIN along with new draft tokens, consuming compute that could have been used for new token generation.

**Does it substantially reduce DFlash speedup?**

This depends on the acceptance rate:

| Scenario | Accepted | Rejected | Re-decode overhead |
|----------|----------|----------|-------------------|
| Full acceptance (14/15) | 14 | 1 | 14 tokens re-verified each cycle |
| Typical (10-12 of 15) | 10-12 | 3-5 | 10-12 tokens re-verified each cycle |
| Poor (3 of 15) | 3 | 12 | 3 tokens re-verified (low overhead) |

For typical DFlash acceptance rates (~70-80%), the re-decode overhead is 10-12 tokens per cycle. Since DFlash drafting itself is ~15 tokens per cycle, the effective new work per cycle is only 3-5 tokens. The re-decode of 10-12 tokens can consume a significant fraction of the cycle time.

**However**, the re-decode is batched with new draft verification, so the attention computation for accepted tokens can potentially overlap with new token computation. The actual overhead depends on whether the backend can pipeline the work.

**What is the actual token count?**

Typical DFlash: 10-14 accepted of 15 drafted, depending on model quality and draft-target alignment. The Qwen3.6 DFlash papers report ~70-85% acceptance rates, meaning 10-13 of 15 tokens accepted on average.

**VRAM cost:** No additional VRAM (uses existing checkpoint mechanism).

**Speed impact:** Re-decode of K tokens per cycle adds latency proportional to K. For K=12, this is roughly 80% of the draft batch size being re-processed.

---

### Option C — Minimum Tape Functionality

**Concept:** Record enough intermediate state during the verification forward pass to replay S0 → SK without full re-decode.

**What "minimum tape" means for linear DFlash:**

NOT DDTree, NOT branches, NOT the full old tape system. Just: enough information to advance from S0 to SK.

**What tensors need to be captured?**

For hybrid models (Qwen3.6, Gemma 4), the recurrent state consists of:

1. **R tensor** (recurrent "running" state): [`src/llama-memory-recurrent.h:118`](src/llama-memory-recurrent.h:118)
   - Shape: `[n_embd_r, n_seqs]` per layer
   - Already captured by `n_rs_seq` snapshots

2. **S tensor** (recurrent "snapshot" state): [`src/llama-memory-recurrent.h:119`](src/llama-memory-recurrent.h:119)
   - Shape: `[n_embd_s, n_seqs]` per layer
   - Already captured by `n_rs_seq` snapshots

3. **Attention KV cache** (for attention layers in hybrid models):
   - Shape: `[n_kv_heads, head_dim, seq_len, n_seqs]` per layer
   - NOT captured by `n_rs_seq` snapshots
   - This is the KV cache for the ACCEPTED tokens

**Key insight:** The `n_rs_seq` snapshots ALREADY capture R and S state per token. What's missing is the attention KV cache for the accepted tokens.

But wait — the attention KV cache for accepted tokens IS preserved. After verification, the accepted tokens' KV entries are at positions P+1 through P+K in the attention cache. When we restore the checkpoint (S0), the attention cache is also restored to pre-draft state, losing the accepted tokens' KV entries.

**So the tape would need to capture:**
- R/S state at position K (already in `n_rs_seq` snapshot)
- Attention KV for positions P+1 through P+K

**How much VRAM would the tape consume?**

For the attention KV cache of K accepted tokens:
- Per layer: `2 * n_kv_heads * head_dim * K * cell_size` bytes (K and V)
- For Qwen3.6 3B with ~36 attention layers, K=12:
  - Approximate: `36 * 2 * 72 * 128 * 12 * 2 bytes` ≈ ~16 MB per sequence

This is modest but not trivial. For a typical serving scenario with multiple parallel sequences, multiply by `n_parallel`.

**Which files/functions would need modification?**

- [`tools/server/server-context.cpp:4219-4264`](tools/server/server-context.cpp:4219) — add tape capture during verification
- [`common/common.h:1181-1229`](common/common.h:1181) — extend `common_prompt_checkpoint` with tape buffer
- [`src/llama-kv-cache.cpp`](src/llama-kv-cache.cpp) — add KV range copy method
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — add R/S range copy method

**Verdict on Option C:** This is the most complex option and requires the most new code. It would need a KV cache range-copy mechanism, tape storage, and tape replay logic. The VRAM savings over Option B are modest (avoiding re-decode compute, not VRAM).

---

*(Appendix follows below)*

---

## Section 7 — Linear DFlash vs DDTree

### 7.0 Scope Separation

The original task asks about recovering accepted recurrent state. This analysis has focused on **linear DFlash** (no branches, single draft chain). DDTree adds complications that are addressed separately.

**Linear DFlash characteristics:**
- Single draft chain: tokens drafted in order 1, 2, 3, ..., N
- Verification produces a single acceptance boundary: tokens 1..K accepted, K+1..N rejected
- No branches, no tree structure, no speculative forking
- The acceptance boundary is a simple cutoff point

**DDTree characteristics (NOT in scope for this analysis):**
- Multiple draft branches from a single position
- Tree-structured acceptance/rejection
- Requires tree-aware tape to track which branches were accepted
- Significantly more complex state recovery

### 7. Linear DFlash Simplifications

For linear DFlash specifically, several simplifications apply:

#### 7.1.1 Attention KV Can Simply Retain Accepted Positions

For hybrid models, after verification accepts tokens at positions P+1 through P+K:

- The attention KV cache at positions P+1..P+K is valid (these tokens were verified)
- The attention KV cache at positions P+K+1..P+N is invalid (rejected tokens)

**Question:** Can we simply delete the rejected positions and keep the accepted ones?

**Answer:** YES, for the attention KV cache. The standard KV cache path uses `seq_rm` to remove rejected positions:

```cpp
// After restoring recurrent state to SK, remove rejected KV entries
common_context_seq_rm(ctx_tgt, slot.id, P + K + 1, P + N + 1);
```

This is the standard behavior for `COMMON_CONTEXT_SEQ_RM_TYPE_PART` (arbitrary range removal). The attention cache is NOT part of the recurrent state that needs checkpointing.

**BUT:** When `use_ckpt_tgt` is true, the checkpoint restore at [`server-context.cpp:4239`](tools/server/server-context.cpp:4239) restores the FULL partial state, including the attention KV cache. After restore, the KV cache is back to pre-draft state (no accepted tokens). The accepted tokens' KV entries are lost.

**This is the core problem:** The checkpoint restore is too aggressive — it restores S0, wiping both the recurrent state AND the attention KV cache. The accepted tokens' KV entries are valid and should be preserved.

#### 7.1.2 Recurrent Backup Only Needs R+S State

For the recurrent memory, the backup contains:
- R tensor data (running state)
- S tensor data (snapshot state)

These are the only tensors that need backup/restore for recurrent models. The attention KV cache is handled separately.

**Question:** Is full KV backup unnecessary for recurrent state recovery?

**Answer:** For pure recurrent models (Mamba, RWKV), there IS no attention KV cache. The backup only contains R/S state, and the `n_rs_seq` snapshots already capture this.

For hybrid models (Qwen3.6, Gemma 4), the backup contains BOTH recurrent state AND attention KV cache. The attention KV portion of the backup is what makes checkpoint restore too aggressive.

#### 7.1.3 Branch Tape/Tree Machinery NOT Required for Linear DFlash

For linear DFlash:
- No tree structure
- No branches
- Single acceptance boundary
- Simple position-based rollback

The tape system from the old fork (DDTree, CopySpec, etc.) was designed for tree-structured speculation with multiple branches. For linear DFlash, we only need:
1. R/S state at position K (already in `n_rs_seq`)
2. Attention KV for positions P+1..P+K (already in the cache, lost on checkpoint restore)

### 7.2 Refined Option A: Selective Restore

Given the analysis above, a refined Option A emerges:

**Instead of restoring the full checkpoint (which wipes both R/S and KV), restore ONLY the recurrent state and preserve the attention KV cache for accepted tokens.**

This requires:

1. **Split the checkpoint into recurrent-only and attention-KV portions**
   - The recurrent portion restores R/S to S0
   - The attention KV is NOT restored (preserves accepted token entries)

2. **Use `seq_rm` to set `rs_idx` to select snapshot SK**
   - This advances the recurrent state from S0 to SK using the snapshot

3. **Use `seq_rm` to remove rejected KV entries**
   - Remove positions P+K+1..P+N from the attention cache
   - Keep positions P+1..P+K (accepted tokens)

**This approach:**
- Eliminates re-decode of accepted tokens
- Preserves valid attention KV entries
- Uses existing `n_rs_seq` infrastructure
- Requires minimal new code

**Files to modify:**
- [`tools/server/server-context.cpp:4221-4264`](tools/server/server-context.cpp:4221) — add selective restore path
- [`common/common.h:1181-1229`](common/common.h:1181) — add method to restore only recurrent state
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — expose `rs_idx` selection through public API
- [`src/llama-kv-cache.cpp`](src/llama-kv-cache.cpp) — ensure `seq_rm` works for accepted range cleanup

**VRAM cost:** Zero additional VRAM (uses existing snapshots and cache).

**Speed impact:** Eliminates re-decode entirely. Accepted tokens are verified once, and their state (both R/S and KV) is preserved for the next cycle.

### 7.3 DDTree Complications (Out of Scope)

For DDTree (if implemented in the future), the following additional complications exist:

1. **Multiple acceptance boundaries** — different branches may accept different numbers of tokens
2. **Tree-structured KV cache** — rejected branches need cleanup, accepted branches need preservation
3. **Branch-aware tape** — need to track which snapshots correspond to which branches
4. **Complex replay** — replaying S0 → SK for each accepted branch independently

These complications require the full tape system (DDTree, branch tracking, etc.) that was removed in v0.4.0. For the current linear DFlash implementation, these are not relevant.

---

## Section 8 — Final Recommendation

### 8.1 Assessment Summary

| Option | VRAM Cost | Speed Impact | Code Complexity | Risk |
|--------|-----------|-------------|-----------------|------|
| **A: Use `n_rs_seq` snapshots** | Zero | Eliminates re-decode | Low | Medium |
| **B: Re-decode (current)** | Zero | K tokens re-verified per cycle | None | None |
| **C: Minimum tape** | ~16 MB/seq | Eliminates re-decode | High | High |
| **A-refined: Selective restore** | Zero | Eliminates re-decode | Medium | Medium |

### 8.2 Recommendation

**For linear DFlash, Option A-refined (Selective Restore) is the best approach:**

1. **Low risk** — uses existing `n_rs_seq` infrastructure
2. **Zero VRAM overhead** — no new allocations
3. **Eliminates re-decode** — accepted tokens verified once
4. **Minimal code change** — modify verification path at [`server-context.cpp:4221`](tools/server/server-context.cpp:4221)

**Implementation steps:**

1. Add `llama_memory_set_rs_idx()` public API to select snapshot group
2. Modify checkpoint restore to split recurrent state and attention KV
3. After verification with partial acceptance:
   a. Restore recurrent state only (not attention KV)
   b. Use `seq_rm` to set `rs_idx` to select snapshot SK
   c. Use `seq_rm` to remove rejected KV positions P+K+1..P+N
4. Clean up `rs_idx` after next decode (already done in `s_copy()`)

**Files to modify:**
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) — verification path
- [`common/common.h`](common/common.h) — checkpoint struct
- [`common/common.cpp`](common/common.cpp) — selective restore methods
- [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) — public API
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — selective restore
- [`src/llama.h`](include/llama.h) — new API declarations

**Estimated scope:** 2-3 days of focused implementation work.

### 8.3 Alternative: Accept Current Behavior

If the re-decode overhead is acceptable for the target workload, Option B (current behavior) requires no code changes. The overhead is:
- ~10-12 tokens re-verified per cycle for typical acceptance rates
- Batched with new draft verification, so some compute overlap
- No VRAM overhead

For workloads where DFlash drafting is the bottleneck (not target verification), the re-decode overhead may be negligible. Benchmark before optimizing.

---

## Appendix A: Key Code References

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `n_rs_seq` configuration | [`common/common.h`](common/common.h:417) | 417-423 | DFlash requests `draft.n_max` snapshots |
| Snapshot storage | [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h:79) | 79-85 | Tensor widening for snapshots |
| `seq_rm` with `rs_idx` | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:214) | 214-222 | Sets rollback index |
| `s_copy()` | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:1330) | 1330-1348 | Reads from snapshot group |
| Checkpoint struct | [`common/common.h`](common/common.h:1181) | 1181-1229 | `common_prompt_checkpoint` |
| Checkpoint decision | [`tools/server/server-task.h`](tools/server/server-task.h:20) | 20-26 | When checkpoint is required |
| Verification path | [`tools/server/server-context.cpp`](tools/server/server-context.cpp:4219) | 4219-4274 | Speculative verification and rollback |
| Checkpoint save | [`tools/server/server-context.cpp`](tools/server/server-context.cpp:3238) | 3238-3319 | Pre-draft checkpoint |
| Checkpoint restore | [`tools/server/server-context.cpp`](tools/server/server-context.cpp:4239) | 4239-4248 | Post-rollback restore |
| seq_rm capability | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:150) | 150-177 | `can_seq_rm()` |
| seq_rm_type enum | [`common/common.h`](common/common.h:1022) | 1022-1027 | `COMMON_CONTEXT_SEQ_RM_TYPE_*` |
|| | | |
| `common_replay_last_token` | [`common/common.cpp`](common/common.cpp:2191) | 2191-2199 | Single-token replay |
| Eagle3 g_embd stash | [`common/speculative.cpp`](common/speculative.cpp:857) | 857-898 | Draft-side state preservation |
| DFlash impl | [`common/speculative.cpp`](common/speculative.cpp:906) | 906-990 | DFlash speculative decoding |
| DFlash model graph | [`src/models/dflash.cpp`](src/models/dflash.cpp:1) | 1-320 | DFlash encoder/decoder graphs |
| DeltaNet recurrent attn | [`src/models/delta-net-base.cpp`](src/models/delta-net-base.cpp:527) | 527-595 | Recurrent attention build |
| Hybrid memory | [`src/llama-memory-hybrid.cpp`](src/llama-memory-hybrid.cpp:1) | 1-400 | Hybrid attention+recurrent memory |
| Recurrent memory | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:1) | 1-1349 | Recurrent memory implementation |
| State write | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:822) | 822-900 | `state_write()` with rs_idx |
| State read | [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:902) | 902-1130 | `state_read()` with rs_idx |
| Test: recurrent rollback | [`tests/test-recurrent-state-rollback.cpp`](tests/test-recurrent-state-rollback.cpp:1) | 1-186 | Unit test for rollback |

---

*End of document.*
