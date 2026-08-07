## Section 9 — Does n_rs_seq=0 Break Anything Else in DFlash?

### 9.1 Complete Audit of n_rs_seq / suffix_rollback_tokens / rs_idx / can_seq_rm / seq_rm Occurrences

Every occurrence was classified into one of five categories:

| Category | Meaning |
|---|---|
| **MUST CHANGE** | DFlash-specific code that assumes RS snapshots exist |
| **HARMLESS** | Works fine with `n_rs_seq = 0` |
| **GENERIC ROLLBACK** | Affects all speculative types equally |
| **MTP/EAGLE3 ONLY** | Does not affect DFlash |
| **UNRELATED** | No impact on DFlash |

---

### Group A: Target Context Configuration

#### [`common/common.h:417-423`](common/common.h:417) — `need_n_rs_seq()`

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

**Classification: GENERIC ROLLBACK**

DFlash is explicitly listed as needing `n_rs_seq = draft.n_max`. This is the source value for ALL speculative types. For DFlash, this returns `draft.n_max` (typically 8-15).

**Key insight:** DFlash requests `n_rs_seq` snapshots but the current runtime shows `n_rs_seq = 0`. This means either:
1. The arch check at [`llama-context.cpp:266-270`](src/llama-context.cpp:266) clamps to 0, OR
2. The model doesn't use recurrent memory (pure attention model with no `llama_memory_recurrent`)

For Qwen3.6 and Gemma 4 (hybrid models), `need_n_rs_seq()` returns `draft.n_max`, but the recurrent memory component receives this value. The KV cache component receives `kv_tail_rollback_tokens = n_rs_seq` at [`llama-context.cpp:264`](src/llama-context.cpp:264).

#### [`common/common.cpp:1770`](common/common.cpp:1770) — Context params conversion

```cpp
cparams.n_rs_seq = params.speculative.need_n_rs_seq();
```

**Classification: GENERIC ROLLBACK**

Passes `need_n_rs_seq()` to target context. Not DFlash-specific.

#### [`src/llama-context.cpp:264-270`](src/llama-context.cpp:264) — Arch support check

```cpp
cparams.kv_tail_rollback_tokens = params.n_rs_seq;
cparams.n_rs_seq = params.n_rs_seq;
if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
    cparams.n_rs_seq = 0;  // CLAMPED
}
```

**Classification: MUST CHANGE (if arch not in whitelist)**

This is the gate that can clamp `n_rs_seq` to 0. If the DFlash target model's architecture is NOT in `llm_arch_supports_rs_rollback()`, then `n_rs_seq` becomes 0 and triggers the checkpoint path.

**Action required:** Verify that Qwen3.6 and Gemma 4 architectures ARE in `llm_arch_supports_rs_rollback()`. If not, DFlash will always use checkpoints.

---

### 9.3 Draft Context Configuration

#### [`common/speculative.cpp:2286`](common/speculative.cpp:2286) — Draft context n_rs_seq = 0

```cpp
cparams.n_rs_seq = 0;
```

**Classification: HARMLESS**

The draft context never needs RS snapshots. This is correct — DFlash drafts don't use recurrent rollback on the draft side.

#### [`tools/server/server-context.cpp:1182`](tools/server/server-context.cpp:1182) — Server draft context

```cpp
cparams_dft.n_rs_seq = 0;
```

**Classification: HARMLESS**

Same as above — draft context doesn't need RS.

---

### 9.4 Recurrent Memory (llama_memory_recurrent)

#### [`src/llama-memory-recurrent.cpp:35`](src/llama-memory-recurrent.cpp:35) — Constructor stores n_rs_seq

```cpp
this->n_rs_seq = n_rs_seq;
rs_idx.assign(n_seq_max, 0);
```

**Classification: HARMLESS**

Stores the value. When `n_rs_seq = 0`, the tensor is sized normally (no snapshot columns).

#### [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) — Tensor sizing

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, tensors are sized for `mem_size * 1` (no snapshot columns). This is correct.

#### [`src/llama-memory-recurrent.cpp:173-174`](src/llama-memory-recurrent.cpp:173) — `can_seq_rm` check

```cpp
const llama_pos rollback = cell.pos - (p0 - 1);
return rollback >= 1 && rollback <= llama_pos(n_rs_seq);
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, this returns `false` for any `rollback >= 1`, meaning partial removal is not supported. This correctly causes `common_context_can_seq_rm()` to return `TYPE_FULL` instead of `TYPE_RS`.

#### [`src/llama-memory-recurrent.cpp:214-219`](src/llama-memory-recurrent.cpp:214) — `seq_rm` partial rollback

```cpp
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
    }
}
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, the inner `if` never triggers. The code falls through to full sequence removal. This is correct behavior.

#### [`src/llama-memory-recurrent.cpp:473-478`](src/llama-memory-recurrent.cpp:473) — `set_rs_idx()`

```cpp
void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0 || (size_t) seq_id >= rs_idx.size()) {
        return;
    }
    rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
}
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, any `idx > 0` is clamped to 0. Effectively a no-op.

#### [`src/llama-memory-recurrent.cpp:503-505`](src/llama-memory-recurrent.cpp:503) — Ubatch split for rollback

```cpp
ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, no tail tokens are kept together. This is correct.

#### [`src/llama-memory-recurrent.cpp:781-786`](src/llama-memory-recurrent.cpp:781) — Capability reporting

```cpp
llama_memory_i::seq_rm_capability llama_memory_recurrent::get_seq_rm_capability() const {
    return {
        /* .full_clear = */ true,
        /* .arbitrary_ranges = */ false,
        /* .suffix_rollback_tokens = */ n_rs_seq,
    };
}
```

**Classification: GENERIC ROLLBACK**

This is the capability that drives the checkpoint decision. When `n_rs_seq = 0`, the memory reports `suffix_rollback_tokens = 0`, causing `TYPE_FULL` and checkpoint usage.

#### [`src/llama-memory-recurrent.cpp:837-848`](src/llama-memory-recurrent.cpp:837) — State save with rs_idx

```cpp
if (n_rs_seq != 0) {
    // save rs_idx state
}
```

**Classification: HARMLESS**

Guarded by `n_rs_seq != 0`. When 0, skips rs_idx save. Correct.

#### [`src/llama-memory-recurrent.cpp:921-927`](src/llama-memory-recurrent.cpp:921) — State restore with rs_idx

```cpp
if (n_rs_seq != 0) {
    // restore rs_idx state
}
```

**Classification: HARMLESS**

Guarded by `n_rs_seq != 0`.

#### [`src/llama-memory-recurrent.cpp:1330-1348`](src/llama-memory-recurrent.cpp:1330) — `s_copy()` for cell copy

```cpp
int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;  // Returns base index when no snapshots
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
```

**Classification: MUST CHANGE (for DFlash backup API)**

This is the `s_copy()` function referenced in Part 3 as the cell_copy API. When `n_rs_seq = 0`, it returns `src0` (the base index). For DFlash backup/restore to work with recurrent state, this function needs to be accessible as a public API. Currently it is a private method of `llama_memory_recurrent_context`.

---

### 9.5 Hybrid Memory

#### [`src/llama-memory-hybrid.cpp:161-162`](src/llama-memory-hybrid.cpp:161) — Combined capability

```cpp
llama_memory_i::seq_rm_capability llama_memory_hybrid::get_seq_rm_capability() const {
    return llama_memory_seq_rm_capability_all({ mem_attn.get(), mem_recr.get() });
}
```

**Classification: GENERIC ROLLBACK**

Takes minimum of attention and recurrent capabilities. If recurrent reports `suffix_rollback_tokens = 0`, the hybrid reports 0.

#### [`src/llama-memory-hybrid.cpp:107-112`](src/llama-memory-hybrid.cpp:107) — Ubatch split

```cpp
const uint32_t n_rs_seq = mem_recr->n_rs_seq;
ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, no tail grouping. Correct.

#### [`src/llama-memory-hybrid-iswa.cpp:104-109`](src/llama-memory-hybrid-iswa.cpp:104) — Same pattern

**Classification: HARMLESS**

Same as hybrid.

---

### 9.6 KV Cache

#### [`src/llama-kv-cache.cpp:1509-1517`](src/llama-kv-cache.cpp:1509) — Capability

```cpp
llama_memory_i::seq_rm_capability llama_kv_cache::get_seq_rm_capability() const {
    if (has_compact_tail()) {
        return {
            /* .full_clear = */ true,
            /* .arbitrary_ranges = */ false,
            /* .suffix_rollback_tokens = */ tail_rollback_tokens,
        };
    }
    return {};  // Default: arbitrary_ranges = true
}
```

**Classification: GENERIC ROLLBACK**

For KVarN caches with compact tail, reports `tail_rollback_tokens`. For standard caches without compact tail, returns default (arbitrary_ranges = true).

**Important:** If the KV cache returns `arbitrary_ranges = true`, then `common_context_can_seq_rm()` returns `TYPE_PART`, meaning NO checkpoint needed. This is the normal path for pure-attention models.

For hybrid models, the KV cache's `arbitrary_ranges = false` (if compact tail) means the capability falls to `suffix_rollback_tokens`, which is `tail_rollback_tokens = kv_tail_rollback_tokens = n_rs_seq`.

---

### 9.7 Delta-Net Base Model (Qwen3.6 / Gemma 4 graph builder)

#### [`src/models/delta-net-base.cpp:479-522`](src/models/delta-net-base.cpp:479) — Conv state build

```cpp
if (cparams.n_rs_seq == 0) {
    // Simple path: copy last conv state to position 0
    const int64_t s_idx = conv_input->ne[0] - conv_states->ne[0];
    // ... copy to slot 0 only ...
} else {
    // Rollback path: copy K+1 states for snapshot columns
    const int64_t K = (int64_t) cparams.n_rs_seq + 1;
    for (int64_t t = 1; t <= K; ++t) {
        // ... copy to snapshot slots ...
    }
}
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, uses the simple path (no snapshot columns). The graph builder correctly handles both cases.

#### [`src/models/delta-net-base.cpp:546`](src/models/delta-net-base.cpp:546) — Keep flag

```cpp
const bool keep = cparams.n_rs_seq > 0;
```

**Classification: HARMLESS**

When `n_rs_seq = 0`, `keep = false`. Correct.

#### [`src/models/delta-net-base.cpp:564`](src/models/delta-net-base.cpp:564) — K calculation

```cpp
const int64_t K = cparams.n_rs_seq + 1;
```

**Classification: HARMLESS**

Only used in the `n_rs_seq > 0` branch.

---

### 9.8 Server Code

#### [`tools/server/server-context.cpp:976-977`](tools/server/server-context.cpp:976) — Member variables

```cpp
common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
```

**Classification: GENERIC ROLLBACK**

Server-wide type determination.

#### [`tools/server/server-context.cpp:3219-3220`](tools/server/server-context.cpp:3219) — Draft phase checkpoint check

```cpp
const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
```

**Classification: GENERIC ROLLBACK**

Simple type check. When `TYPE_FULL`, checkpoint is used.

#### [`tools/server/server-context.cpp:3232-3234`](tools/server/server-context.cpp:3232) — Assert checkpoint exists

```cpp
if (use_ckpt_tgt) {
    GGML_ASSERT(!slot.spec_ckpt.empty());
}
```

**Classification: GENERIC ROLLBACK**

Validates checkpoint was created before draft.

#### [`tools/server/server-context.cpp:3306-3328`](tools/server/server-context.cpp:3306) — Draft checkpoint creation

```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());
```

**Classification: GENERIC ROLLBACK — but DFlash is affected**

When `TYPE_FULL` and `max_rollback = 0`, this returns `true` for any `draft.size() > 0`. DFlash will checkpoint every draft cycle.

#### [`tools/server/server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221) — Verification checkpoint restore

```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);

if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
        // ... restore sampler, loop guard, prompt ...
    }
}
```

**Classification: GENERIC ROLLBACK — but DFlash is affected**

This is the critical path. When DFlash has partial acceptance and `use_ckpt_tgt = true`, the server:
1. Restores the full checkpoint (expensive serialization).
2. Removes tokens after `ckpt.pos_max`.
3. Restores sampler state, loop guard, and prompt.

For DFlash, this means re-serializing the entire target context state on every partial acceptance cycle.

#### [`tools/server/server-context.cpp:3784-3787`](tools/server/server-context.cpp:3784) — Checkpoint decision for partial acceptance

```cpp
do_checkpoint = do_checkpoint && (
        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
        n_swa > 0);
```

**Classification: GENERIC ROLLBACK**

Includes `TYPE_RS` in checkpoint eligibility.

---

### 9.9 Public API

#### [`include/llama.h:426`](include/llama.h:426) — `n_rs_seq` in context params

```cpp
uint32_t n_rs_seq;  // number of recurrent-state snapshots per seq for rollback
```

**Classification: UNRELATED**

Public API definition.

#### [`include/llama.h:679`](include/llama.h:679) — `llama_n_rs_seq()` getter

```cpp
LLAMA_API uint32_t llama_n_rs_seq(const struct llama_context * ctx);
```

**Classification: UNRELATED**

Public API getter.

#### [`include/llama.h:864`](include/llama.h:864) — `suffix_rollback_tokens` in capability

**Classification: UNRELATED**

Public API structure.

#### [`src/llama-context.cpp:4147-4149`](src/llama-context.cpp:4147) — `llama_n_rs_seq()` implementation

**Classification: UNRELATED**

Returns `cparams.n_rs_seq`.

---

### 9.10 Test Code

#### [`tests/test-recurrent-state-rollback.cpp:14-16`](tests/test-recurrent-state-rollback.cpp:14)

```cpp
cparams.n_rs_seq = 8;
cparams.n_batch = std::max(cparams.n_batch, (uint32_t)(cparams.n_rs_seq + 1));
```

**Classification: MTP/EAGLE3 ONLY**

Tests recurrent state rollback. Not DFlash-specific.

#### [`tests/test-recurrent-state-rollback.cpp:75-77`](tests/test-recurrent-state-rollback.cpp:75)

```cpp
if (llama_n_rs_seq(ctx_src) == 0) {
    // skip test
}
```

**Classification: MTP/EAGLE3 ONLY**

Skips test when RS not available.

---

### 9.11 Summary Classification Table

| File:Line | Symbol | Classification | Impact on DFlash with n_rs_seq=0 |
|---|---|---|---|
| [`common/common.h:417-423`](common/common.h:417) | `need_n_rs_seq()` | GENERIC ROLLBACK | Returns `draft.n_max` for DFlash |
| [`common/common.cpp:1770`](common/common.cpp:1770) | Context params | GENERIC ROLLBACK | Passes to target context |
| [`src/llama-context.cpp:264-270`](src/llama-context.cpp:264) | Arch check | **MUST CHANGE?** | Can clamp to 0 if arch not whitelisted |
| [`common/speculative.cpp:2286`](common/speculative.cpp:2286) | Draft n_rs_seq=0 | HARMLESS | Draft never needs RS |
| [`tools/server/server-context.cpp:1182`](tools/server/server-context.cpp:1182) | Server draft | HARMLESS | Same |
| [`src/llama-memory-recurrent.cpp:35`](src/llama-memory-recurrent.cpp:35) | Store n_rs_seq | HARMLESS | Stores 0 |
| [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) | Tensor sizing | HARMLESS | No snapshot columns |
| [`src/llama-memory-recurrent.cpp:173-174`](src/llama-memory-recurrent.cpp:173) | can_seq_rm | HARMLESS | Returns false for partial |
| [`src/llama-memory-recurrent.cpp:214-219`](src/llama-memory-recurrent.cpp:214) | seq_rm partial | HARMLESS | Skips, falls to full |
| [`src/llama-memory-recurrent.cpp:473-478`](src/llama-memory-recurrent.cpp:473) | set_rs_idx | HARMLESS | Clamps to 0 |
| [`src/llama-memory-recurrent.cpp:503-505`](src/llama-memory-recurrent.cpp:503) | Ubatch split | HARMLESS | No tail grouping |
| [`src/llama-memory-recurrent.cpp:781-786`](src/llama-memory-recurrent.cpp:781) | Capability | GENERIC ROLLBACK | Reports 0 suffix_rollback |
| [`src/llama-memory-recurrent.cpp:837-848`](src/llama-memory-recurrent.cpp:837) | State save | HARMLESS | Guarded by n_rs_seq |
| [`src/llama-memory-recurrent.cpp:921-927`](src/llama-memory-recurrent.cpp:921) | State restore | HARMLESS | Guarded by n_rs_seq |
| [`src/llama-memory-recurrent.cpp:1330-1348`](src/llama-memory-recurrent.cpp:1330) | s_copy() | **MUST CHANGE** | Needs public API for DFlash backup |
| [`src/llama-memory-hybrid.cpp:161-162`](src/llama-memory-hybrid.cpp:161) | Hybrid capability | GENERIC ROLLBACK | Min of children |
| [`src/llama-memory-hybrid.cpp:107-112`](src/llama-memory-hybrid.cpp:107) | Hybrid ubatch | HARMLESS | No tail grouping |
| [`src/llama-kv-cache.cpp:1509-1517`](src/llama-kv-cache.cpp:1509) | KV capability | GENERIC ROLLBACK | Reports tail_rollback_tokens |
| [`src/models/delta-net-base.cpp:479-522`](src/models/delta-net-base.cpp:479) | Conv state | HARMLESS | Simple path when 0 |
| [`src/models/delta-net-base.cpp:546`](src/models/delta-net-base.cpp:546) | Keep flag | HARMLESS | false when 0 |
| [`tools/server/server-context.cpp:976-977`](tools/server/server-context.cpp:976) | Server members | GENERIC ROLLBACK | Type variables |
| [`tools/server/server-context.cpp:3219-3220`](tools/server/server-context.cpp:3219) | Draft checkpoint | GENERIC ROLLBACK | TYPE_FULL triggers |
| [`tools/server/server-context.cpp:3306-3328`](tools/server/server-context.cpp:3306) | Draft save | **GENERIC — DFlash affected** | Checkpoints every draft |
| [`tools/server/server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221) | Verify restore | **GENERIC — DFlash affected** | Checkpoints every partial accept |
| [`tools/server/server-task.h:20-26`](tools/server/server-task.h:20) | Decision function | GENERIC ROLLBACK | Core decision logic |
| [`common/common.cpp:1666-1686`](common/common.cpp:1666) | can_seq_rm | GENERIC ROLLBACK | Maps capability to type |

---

### 9.12 Key Question: Does DFlash-Specific Code Directly Assume n_rs_seq > 0?

**Answer: NO — DFlash-specific code does NOT directly assume `n_rs_seq > 0`.**

The DFlash speculative implementation in [`common/speculative.cpp:905-1201`](common/speculative.cpp:905):

1. **[`common_speculative_impl_draft_dflash::accept()`](common/speculative.cpp:1195-1197)** — Is a no-op. DFlash does not use `set_rs_idx()` or `rs_idx` in its accept path.

2. **[`common_speculative_impl_draft_dflash::draft()`](common/speculative.cpp:1109-1193)** — Does not reference `n_rs_seq`, `rs_idx`, or snapshot state.

3. **[`common_speculative_impl_draft_dflash::process()`](common/speculative.cpp:1012-1107)** — Does not reference `n_rs_seq`.

4. **[`common_speculative_impl_draft_dflash::begin()`](common/speculative.cpp:994-1010)** — Does not reference `n_rs_seq`.

DFlash's `accept()` method is explicitly a no-op:

```cpp
// common/speculative.cpp:1195-1197
void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
}
```

**The only "DFlash-specific" code affected by `n_rs_seq = 0` is the generic checkpoint machinery** that the server uses for ALL speculative types when the memory reports `suffix_rollback_tokens = 0`.

---

### 9.13 Root Cause Summary

The chain of events when `n_rs_seq = 0` for DFlash:

```
need_n_rs_seq() returns draft.n_max
        |
        v
cparams.n_rs_seq = draft.n_max
        |
        v
llama-context.cpp:266 — arch check
        |
        +-- arch NOT in whitelist --> n_rs_seq clamped to 0
        |
        +-- arch IS in whitelist --> n_rs_seq = draft.n_max (OK)
        |
        v
llama_memory_recurrent stores n_rs_seq
        |
        v
get_seq_rm_capability() reports suffix_rollback_tokens = n_rs_seq
        |
        +-- = 0 --> TYPE_FULL --> checkpoints every cycle
        |
        +-- > 0 --> TYPE_RS --> seq_rm for partial rollback
        |
        v
common_context_can_seq_rm() returns TYPE
        |
        v
server_speculative_rollback_requires_checkpoint()
        |
        +-- TYPE_FULL --> true (checkpoint)
        +-- TYPE_RS, rollback <= max --> false (seq_rm)
```

**The single critical question is: Is the DFlash target model's architecture in `llm_arch_supports_rs_rollback()`?**

If YES → `n_rs_seq = draft.n_max` → `TYPE_RS` → no checkpoint needed → DFlash works correctly.
If NO → `n_rs_seq = 0` → `TYPE_FULL` → checkpoint every cycle → DFlash works but with performance penalty.

---

### 9.14 Recommendations

1. **Verify `llm_arch_supports_rs_rollback()` includes Qwen3.6 and Gemma 4 architectures.** If not, add them.

2. **If `n_rs_seq` must be 0 for DFlash** (because the target is a pure-attention model without recurrent state), implement the server-level override from Section 8.4.2 to prevent unnecessary checkpoint serialization.

3. **Make `s_copy()` public** (from [`llama-memory-recurrent.cpp:1330`](src/llama-memory-recurrent.cpp:1330)) for the DFlash cell_copy API discussed in Part 3.

4. **Consider adding DFlash-specific capability** to `seq_rm_capability` to signal "this memory supports lightweight backup for DFlash rollback" without requiring RS snapshots.
