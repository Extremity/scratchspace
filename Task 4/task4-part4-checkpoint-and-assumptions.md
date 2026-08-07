# Task 4.4: Checkpoint Machinery Interaction and n_rs_seq=0 Assumptions

## Overview

This document investigates two questions for upstream DFlash:

1. **Section 8:** How does checkpoint machinery interact with DFlash when `n_rs_seq=0`, and what is the smallest clean fix?
2. **Section 9:** Does `n_rs_seq=0` break anything else in DFlash-specific code?

---

## Section 8 — Checkpoint Machinery with n_rs_seq=0

### 8.1 How the Checkpoint Decision Works

The server's speculative rollback strategy is determined at server initialization and re-evaluated at each draft cycle. The decision chain flows through four layers:

#### Layer 1: Memory Capability Reporting

Each memory module reports its `seq_rm_capability` via [`llama_memory_i::get_seq_rm_capability()`](src/llama-memory.h:157):

```cpp
// src/llama-memory.h:151-155
struct seq_rm_capability {
    bool full_clear = true;
    bool arbitrary_ranges = true;
    uint32_t suffix_rollback_tokens = UINT32_MAX;
};
```

Key implementations:

| Memory Module | File:Line | `suffix_rollback_tokens` |
|---|---|---|
| [`llama_memory_recurrent`](src/llama-memory-recurrent.cpp:781-786) | `n_rs_seq` |
| [`llama_kv_cache`](src/llama-kv-cache.cpp:1509-1517) | `tail_rollback_tokens` (if `has_compact_tail()`) |
| [`llama_kv_cache_kvarn`](src/llama-kv-cache-kvarn.cpp:1672-1673) | Delegates to metadata |
| [`llama_memory_hybrid`](src/llama-memory-hybrid.cpp:161-162) | Min of children via [`llama_memory_seq_rm_capability_all()`](src/llama-memory.h:267-279) |

#### Layer 2: Common Capability Query

[`common_context_can_seq_rm()`](common/common.cpp:1666-1686) maps capability to a `COMMON_CONTEXT_SEQ_RM_TYPE`:

```cpp
// common/common.cpp:1666-1686
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }
    const auto capability = llama_memory_get_seq_rm_capability(mem);
    if (capability.arbitrary_ranges) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_PART;  // no checkpoint needed
    }
    if (capability.suffix_rollback_tokens > 0) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;    // bounded partial removal
    }
    if (capability.full_clear) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_FULL;  // checkpoint needed
    }
    return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
}
```

```cpp
// common/common.cpp:1688-1695
uint32_t common_context_seq_rm_max_rollback(llama_context * ctx) {
    const auto capability = llama_memory_get_seq_rm_capability(mem);
    return capability.arbitrary_ranges ? UINT32_MAX : capability.suffix_rollback_tokens;
}
```

#### Layer 3: Server Initialization

At [`server_context_impl::init()`](tools/server/server-context.cpp:1389-1396):

```cpp
// tools/server/server-context.cpp:1389-1396
ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
    SRV_WRN("%s", "speculative decoding not supported by this context\n");
}
if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
    SRV_TRC("%s", "speculative decoding will use checkpoints\n");
}
```

#### Layer 4: Runtime Checkpoint Decision

**At draft time** ([`server-context.cpp:3306-3328`](tools/server/server-context.cpp:3305-3329)):

```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());
```

**At verification/acceptance time** ([[`server-context.cpp:4221-4222`](tools/server/server-context.cpp:4221-4222)](tools/server/server-context.cpp:4221)):

```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);
```

The decision function ([`server_speculative_rollback_requires_checkpoint()`](tools/server/server-task.h:20-26)):

```cpp
// tools/server/server-task.h:20-26
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
          (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

### 8.2 What Happens When n_rs_seq=0 for DFlash

When `n_rs_seq = 0`:

| Memory Component | `suffix_rollback_tokens` | Reason |
|---|---|---|
| `llama_memory_recurrent` | **0** | Directly reports `n_rs_seq` |
| `llama_kv_cache` (KVarN) | `tail_rollback_tokens` | From `kv_tail_rollback_tokens` |

For hybrid models (Qwen3.6, Gemma 4), the combined capability uses the **minimum** of children:

```cpp
// src/llama-memory.h:267-279 — llama_memory_seq_rm_capability_all()
result.suffix_rollback_tokens = std::min(result.suffix_rollback_tokens,
                                          child->get_seq_rm_capability().suffix_rollback_tokens);
```

So when recurrent reports `suffix_rollback_tokens = 0`, the hybrid memory reports `suffix_rollback_tokens = 0`.

This causes `common_context_can_seq_rm()` to skip the `TYPE_RS` branch and fall through to `TYPE_FULL` (since `full_clear = true`).

**Result:**

- `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL`
- `common_context_seq_rm_max_rollback(ctx_tgt) = 0`
- `server_speculative_rollback_requires_checkpoint()` returns **`true`** for ANY `n_rollback > 0`

This means **every speculative cycle that has a partial acceptance triggers full checkpoint serialization and restore**:

```cpp
// tools/server/server-context.cpp:3312-3324 — checkpoint save (draft time)
if (use_ckpt_tgt) {
    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
}

// tools/server/server-context.cpp:4226-4263 — checkpoint restore (verification time)
if (use_ckpt_tgt) {
    ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
    // ... restore sampler, loop guard, prompt tokens ...
}
```

### 8.3 Where n_rs_seq IS Set for DFlash

DFlash **does** request `n_rs_seq` snapshots. The value comes from [`common_params_speculative::need_n_rs_seq()`](common/common.h:417-423):

```cpp
// common/common.h:417-423
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });
    return needs_rs_seq ? draft.n_max : 0u;
}
```

This flows to [`common_context_params_to_llama()`](common/common.cpp:1770):

```cpp
cparams.n_rs_seq = params.speculative.need_n_rs_seq();
```

And reaches [`llama_context.cpp:264-270`](src/llama-context.cpp:264):

```cpp
cparams.kv_tail_rollback_tokens = params.n_rs_seq;
cparams.n_rs_seq = params.n_rs_seq;
if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
    cparams.n_rs_seq = 0;  // CLAMPED if arch doesn't support it
}
```

**Critical:** If the DFlash target model's architecture does NOT appear in `llm_arch_supports_rs_rollback()`, then `n_rs_seq` is clamped to 0, triggering the checkpoint path.

The draft context explicitly sets `n_rs_seq = 0` at [`common/speculative.cpp:2286`](common/speculative.cpp:2286):

```cpp
cparams.n_rs_seq = 0;  // draft context never needs RS snapshots
```

---

## Section 8.4 — Options to Prevent Routine Checkpoint Serialization

### Option A: DFlash-specific rollback mode (New `COMMON_CONTEXT_SEQ_RM_TYPE`)

**Idea:** Add `COMMON_CONTEXT_SEQ_RM_TYPE_DFLASH = 4` that signals "DFlash has lightweight backup rollback, no checkpoint needed."

**Pros:**
- Minimal changes to server logic — just one new enum value.
- `server_speculative_rollback_requires_checkpoint()` returns `false` for the new type.
- No checkpoint save/restore for DFlash.

**Cons:**
- Requires the memory capability system to know about DFlash specifically.
- Adds a speculative-type dependency into the memory layer.

**Implementation:**
1. Add enum value in [`common/common.h:1022-1027`](common/common.h:1022).
2. Add case in [`common_context_can_seq_rm()`](common/common.cpp:1666) — but this requires knowing whether the context is using DFlash, which the memory module doesn't currently know.
3. **Problem:** `common_context_can_seq_rm()` only receives `llama_context*`. It cannot determine speculative type.

**Verdict:** Viable but requires passing speculative type info down to the capability query, or adding a server-level override.

### 8.4.2 — Server-Level Override for DFlash

**Idea:** After `ctx_tgt_seq_rm_type` is initialized at [`server-context.cpp:1389`](tools/server/server-context.cpp:1389), check if the active speculative type is DFlash and override to `TYPE_RS` with appropriate `max_rollback`.

**Pros:**
- No changes to memory layer.
- Clean separation: server knows about DFlash, memory doesn't need to.

**Cons:**
- The server still needs `common_context_seq_rm_max_rollback()` to return the right value for DFlash.
- If `suffix_rollback_tokens = 0`, the `TYPE_RS` path in `server_speculative_rollback_requires_checkpoint()` still returns `true` when `proposed_rollback > 0`.

**Refinement:** Override both the type AND track a separate `dflash_max_rollback` in the server.

**Verdict:** Practical. Requires ~20 lines of server code.

### 8.4.3 — Bypassing Generic Checkpoint Decision for DFlash

**Idea:** In the server's speculative path, add a DFlash-specific branch that skips checkpoint save/restore entirely and uses DFlash-native backup/restore instead.

**Location:** [`server-context.cpp:3306-3328`](tools/server/server-context.cpp:3306) (draft) and [`server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221) (verification).

**Pros:**
- Most flexible — allows DFlash to use its own backup mechanism.
- No changes to generic checkpoint system.

**Cons:**
- Requires DFlash backup/restore API to exist (the `cell_copy()` API discussed in Part 3).
- Duplicates rollback logic in server.

**Verdict:** Best long-term solution but requires the backup API first.

### 8.4.4 — New Memory Capability: "DFlash has lightweight backup"

**Idea:** Add a boolean field `bool lightweight_backup_rollback = false` to `seq_rm_capability`. When true, the server treats the context as `TYPE_RS` with `max_rollback = draft.n_max`.

**Pros:**
- Memory module knows its own capabilities.
- Server stays generic.

**Cons:**
- Requires memory modules to know about DFlash draft size.
- Slightly changes the capability contract.

**Verdict:** Clean but requires the memory layer to understand DFlash draft semantics.

### 8.4.5 — Recommended Approach

**Short-term (Option A-refined from Part 3):** Override `ctx_tgt_seq_rm_type` at the server level when DFlash is active:

```cpp
// After server-context.cpp:1417
if (ctx_dft && spec && is_dflash(spec)) {
    // DFlash uses draft-side backup, not target RS snapshots.
    // Tell the server not to checkpoint for routine rollbacks.
    ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    // The server will use common_context_seq_rm_max_rollback() which returns 0,
    // so we also need a DFlash-specific max_rollback override.
}
```

**Long-term:** Move DFlash rollback into DFlash-specific server path (Option 8.4.3) once `cell_copy()` backup API exists.

---

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

**Classification: HARMLESS** — Stores the value. When `n_rs_seq = 0`, the tensor is sized normally.

#### [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) — Tensor sizing: `mem_size * (1 + n_rs_seq)`

**Classification: HARMLESS** — When `n_rs_seq = 0`, tensors sized for `mem_size * 1`.

#### [`src/llama-memory-recurrent.cpp:173-174`](src/llama-memory-recurrent.cpp:173) — `can_seq_rm` check

**Classification: HARMLESS** — Returns `false` for any `rollback >= 1` when `n_rs_seq = 0`.

#### [`src/llama-memory-recurrent.cpp:214-219`](src/llama-memory-recurrent.cpp:214) — `seq_rm` partial rollback

**Classification: HARMLESS** — Inner `if` never triggers, falls to full sequence removal.

#### [`src/llama-memory-recurrent.cpp:473-478`](src/llama-memory-recurrent.cpp:473) — `set_rs_idx()`

**Classification: HARMLESS** — Any `idx > 0` clamped to 0. No-op.

#### [`src/llama-memory-recurrent.cpp:503-505`](src/llama-memory-recurrent.cpp:503) — Ubatch split

**Classification: HARMLESS** — No tail grouping when `n_rs_seq = 0`.

#### [`src/llama-memory-recurrent.cpp:781-786`](src/llama-memory-recurrent.cpp:781) — Capability reporting

**Classification: GENERIC ROLLBACK** — Reports `suffix_rollback_tokens = n_rs_seq = 0`, causing `TYPE_FULL`.

#### [`src/llama-memory-recurrent.cpp:837-848`](src/llama-memory-recurrent.cpp:837) — State save with rs_idx

**Classification: HARMLESS** — Guarded by `n_rs_seq != 0`.

#### [`src/llama-memory-recurrent.cpp:921-927`](src/llama-memory-recurrent.cpp:921) — State restore with rs_idx

**Classification: HARMLESS** — Guarded by `n_rs_seq != 0`.

#### [`src/llama-memory-recurrent.cpp:1330-1348`](src/llama-memory-recurrent.cpp:1330) — `s_copy()` for cell copy

**Classification: MUST CHANGE (for DFlash backup API)**

This is the `s_copy()` function referenced in Part 3. When `n_rs_seq = 0`, returns `src0`. For DFlash backup/restore to work with recurrent state, this needs to be a public API. Currently private.

---

### 9.5 Hybrid Memory

#### [`src/llama-memory-hybrid.cpp:161-162`](src/llama-memory-hybrid.cpp:161) — Combined capability

**Classification: GENERIC ROLLBACK** — Takes minimum of children. If recurrent reports 0, hybrid reports 0.

#### [`src/llama-memory-hybrid.cpp:107-112`](src/llama-memory-hybrid.cpp:107), [`src/llama-memory-hybrid-iswa.cpp:104-109`](src/llama-memory-hybrid-iswa.cpp:104) — Ubatch split

**Classification: HARMLESS** — No tail grouping when `n_rs_seq = 0`.

---

### 9.6 KV Cache

#### [`src/llama-kv-cache.cpp:1509-1517`](src/llama-kv-cache.cpp:1509) — Capability

**Classification: GENERIC ROLLBACK**

For KVarN caches with compact tail, reports `tail_rollback_tokens`. Standard caches without compact tail return `arbitrary_ranges = true` (TYPE_PART, no checkpoint).

**Important:** For hybrid models with compact tail, `suffix_rollback_tokens = tail_rollback_tokens = kv_tail_rollback_tokens = n_rs_seq`.

---

### 9.7 Delta-Net Base Model (Qwen3.6 / Gemma 4)

#### [`src/models/delta-net-base.cpp:479-522`](src/models/delta-net-base.cpp:479) — Conv state build

**Classification: HARMLESS** — When `n_rs_seq = 0`, uses simple path (no snapshot columns).

#### [`src/models/delta-net-base.cpp:546`](src/models/delta-net-base.cpp:546) — Keep flag

**Classification: HARMLESS** — `keep = false` when 0.

#### [`src/models/delta-net-base.cpp:564`](src/models/delta-net-base.cpp:564) — K calculation

**Classification: HARMLESS** — Only used in `n_rs_seq > 0` branch.

---

### 9.8 Server Code

#### [`tools/server/server-context.cpp:976-977`](tools/server/server-context.cpp:976) — Member variables

**Classification: GENERIC ROLLBACK**

#### [`tools/server/server-context.cpp:3219-3220`](tools/server/server-context.cpp:3219) — Draft phase checkpoint check

**Classification: GENERIC ROLLBACK** — When `TYPE_FULL`, checkpoint is used.

#### [`tools/server/server-context.cpp:3232-3234`](tools/server/server-context.cpp:3232) — Assert checkpoint exists

**Classification: GENERIC ROLLBACK**

#### [`tools/server/server-context.cpp:3306-3328`](tools/server/server-context.cpp:3306) — Draft checkpoint creation

**Classification: GENERIC ROLLBACK — but DFlash affected**

When `TYPE_FULL` and `max_rollback = 0`, returns `true` for any `draft.size() > 0`. DFlash checkpoints every draft cycle.

#### [`tools/server/server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221) — Verification checkpoint restore

**Classification: GENERIC ROLLBACK — but DFlash affected**

Critical path: when DFlash has partial acceptance and `use_ckpt_tgt = true`, the server restores the full checkpoint (expensive serialization), removes tokens after `ckpt.pos_max`, and restores sampler/loop guard/prompt.

#### [`tools/server/server-context.cpp:3784-3787`](tools/server/server-context.cpp:3784) — Checkpoint decision

**Classification: GENERIC ROLLBACK**

---

### 9.9 Public API — UNRELATED

[`include/llama.h:426`](include/llama.h:426), [`include/llama.h:679`](include/llama.h:679), [`include/llama.h:864`](include/llama.h:864), [`src/llama-context.cpp:4147-4149`](src/llama-context.cpp:4147)

---

### 9.10 Test Code — MTP/EAGLE3 ONLY

[`tests/test-recurrent-state-rollback.cpp:14-16`](tests/test-recurrent-state-rollback.cpp:14), [`tests/test-recurrent-state-rollback.cpp:75-77`](tests/test-recurrent-state-rollback.cpp:75)

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

1. **[`common_speculative_impl_draft_dflash::accept()`](common/speculative.cpp:1195-1197)** — Is a no-op. DFlash does not use `set_rs_idx()` or `rs_idx`.
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
