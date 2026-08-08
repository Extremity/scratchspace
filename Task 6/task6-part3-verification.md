# Research Task 6.3 — Source Verification: DFlash Verification/Rollback Lifecycle and Replay Capture Points

**Date:** 2026-08-08
**Status:** Verification Complete

---

## 1. DFlash Verification Lifecycle

### 1.1 Speculative Forward Pass Start — VERIFIED

**Location:** [`tools/server/server-context.cpp:3264-3285`](tools/server/server-context.cpp:3264)

**Claim:** The speculative forward pass begins with `common_speculative_draft()`.

**VERIFIED — Exact match.**

```cpp
// generate the actual drafts (if any)
{
    const int64_t t_draft_start = ggml_time_us();
    common_speculative_draft(spec.get());
    const float shared_draft_ms = (ggml_time_us() - t_draft_start) / 1000.0f;
    // ... adaptive timing attribution ...
}
```

The draft phase is triggered from `process_request()` when slots are in `SLOT_STATE_GENERATING` and `can_speculate()` returns true with empty `spec_draft`. The draft parameters are set at [`server-context.cpp:3249-3256`](tools/server/server-context.cpp:3249):

```cpp
common_speculative_get_draft_params(spec.get(), slot.id) = {
    /* .drafting = */ true,
    /* .n_max    = */ n_draft_max,
    /* .n_past   = */ slot.prompt.n_tokens(),
    /* .id_last  = */ slot.sampled,
    /* .prompt   = */ &slot.spec_prompt,
    /* .result   = */ &slot.spec_draft,
};
```

**Draft evaluation flow summary:**

| Step | Location | Action |
|------|----------|--------|
| 1 | [`server-context.cpp:3238-3258`](tools/server/server-context.cpp:3238) | Set draft parameters, add slot to `drafting` list |
| 2 | [`server-context.cpp:3267`](tools/server/server-context.cpp:3267) | `common_speculative_draft(spec.get())` — generates draft tokens via DFlash draft model |
| 3 | [`server-context.cpp:3288-3330`](tools/server/server-context.cpp:3288) | Create checkpoints if needed (`ckpt.update_tgt()`, `ckpt.update_dft()`) |
| 4 | [`server-context.cpp:3333-3335`](tools/server/server-context.cpp:3333) | Build verification batch with drafted tokens |

---

### 1.2 `common_sampler_sample_and_accept_n()` Call — VERIFIED

**Location:** [`tools/server/server-context.cpp:4213-4214`](tools/server/server-context.cpp:4213)

**Claim:** Speculative verification calls `common_sampler_sample_and_accept_n()` at line 4213.

**VERIFIED — Exact match.**

```cpp
GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
const auto on_accept = make_loop_guard_accept_callback(slot);
auto accepted = common_sampler_sample_and_accept_n(
    slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_accept);
slot.spec_i_batch.clear();
```

**Function signature** ([`common/sampling.cpp:713-757`](common/sampling.cpp:713)):

```cpp
std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context  * ctx,
        const std::vector<int> & idxs,
        const llama_tokens    & draft,
        bool                    grammar_first,
        const common_sampler_accept_callback & on_accept);
```

**Parameters:**
| Param | Value | Purpose |
|-------|-------|---------|
| `gsmpl` | `slot.smpl.get()` | Sampler state for the slot |
| `ctx` | `slot.ctx_tgt` | Target (main model) context |
| `idxs` | `slot.spec_i_batch` | Batch indices for draft tokens + 1 verification token |
| `draft` | `slot.spec_draft` | Drafted token sequence |
| `grammar_first` | `false` | Grammar checking disabled during speculative verification |
| `on_accept` | Loop guard callback | Per-token acceptance observer for reasoning loop detection |

**Acceptance algorithm** ([`common/sampling.cpp:737-756`](common/sampling.cpp:737)):
1. For each draft token `i` from 0 to `draft.size()-1`:
   - Sample main model token at `idxs[i]`.
   - Call `on_accept` callback with sampled token info.
   - If `on_accept` returns false, break.
   - If sampled token differs from `draft[i]`, break (mismatch).
2. If all draft tokens matched, sample one more token at `idxs[draft.size()]` (the new token after the draft).
3. Return the accepted token sequence (always at least 1 token).

---

### 1.3 Accepted/Rejected Counts Determination — VERIFIED

**Location:** [`tools/server/server-context.cpp:4217-4219`](tools/server/server-context.cpp:4217)

**VERIFIED — Exact match.**

```cpp
GGML_ASSERT(accepted.size() >= 1);

const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
```

**Math:**
- `n_draft = slot.spec_draft.size()` — number of draft tokens.
- `accepted.size()` — number of accepted tokens (includes at least 1 new token after draft).
- `n_rollback = n_draft + 1 - accepted.size()` — number of rejected draft tokens to roll back.

When `accepted.size() == n_draft + 1` (all draft tokens matched), `n_rollback == 0`.
When `accepted.size() == 1` (no draft tokens matched), `n_rollback == n_draft`.

---

### 1.4 Rollback Decision — VERIFIED

**Location:** [`tools/server/server-context.cpp:4221-4222`](tools/server/server-context.cpp:4221)

**VERIFIED — Exact match.**

```cpp
const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);
```

**`server_speculative_rollback_requires_checkpoint()`** ([`tools/server/server-task.h:20-26`](tools/server/server-task.h:20)):

```cpp
static inline bool server_speculative_rollback_requires_checkpoint(
        common_context_seq_rm_type type,
        uint32_t                   max_rollback,
        size_t                     proposed_rollback) {
    return type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
           (type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && proposed_rollback > max_rollback);
}
```

**Decision logic:**

| `ctx_tgt_seq_rm_type` | `proposed_rollback > max_rollback` | Result |
|----------------------|-----------------------------------|--------|
| `COMMON_CONTEXT_SEQ_RM_TYPE_NO` | — | `false` (no rollback capability — should not reach here) |
| `COMMON_CONTEXT_SEQ_RM_TYPE_RS` | `false` | `false` (use RS rollback — fast) |
| `COMMON_CONTEXT_SEQ_RM_TYPE_RS` | `true` | `true` (exceeds RS reserve — checkpoint fallback) |
| `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` | — | `true` (always checkpoint — slow) |

---

### 1.5 Checkpoint Rollback Path — VERIFIED

**Location:** [`tools/server/server-context.cpp:4225-4264`](tools/server/server-context.cpp:4225)

**VERIFIED — Exact match.**

```cpp
if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        if (trace > 0) {
            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
        }

        // partial acceptance is not supported by the context -> truncate the draft and restore the state
        slot.spec_draft = std::move(accepted);

        const auto & ckpt = slot.spec_ckpt;

        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

        {
            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
        }

        if (slot.ctx_dft) {
            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
        }

        slot.prompt.tokens.keep_first(ckpt.n_tokens);
        slot.smpl = std::move(smpl_save);
        slot.loop_guard = loop_guard_save;
        slot.loop_guard_interventions = loop_guard_interventions_save;
        slot.loop_guard_triggered = loop_guard_triggered_save;
        slot.loop_guard_action = loop_guard_action_save;
        slot.loop_guard_reason = loop_guard_reason_save;
        slot.reasoning_output_tokens = reasoning_output_tokens_save;
        slot.visible_output_tokens = visible_output_tokens_save;
        slot.has_next_token = has_next_token_save;
        slot.stop = stop_save;
        slot.stop_detail = stop_detail_save;

        return;
    }
}
```

**Checkpoint rollback steps:**
1. Save `accepted` tokens to `slot.spec_draft` for retry next cycle.
2. Load target context checkpoint: `ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`.
3. Remove speculative KV from target: `common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1)`.
4. If draft context exists, load draft checkpoint and remove draft KV similarly.
5. Restore prompt tokens: `slot.prompt.tokens.keep_first(ckpt.n_tokens)`.
6. Restore sampler state: `slot.smpl = std::move(smpl_save)`.
7. Restore loop guard state and all slot metadata.
8. Return early — no further processing this cycle.

**Key observation:** The checkpoint path restores the entire context state including sampler, loop guard, and prompt. This is the slow path.

---

### 1.6 RS Rollback Path (Implicit) — VERIFIED

**Location:** [`tools/server/server-context.cpp:4267-4348`](tools/server/server-context.cpp:4267)

When `use_ckpt_tgt == false` and `n_rollback > 0`, the code falls through to the post-verification processing path. The RS rollback is handled implicitly by the recurrent memory layer's `seq_rm` capability, which truncates the rejected suffix in-place using RS snapshots.

**Post-verification processing (lines 4267-4348):**

```cpp
if (trace > 0) {
    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
}

common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

slot.spec_draft = std::move(accepted);
// ... timing, metrics, adaptive controller ...

// add accepted tokens to the prompt
slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

slot.sampled = ids.back(); // last accepted token

common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
if (slot.ctx_dft) {
    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
}
```

**RS rollback steps:**
1. Call `common_speculative_accept()` to record acceptance statistics.
2. Update prompt with accepted tokens.
3. Remove future KV beyond accepted position: `common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1)`.
4. Same for draft context if present.
5. Process each accepted token through `process_token()` for output streaming.

---

### 1.7 `common_context_seq_rm_max_rollback()` — VERIFIED

**Location:** [`common/common.cpp:1688-1695`](common/common.cpp:1688)

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

**Behavior:**
- Returns `UINT32_MAX` if memory supports arbitrary range removal (full flexibility).
- Returns `capability.suffix_rollback_tokens` for token-limited rollback (RS snapshot count).
- Returns `0` if no memory or no rollback capability.

---

## 2. Rollback Extension Point for Replay Integration

### 2.1 Post-Verification Rollback Function — IDENTIFIED

**Location:** [`tools/server/server-context.cpp:4055-4349`](tools/server/server-context.cpp:4055) — `post_decode()` method.

The `post_decode()` function handles the entire post-verification lifecycle:

| Line Range | Function | Description |
|------------|----------|-------------|
| 4055-4069 | Batch view validation | Ensures speculative batch indices are within the current sub-batch |
| 4076-4178 | Non-speculative sampling | Handles normal (non-speculative) token sampling |
| 4180-4348 | Speculative verification | The main speculative decoding verification and rollback |

**The speculative verification block** (lines 4180-4348) is a single `iterate(slots, ...)` lambda that:
1. Skips slots without draft tokens (line 4182).
2. Saves sampler and loop guard state (lines 4195-4209).
3. Calls `common_sampler_sample_and_accept_n()` (line 4213).
4. Determines rollback strategy (line 4221).
5. Either restores checkpoint (lines 4225-4264) or proceeds with RS rollback (lines 4267-4348).

---

### 2.2 `common_context_seq_rm()` Call Location — VERIFIED

**Location:** [`tools/server/server-context.cpp:4319-4322`](tools/server/server-context.cpp:4319)

```cpp
common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
if (slot.ctx_dft) {
    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
}
```

This removes rejected KV beyond the accepted position. It is called AFTER `common_speculative_accept()` and prompt update, and BEFORE processing accepted tokens.

**Checkpoint path also calls `common_context_seq_rm()`** at [`server-context.cpp:4241`](tools/server/server-context.cpp:4241) (target) and [`server-context.cpp:4247`](tools/server/server-context.cpp:4247) (draft).

---

### 2.3 `on_accept` Callback Integration Point — VERIFIED

**Location:** [`tools/server/server-context.cpp:4212`](tools/server/server-context.cpp:4212)

```cpp
const auto on_accept = make_loop_guard_accept_callback(slot);
```

**`make_loop_guard_accept_callback()`** ([`server-context.cpp:2019-2027`](tools/server/server-context.cpp:2019)):

```cpp
common_sampler_accept_callback make_loop_guard_accept_callback(server_slot & slot) {
    if (!loop_guard_accept_enabled(slot)) {
        return {};
    }

    return [this, &slot](const common_sampler_accept_info & info) {
        return handle_loop_guard_accept(slot, info);
    };
}
```

**Callback signature** ([`common/sampling.h:47`](common/sampling.h:47)):

```cpp
using common_sampler_accept_callback = std::function<bool(const common_sampler_accept_info &)>;
```

**When called:** Inside `common_sampler_sample_and_accept_n()` at [`common/sampling.cpp:726-729`](common/sampling.cpp:726):

```cpp
auto accept = [&](llama_token id) {
    if (on_accept) {
        const auto info = common_sampler_accept_with_info(gsmpl, id, true);
        result.push_back(id);
        return on_accept(info);
    }
    // ...
};
```

**Timing:** The `on_accept` callback fires for each accepted token DURING the verification loop, BEFORE the mismatch detection breaks the loop. This means:
- Accepted tokens trigger `on_accept` in order.
- The first rejected token (mismatch) does NOT trigger `on_accept`.
- The callback can observe acceptance but cannot modify the verification result.

---

### 2.4 Recommended Replay Integration Point

**Recommended location:** Between checkpoint restore (line 4263) and the RS rollback path continuation (line 4267).

**Rationale:**

The current code has two paths after verification:
1. **Checkpoint path** (lines 4225-4264): Full state restore, returns early.
2. **RS path** (lines 4267-4348): Fast in-place rollback via `common_context_seq_rm()`.

For replay integration, the optimal injection point is:

```
After: common_sampler_sample_and_accept_n() returns (line 4214)
Before: common_context_seq_rm() for rejected KV (line 4319)
```

More precisely, between the acceptance determination and the KV cleanup:

| Current Step | Line | Proposed Replay Step |
|--------------|------|---------------------|
| `accepted = common_sampler_sample_and_accept_n(...)` | 4213 | — |
| `n_rollback` calculation | 4219 | — |
| Checkpoint decision | 4221 | — |
| Checkpoint restore (if needed) | 4225-4264 | — |
| **REPLAY: Restore recurrent state from tape for accepted tokens** | **NEW** | Capture k,v,g,b during draft, replay GDN for accepted tokens |
| `common_context_seq_rm()` for rejected KV | 4319 | — |
| Process accepted tokens | 4324-4343 | — |

**For checkpoint path:** Replay would substitute for the checkpoint restore. Instead of serializing/deserializing the full context, replay the DeltaNet forward pass for accepted tokens using captured intermediate tensors.

**For RS path:** Replay would restore the recurrent state for accepted tokens, eliminating the need for RS snapshots. The `common_context_seq_rm()` would still be needed to remove rejected KV.

**Critical constraint:** Replay must produce the same recurrent state (R and S tensors) as the original forward pass for accepted tokens. This requires capturing k, v, g, beta at the exact points where they were produced during the draft forward pass.

---

## 3. Tape Capture Points

### 3.1 Tensor Names in `delta-net-base.cpp` — VERIFIED

**Location:** [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49)

**Claim:** Tensor names are `k_in-{il}`, `v_in-{il}`, `g_in-{il}`, `b_in-{il}`.

**VERIFIED — Exact match.**

```cpp
cb(q, "q_in", il);
cb(k, "k_in", il);
cb(v, "v_in", il);
cb(b, "b_in", il);
cb(g, "g_in", il);
```

**Tensor details in `build_delta_net_chunking()`** (lines 49-53):

| Tensor | Name Pattern | Dimensions (input) | Description |
|--------|-------------|-------------------|-------------|
| `q` | `q_in-{il}` | `[S_k, H_k, n_tokens, n_seqs]` | Query (scaled by `1/sqrt(S_k)`) |
| `k` | `k_in-{il}` | `[S_k, H_k, n_tokens, n_seqs]` | Key |
| `v` | `v_in-{il}` | `[S_v, H_v, n_tokens, n_seqs]` | Value |
| `g` | `g_in-{il}` | `[1 or S_v, H_v, n_tokens, n_seqs]` | Gate (scalar or KDA) |
| `b` | `b_in-{il}` | `[1, H_v, n_tokens, n_seqs]` | Beta |

**Note:** These tensors are captured AFTER scaling q but BEFORE permutation. The permutation at lines 55-59 transforms them to `[S, n_tokens, H, n_seqs]` layout.

### 3.2 Tensor Production in `qwen35.cpp` — VERIFIED

**Location:** [`src/models/qwen35.cpp:338-470`](src/models/qwen35.cpp:338) — `build_layer_attn_linear()`.

**Tensor production flow:**

| Tensor | Source | Line | Name |
|--------|--------|------|------|
| `q_conv` | Conv output + L2 norm | 407-411, 431 | `q_conv` |
| `k_conv` | Conv output + L2 norm | 413-417, 432 | `k_conv` |
| `v_conv` | Conv output | 419-423 | `v_conv` |
| `gate` | Alpha + softplus + A_log.exp() | 376-379 | `gate` |
| `beta` | Sigmoid(ssm_beta projection) | 361-366 | `beta_sigmoid` |

**Pre-delta-net capture names** (lines 446-448):
```cpp
cb(q_conv, "q_conv_predelta", il);
cb(k_conv, "k_conv_predelta", il);
cb(v_conv, "v_conv_predelta", il);
```

**Call to `build_recurrent_attn()`** (line 450):
```cpp
ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
```

**Note:** The tensors passed to `build_recurrent_attn()` are the final q, k, v, g, b values. These flow into `build_delta_net()` and then to either `build_delta_net_chunking()` or `build_delta_net_fused()`.

### 3.3 `cb()` Function — VERIFIED

**Location:** [`src/llama-graph.cpp:1736-1740`](src/llama-graph.cpp:1736)

```cpp
void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}
```

**Callback signature** (`[`src/llama-graph.h:745`](src/llama-graph.h:745)):

```cpp
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;
```

**Purpose:** The `cb()` function fires during graph BUILD (not execution). It allows the caller to observe each tensor as it's created in the computation graph. This is used for:
- Tensor naming (for debugging/profiling).
- Custom backend assignment.
- Graph analysis.

**Critical limitation:** `cb()` is called at graph BUILD time, not at graph EXECUTION time. The tensor exists but has no data yet. Data is only available after `ggml_backend_sched_graph_compute()`.

### 3.4 Eval Callback (`cparams.cb_eval`) — VERIFIED

**Location:** [`src/llama-context.cpp:1611`](src/llama-context.cpp:1611)

```cpp
ggml_backend_sched_reset(sched.get());
ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

**Eval callback signature** ([`ggml/include/ggml-backend.h:314`](ggml/include/ggml-backend.h:314)):

```cpp
typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
```

**Behavior:**
- `ask == true`: Scheduler asks if the caller wants to observe tensor `t`. Return `true` to observe, `false` to skip.
- `ask == false`: Scheduler passes the computed tensor `t` to the caller. Return `false` to cancel graph compute.

**Timing:** Called at graph EXECUTION time, AFTER the tensor has been computed. Data is available.

**Viability for tape capture:** The eval callback CAN capture intermediate tensors during graph execution. It can observe `k_in-{il}`, `v_in-{il}`, `g_in-{il}`, `b_in-{il}` tensors by name and copy their data to a tape buffer.

**Implementation approach:**
1. Register eval callback that filters on tensor names matching `k_in-*`, `v_in-*`, `g_in-*`, `b_in-*`.
2. On `ask == true`, return `true` for matching tensors.
3. On `ask == false`, copy tensor data to tape storage.
4. After graph compute, replay uses captured tensors.

### 3.5 Graph-Embedded `ggml_cpy` — VIABLE

**Location:** [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) — same `cb()` calls.

**Approach:** Instead of using eval callback, embed `ggml_cpy` operations directly in the graph to copy intermediate tensors to persistent buffers.

**How it would work:**
```cpp
// After cb(k, "k_in", il); at line 50:
ggml_tensor * k_tape = ggml_cpy(ctx0, k, tape_buffer_k[il]);
ggml_build_forward_expand(gf, k_tape);
```

**Advantages:**
- No callback overhead.
- Copies are batched with the main graph compute.
- No synchronization needed.

**Disadvantages:**
- Requires modifying the graph builder code.
- Requires pre-allocated tape buffers.
- Adds memory pressure during the forward pass.

**Viability comparison:**

| Approach | Pros | Cons |
|----------|------|------|
| Eval callback | No graph modification, runtime-configurable | Callback overhead, synchronization needed |
| Graph-embedded `ggml_cpy` | Batched with compute, no sync | Requires graph builder modification, buffer allocation |

**Recommendation:** Graph-embedded `ggml_cpy` is more efficient for production use. Eval callback is better for prototyping and debugging.

---

## 4. GDN Replay Mechanism

### 4.1 `ggml_gated_delta_net` Operation — VERIFIED

**Location:** [`ggml/src/ggml.c:6426-6479`](ggml/src/ggml.c:6426)

**VERIFIED — Exact match.**

```cpp
struct ggml_tensor * ggml_gated_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,      // [S_k, H_k, n_tokens, n_seqs]
        struct ggml_tensor  * k,      // [S_k, H_k, n_tokens, n_seqs]
        struct ggml_tensor  * v,      // [S_v, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * g,      // [1 or S_v, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * beta,   // [1, H_v, n_tokens, n_seqs]
        struct ggml_tensor  * state,  // [S_v, S_v, H, n_seqs] — initial state s0 only
        int64_t               K)      // snapshot slot count (>= 1) {
    // ... assertions ...
    const int64_t state_rows = K * S_v * n_seqs;
    const int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    return result;
}
```

**Output layout:** `[S_v * H, n_tokens * n_seqs + state_rows, 1, 1]`
- First `S_v * H * n_tokens * n_seqs` bytes: attention output.
- Remaining `S_v * H * state_rows` bytes: state snapshots (K slots).

**Parameter K:** Controls how many state snapshots are written. K=1 means only the final state. K=n_tokens means per-token state snapshots.

---

### 4.2 CUDA Kernel Token Processing Order — VERIFIED

**Location:** [`ggml/src/ggml-cuda/gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63)

**VERIFIED — Sequential token processing confirmed.**

```cpp
for (int t = 0; t < n_tokens; t++) {
    const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
    const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
    const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
    // ...
    // Update state s_shard with k, v, g, beta
    // Compute attention output attn_col = s_shard @ q
    // Write state snapshot if keep_rs_t and target_slot valid
}
```

**Token loop semantics:**
- Token `t=0` is processed first, using initial state `s0` from `curr_state`.
- Each token updates `s_shard` (the working state) using the DeltaRule: `s[i] = g * s[i] + k[i] * delta`.
- Attention output for token `t` is computed from the UPDATED state: `attn = s @ q`.
- State snapshots are written at `target_slot = n_tokens - 1 - t` (reversed order — most recent at slot 0).

**Replay implication:** The kernel processes tokens sequentially. Each token's output depends on the state accumulated from all previous tokens. This means:
- Replay MUST start from the correct initial state (state before the first token to replay).
- Replay MUST process tokens in the same order as the original forward pass.
- The kernel does NOT support random-access replay (skipping to token N without processing 0..N-1).

---

### 4.3 Replay with q=zeros — FEASIBLE with Caveats

**Analysis:** If `q` is set to zeros during replay, the attention output `attn = s @ q` will be zero for all tokens. However, the state update `s[i] = g * s[i] + k[i] * delta` does NOT depend on `q`. The state evolution is independent of the query.

**Key insight from the kernel:**

```cpp
// State update (does NOT use q):
s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;

// Attention output (uses q):
attn_partial += s_shard[r] * q_reg[r];
```

**Replay with q=zeros produces:**
- Zero attention output (not needed for replay — we only want state).
- Correct state snapshots at each token boundary.

**Requirements for q=zeros replay:**
1. Provide valid k, v, g, beta tensors (captured from original forward pass).
2. Provide correct initial state (state before the replay segment).
3. Set q to zeros (or any value — it won't affect state).
4. Set K to the number of tokens + 1 to capture per-token state snapshots.
5. Process tokens sequentially from the replay start point.

**Viability:** VERIFIED — The GDN kernel supports replay with q=zeros for state reconstruction. The state update path is independent of q.

---

### 4.4 State Snapshot Layout — VERIFIED

**Location:** [`ggml/src/ggml-cuda/gated_delta_net.cu:145-157`](ggml/src/ggml-cuda/gated_delta_net.cu:145)

```cpp
if constexpr (keep_rs_t) {
    const int target_slot = (int) n_tokens - 1 - t;
    if (target_slot >= 0 && target_slot < K) {
        float * curr_state = state + target_slot * state_slot_stride;
        // Write s_shard to curr_state[col * S_v + i]
    }
}
```

**Slot mapping:**
- Slot 0 = state after processing the LAST token (most recent).
- Slot 1 = state after processing the second-to-last token.
- Slot `n_tokens - 1` = state after processing the FIRST token.
- Slots >= `n_tokens` are caller-owned (initial state for continuation).

**Replay use:** After replay with q=zeros and K=n_tokens+1, the state snapshots can be read from the output to restore recurrent state at any accepted token boundary.

---

## 5. Summary of Findings

### Verification Claims Status

| Claim | Status | Detail |
|-------|--------|--------|
| Speculative verification at `server-context.cpp:4213-4214` | **VERIFIED** | `common_sampler_sample_and_accept_n()` at exact lines |
| `on_accept` callback integration point | **VERIFIED** | `make_loop_guard_accept_callback()` at line 4212, fires during verification loop |
| Tensor names `k_in-{il}`, `v_in-{il}`, `g_in-{il}`, `b_in-{il}` | **VERIFIED** | `delta-net-base.cpp:49-53` in `build_delta_net_chunking()` |
| GDN kernel sequential token processing | **VERIFIED** | `gated_delta_net.cu:63` — sequential `for (int t = 0; t < n_tokens; t++)` loop |
| GDN replay with q=zeros | **FEASIBLE** | State update is independent of q; kernel produces correct state snapshots |
| Eval callback can capture intermediate tensors | **VIABLE** | `ggml_backend_sched_set_eval_callback` fires at execution time with computed data |
| Graph-embedded `ggml_cpy` viable | **VIABLE** | Can embed copy operations in graph builder for tape capture |

### Replay Integration Recommendations

1. **Capture point:** Use graph-embedded `ggml_cpy` after `cb(k, "k_in", il)` etc. in `delta-net-base.cpp` to capture k, v, g, beta tensors to tape buffers during the draft forward pass.

2. **Replay trigger:** After `common_sampler_sample_and_accept_n()` returns, if `n_rollback > 0`, replay GDN for accepted tokens using captured tensors and q=zeros.

3. **State restoration:** Read state snapshots from GDN replay output to restore R/S tensors for accepted tokens, eliminating the need for RS snapshots.

4. **KV cleanup:** After replay restores recurrent state, call `common_context_seq_rm()` to remove rejected KV. This is the same as the current RS path.

5. **Checkpoint elimination:** With replay, the checkpoint path becomes unnecessary for DFlash. The `use_ckpt_tgt` check can be bypassed when replay is available.

---

*End of Part 1 (Verification Lifecycle + Rollback Extension Points + Capture/Replay Analysis)*
