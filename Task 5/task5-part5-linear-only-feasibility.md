# Task 5.3 Part 5: Linear-Only Replay Feasibility Assessment

**Date:** 2026-08-07
**Source:** Current upstream workspace + `old-versions/beellama.cpp-preview-v0.3.2/`

---

## Section 1: Current Upstream DFlash Execution — Is It Linear?

### 1.1 Current speculative decoding flow

The current upstream DFlash speculative decoding follows a linear pipeline:

1. **Draft phase** — Draft model generates N candidate tokens sequentially.
2. **Verify phase** — Target model processes all N+1 tokens (draft + first draft token) in a single forward pass.
3. **Accept phase** — Sampler verifies each draft token against target model output. Returns list of accepted tokens (prefix of the draft chain).
4. **Rollback phase** — If not all draft tokens accepted, restore state to the last accepted token.

The verification forward pass at [`tools/server/server-context.cpp:4213-4214`](tools/server/server-context.cpp:4213):
```cpp
auto accepted = common_sampler_sample_and_accept_n(
    slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_accept);
```

This is a LINEAR chain: tokens 0, 1, 2, ..., N are verified in sequence. The `spec_i_batch` contains positions for all draft tokens plus the verification token. There is no tree structure in current upstream DFlash.

**Key observation:** Current upstream DFlash generates a flat, sequential list of speculative tokens. The verification processes them as a linear batch. The acceptance returns a prefix of accepted tokens. This is inherently linear.

### 1.1 Current upstream does NOT support tree speculation

Current upstream DFlash has no tree speculation. The speculative types enum at [`common/common.h:173-179`](common/common.h:173):
```cpp
COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction
COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash speculative decoding
COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
```

None of these types include tree-based speculation. The `draft-dflash` type generates a linear chain of N draft tokens and verifies them as a batch.

The server verification code at [`server-context.cpp:4191-4269`](tools/server/server-context.cpp:4191) handles partial acceptance by:
- Checking if checkpoint rollback is needed ([`server-context.cpp:4221-4263`](tools/server/server-context.cpp:4221))
- Using `common_context_seq_rm` to remove rejected tokens
- No tree-aware rollback logic exists

---

## Section 2: Old Tape Machinery — What Was Tree-Specific?

### 2.1 Old tree_bufs system

The old code had a `tree_bufs` structure for tree-based DFlash speculation. From [`llama-context.cpp:5405-5510`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:5405):

| Component | Purpose | Linear-Only Needed? |
|-----------|---------|-------------------|
| `tree_bufs.parent_ids_gpu` | GPU buffer storing parent IDs for tree topology | **NO** — linear chain has implicit parent (token i-1) |
| `tree_bufs.ssm_intermediates` | GPU buffers for SSM intermediate states per tree node | **NO** — linear replay doesn't need per-node intermediates |
| `tree_bufs.buffer` | GPU allocation for tree buffers | **NO** |
| `tree_bufs.ggml_ctx` | ggml context for tree tensor allocation | **NO** |
| `tree_bufs.active` | Flag indicating tree mode is active | **NO** |
| `tree_bufs.disabled` | Kill switch for multi-GPU fallback | **NO** |
| `tree_rollback()` | Tree-aware rollback with parent traversal | **NO** — linear rollback is simple prefix truncation |
| `dflash_prepare_branch()` | Prepare KV state for tree branch execution | **NO** — no branches in linear mode |
| `set_tree_parent_ids()` | Upload tree topology to GPU | **NO** |
| `allocate_tree_buffers()` | Allocate tree GPU buffers | **NO** |

**Total tree-specific code:** Approximately 300+ lines across [`llama-context.cpp:5351-5581`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:5351) and related functions.

### 2.2 Old tape structure

The old tape structure stored data per slot and per layer:

```cpp
struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;
    std::vector<int> layer_ids;
    int n_tokens = 0;
    int max_tokens = 0;
};

struct dflash_tape_gpu_layer {
    ggml_backend_buffer_t buf = nullptr;
    void * data = nullptr;
    size_t size = 0;
};
```

This structure is LINEAR (sequential tokens indexed 0..n_tokens-1). The tree-specific code was in `tree_bufs`, NOT in the tape structure. The tape itself stores a flat sequence of captured intermediates.

### 2.3 What old tape replay actually did

The old `tape_replay()` at [`llama-context.cpp:2898-2912`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2898):

```cpp
void llama_context::tape_replay(llama_seq_id seq_id, int n_accepted) {
    // GPU-resident tape path: data already on GPU from graph-embedded copies
    dflash_tape_gpu * const gpu_tape = dflash_capture->active_tape();
    const bool use_gpu_tape = (gpu_tape != nullptr && ...);

    if (use_gpu_tape && tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)) {
            // Direct CUDA kernel launch — linear replay of first K accepted tokens
            ...
            return;
        }

    // Fallback: CPU-based replay using ggml graph
    ...
}
```

The replay function processes `n_accepted` tokens sequentially from the tape. It does NOT traverse a tree. The `n_accepted` parameter is a count of the first K tokens in the linear chain.

### 2.4 Old CPU tape replay

The CPU tape replay at [`llama-context.cpp:3878-4030`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3878) also processes tokens linearly:

```cpp
for (int li = 0; li < (int) rec_ids.size(); ++li) {
    // For each recurrent layer:
    for (int tok = 0; tok < n_accepted; ++tok) {
        // Replay token tok through layer li
        // Read k[tok], v[tok], gate[tok], beta[tok] from tape
        // Apply GDN state update
    }
}
```

The loop is `for tok in 0..n_accepted-1`, a linear sequential replay.

---

## Section 3: Linear-Only Feasibility Verdict

### 3.1 Can replay be linear-only?

**YES.** Replay can be linear-only. The analysis:

1. **Current upstream DFlash generates linear chains.** No tree speculation exists in current upstream. The draft model produces N sequential tokens, and verification processes them as a linear batch.

2. **Old tape replay was already linear.** The `tape_replay()` function at [`llama-context.cpp:2898`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2898) replayed the first K accepted tokens sequentially. The tape structure stores a flat array of per-token intermediates.

3. **The tree machinery was orthogonal to tape replay.** The `tree_bufs` system, `tree_rollback()`, `dflash_prepare_branch()`, etc., were for a DIFFERENT feature: tree-based speculative decoding where the draft model generates a tree of candidate tokens instead of a linear chain. This tree speculation was NEVER part of current upstream DFlash.

4. **GDN state update is inherently sequential.** The GDN kernel at [`gated_delta_net.cu:63`](ggml/src/ggml-cuda/gated_delta_net.cu:63) processes tokens sequentially: `for (int t = 0; t < n_tokens; t++)`. Token t+1 uses the state updated by token t. This means replay MUST be sequential regardless of whether the draft was linear or tree-based.

### 3.2 What old tape machinery can be discarded for linear-only replay

| Old Component | Can Discard? | Reason |
|--------------|------------|-------|
| `tree_bufs` (entire struct) | **YES** | Tree-specific, not used by linear replay |
| `tree_rollback()` | **YES** | Tree-specific rollback |
| `dflash_prepare_branch()` | **YES** | Branch preparation for tree speculation |
| `set_tree_parent_ids()` | **YES** | Tree topology upload |
| `allocate_tree_buffers()` | **YES** | Tree buffer allocation |
| `clear_tree_parent_ids()` | **YES** | Tree cleanup |
| `dflash_drollback_rollback()` tree path | **YES** | The tree branch of rollback logic |
| `tree_mask` | **YES** | Tree attention mask |
| `tree_bufs.ssm_intermediates` | **YES** | Per-tree-node intermediates |

**Total discardable: ~300+ lines of tree-specific code.**

### 3.3 What old tape machinery must be retained for linear-only replay

| Old Component | Retain? | Reason |
|--------------|---------|-------|
| `dflash_tape_gpu` struct | **YES** | Core linear tape structure |
| `dflash_tape_gpu_layer` struct | **YES** | Per-layer tape buffer |
| `allocate_tape_gpu()` | **YES** (simplified) | GPU tape buffer allocation |
| `dflash_eval_callback` tape capture | **YES** (or graph-embedded copy) | Capture mechanism |
| `tape_replay()` | **YES** (simplified) | Core replay logic |
| `tape_replay_gdn_direct_gpu()` | **YES** | Optimized CUDA replay |
| `tape_replay_conv_gpu()` | **YES** | Conv state replay |
| `tape_replay_sync()` | **YES** | Async replay synchronization |
| `set_tape_recording()` | **YES** | Enable/disable tape capture |
| `dflash_ensure_recurrent_setup()` | **YES** | Identify recurrent layers |
| `tape_name_map` | **YES** | Map tensor names to tape slots |

### 3.4 Simplified linear-only replay design

The minimal linear-only replay design:

```
DFlash speculative forward
         ↓
produce normal DeltaNet intermediates (k_in, v_in, g_in, b_in)
         ↓
capture k, v, gate, beta into linear tape buffer (one entry per token)
         ↓
verification (current upstream sampler)
         ↓
determine K = number of accepted tokens
         ↓
if K < n_draft (partial acceptance):
    restore backup R/S state (Task 4 backup cells)
         ↓
    replay first K accepted tokens through GDN using captured tape data
         ↓
    replay conv state for first K tokens
         ↓
    final recurrent state = result of K-token replay
else (full acceptance):
    no replay needed — state is already correct
```

**Data structures needed:**
```cpp
struct linear_tape {
    // Per layer: captured intermediates for n_draft tokens
    std::vector<ggml_backend_buffer_t> layer_buffers;  // one per recurrent layer
    std::vector<int> layer_ids;                        // recurrent layer indices
    int n_tokens;                                      // tokens captured this pass
    int max_tokens;                                    // buffer capacity
};

// Per-layer buffer layout (contiguous F32):
// [k[0]..k[N-1]] [v[0]..v[N-1]] [gate[0]..gate[N-1]] [beta[0]..beta[N-1]]
// where each element is shaped per the tensor definitions in Part 4
```

### 3.5 Dependency analysis — What prevents linear-only replay?

**No dependency prevents linear-only replay.** The analysis:

1. **GDN kernel supports sequential replay.** The CUDA kernel at [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) already processes tokens sequentially. Replay just calls the same kernel with captured k, v, gate, beta and a dummy Q.

2. **Current upstream has no tree speculation.** There is nothing to discard. The linear-only design IS the current upstream design.

3. **Eval callback infrastructure exists.** The `cparams.cb_eval` mechanism at [`llama-context.cpp:1611`](src/llama-context.cpp:1611) is available for capture without graph modification.

4. **Graph-embedded copy is an option.** If eval callback overhead is a concern, `ggml_cpy` nodes can be added to the graph at [`delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) for DFlash-only capture.

5. **Backup cells from Task 4 provide rollback state.** The replay starts from the backup R/S state (captured before speculative forward) and applies K accepted tokens on top.

---

## Section 4: Summary

### 4.1 Capture points (Part 4 summary)

| Tensor | Node Name | Shape | Capture Method |
|--------|-----------|-------|---------------|
| k | `k_in-{il}` | `[S_k, H_k, n_tokens, n_seqs]` | Eval callback or graph-embedded `ggml_cpy` |
| v | `v_in-{il}` | `[S_v, H_v, n_tokens, n_seqs]` | Same as k |
| gate | `g_in-{il}` | `[1 or S_v, H_v, n_tokens, n_seqs]` | Same as k |
| beta | `b_in-{il}` | `[1, H_v, n_tokens, n_seqs]` | Same as k |

All four tensors exist as named ggml graph nodes. None are retained after graph execution. Capture requires eval callback or graph-embedded copy. Preferred design: graph-embedded copy for GPU path, eval callback for CPU fallback.

### 4.2 Linear-only feasibility (Part 5 summary)

| Question | Answer |
|----------|--------|
| Can replay be linear-only? | **YES** — current upstream DFlash is already linear |
| What old machinery can be discarded? | **~300 lines** of tree-specific code (`tree_bufs`, `tree_rollback`, `dflash_prepare_branch`, etc.) |
| What must be retained? | Tape struct, capture mechanism, replay kernel, conv replay |
| What dependency prevents linear-only? | **NONE** — no dependency blocks it |
| Is GDN kernel compatible? | **YES** — kernel processes tokens sequentially |
| Is eval callback available? | **YES** — `cparams.cb_eval` exists in current upstream |

### 4.3 Recommended design

```
Linear DFlash with minimal replay:

1. Before speculative forward: capture backup R/S state (Task 4 backup cells).
2. During speculative forward: capture k, v, gate, beta via graph-embedded ggml_cpy to linear tape buffer.
3. After verification: if partial acceptance, restore backup R/S state and replay K accepted tokens using captured tape data.
4. Replay uses the same GDN CUDA kernel with Q=zeros.
5. Conv state replay uses captured conv_input data from tape.
```

This design requires:
- ~100 lines for tape struct and buffer allocation
- ~200 lines for eval callback OR graph-embedded copy (depending on approach)
- ~200 lines for replay logic (GDN kernel launch + conv replay)
- ~50 lines for integration with server verification flow
- **Total: ~550 lines** (vs ~1,100+ lines for old full tape replay including tree support)

The key insight: the old tape replay was ~50% tree-specific machinery that is not needed for linear DFlash. By discarding tree support and focusing on linear-only replay, the implementation can be cut in half while providing the same rollback performance for the common case.

---

*End of Task 5.3 Parts 4-5*
