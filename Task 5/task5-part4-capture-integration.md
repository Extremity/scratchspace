# Task 5.3 Part 4: Capture Points in Current Upstream + Linear-Only Feasibility

**Date:** 2026-08-07
**Source:** Current upstream workspace analysis

---

## Section 1: Capture Point Analysis for Each Intermediate (k, v, gate, beta)

### 1.1 Overview: Where the intermediates are produced

All four replay-critical intermediates (k, v, gate, beta) are produced in the model-specific graph builder before being consumed by the DeltaNet recurrent attention operation. The production and naming happens in two stages:

**Stage A: Model-specific layer builder** (e.g., [`src/models/qwen35.cpp:338-450`](src/models/qwen35.cpp:338))
- Computes raw projections from layer input
- Applies conv, normalization, and activation
- Produces pre-delta tensors: `q_conv`, `k_conv`, `v_conv`, `gate`, `beta`

**Stage B: DeltaNet base builder** ( [`src/models/delta-net-base.cpp:527-606`](src/models/delta-net-base.cpp:527))
- Receives pre-delta tensors from Stage A
- Names them as graph nodes via `cb()` callback
- Passes them to `build_delta_net()` which invokes either fused GDN or autoregressive/chunked paths

The named graph nodes are the CAPTURE POINTS. The `cb()` calls at [`src/models/delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) assign names:
```cpp
cb(q, "q_in", il);
cb(k, "k_in", il);      // <-- replay capture point for k
cb(v, "v_in", il);      // <-- replay capture point for v
cb(b, "b_in", il);      // <-- replay capture point for beta
cb(g, "g_in", il);      // <-- replay capture point for gate
```

These same names appear in both the chunking path (line 49-53) and the autoregressive path (line 327-331), confirming all execution paths name these nodes consistently.

---

### 1.1 k tensor (key, post-l2_norm)

| Attribute | Value | Source |
|-----------|-------|--------|
| **Graph node name** | `k_in-{il}` | [`delta-net-base.cpp:50`](src/models/delta-net-base.cpp:50) |
| **Produced in** | `llm_build_delta_net_base::build_delta_net_*()` | [`delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) |
| **Predecessor in graph** | `k_conv` after `ggml_l2_norm()` | [`qwen35.cpp:432`](src/models/qwen35.cpp:432) |
| **Shape** | `[S_k, H_k, n_tokens, n_seqs]` | [`delta-net-base.cpp:37`](src/models/delta-net-base.cpp:37) assertion |
| **Datatype** | F32 (l2_norm output is F32) | Inferred from `ggml_l2_norm` return type |
| **Device** | Same as model layer device | Graph nodes inherit backend from context |
| **Lifetime** | Transient — exists during graph execution window only | See Section 2 |
| **Retained after graph exec?** | **NO** — tensor data lives in ggml compute buffer, freed after `ggml_backend_sched_graph_compute()` returns | Directly observed: no persistent reference |
| **Can be copied to persistent buffer?** | **YES** — tensor is accessible during graph execution via eval callback or graph-embedded `ggml_cpy` | Old code did this via `dflash_eval_callback` |
| **Requires core graph modification?** | **NO for eval callback** (external hook). **YES for graph-embedded copy** (adds `ggml_cpy` node) | See Section 3 |
| **Can be captured only for DFlash?** | **YES** — eval callback can check `cap->tape_enabled` flag. Graph-embedded copy can be conditionally built | Old code: [`llama-context.cpp:1953-1956`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:1953) |

**Note:** The k tensor captured for replay is POST-l2_norm. The l2_norm operation at [`qwen35.cpp:432`](src/models/qwen35.cpp:432) transforms the raw convolved key into unit vectors. The GDN kernel expects post-norm k. Capturing pre-norm k and recomputing l2_norm would add unnecessary complexity.

---

### 1.2 v tensor (value)

| Attribute | Value | Source |
|-----------|-------|--------|
| **Graph node name** | `v_in-{il}` | [`delta-net-base.cpp:51`](src/models/delta-net-base.cpp:51) |
| **Produced in** | `llm_build_delta_net_base::build_delta_net_*()` | [`delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) |
| **Predecessor in graph** | `v_conv` view from `conv_output_silu` | [`qwen35.cpp:419-423`](src/models/qwen35.cpp:419) |
| **Shape** | `[S_v, H_v, n_tokens, n_seqs]` | [`delta-net-base.cpp:38`](src/models/delta-net-base.cpp:38) assertion |
| **Datatype** | F32 (conv output) | Inferred from graph path |
| **Device** | Same as model layer device | Graph nodes inherit backend |
| **Lifetime** | Transient — exists during graph execution window only | Same as k |
| **Retained after graph exec?** | **NO** | Same as k |
| **Can be copied to persistent buffer?** | **YES** | Same mechanism as k |
| **Requires core graph modification?** | **NO for eval callback**. **YES for graph-embedded copy** | Same as k |
| **Can be captured only for DFlash?** | **YES** | Same as k |

---

### 1.3 gate tensor

| Attribute | Value | Source |
|-----------|-------|--------|
| **Graph node name** | `g_in-{il}` | [`delta-net-base.cpp:53`](src/models/delta-net-base.cpp:53) |
| **Produced in** | `llm_build_delta_net_base::build_delta_net_*()` | [`delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) |
| **Predecessor in graph** | `gate = ggml_mul(alpha_softplus, ssm_a)` reshaped to 4D | [`qwen35.cpp:376-379`](src/models/qwen35.cpp:376) |
| **Shape** | `[1 or S_v, H_v, n_tokens, n_seqs]` (scalar gate: dim-0 is 1; KDA: dim-0 is S_v) | [`delta-net-base.cpp:40-41`](src/models/delta-net-base.cpp:40) |
| **Datatype** | F32 | Inferred from graph path |
| **Device** | Same as model layer device | Graph nodes inherit backend |
| **Lifetime** | Transient | Same as k |
| **Retained after graph exec?** | **NO** | Same as k |
| **Can be copied to persistent buffer?** | **YES** | Same as k |
| **Requires core graph modification?** | **NO for eval callback**. **YES for graph-embedded copy** | Same as k |
| **Can be captured only for DFlash?** | **YES** | Same as k |

**Critical detail:** The gate is PRE-exp. The GDN kernel applies `exp(g)` at runtime ([`gated_delta_net.cu:96`](ggml/src/ggml-cuda/gated_delta_net.cu:96)). The chunking path applies `ggml_exp()` at [`delta-net-base.cpp:339`](src/models/delta-net-base.cpp:339), but the named node `g_in-{il}` is registered BEFORE the exp operation (line 53 precedes the chunking function's exp at line 339). For the fused GDN path, gate is passed raw to the CUDA kernel which applies exp internally.

---

### 1.4 beta tensor

| Attribute | Value | Source |
|-----------|-------|--------|
| **Graph node name** | `b_in-{il}` | [`delta-net-base.cpp:52`](src/models/delta-net-base.cpp:52) |
| **Produced in** | `llm_build_delta_net_base::build_delta_net_*()` | [`delta-net-base.cpp:49-53`](src/models/delta-net-base.cpp:49) |
| **Predecessor in graph** | `beta` after sigmoid: `ggml_sigmoid(beta_raw)` reshaped to 4D | [`qwen35.cpp:365-366`](src/models/qwen35.cpp:365) |
| **Shape** | `[1, H_v, n_tokens, n_seqs]` | [`delta-net-base.cpp:42`](src/models/delta-net-base.cpp:42) assertion |
| **Datatype** | F32 | Inferred from graph path |
| **Device** | Same as model layer device | Graph nodes inherit backend |
| **Lifetime** | Transient | Same as k |
| **Retained after graph exec?** | **NO** | Same as k |
| **Can be copied to persistent buffer?** | **YES** | Same as k |
| **Requires core graph modification?** | **NO for eval callback**. **YES for graph-embedded copy** | Same as k |
| **Can be captured only for DFlash?** | **YES** | Same as k |

**Critical detail:** The beta captured at `b_in-{il}` is POST-sigmoid. The sigmoid is applied at [`qwen35.cpp:365`](src/models/qwen35.cpp:365) BEFORE the tensor reaches the DeltaNet base builder. The fused GDN kernel uses beta directly as a scalar multiplier without applying sigmoid ([`gated_delta_net.cu:96`](ggml/src/ggml-cuda/gated_delta_net.cu:96)). This means the captured beta is ready-to-use for replay — the replay GDN call can use the captured beta directly.

**Discrepancy with old code:** The old CPU replay at [`llama-context.cpp:4194`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4194) applied `sigmoid()` during replay: `b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]))`. This means the old tape captured PRE-sigmoid beta. The current graph builder applies sigmoid before passing beta to `build_recurrent_attn()`, making the `b_in-{il}` node POST-sigmoid. This is a behavior change from old to current that must be accounted for in replay implementation.

---

### 1.5 Summary table

| Tensor | Node Name | Shape | Datatype | Pre/Post Transform | Old Tape Name |
|--------|-----------|-------|----------|--------------------|A--3--|
| k | `k_in-{il}` | `[S_k, H_k, n_tokens, n_seqs]` | F32 | Post l2_norm | `k_conv_predelta-{il}` |
| v | `v_in-{il}` | `[S_v, H_v, n_tokens, n_seqs]` | F32 | Raw conv output | `v_conv_predelta-{il}` |
| gate | `g_in-{il}` | `[1 or S_v, H_v, n_tokens, n_seqs]` | F32 | Pre exp | `gate-{il}` |
| beta | `b_in-{il}` | `[1, H_v, n_tokens, n_seqs]` | F32 | Post sigmoid | `beta-{il}` |

**Note:** Old tape names (from [`llama-context.cpp:2278-2282`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2278)) used different naming (`k_conv_predelta`, `v_conv_predelta`). Current graph uses cleaner names (`k_in`, `v_in`, `g_in`, `b_in`). The old `qkv_mixed_pretranspose` was also captured but is NOT needed for replay (only k, v, gate, beta are required).

---

## Section 2: Tensor Lifetime and Availability

### 2.1 Current ggml graph execution lifecycle

The graph execution flow in current upstream is:

1. **Graph construction** — [`llama_model::build_graph()`](src/llama-model.cpp:2417) builds the full compute graph. All `cb()` calls happen here, assigning names to tensors. At this point tensors are metadata-only (shape, type, name) with no data.

2. **Graph allocation** — [`ggml_backend_sched_alloc_graph()`](ggml/src/ggml-sched.cpp) allocates buffer space for all intermediate tensors. Memory is reserved but not yet written.

3. **Set inputs** — [`res->set_inputs(&ubatch)`](src/llama-context.cpp:1643) copies input embeddings into input tensors.

4. **Graph execution** — [`ggml_backend_sched_graph_compute_async()`](src/llama-context.cpp:2739) executes the compute graph. Operations run in topological order. Each operation reads its inputs, computes, and writes to its output buffer.

5. **Eval callback** — During step 4, if `cparams.cb_eval` is set, the callback fires after each operation completes. This is set at [`llama-context.cpp:1611`](src/llama-context.cpp:1611): `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data)`.

6. **Post-execution** — After step 4 returns, intermediate tensor buffers are eligible for reuse. The scheduler frees buffers as operations complete, and after the graph finishes, most intermediates are no longer accessible.

### 2.2 The capture window

The capture window is step 5 above: the eval callback fires DURING graph execution, after each tensor's compute operation completes but before its buffer is freed. This is the ONLY point where intermediates can be captured without modifying the graph.

In current upstream, the eval callback mechanism EXISTS but is NOT used by DFlash:
- The callback infrastructure is at [`llama-context.cpp:1611`](src/llama-context.cpp:1611).
- The `cparams.cb_eval` field is initialized from `params.cb_eval` at [`llama-context.cpp:299`](src/llama-context.cpp:299).
- Default value is `nullptr` ([`llama-context.cpp:3946`](src/llama-context.cpp:3946)).
- Current DFlash code does NOT set `cb_eval` for tape capture.

In old code, the eval callback was the PRIMARY capture mechanism:
- [`old-versions/.../llama-context.cpp:1839`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:1839): `dflash_eval_callback` function.
- The callback checked `cap->tape_name_map` to determine if a tensor should be captured.
- On match, it copied tensor data to the persistent tape buffer.

### 2.3 Can intermediates be accessed post-execution?

**NO.** After `ggml_backend_sched_graph_compute_async()` returns, the intermediate tensor buffers are managed by the scheduler's allocation pool and may be overwritten by the next graph execution. There is no guarantee of data persistence.

The only way to access intermediates post-execution is to have copied them during the eval callback window (step 5) or to have embedded `ggml_cpy` operations in the graph itself that write to persistent buffers.

---

## Section 3: Proposed Capture Mechanisms

### 3.1 Option A: Restore eval callback (old approach)

**Description:** Re-introduce `dflash_eval_callback` from old code. Set `cparams.cb_eval` when DFlash tape recording is active.

**Pros:**
- No graph modification required.
- Works with existing graph builder.
- Old code already proved this works ([~200 lines](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:1839)).
- Capture is conditional on `tape_enabled` flag — zero overhead when disabled.

**Cons:**
- Requires re-adding ~200 lines of callback code.
- Callback fires for EVERY tensor in the graph (not just DFlash targets), adding per-tensor overhead.
- CPU-side copy from GPU tensor may require synchronization.

**Implementation sketch:**
```cpp
// In llama_context initialization when DFlash is active:
cparams.cb_eval = dflash_tape_eval_callback;
cparams.cb_eval_user_data = &tape_capture_data;

// Callback fires during graph execution:
static bool dflash_tape_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (dflash_tape_capture *) user_data;
    if (!cap->tape_enabled) return false;
    auto it = cap->tape_name_map.find(t->name);
    if (it == cap->tape_name_map.end()) return false;
    // Copy t->data to persistent GPU tape buffer
    return true;
}
```

### 3.2 Option B: Graph-embedded copy (GPU tape approach)

**Description:** Add `ggml_cpy` operations to the graph that write intermediates to persistent GPU buffers. The copy ops are only added when DFlash tape recording is active.

**Pros:**
- No eval callback overhead (no per-tensor function call).
- Copy happens on GPU asynchronously — no CPU sync needed.
- Data is in persistent GPU buffer immediately after graph execution.

**Cons:**
- Requires modifying the graph builder (`delta-net-base.cpp` or model-specific builder).
- Adds copy operations to the graph (increased compute time during speculative forward).
- Requires persistent GPU buffer allocation for tape data.

**Implementation sketch:**
```cpp
// In build_recurrent_attn() or build_delta_net_*():
if (cparams.tape_gpu != nullptr && cparams.tape_enabled) {
    // Add copy ops for k, v, gate, beta to persistent tape buffer
    ggml_tensor * tape_k_dst = ggml_view_4d(ctx0, cparams.tape_gpu, ...);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, tape_k_dst));
    // Similar for v, g, b
}
```

### 3.3 Option C: Hybrid (preferred)

**Description:** Use graph-embedded copies for the normal DFlash path (GPU-available), falling back to eval callback when GPU tape is not available (multi-GPU, CPU-only, etc.).

This matches the old code's design: [`llama-context.cpp:2898-2912`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2898) checked `use_gpu_tape` and fell back to CPU tape + eval callback.

**Recommendation:** Option C is the most robust. It provides optimal performance for the common single-GPU case while maintaining compatibility with edge cases.

---

(Continued in next section: Linear-Only Feasibility Assessment)
