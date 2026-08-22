# 7. KV / KVarN / Multi-GPU Implications — Local Device-Chain vs 0.4.4 Tensor-Split

This is the pivotal architectural comparison: does 0.4.4's new KV placement
supersede the local `--beefix-kv-device-chain`?

## 7.1 The two mechanisms, precisely

### Local device-chain (Y) — `src/llama-kv-cache-spill.h`

- **Granularity:** whole KV *layers*.
- **Algorithm:** `kv_device_chain_assign()` walks layers in order; for each
  layer it walks an ordered device chain (`CUDA0,CUDA1,CPU`) and assigns the
  layer to the **first device with remaining budget** (first-fit bin packing).
- **Budgets:** caller-supplied per-device free memory (VRAM via
  `ggml_backend_dev` queries) minus a safety margin (`margin_fraction`
  default 15%, `margin_min` 256 MiB; reducible/zeroable — F2 sets 0% when a
  reservation is present).
- **Placement decision point:** before `ctx_for_buft(buft)` /
  `ggml_backend_alloc_ctx_tensors_from_buft()` — i.e., it rewrites the
  per-layer buffer-type selection that upstream derives from
  `model.dev_layer(il)`.
- **Semantics:** *spill* — keep as much KV as possible on the fast device,
  push the rest down the chain. No cross-device data movement per token;
  each layer's attention reads only its own device's KV. Works with
  `--fit`-style budgeting because it *is* the budgeting.
- **Validation:** runtime-validated on the target hardware (24 GB + 10 GB
  consumer GPUs) per prior project investigation (Memory MCP: "KV Device
  Chain Feature" — all standard/KVarN/hybrid/ISWA paths verified).

### Upstream 0.4.4 tensor-split (Z) — `src/llama-kv-cache-placement.{h,cpp}`

- **Granularity:** KV *components* (standard K/V, K/V tails, KVarN K/V
  records and stages) split **along complete KV heads** (axis 0 for rows /
  axis 1 for KVarN sliced heads).
- **Algorithm:** `llama_tensor_split_counts(n_elements, weights, granularity)`
  — cumulative-ratio partition: device i gets `cumulative_i/total` of the
  elements, rounded down to granularity; final shard takes the remainder.
- **Mechanism:** a "meta device" backend (ggml-backend-meta, +409 lines in
  Z) that shards tensors across backends; the model's tensor topology
  (including auxiliary speculative contexts) follows the target topology.
- **Semantics:** *tensor parallelism* — every layer's KV is divided across
  GPUs, so **every attention op does cross-device reductions** (NCCL when
  built with it; PCIe otherwise). This is a compute-topology choice, not a
  memory-spill choice.
- **Constraints (CONFIRMED, docs/multi-gpu.md):**
  - EXPERIMENTAL; "less mature than pipeline parallelism."
  - `--fit` **not supported** — manual `--ctx-size` required.
  - Requires `-fa on`.
  - Not implemented for a deny-list of archs (DeepSeek2/3.2/4, GLM-DSA,
    Nemotron-H, Mamba/Jamba/Falcon-H1, Kimi-Linear, LFM2, Minimax-M2/M3,
    Mistral4, Granite-Hybrid, ...) — **QWEN35/QWEN35MOE are allowed**
    (not in the deny-list, `llm_arch_supports_sm_tensor` default-true).
  - 0.4.4's own validation note: Qwen3.6-27B checks used "two logical CUDA0
    shards... not physical two-GPU or peer-transfer results. Keep tensor
    KVarN and precision tails labeled experimental until the external
    two-GPU checklist is completed on two distinct device IDs."

## 7.2 Do they solve the same problem?

**Partially, and differently.**

| Dimension | Local device-chain | 0.4.4 tensor-split |
|---|---|---|
| Goal | Fit more KV in constrained VRAM | Balance compute across GPUs (and fit more KV) |
| Unit | Whole layer | KV head (component) |
| Cross-device traffic | None (per-layer locality) | Per-layer reductions (NCCL/PCIe) |
| Interconnect sensitivity | Low (spill only) | High (TP is interconnect-bound) |
| `--fit` | Compatible (it *is* the planner) | Disabled |
| Heterogeneous GPUs (24+10 GB) | Natural (chain + budgets) | Works via weights, but TP on an x4 PCIe 10 GB card is throughput-limited |
| Recurrent/SSM layers | Unaffected (no KV; state stays on layer device) | Not in deny-list for QWEN35, but hybrid state handling under meta-device is exactly the "unvalidated" territory |
| DFlash drafter placement | Independent (draft on its own device) | Docs: drafter "on one explicit device while its borrowed target embeddings... execute on the target's tensor topology" — supported but experimental |
| Maturity on this hardware | Runtime-validated | Explicitly unvalidated on physical 2-GPU |

The target hardware (RTX 3090 24 GB PCIe x16 + RTX 3080 10 GB PCIe x4 via
M.2 adapter, DWM VRAM cap ~0.3–0.5 GB on the second card) is a
**heterogeneous, interconnect-constrained** configuration. Tensor
parallelism is explicitly "much more bottlenecked by the GPU interconnect
speed" (docs/multi-gpu.md). A 10 GB card on PCIe x4 through a chipset
adapter is the worst case for TP reduction traffic. The device-chain's
spill model (hot KV + weights on the 3090, overflow KV on the 3080/CPU, no
per-token cross-device traffic) is the better fit for this box.

**Conclusion:** 0.4.4's tensor-split is *not* a drop-in superset of the
device-chain for this workload. It is an alternative for a different
objective (balanced TP on fast interconnects). The device-chain remains the
correct mechanism for constrained-VRAM spill on heterogeneous consumer
GPUs. *(STRONG INFERENCE for performance; CONFIRMED for all maturity/
constraint facts.)*

## 7.3 Classification of the KV differences (forced vs optional)

| # | Difference | Class | Rationale |
|---|---|---|---|
| 1 | Device-chain layer-granularity spill vs TP head-granularity split | **B (design choice)** | Upstream chose TP as the multi-GPU KV story; spill-by-layer is a legitimate alternative design that upstream never built. Not forced by any 0.4.4 API; not redundant (different objective). |
| 2 | Device-chain budget/margin logic vs `--fit` | **B (design choice)** | `--fit` cannot do ordered-chain first-fit with per-device safety margins; the device-chain *replaces* fit for KV when active. |
| 3 | F2 VRAM reservation feeding device budgets | **D-adjacent / B** | 0.4.4's `--fit` accounts for the draft model only in the fit phase, not as an explicit reservation in a chain budget. Not superseded. |
| 4 | KVarN record/stage/tail component model (new in 0.4.2/0.4.3) | **A (forced by modern upstream)** | The KVarN internal layout (records vs stages vs exact tails, `llama-kv-tail-request`) changed; any KVarN device-chain port must operate on the new component model. This is the main *forced* rework in F3. |
| 5 | Prompt-cache/transactional restore rework | **A (forced)** | Checkpoint/restore semantics changed (0.4.3); F4's backup/restore and any device-chain state assumptions must be re-validated against the new transactional restore. |
| 6 | `llama-kv-cache-placement` meta-device split callback | **C (integration convenience, for upstream)** / **E (for us)** | Upstream built it to integrate TP with the model loader. Whether we ever use it for spill is an open design question (see 7.4). |
| 7 | DFlash RS-buffer behavior (still `need_n_rs_seq` in Z) | **D (not superseded)** | F4 still needed. |

## 7.4 The open design question (resolves at implementation time)

0.4.4's `llama_kv_cache_component_from_name()` + split-callback machinery is
a clean *upstream* hook for per-component device assignment. A future
device-chain could, in principle, be re-expressed as "assign each KV
component to a device by budget" using the component taxonomy, which would
make the port more upstream-aligned than re-threading a `const char *`
through every constructor.

This is **not required** for migration (re-threading works), but it is the
recommended target shape for the re-plumbed device-chain because:
- it uses the component roles upstream already defines (standard K/V/tail,
  KVarN records/stages/tail),
- it avoids adding a new public-API string field to
  `llama_context_params` (the current F3 approach),
- it composes with the new `llama-kv-tail-request` fitting primitive.

**Decision for the migration plan:** port F3 in two phases —
(1) re-thread the existing `kv_device_chain` parameter through 0.4.4's
constructors to restore current behavior (bounded, mechanical);
(2) optionally re-express the planner on the component taxonomy as
follow-up work (design improvement, not migration-critical).

## 7.5 KVarN-specific port notes

- 0.4.4 KVarN changed substantially (`llama-kv-cache-kvarn.cpp` +924,
  `llama-kvarn.cpp` +106, CUDA dispatch +367, route policy +206). The Y
  device-chain KVarN block (`llama-kv-cache-kvarn.cpp:1072-1248` in Y)
  computes per-device budgets over KVarN record/stage/tail sizes and calls
  `kv_device_chain_assign`. In Z, the sizes come from the new
  `llama-kv-tail-request`/descriptor model — the *planner call* is unchanged,
  the *size computation* must follow the new descriptors.
- The KVarN fast-decode CUDA pairs and `fattn-kvarn-route-policy` are
  upstream-owned; the device-chain does not touch them (it only chooses
  *which device* holds each component), so no kernel-level conflict.
