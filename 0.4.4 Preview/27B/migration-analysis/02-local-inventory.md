# 3. X → Y — Complete Local Modification Inventory (Extremity-attributed)

Every item below is attributable to `Extremity` commits (pre-merge `589fc8b89`
and/or post-merge `deeda007d..Y`). For each: problem solved, mechanism,
upstream internals touched, invasiveness, coupling, current value, and
whether 0.4.4 already provides the capability.

---

## F1. Independent DFlash draft context size — `--beefix-spec-draft-ctx`

- **Problem:** Upstream copies the target's `n_ctx` into the draft model
  context. For a small DFlash drafter serving a large-context target, the
  draft KV cache is sized for the full target context → absurd VRAM.
  *(Flagged by project owner as crucial.)*
- **Mechanism:** `common/arg.cpp:4262` adds the arg, sets
  `params.speculative.draft.n_ctx`; `common/speculative.cpp:2238`
  (`common_speculative_params_to_llama`) propagates it via
  `result.n_ctx = params_spec.n_ctx` into the draft `llama_context_params`.
- **Upstream internals touched:** 1 arg definition + 1 propagation line +
  field existence in the draft params struct.
- **Invasiveness:** Minimal (CONFIRMED). Opt-in; default = upstream behavior.
- **Coupling:** Independent.
- **Still valuable:** Yes — the core VRAM lever for the target workload.
- **0.4.4 provides equivalent?** **NO.** `grep draft.n_ctx` in Z's
  `common/speculative.cpp` → 0 matches; Z's `common_speculative_params_to_llama`
  does not propagate an independent draft `n_ctx`. *(CONFIRMED)*
- **Port effort to 0.4.4:** Trivial — re-add arg + re-add the one propagation
  line in the new `common_speculative_params_to_llama` (Z's version at
  ~line 2375 has the same `result.n_ctx`/`params_spec` shape).

## F2. Speculative VRAM reservation + measurement — `--beefix-spec-draft-res`, `--beefix-draft-spec-measure`

- **Problem:** The KV planner sizes the target KV cache from free VRAM, but
  the draft model is loaded *after* planning, so its VRAM is not accounted
  for → OOM or undersized target KV. Also no way to measure the draft's real
  footprint.
- **Mechanism (post-merge, `cc25aa90b`):**
  - `--beefix-spec-draft-res` (MiB): reservation subtracted from free VRAM
    before the KV budget calculation in both standard KV and KVarN device
    planning; margin logic adjusted (0% when reservation set or no draft;
    15% only when draft exists without reservation).
  - `--beefix-draft-spec-measure`: `common_speculative_measure_vram()` loads
    the draft model through the real init path and reports exact byte usage,
    then exits (diagnostic).
  - Validation: single-device rejection, `--no-kv-offload` override,
    `--split-mode tensor` fatal, `--fit` warning; cyan startup + completion
    messaging with 2% tolerance check.
- **Upstream internals touched:** `common/arg.cpp`, `common/common.{h,cpp}`,
  `common/speculative.{cpp,h}`, `src/llama-kv-cache.cpp`, `src/llama-kv-cache-kvarn.cpp`
  (budget/margin sites), `tools/server/server.cpp`, `tools/server/server-context.cpp`.
- **Invasiveness:** Low–medium; additive, opt-in, all default-off.
- **Coupling:** Sits on top of F3 (device-chain budgeting) — reservation feeds
  the same `device_budgets` arrays.
- **Still valuable:** Yes — required for correct two-GPU + draft VRAM math.
- **0.4.4 provides equivalent?** **NO.** `grep beefix` in Z → 0 matches.
  *(CONFIRMED)*
- **Port effort:** Low–medium. Re-apply the budget/margin hunks at the new
  (0.4.4) planning sites; re-verify the measurement path against 0.4.4's
  reworked speculative init.

## F3. KV device-chain / multi-GPU KV spill — `--beefix-kv-device-chain`

- **Problem:** KV cache placement is tied to model layer placement
  (`model.dev_layer(il)` → `ggml_backend_dev_buffer_type(dev)`). With a
  24 GB + 10 GB consumer two-GPU box and large contexts, KV must spill to the
  second GPU/CPU without moving whole layers manually.
- **Mechanism (pre-merge, `589fc8b89` + post-merge fixes):**
  - `src/llama-kv-cache-spill.h` (276 lines): device-agnostic
    "walk layers → walk chain → first device with budget" planner
    (`kv_resolve_device_chain`, `kv_device_chain_assign`,
    `kv_device_chain_config` with `margin_fraction`/`margin_min`).
  - Standard KV: `llama-kv-cache.cpp:796-1004` builds per-device budgets from
    `ggml_backend_dev` free memory, calls the planner, assigns per-layer
    `buft` before `ctx_for_buft`/allocation.
  - KVarN: `llama-kv-cache-kvarn.cpp:1072-1248` — same pattern for KVarN
    record/stage/tail components.
  - Plumbed through constructors: `llama-kv-cache.{h,cpp}`,
    `llama-kv-cache-iswa.{h,cpp}`, `llama-memory-hybrid*.h/.cpp`,
    `llama-memory.h`, `llama-model.cpp` (8 call sites), `llama-context.cpp`,
    `llama-cparams.h`, `llama.h`, `common/arg.cpp`, `common/common.cpp`.
  - Tail behavior: KV tail follows body device placement (no separate tail
    device handling needed) — verified in prior investigation.
- **Upstream internals touched:** The KV cache constructor chain (8 files) +
  public API field (`llama_context_params.kv_device_chain`).
- **Invasiveness:** Medium. The *algorithm* is fully isolated in `spill.h`;
  the *integration* is parameter threading through ~15 constructor signatures
  and ~8 model call sites (83 `kv_device_chain` references total).
- **Coupling:** F2 feeds its budgets. Independent of F1/F4.
- **Still valuable:** Yes — the primary multi-GPU KV mechanism, runtime-
  validated on the target hardware.
- **0.4.4 provides equivalent?** **PARTIALLY / NOT CLEANLY.** 0.4.4 adds
  `--split-mode tensor` (tensor-parallel KV split via `llama-kv-cache-placement.{h,cpp}`
  + meta backend). See `04-kv-kvarn-multigpu.md` for the full comparison.
  QWEN35 is *not* in the tensor-split deny-list (`llm_arch_supports_sm_tensor`
  default-true), but the mode is EXPERIMENTAL, disables `--fit`, and 0.4.4's
  own docs state the Qwen3.6-27B checks were "not physical two-GPU or
  peer-transfer results" and must stay "labeled experimental until the
  external two-GPU checklist is completed." The local device-chain is
  layer-granularity spill (different model) and is already validated.
- **Port effort:** **Highest of the four.** Re-thread `kv_device_chain`
  through 0.4.4's rewritten KV constructors (standard/KVarN/ISWA/hybrid/
  DSV4) and re-apply the budget/margin blocks at the new planning sites.
  `spill.h` itself ports as-is.

## F4. Custom VRAM-efficient DFlash (Task 6R) — `--beefix-dflash-custom`

- **Problem:** Upstream DFlash on a recurrent (GDN) target allocates
  `n_rs_seq` RS snapshot rows on the *target* context (~5.4 GB for
  Qwen3.6-27B @ 32k, `n_rs_seq=8`, auto `n_parallel=4`) even though DFlash's
  `accept()` is a NOOP for RS rollback (server uses checkpoint rollback).
  The old 0.3.2 custom DFlash used backup cells + zero-VRAM GPU tape replay
  instead.
- **Mechanism (post-merge, `dc867534c`):**
  - `common/server-dflash-custom.{h,cpp}` (249 + 862 lines): per-slot GPU tape
    (F32, pre-allocated, device-aware placement per model layer) capturing
    rank-factored GDN intermediates (k, v, gate, beta, qkv) during the draft
    forward pass; backup/restore of recurrent R/S cells; replay graph that
    re-applies accepted tokens via `ggml_gated_delta_net()` without a full
    target forward pass.
  - `ggml/src/ggml-cuda/dflash-custom-conv.{cu,cuh}` (185 + 78): CUDA kernel
    for conv-state rebuild during replay.
  - Graph hook: `src/models/qwen35.cpp` `build_layer_attn_linear` — when
    `cparams.tape_gpu != nullptr`, inserts `ggml_cpy` capture ops into the
    graph (isolated ~80-line block, CONFIRMED to be the only qwen35.cpp local
    hunk; the rest of the Y→Z qwen35.cpp delta is upstream MTP-flag changes).
  - `src/llama-memory-recurrent.{h,cpp}`: `n_backup_cells` support +
    `backup_offset()`; `src/llama-context.{h,cpp}`: `cparams.tape_gpu`;
    `tools/server/server-context.cpp`: per-slot lifecycle (init/free/replay
    on accept, `replay_failed` permanent-disable fallback).
  - Strictly opt-in: stock DFlash path untouched when the flag is absent
    (verified by the Task 6R audit: `need_n_rs_seq()` unchanged from upstream
    in the stock path).
- **Upstream internals touched:** qwen35.cpp graph builder (1 block),
  `llama_memory_recurrent` (backup cells), `llama_context_params` (tape_gpu),
  server slot lifecycle, one CUDA kernel.
- **Invasiveness:** Medium. Self-contained modules + additive hooks; no stock
  code path changes.
- **Coupling:** Depends on Qwen3.5 GDN graph internals (hook point) and
  `llama_memory_recurrent` layout (backup cells).
- **Still valuable:** Yes — recovers ~5 GB VRAM vs stock DFlash on the
  recurrent target and preserves the 0.3.2-era capability.
- **0.4.4 provides equivalent?** **NO.** Z's `common/common.h:409-412`
  `need_n_rs_seq()` still includes `DRAFT_DFLASH` (and adds `DRAFT_DSPARK`);
  Z's `common/speculative.cpp:2406` `n_rs_seq = 0` applies to the *draft*
  context only, not the target. No tape/backup-cell machinery exists in Z.
  *(CONFIRMED)*
- **Port effort:** Medium. Re-target the qwen35.cpp hook at 0.4.4's
  `build_layer_attn_linear` (changed: MTP flag handling, but the GDN
  intermediate tensors k_conv/v_conv/gate/beta/qkv_mixed still exist);
  re-verify `llama_memory_recurrent` backup-cell API (93 lines changed in Z);
  modules + CUDA kernel port largely as-is.

## F5. Verbose "Beefix" debug logging

- `aa7367659` + fixes (`033ed8a0a`, `e67fdb054`): `LLAMA_LOG_DEBUG`
  diagnostics for spill/margin/reservation decisions (visible in F2/F3 code
  above as `[Beefix: Spill]`, `[Beefix: Margin]`, `[Beefix: DeviceChain]`).
- **Value:** Low standalone; ships inside F2/F3. **Port:** incidental.

## F6. Test infrastructure

- `test-runner.py` (1,913 lines): Python fork-feature test runner (extends
  server test utilities).
- `tests/dflash-custom-test.py` (737 lines): custom DFlash tests.
- `b572baab6`: `--spec-draft-model` support for DFlash in the runner.
- **Value:** Medium — encodes the validated runtime scenarios. **Port:**
  copy as-is (standalone, no C++ coupling).

## F7. Documentation

- `docs/beellama-args.md`, `docs/beellama-features.md` updates describing the
  beefix flags. **Port:** update alongside features.

---

## Inventory summary

| ID | Feature | Lines (approx) | Isolation | Port effort | Superseded by 0.4.4? |
|---|---|---|---|---|---|
| F1 | Draft ctx size | ~30 | Fully isolated | Trivial | **No** |
| F2 | VRAM reservation/measure | ~500 | Additive, opt-in | Low–med | **No** |
| F3 | KV device-chain | ~1,100 (276 algo + plumbing) | Algo isolated; plumbing wide | **High** | Partial (tensor-split is experimental, unvalidated) |
| F4 | Custom DFlash | ~2,300 | Self-contained modules + 1 graph hook | Medium | **No** |
| F5 | Debug logging | ~100 | Incidental | Trivial | N/A |
| F6 | Test infra | ~2,650 | Standalone | Trivial (copy) | N/A |
| F7 | Docs | ~275 | Standalone | Trivial | N/A |

Total: ~7,000 lines, of which ~4,600 are the four functional subsystems.
The two *crucial* features (F1, F4) and the two *load-bearing* features
(F2, F3) are all still needed on 0.4.4.
