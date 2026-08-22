# Reference S1 — 0.4.4 KV Device-Chain Re-Threading Surface

Research report produced by an Architect subtask during the 0.4.4 migration
planning (Task 6M). All file:line references verified against:

- Y = `75ebe54544c15d0dbd7b3a15884c939654d1ce86` (workspace HEAD)
- Z = `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` (`other-versions\beellama_0.4.4-preview`)
- merge-base = `176c1a16a54f955e5a803b948c746e0a4f58b447` (0.4.1 actual)

No contradictions of the planning context were found.

---

## 1. Constructor Inventory (Z)

All Y-local params are `const char * kv_device_chain = nullptr, size_t beefix_spec_draft_res = 0, bool spec_draft_active = false`. In every Z ctor they slot in as trailing defaulted params after the last existing parameter.

### 1.1 llama_kv_cache (standard KV)
- Ctor decl: [llama-kv-cache.h:105-128](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.h:105) — 20 params, ends `uint32_t tail_visibility_window = 0);` → params slot after line 128.
- Ctor def: [llama-kv-cache.cpp:380](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:380). Z-only pre-loop machinery: route-probe block [435-533](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:435) (only when `tail_tokens > 0`; pre-computed `route_buft` from `model.dev_layer(il)` at 522-524, `route_probe_specs.push_back` 527-532); `tail_plan` at 576.
- Placement loop: [802-915](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:802). Decision at **853-862**: `if (offload) { dev = model.dev_layer(il); buft = ggml_backend_dev_buffer_type(dev); }`; `ctx_for_buft(buft)` at **866**.
- Allocation loop: [942-961](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:942).
- NEW post-allocation tail-route validation: [963-1014](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:963) — `tensor_buft` probe 963-965, `buft_is_meta` 966-969, `validate_meta_body` 970-979, **one-layer-owner check 999-1003**, `spec.buft = k_buft` 1004.
- NEW tail shadow placement: [1050-1101](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:1050) — shadows allocated on the *body* buft (`tail_ctx_for_buft(k_buft)`), so tails ride along with the body's chain assignment.
- Planning-pass insertion point: between [800](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:800) and [802](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache.cpp:802).
- Y reference: [src/llama-kv-cache.h:105-131](../../../src/llama-kv-cache.h:105); planning pass [src/llama-kv-cache.cpp:786-949](../../../src/llama-kv-cache.cpp:786); placement decision [1004-1016](../../../src/llama-kv-cache.cpp:1004).

### 1.2 llama_kv_cache_kvarn
- Ctor decl: [llama-kv-cache-kvarn.h:207-225](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.h:207) — 18 params, ends `uint32_t tail_rollback_tokens = 0);` → params slot after line 225.
- Ctor def: [llama-kv-cache-kvarn.cpp:1135-1623](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1135).
- Z-only capability fns (file-local): `kvarn_backend_supports_native_tail` [53-87](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:53) (meta-recursive), `kvarn_backend_supports_tail_write` [89-117](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:89) (meta-recursive), `llama_kvarn_backend_supports_ops` [260-285](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:260), `kvarn_record_bytes` [289-290](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:289).
- Inner metadata cache: [1181-1203](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1181) (`offload=false`, no chain) + NEW `metadata->set_allocation_group_size` [1210-1212](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1210).
- Record sizing: [1258-1265](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1258) (`n_record_groups = n_groups_per_stream * n_stream`, `n_stage_tokens = KVAR_N_GROUP * stage_groups * n_stream`).
- Layer loop: [1268-1392](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1268): `dev = offload ? model.dev_layer(il) : nullptr` **1276**; fail-closed ops check **1277-1281**; `buft` **1282**; `ctx_for_buft` **1283**; **per-layer device flags 1298-1309** (`native_tail` via `kvarn_backend_supports_native_tail(dev, exact_tail_type, head_dim_k, head_dim_v)`, plus `native_attention`, `mixed_tail_native`, `native_original_v`, `native_rotated_max_query_tokens`); component tensors k/v_records I8 1313-1314, k/v_stage F16 1315-1316; flags stored 1375-1378.
- Allocation: [1421-1441](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1421); NEW body validation [1450-1478](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1450) (K/V records share owner; meta requires axis-1 complete-head split).
- NEW tail routing: [1480-1618](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1480) — routes from body buft 1490; `device_route` = non-CPU 1493; **fail-closed `kvarn_backend_supports_tail_write` 1505-1513**; native-tail check 1515-1521; `metadata->set_tail_routes` 1543; tail tensors on body buft 1570-1582; tail ownership validation 1604-1615.
- `make_metadata_cache()`: [1625-1659](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1625) (second `llama_kv_cache`, CPU-only).
- Planning-pass insertion point: between [1266](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1266) and [1268](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-kvarn.cpp:1268).
- Y reference: ctor [src/llama-kv-cache-kvarn.cpp:931-952](../../../src/llama-kv-cache-kvarn.cpp:931); planning pass [1063-1235](../../../src/llama-kv-cache-kvarn.cpp:1063) (budgets 1093-1150 incl. `llama_kvarn_backend_supports_ops` at 1143; I8-record + F16-stage dummy sizing 1152-1205; assign 1218-1220); placement [1237-1256](../../../src/llama-kv-cache-kvarn.cpp:1237).

### 1.3 llama_kv_cache_iswa
- Ctor decls: overload 1 [16-40](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-iswa.h:16) (ends `tail_native_exact_swa = false`), overload 2 (explicit hparams, DSV4) [44-69](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-iswa.h:44). Params slot after `tail_native_exact_swa` in **both**.
- Ctor def: [llama-kv-cache-iswa.cpp](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-iswa.cpp) — `make_cache` lambda [150-191](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-iswa.cpp:150): compact-exact `llama_kv_cache` 168-173, `llama_kv_cache_kvarn` 177-181, standard `llama_kv_cache` 184-190; `kv_base = make_cache(...)` **205**, `kv_swa = make_cache(...)` **209**.
- Y reference: [src/llama-kv-cache-iswa.h:34,64](../../../src/llama-kv-cache-iswa.h:34) (chain in both overloads); `make_cache` [src/llama-kv-cache-iswa.cpp:159-204](../../../src/llama-kv-cache-iswa.cpp:159) (chain at 183/191/203).

### 1.4 llama_kv_cache_msa — NEW in Z (absent from Y)
- Ctor decl: [llama-kv-cache-msa.h:16-35](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-msa.h:16) — 19 params, ends `tail_rollback_tokens = 0` → slot after line 35.
- Ctor def: [llama-kv-cache-msa.cpp:13-56](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-msa.cpp:13) — `kv_base` `llama_kv_cache` 39-43, `kv_idx` (F32 indexer) 52-55.

### 1.5 llama_kv_cache_dsv4
- Ctor decl: [llama-kv-cache-dsv4.h:90-108](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.h:90) — 18 params → slot after line 108.
- Ctor def: [llama-kv-cache-dsv4.cpp:1177](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:1177) — `kv_raw` (iswa) [1226-1233](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:1226), `kv_csa` [1267-1270](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:1267), `kv_hca` [1275-1278](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:1275), `kv_lid` [1283-1286](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:1283), comp states 1290-1304.
- Separate placement path: `llama_dsv4_comp_state` ctor has its **own** per-layer loop [908-949](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsv4.cpp:908) (decision 917-922) — same `dev_layer(il)`→`buft` pattern.
- Y reference: Y's dsv4 ctor has **no** chain params; Y threaded only a `nullptr` into the inner iswa call at [src/llama-kv-cache-dsv4.cpp:1034-1040](../../../src/llama-kv-cache-dsv4.cpp:1034).

### 1.6 llama_kv_cache_dsa
- Ctor decl: [llama-kv-cache-dsa.h:17-36](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsa.h:17) — 19 params → slot after line 36.
- Ctor def: [llama-kv-cache-dsa.cpp:14-60](../../../other-versions/beellama_0.4.4-preview/src/llama-kv-cache-dsa.cpp:14) — `kv_mla` 38-42, `kv_lid` 56-59.
- Y reference: Y's dsa ctor has no chain params; Y never threaded DSA.

### 1.7 Hybrids
- **llama_memory_hybrid**: primary ctor [21-47](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-hybrid.h:21) (ends `tail_rollback_tokens = 0` → slot after line 47); secondary ctor (model, mem_attn, mem_recr) [49-52](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-hybrid.h:49) takes **no** params. Inner `mem_attn(new llama_kv_cache(...))` at [llama-memory-hybrid.cpp:39](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-hybrid.cpp:39). **Z has NO `n_backup_cells` param** (Y does — that is F4's, re-added in Phase 5, not F3's).
- **llama_memory_hybrid_iswa**: ctor [21-51](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-hybrid-iswa.h:21) (ends `tail_native_exact_swa = false` → slot after line 51). Inner `mem_attn(new llama_kv_cache_iswa(...))` at [llama-memory-hybrid-iswa.cpp:44](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-hybrid-iswa.cpp:44).
- Y flow reference: [src/llama-memory-hybrid.cpp:11-81](../../../src/llama-memory-hybrid.cpp:11) — Y forwards **chain only** (line 67), not res/active (hybrid-path margin is always 0 in Y); [src/llama-memory-hybrid-iswa.cpp:12-87](../../../src/llama-memory-hybrid-iswa.cpp:12) — chain forwarded at line 66.

---

## 2. Call-Site Inventory (Z)

All top-level sites are in `create_memory` at [llama-model.cpp:2115-2585](../../../other-versions/beellama_0.4.4-preview/src/llama-model.cpp:2115). FORWARD = main target-context cache (the Y F3 surface); PASS-NULL = auxiliary/draft context (KVarN is target-context-only per [llama-context.cpp:4307-4311](../../../other-versions/beellama_0.4.4-preview/src/llama-context.cpp:4307)) or architecture Y never threaded.

| # | Site (Z llama-model.cpp) | Ctor | Flag | Rationale |
|---|---|---|---|---|
| 1 | 2146-2165 MINIMAX_M3 | llama_kv_cache_msa | PASS-NULL | New in Z; Y has no MSA. Extend later if needed |
| 2 | 2177-2193 GLM_DSA/DEEPSEEK32 MTP | llama_kv_cache | PASS-NULL | Auxiliary MTP draft-head context |
| 3 | 2203-2222 GLM_DSA/DEEPSEEK32 main | llama_kv_cache_dsa | PASS-NULL | Y never threaded DSA |
| 4 | 2234-2250 DEEPSEEK4 MTP | llama_kv_cache_iswa | PASS-NULL | Auxiliary MTP context |
| 5 | 2252-2270 DEEPSEEK4 main | llama_kv_cache_dsv4 | PASS-NULL | Y never threaded DSV4; comp_state has separate placement loop |
| 6 | 2279-2295 DFLASH (dsv4_hc) | llama_kv_cache_iswa | PASS-NULL | Draft-ring auxiliary context |
| 7 | 2314-2322 recurrent | llama_memory_recurrent | N/A | Recurrent state; F4's n_backup_cells lands here in Phase 5 |
| 8 | 2349-2375 hybrid+SWA | llama_memory_hybrid_iswa | FORWARD | Main target hybrid (Y forwarded chain here) |
| 9 | 2380-2387 hybrid+kvarn native-exact | llama_kv_cache | FORWARD | Main target attention component (all 3 params) |
| 10 | 2389-2395 hybrid+kvarn | llama_kv_cache_kvarn | FORWARD | Main target KVarN (all 3 params) |
| 11 | 2406 | llama_memory_hybrid(model, mem_attn, mem_recr) | N/A | Wraps pre-built caches; no params |
| 12 | 2408-2430 plain hybrid | llama_memory_hybrid | FORWARD | Main target (chain only in Y) |
| 13 | 2480-2504 GEMMA4_ASSISTANT | llama_kv_cache_iswa | FORWARD | Main target w/ cross-context sharing |
| 14 | 2506-2530 standard iswa | llama_kv_cache_iswa | FORWARD | Main target |
| 15 | 2537-2544 kvarn native-exact | llama_kv_cache | FORWARD | Main target (all 3 params) |
| 16 | 2546-2552 kvarn | llama_kv_cache_kvarn | FORWARD | Main target (all 3 params) |
| 17 | 2555-2577 standard kv | llama_kv_cache | FORWARD | Primary F3 path (all 3 params) |

Inner construction sites:

| Site | Ctor | Flag | Rationale |
|---|---|---|---|
| llama-kv-cache-iswa.cpp:168-173 | llama_kv_cache (compact-exact) | FORWARD | iswa is a main-target wrapper |
| llama-kv-cache-iswa.cpp:177-181 | llama_kv_cache_kvarn | FORWARD | Same |
| llama-kv-cache-iswa.cpp:184-190 | llama_kv_cache (standard) | FORWARD | Same |
| llama-kv-cache-msa.cpp:39-43, 52-55 | llama_kv_cache | PASS-NULL | MSA out of scope |
| llama-kv-cache-dsa.cpp:38-42, 56-59 | llama_kv_cache | PASS-NULL | DSA out of scope |
| llama-kv-cache-dsv4.cpp:1226-1233 | llama_kv_cache_iswa (kv_raw) | PASS-NULL | DSV4 out of scope |
| llama-kv-cache-dsv4.cpp:1267-1286 | llama_kv_cache x3 (csa/hca/lid) | PASS-NULL | DSV4 out of scope |
| llama-kv-cache-dsv4.cpp:1290-1304 | llama_dsv4_comp_state x3 | PASS-NULL | Own placement loop 908-949; follow-up item |
| llama-kv-cache-kvarn.cpp:1181-1203 | llama_kv_cache (metadata) | PASS-NULL | CPU-only metadata cache (offload=false) |
| llama-kv-cache-kvarn.cpp:1625-1659 | llama_kv_cache (make_metadata_cache) | PASS-NULL | CPU-only metadata |
| llama-memory-hybrid.cpp:39 | llama_kv_cache (mem_attn) | FORWARD | Forward from hybrid ctor |
| llama-memory-hybrid-iswa.cpp:44 | llama_kv_cache_iswa (mem_attn) | FORWARD | Forward from hybrid-iswa ctor |

Y call-site reference (pattern to restore): src/llama-model.cpp — hybrid-iswa 2159-2161 (chain only), native-exact kv 2179-2182 (all 3), kvarn 2188-2195 (all 3), hybrid 2234-2235 (chain only), iswa 2321-2323 & 2348-2350 (chain), native-exact 2370-2373, kvarn 2379-2386, standard kv 2412-2415 (all 3). Y also passes `n_backup_cells` (e.g. 2110) — that is F4's and belongs to Phase 5.

---

## 3. Budget / Margin Planning Sites (Z)

### 3.1 Free-memory APIs (identical in Z and Y)
- [ggml_backend_dev_memory](../../../other-versions/beellama_0.4.4-preview/ggml/include/ggml-backend.h:223): `void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);`
- [ggml_backend_buft_get_alloc_size](../../../other-versions/beellama_0.4.4-preview/ggml/include/ggml-backend.h:41) — dummy-tensor sizing.

### 3.2 Planning-pass locations
1. **Standard KV:** insert between llama-kv-cache.cpp:800 and 802. Port Y's pass from src/llama-kv-cache.cpp:786-949 (gated `if (offload && kv_device_chain)` at 798; budgets via `ggml_backend_dev_memory` at 823; draft reservation at 828; margin at 834-840; dummy-tensor sizing 868-919; `kv_device_chain_assign` 932-934). Placement decision becomes the Y 3-way at src/llama-kv-cache.cpp:1004-1016 replacing Z's 853-862.
   - Z-specific interaction: the route-probe block 435-533 pre-computes `route_buft` from `model.dev_layer(il)` (522-524) *before* the planning pass. The post-allocation validation 963-1014 re-derives routes from the actual allocated `k` tensor buft (`spec.buft = k_buft` at 1004), so the chain composes — but the early probe validates capabilities against the preferred device, not the spill destination (Risk R1).
2. **KVarN:** insert between llama-kv-cache-kvarn.cpp:1266 and 1268. Port Y's pass from src/llama-kv-cache-kvarn.cpp:1063-1235.
   - Exact Z byte-size equivalents: `kvarn_record_bytes(int bits)` (file-local, same as Y); record/stage counts at 1258-1265; per-layer bytes = 2 x (record I8 + stage F16) dummy tensors, mirroring Y's 1152-1205.
   - **Mandatory Z delta vs Y (1):** the layer loop's per-layer device flags 1298-1309 are computed from `dev` (1276). When the chain is active, flags must be computed from the **planned** layer device, not `model.dev_layer(il)` — otherwise `native_tail`/`native_attention` eligibility silently changes per spill.
   - **Mandatory Z delta vs Y (2):** Z's tail routing is fail-closed on `kvarn_backend_supports_tail_write` (1505) for non-CPU route devices. The planner's device filter must be `llama_kvarn_backend_supports_ops(dev) && (dev == CPU || kvarn_backend_supports_tail_write(dev, exact_type, head_dim))` — stronger than Y's ops-only filter (Y 1143).

### 3.3 Margin / reservation logic (port verbatim from Y)
- [kv_device_chain_config](../../../src/llama-kv-cache-spill.h:143) — `margin_fraction = 0.15`, `margin_min = 256 MiB`.
- Margin rule (Y llama-kv-cache.cpp:834-840 / kvarn 1120-1130): `if (beefix_spec_draft_res > 0 || !spec_draft_active) margin = 0; else margin = max(margin_min, effective_free * margin_fraction)`; draft reservation subtracted from effective free (Y 828).
- Core algorithm to copy verbatim: [src/llama-kv-cache-spill.h](../../../src/llama-kv-cache-spill.h) (276 lines; `kv_string_split` 32, `kv_list_available_devices` 46, `kv_resolve_device_chain` 64-140, `kv_device_chain_plan` 158-166, `kv_device_chain_assign` 186-276). Header-only → no CMake change.

---

## 4. Public API + Arg Plumbing (Z)

| # | File (Z) | Insertion point | What to add | Y reference |
|---|---|---|---|---|
| 1 | include/llama.h | before line 502 (after `kv_tail_request` 501) | `const char * kv_device_chain; size_t beefix_spec_draft_res; bool spec_draft_active;` in `llama_context_params` | include/llama.h:493,497,499 |
| 2 | src/llama-cparams.h | after `kv_tail_type` (73), before `cb_eval` (75) | same 3 fields (defaults nullptr/0/false) | src/llama-cparams.h:70,76 |
| 3 | src/llama-memory.h | in `llama_memory_params` 22-45, after `kv_tail_type` (42), before `mem_other` (44) | same 3 fields | src/llama-memory.h:36-43 |
| 4 | src/llama-context.cpp | (a) cparams copy 292-332; (b) `llama_memory_params` build 720-735; (c) `llama_context_default_params` 4240-4287 — designated-init list ends at 4283-4284, add the 3 fields | 3 copies + 3 defaults | src/llama-context.cpp:308-310, 636-639, 3978 |
| 5 | common/common.h | (a) `common_params_speculative_draft` 326-351 → `size_t beefix_spec_draft_res = 0;`; (b) `common_params` near KV fields 616-630 → `std::string kv_device_chain; size_t beefix_spec_draft_res = 0; bool spec_draft_active = false;` | 2 struct additions | common/common.h:357-358, 641-645 |
| 6 | common/arg.cpp | (a) `--beefix-kv-device-chain` right after `-kvo/--kv-offload` (2568-2575); (b) `--beefix-spec-draft-res` in speculative section (starts 4158; `--spec-type` 4495-4511) | arg definitions; help text verbatim from Y | common/arg.cpp:2444-2453, 4271-4279 |
| 7 | common/common.cpp | in `common_context_params_to_llama` 1814-1855, after `cparams.kv_tail_type` (1852) | `cparams.kv_device_chain = params.kv_device_chain.empty() ? nullptr : params.kv_device_chain.c_str();` + res + spec_draft_active | common/common.cpp:1815-1820 |
| 8 | common/speculative.cpp | in `common_base_params_to_speculative` 2337-2376, near `result.kv_tail_tokens = "0"` (2370) | `result.beefix_spec_draft_res = params_spec.beefix_spec_draft_res;` | common/speculative.cpp:2252 |
| 9 | tools/server/server-context.cpp | at `has_draft` (1187) | `params_base.spec_draft_active = has_draft;` | tools/server/server-context.cpp:1107-1109 |
| 10 | src/llama-kv-cache-spill.h | copy verbatim to Z `src/` | entire 276-line header | — |

Z-only context (no action): `llama_kv_tail_request` API (llama.h:501, src/llama-kv-tail-request.cpp) — tail requests are resolved during context creation and `cparams.kv_tail_tokens` is zeroed when a request is used (common/common.cpp:1333-1347); the chain planner runs inside the ctor with the *resolved* `tail_tokens`, so no interaction needed.

---

## 5. Risk Flags

- **R1 — Tail-route "one layer owner" validation x chain (HIGH).** Z standard KV validates post-allocation that tail-route layers share one buft owner (963-1014, owner check 999-1003). A chain that spills tail-bearing layers onto different devices can violate this. Additionally the early route probe (435-533) evaluates capabilities against `model.dev_layer(il)` (preferred device), not the planned spill device — it may fail-closed on a route that would be valid on the spill device, or pass early and fail late. Design decision needed: (a) accept post-allocation validation as authoritative and treat early-probe failures as warnings when chain active, or (b) re-probe against `kv_layer_buft[il]` when chain active.
- **R2 — KVarN tail-route fail-closed (HIGH).** `kvarn_backend_supports_tail_write` (89) gates non-CPU tail routes (1505-1513). Planner device filter must include this (see 3.2); a spill device without tail-write support must be excluded from *tail-bearing* layers specifically.
- **R3 — Per-layer device-dependent KVarN flags (HIGH).** Flags at 1298-1309 (`native_tail`, `native_attention`, `mixed_tail_native`, `native_original_v`, `native_rotated_max_query_tokens`) are device-dependent; spilling a layer changes runtime code paths. Must recompute from planned device (mandatory delta, 3.2).
- **R4 — Meta-device / tensor-split interaction (MEDIUM).** Z tail validation has explicit meta handling (`buft_is_meta` 966-969, `validate_meta_body` 970-979). `kv_resolve_device_chain` resolves explicit device names to concrete bufts, so chain-assigned layers never land on meta bufts; the default (non-chain) path is untouched. The KVarN capability fns are meta-recursive (53-87, 89-117), so probing a planned concrete device is safe.
- **R5 — SWA/ISWA double planning (MEDIUM).** iswa constructs base then SWA caches sequentially (205, 209); each runs its own planning pass against *live* free memory, so the second pass naturally sees the first's allocation. Verify no double-reservation of `beefix_spec_draft_res` (it must be reserved once per target context, not per inner cache).
- **R6 — `unified` flag (LOW).** Sizing uses real per-layer dummy tensors, so unified vs per-sequence is handled naturally.
- **R7 — KVarN target-context-only (LOW, by design).** Z disables KVarN in auxiliary contexts (llama-context.cpp:4307-4311); PASS-NULL draft/MTP sites keep res/active semantics consistent.
- **R8 — Y hybrid-path margin asymmetry (LOW, preserve or consciously fix).** Y forwards chain but **not** res/active through hybrid wrappers (src/llama-memory-hybrid.cpp:67), so hybrid-target runs never reserve draft margin. Porting as-is preserves the limitation; forwarding all 3 is a 2-line change per wrapper if desired.
- **R9 — DSV4 comp_state separate placement loop (LOW, out of scope).** 908-949 duplicates the placement pattern; small ratio-compressed state; candidate follow-up only.
- **R10 — Y-only `n_backup_cells` in hybrid/recurrent ctors (informational).** Present in Y (e.g. src/llama-model.cpp:2110); absent in Z. This is F4's param — re-added in Phase 5, NOT part of F3 threading.

---

## 6. Relative Effort Estimate (ranked, no time estimates)

| Rank | Work item | Type | Notes |
|---|---|---|---|
| 1 | KVarN: planning pass + per-layer flags from planned device + tail-write capability filter (3.2, R2/R3) | Design-heavy | Only ctor where the Z layer loop itself must change, not just gain a pre-pass |
| 2 | Standard KV: planning pass + route-probe interaction decision (R1) | Design-heavy | One open design decision (early probe vs post-allocation authority); rest is port |
| 3 | Wrapper ctor threading: iswa x2, hybrid, hybrid-iswa, msa, dsa, dsv4 (1) | Mechanical, many files | ~7 headers + 5 cpp; pure param add + forward |
| 4 | llama-model.cpp call sites (2, 17 top-level + inner sites) | Mechanical | FORWARD/PASS-NULL per table |
| 5 | Public API + arg plumbing (4, 10 items) | Mechanical | ~20 small edits across 9 files |
| 6 | Copy src/llama-kv-cache-spill.h verbatim | Trivial | Header-only, no CMake change |
| 7 | Build + ctest + multi-GPU manual verification | User-performed | Per project rules, builds are user-run |

Recommended implementation order: 6 → 5 → 1 → 2 → 3 → 4, so the two design-heavy ctors land while the plumbing they depend on is already in place.
