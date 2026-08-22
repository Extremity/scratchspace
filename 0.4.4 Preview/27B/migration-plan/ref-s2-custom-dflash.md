# Reference S2 — Custom DFlash (F4) Migration Surface

Research report produced during the 0.4.4 migration planning (Task 6M).
All file:line references verified against:

- Y = `75ebe54544c15d0dbd7b3a15884c939654d1ce86` (workspace HEAD)
- Z = `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` (`other-versions\beellama_0.4.4-preview`)
- merge-base = `176c1a16a54f955e5a803b948c746e0a4f58b447` (0.4.1 actual)

Includes the dry-run merge results for the F4 surface (see §7).

---

## 1. F4 Component Map (Y)

| Component | File (Y) | Lines | Role |
|---|---|---|---|
| Tape struct + cell copy + replay graph builder | common/server-dflash-custom.{h,cpp} | 249+862 | Per-slot GPU tape (F32, pre-allocated, per-layer device-aware), `dflash_custom_cell_copy()`, backup/restore, replay graph re-applying accepted tokens via `ggml_gated_delta_net()` |
| Conv-state rebuild kernel | ggml/src/ggml-cuda/dflash-custom-conv.{cu,cuh} | 185+78 | CUDA kernel rebuilding conv state during replay |
| Graph hook | src/models/qwen35.cpp | ~80 | `ggml_cpy` capture ops when `cparams.tape_gpu != nullptr` |
| Backup cells | src/llama-memory-recurrent.{h,cpp} | ~30 | `n_backup_cells` rows appended to R/S tensors; `backup_offset()` |
| cparams | src/llama-cparams.h | ~15 | `n_backup_cells`, `tape_gpu` (forward-declared `struct server_dflash_tape_gpu *`) |
| public API | include/llama.h | ~1 | `llama_context_params.n_backup_cells` |
| context plumbing | src/llama-context.{h,cpp} | ~10 | copy + arch-support validation (`llm_arch_supports_rs_rollback`) |
| model plumbing | src/llama-model.cpp | ~20 | `n_backup_cells` through `create_memory` to `llama_memory_recurrent` ctor |
| server slot lifecycle | tools/server/server-context.cpp | ~240 | tape init/free per slot, backup on speculative start, restore/replay on accept, `replay_failed` permanent-disable fallback, `spec_draft_active` setting |
| flag | common/arg.cpp | ~15 | `--beefix-dflash-custom` (sets `common_params_speculative_draft.beefix_dflash_custom` + `n_backup_cells = n_parallel` in common.cpp:1829) |
| build | CMakeLists.txt, tools/server/CMakeLists.txt | ~5 | `add_compile_definitions(BEE_DFLASH_CUSTOM)` (global, currently unconditional — see §6 R3), server lib compiles `../../common/server-dflash-custom.cpp` + include of `${CMAKE_SOURCE_DIR}/src` |
| tests | tests/dflash-custom-test.py, test-runner.py | 737+1913 | runtime scenarios |

## 2. Y Mechanism Detail

### 2.1 Backup cells (llama-memory-recurrent)
Y's full local diff (verified):
- Ctor gains `uint32_t n_backup_cells` after `n_rs_seq`.
- `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` (was `mem_size * (1 + n_rs_seq)`) for both R and S tensors.
- `mem_size` cached (private) for `backup_offset() = mem_size * (1 + n_rs_seq)`.
- `size`/`used` comments updated: total cells EXCLUDE backup rows (backup rows live beyond the normal allocator range).
- `cell_copy()` was NOT added to the class; extracted to free function `dflash_custom_cell_copy()` in server-dflash-custom.cpp (deliberate minimization of class modification).
- Info log of backup cell MiB when enabled.

### 2.2 Tape capture hook (qwen35.cpp)
Y's full local diff (verified): one include (`../../common/server-dflash-custom.h`) + one ~77-line block in `build_layer_attn_linear` inserted **after** the `cb(k_conv, "k_conv_predelta", il)` / `cb(v_conv, "v_conv_predelta", il)` callbacks and **before** `build_recurrent_attn(...)`:
- Guard: `cparams.tape_gpu != nullptr`.
- Layer lookup in `tgpu->layer_ids`; skip if absent or `n_seq_tokens > tgpu->max_tokens`.
- Captures 5 rank-factored GDN intermediates per layer via `ggml_view_3d`/`ggml_view_2d` (first sequence), `ggml_cont`, then graph-embedded `ggml_build_forward_expand(gf, ggml_cpy(...))` into pre-allocated tape tensors `tl.{k,v,gate,beta,qkv}`:
  - `k_conv` [head_k_dim, num_k_heads, n_seq_tokens, n_seqs]
  - `v_conv` [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
  - `gate` [1, num_v_heads, n_seq_tokens, n_seqs]
  - `beta` (post-sigmoid) [1, num_v_heads, n_seq_tokens, n_seqs]
  - `qkv_mixed` [conv_channels, n_seq_tokens, n_seqs]
- Zero compute overhead: copies overlap with the layer's own compute.

### 2.3 Replay + verification + failure semantics (server-dflash-custom.cpp)
- On speculative start: backup the slot's live R/S cells into the backup rows (`dflash_custom_cell_copy` → rows at `backup_offset()`), remember conv state.
- Draft forward runs with the tape hook active (tape captured into the slot's tape tensors on the per-layer devices).
- On accept: replay accepted tokens through a replay graph built from tape data — re-applies the GDN recurrence via `ggml_gated_delta_net()` with the captured (k,v,gate,beta,qkv) instead of re-running the full target forward; conv state rebuilt by the CUDA kernel (dflash-custom-conv.cu).
- Target checkpoint rollback (upstream mechanism) reverts KV; the recurrent state is restored from the backup rows; then replay re-advances recurrent state to the accepted length.
- `replay_failed` flag: permanent per-slot disable of custom replay, falling back to stock checkpoint-restore behavior (no crash, degraded VRAM).

### 2.4 Stock-path isolation (verified in Y)
- `need_n_rs_seq()` unchanged from upstream in the stock path (Task 6R audit).
- All F4 code paths gated on `--beefix-dflash-custom` (cparams.tape_gpu != nullptr / n_backup_cells > 0 / beefix_dflash_custom flag).
- Without the flag: no tape allocation, no backup rows, no conv kernel, stock DFlash identical to upstream.

### 2.5 F1/F2 flag and plumbing names (verified in Y)
- `--beefix-spec-draft-ctx N` → `params.speculative.draft.n_ctx` (default 512 in common.h:329); propagated by ONE line in `common_base_params_to_speculative`: `result.n_ctx = params_spec.n_ctx;` (speculative.cpp:2238).
- `--beefix-spec-draft-res N` (MiB) → `params.speculative.draft.beefix_spec_draft_res`; propagated to `result.beefix_spec_draft_res` (speculative.cpp:2252) and to `cparams.beefix_spec_draft_res` in `common_context_params_to_llama` (common.cpp:1815).
- `--beefix-draft-spec-measure` → `params.beefix_draft_spec_measure`; server calls `common_speculative_measure_vram(params)` (speculative.cpp:2733-2802) and prints per-device delta, then exits.
- `--beefix-dflash-custom` → `params.speculative.beefix_dflash_custom`; in common.cpp `common_context_params_to_llama`: `if (beefix_dflash_custom) { cparams.n_rs_seq = 0; cparams.n_backup_cells = params.n_parallel; }` (common.cpp:1818-1821). THIS is the 5.4 GB saving: the target's RS snapshot rows are eliminated and replaced by n_parallel backup rows.
- `spec_draft_active`: set by the server from `has_draft` (server-context.cpp:1107-1109) → `cparams.spec_draft_active` (common.cpp:1820). Drives the F3 margin decision.
- `--beefix-kv-device-chain DEVICE[,DEVICE...]` → `params.kv_device_chain` → `cparams.kv_device_chain` (F3, see ref S1).

## 3. Z (0.4.4) Anchor Map for F4

### 3.1 Recurrent memory
- Class: [llama-memory-recurrent.h:17](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:17) `llama_memory_recurrent : public llama_memory_i`.
- Ctor [19-27](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:19) — same param list as merge-base (no `n_backup_cells`): `model, mem_size, n_seq_max, n_rs_seq, filter`.
- `size`/`used` [76-77](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:76); `n_rs_seq` [82](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:82); `rs_idx` [83](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:83); `set_rs_idx` [85](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:85).
- `mem_cell` struct [94-115](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:94); `find_slot` [64](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:64) (contiguous slot search — backup rows must stay OUT of this allocator's range, exactly as in Y).
- State I/O: `state_write_meta/data` [137-138](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:137), `state_read_meta/data` [140-141](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:140) — **Z's 0.4.3 transactional-restore rework touches these; backup rows beyond `size` must not be serialized by them (they take explicit `cell_ranges`, so out-of-range backup rows are naturally excluded — confirm during Phase 5).**
- R/S accessors: `get_r_l` [180](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:180), `get_s_l` [181](../../../other-versions/beellama_0.4.4-preview/src/llama-memory-recurrent.h:181).
- Z ctor body: R/S tensor creation at the `n_rows = mem_size * (1 + n_rs_seq)` line in llama-memory-recurrent.cpp (Y line 100; Z line differs by +93 lines — locate by text, not number).

### 3.2 qwen35.cpp GDN graph
- `build_layer_attn_linear` def: [qwen35.cpp:339](../../../other-versions/beellama_0.4.4-preview/src/models/qwen35.cpp:339); call site [172](../../../other-versions/beellama_0.4.4-preview/src/models/qwen35.cpp:172).
- `qkv_mixed` computed [237-244](../../../other-versions/beellama_0.4.4-preview/src/models/qwen35.cpp:237) (`build_lora_mm` + reshape + `cb(qkv_mixed, "linear_attn_qkv_mixed", il)`).
- `conv_states_all = mctx_cur->get_r_l(il)` [382](../../../other-versions/beellama_0.4.4-preview/src/models/qwen35.cpp:382); `conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il)` [389](../../../other-versions/beellama_0.4.4-preview/src/models/qwen35.cpp:389).
- Y's hook anchor (after `cb(k_conv, "k_conv_predelta", il)` / `cb(v_conv, "v_conv_predelta", il)`) exists in Z — locate by those two `cb(...)` calls.
- **All 5 captured tensors (k_conv, v_conv, gate, beta, qkv_mixed) exist in Z with the same names/shapes.** Z's local changes to qwen35.cpp vs merge-base are MTP-flag handling elsewhere in the file; the GDN core is intact.

### 3.3 Speculative / RS infrastructure
- `need_n_rs_seq()`: [common/common.h:409-412](../../../other-versions/beellama_0.4.4-preview/common/common.h:409) — still includes `DRAFT_DFLASH` and adds `DRAFT_DSPARK`. So stock DFlash still allocates `n_rs_seq` RS snapshot rows on the target (the 5.4 GB problem F4 solves) — **F4 is still needed in Z** (CONFIRMED).
- Draft-context `n_rs_seq = 0`: [common/speculative.cpp:2406](../../../other-versions/beellama_0.4.4-preview/common/speculative.cpp:2406) — draft-only; does not affect the target.
- 0.4.3 transactional restore: prompt-cache/restore rework (one safe-prefix planner across standard/recurrent/KVarN, self-contained checkpoints) — the rollback semantics F4's backup/restore/replay composes with. New API: `restore_checkpoint_transaction(slot, ckpt, ctx_tgt, ctx_dft, ...)`. This is the one genuine semantic risk (see §6 R1).

### 3.4 Server
- `has_draft` at [server-context.cpp:1187](../../../other-versions/beellama_0.4.4-preview/tools/server/server-context.cpp:1187); `params_base.n_outputs_max_per_seq = output_limits.per_seq` [1191](../../../other-versions/beellama_0.4.4-preview/tools/server/server-context.cpp:1191).
- `kv_tail_requested` accounting [2992](../../../other-versions/beellama_0.4.4-preview/tools/server/server-context.cpp:2992), API field [5332-5334](../../../other-versions/beellama_0.4.4-preview/tools/server/server-context.cpp:5332).
- New slot field `spec_is_replay` set by Z's accept/restore path (server-context.cpp:4680 in the dry-run merge) — required by Z's downstream code; the F4 fallback branch must set it too.

## 4. F4 Merge Surface (dry-run merge results)

From the scratch-clone dry-run 3-way merge (base=176c1a16a, ours=Y, theirs=Z).
Note: "whole-file EOL conflict" means the single conflict region spans the entire file
because Y's blob is CRLF and Z's is LF; the per-file protocol in §5 resolves them.

| F4 file | Merge result | Action |
|---|---|---|
| common/server-dflash-custom.{h,cpp} | auto-added (A) | none |
| ggml/src/ggml-cuda/dflash-custom-conv.{cu,cuh} | auto-added (A) | verify CMake picks them up (they're in the cuda source glob — confirm) |
| src/llama-cparams.h | auto-merged (M) | `n_backup_cells` + `tape_gpu` + chain fields all present |
| include/llama.h | UU, 2 localized conflicts (434-454, 513-527) | both are "both-sides-added-adjacent" in `llama_context_params` (Y's `n_backup_cells`/chain fields vs Z's `n_outputs_max_per_seq`/`kv_tail_request`) — keep both sides |
| common/common.h | auto-merged (M) | both beefix fields present |
| common/speculative.h | auto-merged (M) | `speculative_vram_measurement` + decl present |
| common/speculative.cpp | UU, 1 localized conflict (2374-2379) | Y's `result.beefix_spec_draft_res = ...` vs Z's `result.n_outputs_max_per_seq = 1;` — keep both lines. Y's `result.n_ctx` (2348) + measure_vram impl auto-merged |
| src/llama-context.cpp | UU, 1 localized conflict (4301-4305) | Y's `/*.kv_device_chain =*/ nullptr` vs Z's `/*.kv_tail_request =*/ nullptr` in default params — keep both lines. Y's cparams copies (291-296, 331, 736-737) auto-merged |
| src/llama-memory-recurrent.{h,cpp} | auto-merged (M) | `n_backup_cells` ctor param, row expansion, `backup_offset()` present |
| src/models/qwen35.cpp | auto-merged (M) | Y's tape hook + Z's MTP changes both present |
| src/llama-model.cpp | UU, whole-file EOL conflict | re-apply Y's 75 local lines (create_memory n_backup_cells + call sites + LLAMA_API on dev_layer) per §5 protocol |
| tools/server/server-context.cpp | UU, **1 SEMANTIC conflict (4632-4711)** | the only true semantic conflict in the entire merge — see §4.1. Y's other slot-lifecycle hunks (247, 1200-1202, 1334-1426, 1558, 3548-3549) auto-merged |
| CMakeLists.txt / tools/server/CMakeLists.txt | auto-merged (M) | `BEE_DFLASH_CUSTOM` definition + server-lib source present |

**Conclusion: F4's textual merge surface is small.** 3 of 4 localized conflicts are trivial keep-both; the 8 EOL files resolve per §5. The one genuine semantic conflict (§4.1) is bounded and has a prescribed resolution. The remaining F4 work is **semantic validation**, not re-implementation.

### 4.1 The one semantic conflict — server-context.cpp:4632-4711

Location: the speculative **accept/fallback** block (after the replay attempt, before `common_speculative_accept`).

- **Y (ours) side:** branches on `replay_succeeded`. On failure: trace log, then manual checkpoint restore — `ckpt.load_tgt(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` + `common_context_seq_rm(ctx_tgt, ...)`, same for `ctx_dft`, `slot.prompt.tokens.keep_first`, restore of `smpl`/`loop_guard`/`reasoning_output_tokens`/`visible_output_tokens`/`has_next_token`/`stop` saves, `return`. On success: `slot.spec_draft = std::move(accepted)` and continue **without** checkpoint restore.
- **Z (theirs) side:** unconditional restore using the NEW 0.4.3 transactional API: sets `slot.spec_is_replay = true`, `slot.spec_draft = std::move(accepted)`, then `restore_checkpoint_transaction(slot, ckpt, slot.ctx_tgt, slot.ctx_dft, true, slot.ctx_dft != nullptr, true)` with `SLT_ERR + slot.release() + return` on failure; `slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1)`; `slot.prompt.tokens.keep_first(ckpt.n_tokens)`; `common_sampler_copy(smpl_save.get(), slot.smpl.get())` (Z's sampler copy API, not Y's `std::move`); same loop_guard/reasoning/visible/has_next_token/stop save restores; `return`.

**Prescribed resolution (primary Architect decision):** keep Y's `if (!replay_succeeded) { ... }` branch structure (this IS the F4 feature — replay success skips restore entirely), but implement the fallback branch with **Z's** primitives:
1. `slot.spec_is_replay = true;` (Z's flag — required by Z's downstream code).
2. Replace Y's manual `ckpt.load_tgt/load_dft + common_context_seq_rm` sequence with Z's `restore_checkpoint_transaction(slot, ckpt, slot.ctx_tgt, slot.ctx_dft, true, slot.ctx_dft != nullptr, true)` + the `SLT_ERR/slot.release()/return` failure handling.
3. Use Z's `slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1)` and `common_sampler_copy(smpl_save.get(), slot.smpl.get())`.
4. Keep Y's trace logging and the replay-success path (`slot.spec_draft = std::move(accepted);` then fall through to `common_speculative_accept`).

Rationale: Y's manual restore sequence is exactly the code Z's transactional rework replaced; re-implementing Y's sequence would re-introduce the pre-0.4.3 restore model that the migration exists to adopt. Using Z's transaction API in the fallback branch preserves F4's semantics (replay success → no restore; replay failure → full upstream-grade restore) and inherits Z's robustness (transactional safety, slot.release on failure). The backup-cell restore (`dflash_custom_backup` at 3548-3549, which auto-merged) is unaffected — it runs before the draft forward, not in this block.

This resolution must then be validated by the Phase 5 replay/rollback tests (R1 in §6).

## 5. Resolution Protocol for the 8 EOL-Conflict Files (applies to all features)

The 8 whole-file conflicts (llama-kv-cache.cpp, llama-kv-cache-kvarn.cpp, llama-model.cpp, llama-kv-cache.h, llama-kv-cache-iswa.cpp, llama-memory-hybrid.cpp, llama-memory-hybrid.h, llama-kv-cache-dsv4.cpp) exist ONLY because Y's blobs are CRLF and Z's are LF.

**Preferred (pre-merge) fix — eliminates the conflicts entirely:**
1. Add a new `.gitattributes` (no conflict possible): `* text=auto eol=lf` (or scoped: `*.c *.cpp *.h *.cu *.cuh *.py *.sh *.txt *.md text eol=lf`).
2. `git add .gitattributes`, commit.
3. Merge. Git normalizes base/ours/theirs through the attribute → the 3-way merge becomes content-only. Expected residual conflicts: a small number of semantic hunks (Y's planning passes sit adjacent to Z's route-probe code in llama-kv-cache.cpp/kvarn.cpp; Y's create_memory call sites sit adjacent to Z's new MSA/DSV4/DEEPSEEK4 sites in llama-model.cpp).

**Fallback (if .gitattributes is declined) — per-file protocol:**
1. `git checkout --theirs <file>` (take Z/LF side).
2. Generate Y's local hunks ignoring EOL: `git diff --ignore-cr-at-eol <base> <Y> -- <file> > patch`.
3. `git apply --ignore-whitespace patch` (check with `--check` first).
4. Any hunk that fails to apply (context drift from Z's changes) is resolved manually — all Y hunks in these files are additive parameter/line insertions, so manual resolution is a line-level task.
5. Verify: file is LF; Y symbols present (`kv_device_chain`, `n_backup_cells`, `beefix_spec_draft_res`, `spec_draft_active`, `tape_gpu`); Z symbols present (`n_outputs_max_per_seq`, `kv_tail_request`, `DRAFT_DSPARK`, route-probe code).

Y's local line counts for the 8 files (content, ignoring EOL): llama-kv-cache.cpp 181, llama-kv-cache-kvarn.cpp 196, llama-model.cpp 75, llama-kv-cache.h ~10, llama-kv-cache-iswa.cpp ~20, llama-memory-hybrid.cpp ~15, llama-memory-hybrid.h ~10, llama-kv-cache-dsv4.cpp 1.

## 6. F4 Risks

- **R1 (HIGH) — 0.4.3 transactional restore semantics.** Y's backup/restore/replay was validated against 0.4.1 checkpoint rollback. Z's restore rework (safe-prefix planner, self-contained checkpoints, `restore_checkpoint_transaction`) changes when/how recurrent state is restored during slot save/restore and prompt-cache reuse. F4's backup rows must remain consistent across a Z-restore: verify `state_read_data` (restore_head semantics) does not clobber or shift backup rows, and that a Z-initiated restore of a custom-DFlash slot re-runs the backup protocol or invalidates the tape. This is THE validation gate for Phase 5. The §4.1 resolution puts the fallback branch on Z's transaction API, which mitigates but does not eliminate this risk.
- **R2 (MEDIUM) — qwen35 hook context drift.** The hook auto-merged, but Z's MTP rework changed nearby code. Re-verify the 5 tensor names/shapes at the hook site and that `cparams` is in scope exactly as Y expects (the merged file must compile; the shape check is the semantic gate).
- **R3 (MEDIUM) — `BEE_DFLASH_CUSTOM` unconditional compile definition.** Y's CMakeLists.txt adds `add_compile_definitions(BEE_DFLASH_CUSTOM)` globally and unconditionally. For the "stock path = upstream" invariant, the definition should be behind a CMake option (default OFF) or removed in favor of pure runtime flag gating (the code already runtime-gates on cparams). Decide in Phase 7; recommend converting to `option(BEE_DFLASH_CUSTOM ... OFF)` so stock builds have zero F4 surface.
- **R4 (LOW) — `llm_arch_supports_rs_rollback` validation.** Y disables `n_backup_cells` with a warning on unsupported archs (llama-context.cpp:293-296). Confirm the function exists unchanged in Z (it does — auto-merged context).
- **R5 (LOW) — CUDA kernel build inclusion.** dflash-custom-conv.cu is in ggml/src/ggml-cuda/ (glob-included). Confirm it compiles under Z's CUDA CMake (newer nvcc flags) and is gated so non-CUDA builds don't reference it.
- **R6 (LOW) — tape device placement under F3.** Tape tensors are placed per model layer device (Y: "device-aware placement per model layer"). With F3's device-chain active, the tape should follow the layer's *planned* KV device or the model layer device — decide in Phase 5 (recommend: model layer device, since the tape captures the target's draft forward, and document the interaction).

## 7. F4 Completion Criteria

1. `--beefix-dflash-custom` build compiles on CUDA; stock build (flag absent) compiles with zero F4 symbols referenced (R3).
2. Stock DFlash (no flag) behavior identical to Z upstream: same `n_rs_seq` target allocation, same acceptance, same VRAM (regression gate).
3. Custom DFlash: target VRAM ≈ Z stock DFlash − 5.4 GB (the n_rs_seq rows) + backup rows (~n_parallel × layers × (embd_r+embd_s) × 4 B) + tape (5 tensors × n_draft_tokens) — measured, not assumed.
4. Replay correctness: accepted-token outputs bit-identical (or within tolerance) to stock DFlash outputs on the same prompt (the dflash-custom-test.py scenarios).
5. Rollback/replay under multi-slot + prompt-cache reuse (Z's new restore path) — the R1 gate.
6. `replay_failed` fallback path exercised (induce failure) → permanent disable, no crash, stock behavior resumes.
