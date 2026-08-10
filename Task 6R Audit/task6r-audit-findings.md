# Task 6R DFlash Custom Mode — Final Integration Audit Findings

**Date:** 2026-08-10
 **Auditor:** Roo (Architect Mode)
**Status:** Audit complete. MATERIAL ISSUES FOUND.

---

## Concise Overview

### Audit Summary

| Audit | Scope | Result |
|-------|-------|--------|
| 1a | Subtask 1→2 handoff (n_rs_seq + backup cells) | ❌ **1 CRITICAL issue** |
| 1b | Subtask 2→3→4 handoff (tape, device, layer_ids) | ✅ All correct |
| 2 | Runtime lifecycle, opt-in, fallback safety | ✅ All correct |
| 3 | Generality and model independence | ✅ Predominantly generic |
| 4 | Testing and implementation completeness | ⚠️ Complete implementation, limited test coverage |

### Critical Issue

**`n_backup_cells` never populated** ([`common/common.cpp:1765`](common/common.cpp:1765) — `common_context_params_to_llama()`):

The `--beefix-dflash-custom` flag correctly sets `n_rs_seq = 0` but **never sets `n_backup_cells` to a non-zero value**. Since the default is `0` and no code path assigns it, backup cells are never allocated. This means:
- RS buffer reduction works (`n_rs_seq=0`)
- Tape capture works (`qwen35.cpp` graph-embedded copies)
- **Replay can never succeed** — `dflash_custom_replay()` returns `false` when `n_backup_cells == 0`
- Server falls back to checkpoint rollback every cycle (correct but defeats the purpose)

**Fix:** Add `cparams.n_backup_cells = params.n_parallel;` in [`common/common.cpp:1775-1781`](common/common.cpp:1775) within the existing `if (beefix_dflash_custom && has_dflash)` block.

### Other Issues

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | `n_backup_cells` never populated | **CRITICAL** | [`common/common.cpp:1765`](common/common.cpp:1765) |
| 2 | Replay correctness not validated by tests | MODERATE | Test script cannot distinguish replay vs. fallback |
| 3 | `conv_channels` formula Qwen-specific | LOW | [`server-dflash-custom.cpp:65`](common/server-dflash-custom.cpp:65) |

### Verified Correct (No Issues)

| Area | Verdict |
|------|---------|
| `n_rs_seq=0` propagation through constructors | ✅ Correct |
| Backup offset arithmetic | ✅ Correct |
| Tape tensor shape compatibility (capture ↔ replay) | ✅ Correct |
| Device placement (tape and R/S on same GPU) | ✅ Correct |
| Layer ID mapping consistency | ✅ Correct |
| Strict opt-in (triple-gated) | ✅ Correct |
| Stock DFlash unchanged without flag | ✅ Correct |
| Replay lifecycle ordering | ✅ Correct |
| Fallback safety (try-catch → checkpoint) | ✅ Correct |
| Permanent disable after 3 failures | ✅ Correct |
| Edge cases (zero/full acceptance, first cycle) | ✅ Correct |
| Model independence (6/7 areas generic) | ✅ Correct |
| Implementation completeness (9/9 functions, 4/4 structs) | ✅ Correct |

---

## Detailed Findings by Audit Area

### AUDIT 1A: Subtask 1 → Subtask 2 Handoff

**Objective:** Verify `n_rs_seq=0` and `n_backup_cells` correctly propagate from the CLI flag through the recurrent memory constructor.

#### Complete Data Flow Trace

```
--beefix-dflash-custom (CLI)
      │
      ▼
[common/arg.cpp:4384] params.speculative.beefix_dflash_custom = true
      │
      ▼
[common/common.cpp:1770] cparams.n_rs_seq = params.speculative.need_n_rs_seq()  // sets to draft.n_max (e.g., 8)
      │
      ▼
[common/common.cpp:1775-1781] if (beefix_dflash_custom && has_dflash) {
                                 cparams.n_rs_seq = 0;
                               }
      │
      ▼ (n_backup_cells NOT set here)
      │
      ▼
[src/llama-context.cpp:265] cparams.n_rs_seq = params.n_rs_seq;  // passes 0 through ✓
      │
      ▼
[src/llama-context.cpp:271] cparams.n_backup_cells = params.n_backup_cells;  // passes 0 (default) ✗
      │
      ▼
[src/llama-model.cpp:2108-2109] new llama_memory_recurrent(..., cparams.n_rs_seq, cparams.n_backup_cells, ...)
      │
      ▼
[src/llama-memory-recurrent.cpp:104] const uint32_t n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells;
```

#### Check Results

| Check | Result | Evidence |
|-------|--------|----------|
| `beefix_dflash_custom` causes `n_rs_seq=0` | ✅ PASS | [`common/common.cpp:1775-1781`](common/common.cpp:1775) — override within `if (beefix_dflash_custom && has_dflash)` block |
| `n_rs_seq=0` reaches constructor | ✅ PASS | [`src/llama-context.cpp:265`](src/llama-context.cpp:265) passes through. [`src/llama-model.cpp:2108`](src/llama-model.cpp:2108) passes to constructor. Stored at [`src/llama-memory-recurrent.cpp:36`](src/llama-memory-recurrent.cpp:36). |
| Allocation formula reduces correctly | ✅ PASS (when `n_backup_cells=0`) | [`src/llama-memory-recurrent.cpp:104`](src/llama-memory-recurrent.cpp:104): `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells`. With `n_rs_seq=0`: `n_rows = mem_size + n_backup_cells`. |
| `n_backup_cells` passed correctly | ❌ **FAIL** | Plumbing exists but value never set: [`include/llama.h:427`](include/llama.h:427) field exists, [`src/llama-context.cpp:3941`](src/llama-context.cpp:3941) defaults to 0, [`common/common.cpp:1765`](common/common.cpp:1765) **never sets it**, [`src/llama-context.cpp:271`](src/llama-context.cpp:271) passes through 0, [`src/llama-model.cpp:2109`](src/llama-model.cpp:2109) passes 0 to constructor. |
| No override back to non-zero | ✅ PASS | Only assignments after override: [`src/llama-context.cpp:265`](src/llama-context.cpp:265) passes through, [`src/llama-context.cpp:269`](src/llama-context.cpp:269) only resets to 0 if arch doesn't support RS, [`common/speculative.cpp:2286`](common/speculative.cpp:2286) sets draft context `n_rs_seq=0`. |

#### Impact of Issue 1

With `n_backup_cells = 0`:
- Allocation: `n_rows = mem_size * 1 + 0 = mem_size` — **no backup rows allocated**
- `dflash_custom_backup()` at [`server-dflash-custom.cpp:254`](common/server-dflash-custom.cpp:254) returns early: `mem->n_backup_cells < n_cells` is true
- `dflash_custom_replay()` at [`server-dflash-custom.cpp:345`](common/server-dflash-custom.cpp:345) returns `false`: `n_cells == 0`
- Result: **Replay always fails, server always uses checkpoint fallback** (correct but slower)

---

### AUDIT 1B: Subtask 2 → Subtask 3 → Subtask 4 Handoff

**Objective:** Verify device placement, backup_offset consistency, tape tensor shapes, and layer ID mapping across backup cells, GPU tape capture, and replay.

#### 1. Backup Offset Consistency

| Item | Finding |
|------|---------|
| [`backup_offset()`](src/llama-memory-recurrent.h:93-95) | Returns `mem_size * (1 + n_rs_seq)`. With `n_rs_seq=0`: `mem_size * 1`. |
| R/S allocation ([`llama-memory-recurrent.cpp:104`](src/llama-memory-recurrent.cpp:104)) | `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells`. With `n_rs_seq=0`: `n_rows = mem_size + n_backup_cells`. |
| Row layout | `[0, mem_size-1]` = active. `[mem_size, mem_size + n_backup_cells - 1]` = backup. |
| [`dflash_custom_backup()`](common/server-dflash-custom.cpp:253-263) | Copies row `i` → `backup_offset() + i` = `mem_size + i`. **Correct.** |
| [`dflash_custom_restore()`](common/server-dflash-custom.cpp:271-281) | Copies row `backup_offset() + i` → row `i`. **Correct.** |

**Row arithmetic** (with `n_rs_seq=0`, `mem_size=4096`, `n_backup_cells=4`):

| Range | Purpose |
|-------|---------|
| `[0, 4095]` | Active cells |
| `[4096, 4099]` | Backup cells |
| `n_rows = 4100` | Tensor height |

**Verdict: ✅ CORRECT** — No off-by-one errors. Row indices match allocation formula exactly.

#### 2. Tape Tensor Shape Compatibility

**Capture side** ([`qwen35.cpp:460-527`](src/models/qwen35.cpp:460)):

| Tensor | Source Shape | Tape Tensor Shape | Capture Dest View | Match? |
|--------|-------------|-------------------|-------------------|--------|
| k | `[S_k, H_k, n_seq_tokens]` (view of `k_conv`) | `[S_k, H_k, max_tokens]` | `[S_k, H_k, n_seq_tokens]` (view of tape) | ✅ |
| v | `[S_v, H_v, n_seq_tokens]` (view of `v_conv`) | `[S_v, H_v, max_tokens]` | `[S_v, H_v, n_seq_tokens]` | ✅ |
| gate | `[1, H_v, n_seq_tokens]` (view of `gate`) | `[1, H_v, max_tokens]` | `[1, H_v, n_seq_tokens]` | ✅ |
| beta | `[1, H_v, n_seq_tokens]` (view of `beta`) | `[1, H_v, max_tokens]` | `[1, H_v, n_seq_tokens]` | ✅ |
| qkv | `[conv_channels, n_seq_tokens]` (view of `qkv_mixed`) | `[conv_channels, max_tokens]` | `[conv_channels, n_seq_tokens]` | ✅ |

**Replay side** ([`server-dflash-custom.cpp:424-462`](common/server-dflash-custom.cpp:424)):

| Tape Tensor | Reshape | View | nb Values | Correct? |
|-------------|-----------|------|-----------|----------|
| k `[S_k, H_k, max_tokens]` | `[S_k, H_k, max_tokens, 1]` | `[S_k, H_k, n_accepted, 1]` | `nb[0]=row_size(S_k)`, `nb[1]=row_size(S_k*H_k)`, `nb[2]=row_size(S_k*H_k)` | ✅ |
| v `[S_v, H_v, max_tokens]` | `[S_v, H_v, max_tokens, 1]` | `[S_v, H_v, n_accepted, 1]` | `nb[0]=row_size(S_v)`, `nb[1]=row_size(S_v*H_v)`, `nb[2]=row_size(S_v*H_v)` | ✅ |
| gate `[1, H_v, max_tokens]` | `[1, H_v, max_tokens, 1]` | `[1, H_v, n_accepted, 1]` | `nb[0]=row_size(1)`, `nb[1]=row_size(H_v)`, `nb[2]=row_size(H_v)` | ✅ |
| beta `[1, H_v, max_tokens]` | `[1, H_v, max_tokens, 1]` | `[1, H_v, n_accepted, 1]` | Same as gate | ✅ |

**Stride analysis:** For F32 tape tensors created with `ggml_new_tensor_3d()`, layout is column-major (ggml default). The nb values in replay views use `ggml_row_size()` which computes correct byte strides. Key: `nb[2]` (stride between token columns in 4D view) equals `nb[1]` (stride between heads in 3D tape) — correct because 3D tape has no padding between token columns.

The `ggml_cont()` calls in capture ([`qwen35.cpp:499-503`](src/models/qwen35.cpp:499)) ensure source data is contiguous before `ggml_cpy` to tape.

**Verdict: ✅ CORRECT**

#### 3. Device Placement

**Tape allocation** ([`server-dflash-custom.cpp:106-159`](common/server-dflash-custom.cpp:106)):
```cpp
ggml_backend_dev_t dev = model_dev_layer(model, (int)il);  // Line 111
tl.dev = dev;
ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);  // Line 132
tl.buf = ggml_backend_alloc_ctx_tensors_from_buft(tl.ctx, buft);      // Line 147
```

**R/S tensor allocation** ([`llama-memory-recurrent.cpp:88-93`](src/llama-memory-recurrent.cpp:88)):
```cpp
auto * dev = model.dev_layer(i);  // Line 89
buft = ggml_backend_dev_buffer_type(dev);  // Line 90
```

For layer `il`, tape tensors and R/S tensors share `model.dev_layer(il)`. Same-device copies for `cell_copy()` and replay state write-back.

**Verdict: ✅ CORRECT**

#### 4. Layer ID Mapping

**Construction** ([`server-dflash-custom.cpp:94-101`](common/server-dflash-custom.cpp:94)):
```cpp
for (uint32_t il = 0; il < n_layers; ++il) {
    if (model_is_recr(model, (int)il)) {
        tape->layer_ids.push_back(il);
        ++tape_idx;
    }
}
```

**Capture lookup** ([`qwen35.cpp:464-470`](src/models/qwen35.cpp:464)):
```cpp
for (int i = 0; i < (int)tgpu->layer_ids.size(); ++i) {
    if (tgpu->layer_ids[i] == (uint32_t)il) {
        li = i;
        break;
    }
}
```

**Replay iteration** ([`server-dflash-custom.cpp:403-410`](common/server-dflash-custom.cpp:403)):
```cpp
for (size_t ti = 0; ti < tape_layers.size(); ++ti) {
    const auto & tl = tape_layers[ti];
    int il = (int)layer_ids[ti];
    if (!mem->s_l[il]) continue;
}
```

Both use the same `layer_ids` array. Capture: model layer `il` → tape index `li` (search). Replay: tape index `ti` → model layer `il` (direct).

**Verdict: ✅ CORRECT**

---

### AUDIT 2: Runtime Behavior, Opt-in, and Fallback Safety

**Objective:** Verify the complete runtime lifecycle: initialization → backup → draft/capture → verification → replay OR checkpoint rollback.

#### 1. Strict Opt-in

| Check | File:Lines | Verdict |
|-------|-----------|---------|
| `dflash_custom_init()` only with flag | [`server-context.cpp:1459-1476`](tools/server/server-context.cpp:1459) | ✅ Gated on `beefix_dflash_custom && spec` |
| `tape_gpu` gating in `qwen35.cpp` | [`qwen35.cpp:460`](src/models/qwen35.cpp:460) | ✅ `cparams.tape_gpu != nullptr` only set by `dflash_custom_set_tape_gpu()` |
| `dflash_custom_is_enabled()` triple check | [`server-dflash-custom.h:152-154`](common/server-dflash-custom.h:152) | ✅ `state != nullptr && state->enabled && state->tape != nullptr` |

Flow chain:
1. `--beefix-dflash-custom` → creates `dflash_custom` state at line 1462
2. `dflash_custom_is_enabled()` → true only if state exists and enabled and tape allocated
3. `dflash_custom_set_tape_gpu(ctx_tgt, tape)` → sets `cparams.tape_gpu` to non-null
4. `qwen35.cpp:460` → capture executes only when `cparams.tape_gpu != nullptr`
5. After draft: `tape_gpu` reset to `nullptr` at line 3349

**Verdict: ✅ CORRECT** — Zero Task 6R code execution without the flag.

#### 2. Stock DFlash Unchanged

| Check | File:Lines | Verdict |
|-------|-----------|---------|
| `n_rs_seq=0` only with flag + DFlash | [`common.cpp:1775-1781`](common/common.cpp:1775) | ✅ Requires both `beefix_dflash_custom` AND DFlash in types list |
| No Task 6R allocations without flag | Multiple files | ✅ `n_backup_cells` defaults to 0, no tape allocated, no capture |

Without flag: `n_rs_seq` remains at `need_n_rs_seq()` which returns `draft.n_max` for DFlash ([`common/common.h:423-429`](common/common.h:423)).

**Verdict: ✅ CORRECT**

#### 3. Replay Lifecycle

| Phase | File:Lines | Verdict |
|-------|-----------|---------|
| Pre-draft backup | [`server-context.cpp:3293-3326`](tools/server/server-context.cpp:3293) | ✅ Before `common_speculative_draft()` at line 3331 |
| Tape activation | [`server-context.cpp:3323-3325`](tools/server/server-context.cpp:3323) | ✅ After backup, before draft |
| Tape reset | [`server-context.cpp:3349`](tools/server/server-context.cpp:3349) | ✅ Immediately after draft, `tokens_captured` updated |
| Post-verify replay | [`server-context.cpp:4296-4327`](tools/server/server-context.cpp:4296) | ✅ Inside `if (n_rollback > 0) { if (use_ckpt_tgt) {` |
| Fallback to checkpoint | [`server-context.cpp:4329-4367`](tools/server/server-context.cpp:4329) | ✅ `ckpt.load_tgt()` → `seq_rm()` → restore sampler |

**Verdict: ✅ CORRECT**

#### 4. Consecutive Failure and Permanent Disable

| Check | File:Lines | Verdict |
|-------|-----------|---------|
| Failure counter (return false) | [`server-context.cpp:4313`](tools/server/server-context.cpp:4313) | ✅ `fail_count++`, disable at `>= 3` |
| Failure counter (exception) | [`server-context.cpp:4321`](tools/server/server-context.cpp:4321) | ✅ Same pattern in catch block |
| Success resets counter | [`server-context.cpp:4308`](tools/server/server-context.cpp:4308) | ✅ `fail_count = 0` |
| Skip after `replay_failed` | [`server-context.cpp:4303`](tools/server/server-context.cpp:4303) | ✅ `!slot.dflash_custom->replay_failed` in condition |

**Verdict: ✅ CORRECT**

#### 5. Checkpoint Rollback Availability

| Check | File:Lines | Verdict |
|-------|-----------|---------|
| Checkpoint before speculation | [`server-context.cpp:3358-3394`](tools/server/server-context.cpp:3358) | ✅ Created after draft, before verification |
| Replay nested in `use_ckpt_tgt` | [`server-context.cpp:4296-4297`](tools/server/server-context.cpp:4296) | ✅ Only attempted when checkpoint exists |

**Verdict: ✅ CORRECT**

#### 6. Edge Cases

| Edge Case | File:Lines | Verdict |
|-----------|-----------|---------|
| `n_accepted == 0` (zero acceptance) | [`server-dflash-custom.cpp:309`](common/server-dflash-custom.cpp:309) | ⚠️ **CORRECT but suboptimal** — returns `false`, increments `fail_count`. After 3 zero-acceptance cycles, replay disabled even though it would work for partial acceptance. Not a correctness issue (fallback to checkpoint is safe). |
| `n_accepted == n_draft` (full acceptance) | [`server-context.cpp:4296`](tools/server/server-context.cpp:4296) | ✅ `n_rollback == 0`, skips replay entirely |
| First cycle (no tape data) | [`server-dflash-custom.cpp:317`](common/server-dflash-custom.cpp:317) | ✅ `tokens_captured` updated after draft before replay attempted |

---

### AUDIT 3: Generality and Model Independence

**Objective:** Determine whether Task 6R is a reusable DFlash framework capability or accidentally tied to Qwen3.6-27B.

#### Area-by-Area Assessment

| Area | Status | Details |
|------|--------|---------|
| 1. Capture in [`qwen35.cpp:460-529`](src/models/qwen35.cpp:460) | Model-Specific | **Architecturally necessary** — capture point MUST be in model's graph builder. Operates on generic GDN intermediates (k_conv, v_conv, gate, beta, qkv_mixed). New models add capture in their own builder. |
| 2. Dimension extraction [`server-dflash-custom.cpp:49-70`](common/server-dflash-custom.cpp:49) | Generic | All from runtime hparams: `hp.ssm_d_state`, `hp.ssm_dt_rank`, `hp.ssm_n_group`, `hp.ssm_d_inner`. **Accidental coupling:** `conv_channels = d_inner + 2 * H_k * S_k` at line 65 matches Qwen3.6's projection layout but may not hold for other SSM architectures. |
| 3. Recurrent layer detection [`server-dflash-custom.cpp:74-85`](common/server-dflash-custom.cpp:74) | Generic | Uses `hp.is_recr(il)` — generic hparams method. Any model setting `is_recr_impl[il] = true` will be detected. |
| 4. GDN kernel [`server-dflash-custom.cpp:482-483`](common/server-dflash-custom.cpp:482) | Generic | Calls upstream `ggml_gated_delta_net()` with correct API. Used by ALL recurrent models with GDN. |
| 5. Backup cells [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h:82) | Generic | Operates on R/S tensors generically. No shape assumptions. |
| 6. Server integration [`server-context.cpp:207`](tools/server/server-context.cpp:207) | Generic | No Qwen-specific loading, validation, or configuration. |
| 7. Test script [`tests/dflash-custom-test.py:40-43`](tests/dflash-custom-test.py:40) | Generic | Model paths are configurable defaults. Test logic works with any DFlash model. |

**Verdict: ✅ Predominantly generic (6/7 areas). One architecturally necessary model-specific integration (capture). One accidental coupling (`conv_channels` formula, LOW risk).**

---

### AUDIT 4: Testing and Implementation Completeness

**Objective:** Inspect test coverage and verify all blueprint Part C items are implemented.

#### Test Script Coverage

| Test ID | Blueprint Objective | Script Coverage | Notes |
|--------|---|---|---|
| T1 | No DFlash baseline | ❌ NOT COVERED | Script starts with stock DFlash (T2) |
| T2 | Stock DFlash baseline | ✅ COVERED | `test_stock_dflash()` — n_rs_seq=8, VRAM, generation |
| T3 | Custom mode basic | ✅ COVERED | `test_custom_dflash()` — n_rs_seq=0, VRAM savings, output match |
| T4 | K=0 acceptance | ❌ NOT COVERED | Cannot force zero-acceptance pattern |
| T5 | K=1 acceptance | ❌ NOT COVERED | Cannot force specific acceptance count |
| T6 | K=n-1 acceptance | ❌ NOT COVERED | Same as above |
| T7 | Full acceptance | ⚠️ PARTIAL | Crash resilience test exercises multiple requests but cannot force full acceptance |
| T8 | n_draft_max=4 | ❌ NOT COVERED | Fixed `DRAFT_N_MAX=8` |
| T9 | n_draft_max=15 | ❌ NOT COVERED | Same |
| T10 | n_ctx=2048 | ❌ NOT COVERED | Fixed `CONTEXT_SIZE=8192` |
| T11 | n_ctx=32768 | ❌ NOT COVERED | Same |
| T12 | Force F1 (tape alloc fail) | ❌ NOT COVERED | No failure injection |
| T13 | Force F3 (graph error) | ❌ NOT COVERED | No failure injection |
| T14 | Force F4 (mismatch) | ❌ NOT COVERED | No failure injection |
| T15 | VRAM measurement | ✅ COVERED | Stock and custom VRAM via `nvidia-smi` |
| T16 | Performance benchmark | ❌ NOT COVERED | No `llama-bench` invocation |

**Coverage: 3/16 fully covered (T2, T3, T15). 1 partial (T7). 12 not covered.**

#### Critical Test Gap

**Replay correctness not validated** — The test script cannot distinguish "replay working correctly" from "replay failing every time with checkpoint fallback." Both produce correct output. The output comparison (T3-MATCH) is an indirect check but cannot distinguish replay success from checkpoint fallback.

**Risk:** If replay always returns `false` (as it currently does due to Issue 1), the test passes because checkpoint rollback produces correct output. VRAM savings appear (tape allocated) but replay never fires.

#### Implementation Completeness

**Functions (9/9 implemented):**

| Function | Blueprint Signature | Implemented | Notes |
|----------|---|---|-------|
| `dflash_custom_backup()` | `void(..., llama_memory_recurrent * mem, ...)` | ✅ [`server-dflash-custom.cpp:253`](common/server-dflash-custom.cpp:253) | Const-correct improvement |
| `dflash_custom_restore()` | `void(..., llama_memory_recurrent * mem, ...)` | ✅ [`server-dflash-custom.cpp:271-281`](common/server-dflash-custom.cpp:271) | Const-correct improvement |
| `dflash_custom_tape_alloc()` | `...(llama_model *, ggml_backend_t, int)` | ✅ [`server-dflash-custom.cpp:49-165`](common/server-dflash-custom.cpp:49) | **Improved:** no `ggml_backend_t` param, device-aware placement |
| `dflash_custom_tape_free()` | `void(server_dflash_tape_gpu *)` | ✅ [`server-dflash-custom.cpp:167-182`](common/server-dflash-custom.cpp:167) | Exact match |
| `dflash_custom_replay()` | `bool(state, ctx, n_accepted)` | ✅ [`server-dflash-custom.cpp:307-540`](common/server-dflash-custom.cpp:307) | Exact match |
| `dflash_custom_init()` | `state *(model, n_draft_max, n_parallel)` | ✅ [`server-dflash-custom.cpp:188-215`](common/server-dflash-custom.cpp:188) | **Improved:** no `n_parallel` param (per-slot allocation) |
| `dflash_custom_free()` | `void(state *)` | ✅ [`server-dflash-custom.cpp:217-236`](common/server-dflash-custom.cpp:217) | Exact match |
| `dflash_custom_is_enabled()` | `bool(const common_params_speculative &)` | ✅ [`server-dflash-custom.h:152-154`](common/server-dflash-custom.h:152) | **Improved:** takes `const server_dflash_custom_state *`, inline |

**Data Structures (4/4 implemented):**

| Structure | Blueprint | Implemented | Notes |
|---|---|---|---|
| `server_dflash_tape_gpu_layer` | 5 tensors + buf/ctx/dev | ✅ [`server-dflash-custom.h:44-53`](common/server-dflash-custom.h:44) | Exact match |
| `server_dflash_tape_gpu` | layers, layer_ids, max_tokens, n_tokens | ✅ [`server-dflash-custom.h:62-67`](common/server-dflash-custom.h:62) | Exact match |
| `server_dflash_custom_state` | tape, metadata, replay state, graph context | ✅ [`server-dflash-custom.h:75-97`](common/server-dflash-custom.h:75) | Exact match |
| `llama_memory_recurrent` extensions | `n_backup_cells`, `backup_offset()`, `cell_copy()` | ✅ [`llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | All three added |

**Server Integration Points (all wired):**

| Integration Point | Blueprint | Implemented |
|---|---|---|
| `server_slot::dflash_custom` member | ✅ | ✅ [`server-context.cpp:207`](tools/server/server-context.cpp:207) |
| Pre-draft backup before `common_speculative_draft()` | ✅ | ✅ [`server-context.cpp:3323-3324`](tools/server/server-context.cpp:3323) |
| Post-verify replay with try-catch fallback | ✅ | ✅ Verified in `server-context.cpp` |
| Tape activation/reset | ✅ | ✅ [`server-context.cpp:3324`](tools/server/server-context.cpp:3324) / [`server-context.cpp:3349`](tools/server/server-context.cpp:3349) |
| `tokens_captured` update after draft | Implicit | ✅ [`server-context.cpp:3352`](tools/server/server-context.cpp:3352) |

**Additional elements added during implementation (not in original blueprint but documented in implementation-summary.md):**
- `dflash_custom_set_tape_gpu()` helper
- `set_tape_gpu()` method on `llama_context`
- `LLAMA_API` exports on 8 internal methods for DLL linking
- CMakeLists.txt update for server library

---

## Recommended Priority Actions

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **P0** | Fix Issue 1: Add `cparams.n_backup_cells = params.n_parallel;` in [`common/common.cpp:1775-1781`](common/common.cpp:1775) | ~3 lines | Enables replay to actually function |
| **P1** | Add replay-path logging in [`server-context.cpp:4303-4327`](tools/server/server-context.cpp:4303) so test script can verify replay execution | ~5 lines | Validates replay actually fires |
| **P2** | Document `conv_channels` formula at [`server-dflash-custom.cpp:65`](common/server-dflash-custom.cpp:65) as Qwen-specific | ~2 lines comment | Future-proof for non-Qwen SSM models |

---

*End of audit findings document.*
