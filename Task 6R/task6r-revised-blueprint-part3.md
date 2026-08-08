# DFlash Custom Mode — Revised Implementation Blueprint (Part 3 of 3)

**Date:** 2026-08-08
**Continuation of:** `task6r-revised-implementation-blueprint.md`, `task6r-revised-blueprint-part2.md`
**Based on:** Task 6R.1-6R.4 findings

---

## PART E: Implementation Subtasks

### E.1 Dependency Graph

```
Subtask 1: Opt-in flag + n_rs_seq override
     ↓ (provides n_rs_seq=0 for DFlash)
Subtask 2: Backup cell infrastructure
     ↓ (provides cell_copy() and extended tensors)
Subtask 3: GPU tape allocation + capture mechanism
     ↓ (provides rank-factored intermediates during draft)
Subtask 4: Replay orchestration
     ↓ (uses backup + tape to replay GDN)
Subtask 5: Server integration + fallback
     ↓ (wires everything into post_decode())
Subtask 6: Testing + benchmarking
```

### E.2 Subtask 1: Opt-in Flag and n_rs_seq Override

**Objective:** Enable `--beefix-dflash-custom` flag that sets `n_rs_seq=0` for DFlash, eliminating the 5.4 GB RS snapshot buffer.

**Files Involved:**
- [`common/common.h`](common/common.h) — add `beefix_dflash_custom` bool to `common_params_speculative`
- [`common/arg.cpp`](common/arg.cpp) — add CLI argument parser
- [`common/common.cpp`](common/common.cpp) — override `n_rs_seq` at [`common_context_params_to_llama()`](common/common.cpp:1770)

**Functions/Symbols:**
- `common_params_speculative::beefix_dflash_custom` (new member)
- `common_context_params_to_llama()` — modified to check flag and override

**Prerequisites:** None

**Relevant Research Documents:**
- [`task6-part1-verification.md`](plans/dflash-solutions/task6-part1-verification.md) — verified `need_n_rs_seq()` at [`common/common.h:417`](common/common.h:417) and override point at [`common/common.cpp:1770`](common/common.cpp:1770)

**Expected Behavior:**
- Server starts with `--spec-type draft-dflash --beefix-dflash-custom`
- Log shows `n_rs_seq = 0` instead of `n_rs_seq = 8`
- RS buffer drops from 5,387 MB to 599 MB
- DFlash still functions (checkpoint-only rollback)

**Testing Method:**
1. Start server with and without `--beefix-dflash-custom`.
2. Verify log output shows correct `n_rs_seq` value.
3. Measure VRAM with `nvidia-smi` — should drop by ~4.8 GB.
4. Generate text — verify output is correct (checkpoint rollback works).

**Est. Lines:** ~20

---

### E.3 Subtask 2: Backup Cell Infrastructure

**Objective:** Extend recurrent memory tensors with backup cell rows and implement `cell_copy()` for device-native row-to-row copies.

**Files:**
- [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) — add `n_backup_cells`, `backup_offset()`, `cell_copy()` declaration
- [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) — extended allocation, `cell_copy()` implementation
- [`src/llama-model.cpp`](src/llama-model.cpp) — pass backup cell count to constructor

**Functions/Symbols:**
- `llama_memory_recurrent::n_backup_cells` (new member)
- `llama_memory_recurrent::backup_offset()` (new method)
- `llama_memory_recurrent::cell_copy(uint32_t src, uint32_t dst)` (new method)
- `llama_memory_recurrent` constructor — modified allocation formula

**Prerequisites:** Subtask 1 complete

**Relevant Research Documents:**
- [`task6-part2-verification.md`](plans/dflash-solutions/task6-part2-verification.md) — verified R/S state tensors, cell indexing, `s_copy()` behavior
- [`task4-part2-allocation-and-rollback.md`](plans/dflash-solutions/task4-part2-allocation-and-rollback.md) — backup cell design

**Expected Behavior:**
- R/S tensors have `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` rows.
- `cell_copy()` copies R and S data between any two rows using tensor views + `ggml_backend_tensor_copy()`.
- Backup rows are not used by normal allocation (`find_slot()` skips backup range).

**Testing Method:**
1. Unit test: `cell_copy(0, backup_offset())` followed by memcmp of source and destination.
2. Verify backup rows are not allocated during normal operation.
3. Verify VRAM increases by `n_backup_cells × ~149.9 MB`.

**Est. Lines:** ~80

---

### E.4 Subtask 3: GPU Tape Allocation and Capture Mechanism

**Objective:** Allocate GPU tape tensors with device-aware placement and add graph-embedded `ggml_cpy` operations to capture rank-factored GDN intermediates during draft forward pass.

**Files:**
- [`common/server-dflash-custom.h`](common/server-dflash-custom.h) — NEW: tape structs, function declarations
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) — NEW: tape allocation, metadata
- [`src/models/qwen35.cpp`](src/models/qwen35.cpp) — graph-embedded capture after tensor computation

**Functions/Symbols:**
- `server_dflash_tape_gpu_layer` (new struct)
- `server_dflash_tape_gpu` (new struct)
- `dflash_custom_tape_alloc()` (new function) — device-aware tape tensor allocation
- `dflash_custom_tape_free()` (new function) — cleanup
- `build_layer_attn_linear()` in `qwen35.cpp` — modified to add `ggml_cpy` after k_conv, v_conv, gate, beta, qkv_mixed computation

**Prerequisites:** Subtasks 1-2 complete

**Relevant Research Documents:**
- [`task6r-part2-recurrent-tape-mechanics.md`](plans/dflash-solutions/task6r-part2-recurrent-tape-mechanics.md) — tape dimensions, old `allocate_tape_gpu()` implementation
- [`task6r-part4-discrepancy-and-current-mapping.md`](plans/dflash-solutions/task6r-part4-discrepancy-and-current-mapping.md) — current upstream tensor names, graph builder comparison
- [`old-versions/.../qwen35.cpp:564-579`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:564) — old graph-embedded capture code (reference)

**Expected Behavior:**
1. GPU tape tensors allocated at server init with device-aware placement.
2. During draft forward pass, `ggml_cpy` operations copy k, v, gate, beta, qkv_mixed to tape tensors.
3. Tape data persists after graph execution (tensors are pre-allocated, not freed).
4. Tape tensors are on the same GPU as the corresponding model layer.

**Key Implementation Detail — Capture Point in `qwen35.cpp`:**

The capture operations should be inserted at [`src/models/qwen35.cpp:446-450`](src/models/qwen35.cpp:446), immediately after the `cb()` calls that name the tensors:

```cpp
// Existing code (lines 446-450):
cb(q_conv, "q_conv_predelta", il);
cb(k_conv, "k_conv_predelta", il);
cb(v_conv, "v_conv_predelta", il);
// ... gate and beta named earlier ...
ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

// NEW: tape capture (conditional on custom mode):
if (cparams.tape_gpu != nullptr) {
    auto & tl = cparams.tape_gpu->layers[tape_layer_index(il)];
    // Copy k_conv, v_conv, gate, beta, qkv_mixed to tape tensors
    // using ggml_cpy with appropriate views for the current token range
}
```

**Tape Name Map:**

| Old Name | Current Name | Tensor |
|----------|-------------|--------|
| `k_conv_predelta-{il}` | `k_conv_predelta-{il}` | k after l2_norm |
| `v_conv_predelta-{il}` | `v_conv_predelta-{il}` | v after conv/silu |
| `gate-{il}` | `gate-{il}` | gate (pre-exp) |
| `beta_sigmoid-{il}` | `beta_sigmoid-{il}` | beta after sigmoid |
| `qkv_mixed_pretranspose-{il}` | `linear_attn_qkv_mixed-{il}` | raw QKV (name changed) |

**Testing Method:**
1. Verify tape tensors are allocated on correct GPU devices.
2. After draft forward pass, verify tape tensors contain non-zero data.
3. Verify tape data matches expected dimensions (S_k, H_k, S_v, H_v, conv_channels).
4. Measure capture overhead — should be ~0 ms (overlaps with compute).

**Est. Lines:** ~120 (80 in new files + 35 in qwen35.cpp + 5 in common.h)

---

### E.5 Subtask 4: Replay Orchestration

**Objective:** Implement GDN replay for accepted tokens using backup state + captured tape data.

**Files:**
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) — `dflash_custom_replay()`, `dflash_custom_backup()`, `dflash_custom_restore()`

**Functions/Symbols:**
- `dflash_custom_backup()` — copy active R/S rows to backup rows
- `dflash_custom_restore()` — copy backup rows back to active rows
- `dflash_custom_replay()` — build and execute GDN replay graph

**Prerequisites:** Subtasks 2-3 complete

**Relevant Research Documents:**
- [`task6r-part3-tape-replay-and-rollback.md`](plans/dflash-solutions/task6r-part3-tape-replay-and-rollback.md) — replay mechanics, GDN math
- [`task5-part6-ggml-and-integration.md`](plans/dflash-solutions/task5-part6-ggml-and-integration.md) — ggml primitives for replay
- [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) — GDN kernel (verifies q-independence)

**Expected Behavior:**
1. `dflash_custom_backup()` copies active R/S state to backup cells before draft.
2. After verification, `dflash_custom_replay()`:
   a. Restores backup state to active cells.
   b. Builds GDN replay graph: for each recurrent layer, creates q_zeros, tape views, state view, and GDN node.
   c. Executes replay graph on GPU.
   d. Writes updated state to active R/S tensors.
3. Replay produces the same S-state as the original forward pass for accepted tokens.

**Replay Graph Construction:**

```cpp
for (each recurrent layer il):
    // Create zero q tensor (state update is q-independent)
    q_zeros = ggml_new_tensor_3d(ctx, F32, S_k, H_k, n_accepted)
    ggml_set_f32(q_zeros, 0.0f)

    // Create views into tape tensors for accepted tokens
    k_view = ggml_view_3d(tape_k[layer], S_k, H_k, n_accepted, ..., token_offset=0)
    v_view = ggml_view_3d(tape_v[layer], S_v, H_v, n_accepted, ..., token_offset=0)
    g_view = ggml_view_3d(tape_g[layer], 1, H_v, n_accepted, ..., token_offset=0)
    b_view = ggml_view_3d(tape_b[layer], 1, H_v, n_accepted, ..., token_offset=0)

    // View into backup state
    s_backup = ggml_view_3d(s_l[il], S_v, S_v, H_v, ..., row=backup_row)

    // GDN replay — produces updated state
    output = ggml_gated_delta_net(ctx, q_zeros, k_view, v_view, g_view, b_view, s_backup, K=n_accepted)
    ggml_build_forward_expand(replay_graph, output)
```

**Testing Method:**
1. Compare replayed S-state vs. original forward pass S-state (element-wise F32 comparison).
2. Verify numerical accuracy within F32 precision (max difference < 1e-5).
3. Measure replay time — should be ~0.1 ms for K=8 tokens.
4. Test with K=0 (no replay), K=1 (single token), K=n_draft (full replay).

**Est. Lines:** ~150

---

### E.6 Subtask 5: Server Integration and Fallback

**Objective:** Wire custom mode into server `post_decode()` lifecycle with fallback to checkpoint rollback.

**Files:**
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) — replay integration in `post_decode()`, pre-draft backup
- [`tools/server/server-context.h`](tools/server/server-context.h) — add custom mode state to `server_slot`
- [`tools/server/server-task.h`](tools/server/server-task.h) — optional constants

**Functions/Symbols:**
- `server_slot::dflash_custom` (new member — `server_dflash_custom_state`)
- `post_decode()` — modified at [`server-context.cpp:4263-4267`](tools/server/server-context.cpp:4263) to insert replay attempt
- Pre-draft backup call before [`common_speculative_draft()`](tools/server/server-context.cpp:3264)

**Prerequisites:** Subtasks 2-4 complete

**Relevant Research Documents:**
- [`task6-part3-verification.md`](plans/dflash-solutions/task6-part3-verification.md) — verification lifecycle at [`server-context.cpp:4213-4274`](tools/server/server-context.cpp:4213)
- [`task6-part4-verification.md`](plans/dflash-solutions/task6-part4-verification.md) — fallback conditions

**Expected Behavior:**
1. Pre-draft: `dflash_custom_backup()` called before `common_speculative_draft()`.
2. Post-verify: If partial acceptance, attempt `dflash_custom_replay()`.
3. If replay succeeds: skip checkpoint path, proceed to KV cleanup.
4. If replay fails: fall back to checkpoint rollback (`ckpt.load_tgt()` → `seq_rm()` → restore sampler).
5. After 3 consecutive failures: permanently disable replay for this slot.

**Integration Code (pseudocode at `server-context.cpp:4263-4267`):**

```cpp
// Between existing checkpoint path and RS path:
if (slot.dflash_custom.enabled && n_rollback > 0 && !slot.dflash_custom.replay_failed) {
    try {
        dflash_custom_replay(&slot.dflash_custom, ctx, n_accepted);
        // Replay succeeded — skip checkpoint path
        use_ckpt_tgt = false;
    } catch (const std::exception & e) {
            SLT_WRN("dflash custom replay failed: %s — falling back to checkpoint\n", e.what());
            slot.dflash_custom.fail_count++;
            if (slot.dflash_custom.fail_count >= 3) {
                slot.dflash_custom.replay_failed = true;
                SLT_WRN("dflash custom replay permanently disabled after 3 consecutive failures\n");
            }
            use_ckpt_tgt = true; // fall through to checkpoint path
        }
    }
}
```

**Testing Method:**
1. End-to-end test: generate text with custom mode enabled.
2. Verify output matches stock DFlash (same prompt, seed, sampling).
3. Force fallback by inducing errors (e.g., corrupt tape data).
4. Verify server never crashes on replay failure.
5. Measure throughput vs. stock DFlash.

**Est. Lines:** ~80

---

### E.7 Subtask 6: Testing and Benchmarking

**Objective:** Comprehensive testing to verify correctness, VRAM savings, and performance.

**Prerequisites:** All subtasks complete

**Est. Lines:** ~50 (test scripts)

(Detailed test plan in Part F below.)

---

## PART F: Fallback Behavior

### F.1 Fallback Conditions

All failure conditions with detection point, fallback action, and scope:

| # | Failure Condition | Detection Point | Fallback Action | Scope |
|---|------------------|----------------|-----------------|-------|
| F1 | GPU tape allocation fails (backend returns nullptr) | Server init | Disable custom mode, use checkpoint rollback | Permanent |
| F2 | Backup cell allocation fails (tensor too large for GPU) | `llama_memory_recurrent` constructor | Constructor throws; server catches, reduces `n_parallel` or disables | Permanent (startup) |
| F3 | Graph construction assertion during replay (dimension mismatch, non-contiguous inputs) | Pre-replay graph build | Catch exception, fall back to checkpoint | Permanent (dimension mismatch) or Transient (context exhaustion) |
| F4 | Mismatch between captured tape data and backup state (tokens_captured < n_accepted) | Pre-replay validation | Fall back to checkpoint, permanently disable replay | Permanent (capture bug) |
| F5 | GDN kernel execution failure (CUDA error) | During replay graph execution | Catch exception, fall back to checkpoint | Permanent |
| F6 | Device-aware placement fails (backend not found for layer device) | Tape allocation | Fall back to single-device tape or disable | Permanent |
| F7 | `cell_copy()` fails (backend copy error) | During backup or restore | Fall back to checkpoint | Per-cycle (transient) or Permanent |

### F.2 Fallback Implementation Pattern

Every custom mode operation is wrapped in try-catch that falls through to the existing checkpoint rollback path:

```cpp
// In server-context.cpp post_decode():
if (slot.dflash_custom.enabled && n_rollback > 0 && !slot.dflash_custom.replay_failed) {
    try {
        dflash_custom_replay(&slot.dflash_custom, ctx, n_accepted);
        use_ckpt_tgt = false; // replay succeeded
    } catch (const std::exception & e) {
        SLT_WRN("dflash custom replay failed: %s — falling back to checkpoint\n", e.what());
        slot.dflash_custom.fail_count++;

        if (slot.dflash_custom.fail_count >= 3) {
            slot.dflash_custom.replay_failed = true;
            SLT_WRN("dflash custom replay permanently disabled after 3 consecutive failures\n");
        }

        use_ckpt_tgt = true; // fall through to checkpoint path
    }
}
```

### F.3 Permanent vs Temporary Disable

| Condition | Type | Recovery |
|-----------|------|----------|
| `fail_count >= 3` consecutive failures | Permanent (for this slot) | Restart server |
| Structural failure (dimension mismatch, dtype error) | Permanent | Fix code, restart |
| Tape allocation failed at init | Permanent | Reduce `n_draft_max` or disable flag |
| Context exhaustion during graph build | Transient | Retry with larger context next cycle |
| Single transient GPU error | Transient | Retry next cycle |

### F.4 Safe Fallback Guarantee

The custom mode path MUST fail safely to checkpoint rollback. The checkpoint is always created before every speculation cycle (existing upstream behavior), so checkpoint rollback is always available as a fallback. The custom mode path can NEVER prevent the server from completing the speculation cycle — it can only improve rollback performance.

---

## PART G: Testing Plan

### G.1 Test Matrix

| Test ID | Configuration | Objective | Pass Criteria |
|---------|--------------|-----------|---------------|
| T1 | No DFlash | Baseline generation | Correct output, no errors |
| T2 | Stock DFlash (no `--beefix-dflash-custom`) | Stock DFlash unchanged | Correct output, ~6.2 GB overhead |
| T3 | Custom mode, basic | Custom mode starts and generates | Correct output, **~2.1 GB overhead** |
| T4 | Custom mode, K=0 acceptance | Zero acceptance handling | Correct output, no crash |
| T5 | Custom mode, K=1 acceptance | Single-token replay | Output matches T2 |
| T6 | Custom mode, K=n_draft-1 | Late partial acceptance | Output matches T2 |
| T7 | Custom mode, full acceptance | Full acceptance (no replay) | Output matches T2 |
| T8 | Custom mode, n_draft_max=4 | Small draft | Correct output |
| T9 | Custom mode, n_draft_max=15 | Large draft | Correct output |
| T10 | Custom mode, n_ctx=2048 | Small context | Correct output |
| T11 | Custom mode, n_ctx=32768 | Large context | Correct output |
| T12 | Force F1 (tape alloc fail) | Fallback to checkpoint | No crash, correct output |
| T13 | Force F3 (graph error) | Fallback to checkpoint | No crash, correct output |
| T14 | Force F4 (mismatch) | Permanent disable | No crash, correct output |
| T15 | VRAM measurement | Verify savings | ≥3 GB saved vs T2 |
| T16 | Performance benchmark | Verify speed | ≥80% of T2 throughput |

### G.2 Baseline Tests

#### G1: No DFlash

```bash
build/bin/llama-server -m qwen3.6-27b.gguf \
  --n_ctx 8192 --n_parallel 4 --port 8080
```

**Expected:** Standard generation, no speculative decoding. VRAM ~18.6 GB.

#### G2: Stock DFlash

```bash
build/bin/llama-server -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 \
  --n_ctx 8192 --n_parallel 4 --port 8080
```

**Expected:** `n_rs_seq = 8`, VRAM ~24.8 GB, correct output.

### G.3 Custom Mode Tests

#### G3: Basic Custom Mode

```bash
build/bin/llama-server -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 \
  --beefix-dflash-custom \
  --n_ctx 8192 --n_parallel 4 --port 8080
```

**Expected:**
- `n_rs_seq = 0`
- Backup cells allocated: `n_backup_cells = 4` **CORRECTED** (was 8, old v0.3.2 used n_parallel=4)
- GPU tape allocated: ~85 MB (fused) or ~117 MB (non-fused) **CORRECTED** (was ~134-186 MB)
- VRAM ~20.2 GB (model + ~2.1 GB DFlash overhead) **CORRECTED** (was ~21.5 GB)
- Generated tokens match G2 exactly (same prompt, seed, sampling)

#### G4-G7: Acceptance Pattern Tests

Use controlled test prompts that produce known acceptance patterns. Compare output with G2 baseline.

| Test | Acceptance Pattern | Expected Path |
|------|-------------------|---------------|
| G4 (K=0) | No tokens accepted | Backup restore + checkpoint fallback |
| G5 (K=1) | 1 token accepted | Backup restore + GDN replay for 1 token |
| G6 (K=n-1) | All but last accepted | Backup restore + GDN replay for n-1 tokens |
| G7 (K=n) | All accepted | No replay needed |

### G.4 Speculative Depth Tests

**CORRECTION (2026-08-08):** Tape sizes updated from S=1536 to S=128, H_v=48, conv_ch=10,240. Per-token fused tape = ~74 KB per layer (was ~116 KB).

| Test | n_draft_max | Expected Tape Size | Expected Behavior |
|------|------------|-------------------|-------------------|
| G8 | 4 | ~22 MB (fused) | Smaller tape, faster |
| G3 | 8 | ~44 MB (fused) | Default |
| G9 | 15 | ~82 MB (fused) | Larger tape |

### G.5 Context Size Tests

| Test | n_ctx | Expected Behavior |
|------|-------|-------------------|
| G10 | 2048 | Small context, fast |
| G3 | 8192 | Default context |
| G11 | 32768 | Large context (KV pressure) |

### G.6 Fallback Tests

| Test | Trigger Method | Expected Result |
|------|---------------|-----------------|
| G12 (F1) | Set `n_draft_max` extremely high (e.g., 1000) to exhaust GPU memory during tape alloc | Server logs warning, disables custom mode, uses checkpoint |
| G13 (F3) | Inject dimension mismatch in tape metadata | Catch exception, fall back to checkpoint |
| G14 (F4) | Set `tokens_captured` to 0 manually | Detect mismatch, permanently disable, use checkpoint |

### G.7 VRAM Measurement

```bash
# Before server starts:
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# After server starts (wait for model load):
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Calculate difference:
# Stock DFlash: ~24,800 MB
# Custom mode: ~20,200 MB
# Savings: ~4,600 MB
```

**Pass criteria:** Custom mode VRAM is at least 3.5 GB lower than stock DFlash. **CORRECTION (2026-08-08):** Updated from 2.5 GB to 3.5 GB to reflect corrected ~4 GB savings.

### G.8 Performance Benchmark

```bash
# Stock DFlash:
build/bin/llama-bench -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 -p 0 -n 256 -t 1

# Custom mode:
build/bin/llama-bench -m qwen3.6-27b.gguf \
  --spec-type draft-dflash \
  --spec-draft-model qwen3.6-27b-dflash.gguf \
  --spec-draft-n-max 8 \
  --beefix-dflash-custom -p 0 -n 256 -t 1
```

**Metrics to record:**
- Tokens/sec (throughput)
- Time to first token
- Replay overhead (ms/cycle) — logged by custom mode
- Acceptance rate

**Pass criteria:** Custom mode throughput is within 90% of stock DFlash. If less than 80%, investigate replay overhead.

---

## APPENDIX A: Source Code References

All claims in this blueprint are verified against these source locations:

| Claim | Source |
|-------|--------|
| `need_n_rs_seq()` includes DFlash | [`common/common.h:417`](common/common.h:417) |
| RS allocation formula | [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) |
| Opt-in override point | [`common/common.cpp:1770`](common/common.cpp:1770) |
| GDN kernel (q-independent) | [`ggml/src/ggml-cuda/gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) |
| GDN op definition | [`ggml/src/ggml.c:6426`](ggml/src/ggml.c:6426) |
| Current graph builder tensor names | [`src/models/qwen35.cpp:446-450`](src/models/qwen35.cpp:446) |
| Current graph builder hyperparameters | [`src/models/qwen35.cpp:58-63`](src/models/qwen35.cpp:58) |
| Old tape allocation | [`old-versions/.../llama-context.cpp:2413-2479`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2413) |
| Old tape structs | [`old-versions/.../llama-context.h:118-160`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h:118) |
| Old graph-embedded capture | [`old-versions/.../qwen35.cpp:564-579`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:564) |
| Old GPU cross ring | [`old-versions/.../cross-ring-interleave.cu:126-178`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml-cuda/cross-ring-interleave.cu:126) |
| Verification lifecycle | [`tools/server/server-context.cpp:4213-4274`](tools/server/server-context.cpp:4213) |
| Replay integration point | [`tools/server/server-context.cpp:4263-4267`](tools/server/server-context.cpp:4263) |
| Pre-draft point | [`tools/server/server-context.cpp:3264`](tools/server/server-context.cpp:3264) |

---

## APPENDIX B: Estimated Code Summary

| Category | Lines | Files |
|----------|-------|-------|
| New code | ~330 | `server-dflash-custom.h`, `server-dflash-custom.cpp` |
| Modified code | ~216 | 9 existing files |
| **Total** | **~546** | **11 files (2 new + 9 modified)** |

---

*End of Part 3 (Parts E-G). This completes the revised implementation blueprint.*
