# Task 6R: Custom DFlash Replay - Test Plan

## Overview

This document defines the test plan for validating the Task 6R custom DFlash replay implementation (`--beefix-dflash-custom`). The implementation provides a VRAM-efficient alternative to the stock checkpoint snapshot system by using backup cells and GPU tape for rollback-then-replay.

## Architecture Summary

The custom DFlash mode replaces the large RS snapshot buffer (~5.4 GB for Qwen3.6-27B) with:

1. **Backup cells** in recurrent memory (pre-draft state copy)
2. **GPU tape** - per-layer F32 tensors capturing GDN intermediates during draft forward pass
3. **Replay graph** - rebuilds GDN state for accepted tokens without re-running the full forward pass

Key files:
- [`common/server-dflash-custom.h`](../common/server-dflash-custom.h) - GPU tape structs and API declarations
- [`common/server-dflash-custom.cpp`](../common/server-dflash-custom.cpp) - Backup, restore, replay implementation
- [`tools/server/server-context.cpp`](../tools/server/server-context.cpp) - Server integration (init, backup, replay invocation)
- [`common/arg.cpp`](../common/arg.cpp) - `--beefix-dflash-custom` CLI flag

## Test Scenarios

### Scenario 1: Initialization and Tape Allocation

**Goal:** Verify custom mode initializes correctly with proper tape dimensions.

**Preconditions:**
- Build with CUDA support
- Qwen3.6 DFlash model pair available (target + draft)
- Sufficient VRAM for model + tape

**Steps:**
1. Start `llama-server` with `--beefix-dflash-custom --spec-type draft-dflash`
2. Observe startup logs

**Expected Output:**
```
[dflash-custom] GPU tape allocated: N recurrent layers, M max tokens
[dflash-custom] Initialized: S_k=128, H_k=16, S_v=128, H_v=48, conv_ch=X, max_tokens=M
```

**Pass Criteria:**
- `[dflash-custom]` init logs present
- Tape dimensions match model hparams (S_k=128, H_k=16 for Qwen3.6)
- No `nullptr` or allocation failure messages
- Server reports `dflash custom mode initialized` in slot info

---

### Scenario 2: Normal Custom Replay Path Execution

**Goal:** Verify the full replay cycle executes successfully for a standard inference request.

**Preconditions:**
- Server running with custom mode enabled
- Trace level > 0 (`--trace 1`)

**Steps:**
1. Send a chat completion request that triggers speculative decoding
2. Ensure at least some draft tokens are accepted (n_accepted >= 1)
3. Observe replay logs

**Expected Output:**
```
[dflash-custom] backup: N cells backed up
[dflash-custom] replay execute: n_accepted=X, n_cells=Y, layers=Z
[dflash-custom] conv rebuild: Z layers, conv_window=W, channels=C, n_accepted=X
[dflash-custom] replay success: n_accepted=X, n_cells=Y, S_k=128, S_v=128, H_k=16, H_v=48
```

**Pass Criteria:**
- No skip/reason code (R0-R9) messages during replay
- `replay success` log present
- Conv rebuild log present (unless model has no conv)
- No fallback to checkpoint path
- Response matches expected output quality

---

### Scenario 3: Checkpoint Rollback Fallback Detection

**Goal:** Verify the system falls back to checkpoint rollback when custom replay fails.

**Preconditions:**
- Server running with custom mode enabled

**Steps:**
1. Induce replay failure (e.g., corrupt tape data scenario, or n_accepted > tokens_captured)
2. Observe failure handling and fallback

**Expected Output:**
```
[dflash-custom] replay skipped (R3): n_accepted=X > tokens_captured=Y
SLT_WRN: dflash custom replay failed - falling back to checkpoint
```

Or after 3 consecutive failures:
```
SLT_WRN: dflash custom replay permanently disabled after 3 consecutive failures
```

**Pass Criteria:**
- Fallback to checkpoint path occurs without crash
- After 3 consecutive failures, custom mode is permanently disabled for that slot
- Server continues serving requests (no hang or OOM)

---

### Scenario 4: Backup Cell Allocation Verification

**Goal:** Verify backup cells are properly allocated and used.

**Preconditions:**
- Server running with custom mode enabled

**Steps:**
1. Monitor backup/restore logs during speculative decoding
2. Verify `mem->n_backup_cells > 0` in backup calls

**Expected Output:**
```
[dflash-custom] backup skipped (R0): mem=0x..., n_cells=X, n_backup_cells=Y
```
(Only appears if backup cells are insufficient - should NOT appear in normal operation)

In normal operation, no R0 skip message should appear during backup.

**Pass Criteria:**
- No R0 skip message during normal backup
- `n_backup_cells` matches the number of parallel cells
- Backup and restore operations are symmetric (same n_cells value)

---

### Scenario 5: Convolution State Replay Correctness

**Goal:** Verify convolution state is correctly rebuilt after replay.

**Preconditions:**
- Model with conv state (Qwen3.6 has conv)
- Server running with custom mode enabled

**Steps:**
1. Run inference with speculative decoding
2. Observe conv rebuild logs
3. Compare output quality with stock checkpoint mode

**Expected Output:**
```
[dflash-custom] conv rebuild: Z layers, conv_window=W, channels=C, n_accepted=X
```

**Pass Criteria:**
- Conv rebuild log present for each replay cycle
- `conv_window` matches `n_embd_r / conv_channels`
- Output tokens match stock checkpoint mode (bitwise comparison of first N tokens)
- No R9 skip message (unless model genuinely has no conv)

---

### Scenario 6: Edge Case - n_accepted = 0

**Goal:** Verify graceful handling when zero draft tokens are accepted.

**Steps:**
1. Craft a request where verification rejects all draft tokens
2. Observe replay behavior

**Expected Output:**
```
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x..., ctx=0x..., n_accepted=0)
```

**Pass Criteria:**
- R1 skip message with `n_accepted=0`
- No crash or undefined behavior
- Fallback to checkpoint path
- Server continues normal operation

---

### Scenario 7: Edge Case - n_accepted > Context Window

**Goal:** Verify handling when n_accepted exceeds captured tokens.

**Steps:**
1. Force a scenario where `n_accepted > tokens_captured`
2. Observe replay behavior

**Expected Output:**
```
[dflash-custom] replay skipped (R3): n_accepted=X > tokens_captured=Y
```

**Pass Criteria:**
- R3 skip message present
- Fallback to checkpoint path
- No buffer overrun or out-of-bounds access

---

### Scenario 8: Edge Case - Tape Corruption Simulation

**Goal:** Verify system handles corrupted/invalid tape data gracefully.

**Steps:**
1. Run inference with custom mode
2. (Conceptual test - verify code paths handle null tape tensors)

**Expected Output:**
```
[dflash-custom] replay skipped (R4): no tape allocated
```
Or if tape exists but layer tensors are null:
```
[dflash-custom] Layer N: qkv tape tensor is null. Conv state rebuild skipped.
```

**Pass Criteria:**
- Null pointer checks prevent crash
- Graceful degradation to fallback path
- Informative error message

---

### Scenario 9: Multi-Round Inference Stability

**Goal:** Verify custom mode remains stable across multiple speculative decoding cycles.

**Steps:**
1. Run a long conversation (20+ turns) with speculative decoding enabled
2. Monitor for memory leaks, failures, or degradation

**Pass Criteria:**
- No increase in VRAM usage beyond initial allocation
- No cumulative failure count growth (fail_count resets on success)
- Consistent token quality across all turns
- No permanent disable of custom mode

---

### Scenario 10: Disable/Re-enable Cycle

**Goal:** Verify custom mode can be disabled and the server continues functioning.

**Steps:**
1. Start server with `--beefix-dflash-custom`
2. Let 3 consecutive failures occur (permanent disable)
3. Continue serving requests

**Pass Criteria:**
- After permanent disable, requests use stock checkpoint path
- Server does not crash
- Performance degrades gracefully (higher VRAM usage due to checkpoint buffer)

---

## Reason Code Reference

| Code | Location | Meaning | Severity |
|------|----------|---------|----------|
| R0 | `dflash_custom_backup()`, `dflash_custom_restore()` | Backup cells insufficient (`n_backup_cells < n_cells`) or null memory pointer | Warning - indicates backup cell allocation issue |
| R1 | `dflash_custom_replay()` | Invalid preconditions: null state, null ctx, or `n_accepted <= 0` | Info - expected for n_accepted=0 scenarios |
| R2 | `dflash_custom_replay()` | Custom mode not enabled (tape null or `enabled=false`) | Warning - flag may not have been processed |
| R3 | `dflash_custom_replay()` | `n_accepted > tokens_captured` - cannot replay more than captured | Warning - indicates draft/capture mismatch |
| R4 | `dflash_custom_replay()` | No tape allocated | Error - indicates init failure |
| R5 | `dflash_custom_replay()` | No recurrent memory (`llama_get_memory()` returns null) | Error - model memory issue |
| R6 | `dflash_custom_replay()` | No recurrent memory component (hybrid cast fails) | Error - memory type mismatch |
| R7 | `dflash_custom_replay()` | `n_backup_cells = 0` - n_backup_cells fix may not be active | Error - backup cell patch missing |
| R8 | `dflash_custom_replay()` | No scheduler available (`ctx->get_sched()` returns null) | Error - context scheduler issue |
| R9 | `dflash_custom_replay()` | Conv rebuild skipped: `conv_window=0` (model has no conv) | Info - expected for non-conv models |

---

## Test Commands

See [`validate-dflash-custom.sh`](validate-dflash-custom.sh) for the automated validation script.

### Manual Test Invocation

```bash
# Basic test with custom mode
./build/bin/llama-server \
    -m target_model.gguf \
    --spec-type draft-dflash \
    --spec-draft-model draft_model.gguf \
    --spec-draft-n-max 8 \
    --beefix-dflash-custom \
    --trace 1 \
    --port 8080 \
    2>&1 | tee server.log

# From another terminal - send test request
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Explain speculative decoding in three sentences."}],
        "temperature": 0.7,
        "max_tokens": 100
    }'
```

### Log Analysis

```bash
# Count replay successes
grep -c "replay success" server.log

# Count replay skips by reason code
grep -oP "replay skipped \(\K[R0-9]+" server.log | sort | uniq -c

# Check for permanent disable
grep -c "permanently disabled" server.log

# Verify no R0 backup failures
grep "backup skipped (R0)" server.log && echo "BACKUP FAILURE DETECTED" || echo "Backup OK"
```

---

## Expected Log Patterns

### Successful Custom Replay Cycle

```
[dflash-custom] backup skipped (R0): ...                    <-- Should NOT appear
[dflash-custom] replay execute: n_accepted=5, n_cells=1, layers=60
[dflash-custom] conv rebuild: 60 layers, conv_window=4, channels=2560, n_accepted=5
[dflash-custom] replay success: n_accepted=5, n_cells=1, S_k=128, S_v=128, H_k=16, H_v=48
```

### Checkpoint Fallback (Single Failure)

```
[dflash-custom] replay skipped (R3): n_accepted=8 > tokens_captured=6
SLT_WRN: dflash custom replay failed - falling back to checkpoint
```

### Permanent Disable (After 3 Consecutive Failures)

```
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x..., ctx=0x..., n_accepted=0)
SLT_WRN: dflash custom replay failed - falling back to checkpoint
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x..., ctx=0x..., n_accepted=0)
SLT_WRN: dflash custom replay failed - falling back to checkpoint
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x..., ctx=0x..., n_accepted=0)
SLT_WRN: dflash custom replay permanently disabled after 3 consecutive failures
```

### Conv State Skip (No-Conv Model)

```
[dflash-custom] conv rebuild skipped (R9): conv_window=0 (model has no conv)
```

### Backup Cell Failure

```
[dflash-custom] backup skipped (R0): mem=0x..., n_cells=4, n_backup_cells=0
```

---

## Success Criteria Summary

| Category | Pass Criteria |
|---------|---------------|
| Initialization | Tape allocated with correct dimensions; no null pointers |
| Normal Replay | `replay success` log present; no R0-R9 skip codes |
| Fallback | Graceful degradation to checkpoint path; no crash |
| Backup/Restore | Symmetric operations; no R0 skip in normal operation |
| Conv Rebuild | Log present for conv models; output matches stock mode |
| Edge Cases | R1 for n_accepted=0; R3 for overflow; no buffer overrun |
| Stability | No memory leak; no cumulative failures; consistent quality |
| Recovery | Server continues after permanent disable |

---

## Known Limitations

1. **CPU-based conv rebuild:** The convolution state rebuild currently runs on CPU. A CUDA kernel (P3 priority) can improve performance but is not required for correctness.
2. **Single sequence:** The replay graph assumes `n_seqs = 1` (per-slot serving). Multi-sequence support is not yet implemented.
3. **K=1 GDN output:** The replay uses K=1 (state-only output, no RS snapshots). This is intentional for VRAM efficiency.
4. **Same-device tape:** Tape tensors must live on the same GPU as the corresponding model layer. Cross-GPU tape placement is not supported.
