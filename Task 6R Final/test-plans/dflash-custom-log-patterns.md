# Task 6R: Custom DFlash Replay - Expected Log Patterns

## Overview

This document provides a comprehensive reference for all log output patterns produced by the Task 6R custom DFlash replay implementation. Use this guide to interpret test output, diagnose issues, and validate correct behavior.

All custom DFlash logs use the prefix `[dflash-custom]`. Server-level warnings about custom replay use the `SLT_WRN` macro format.

---

## Log Pattern Categories

### 1. Initialization Logs

Produced when `dflash_custom_init()` is called at server startup with `--beefix-dflash-custom`.

#### Successful Initialization

```
[dflash-custom] GPU tape allocated: 60 recurrent layers, 8 max tokens
[dflash-custom] Initialized: S_k=128, H_k=16, S_v=128, H_v=48, conv_ch=2560, max_tokens=8
```

**Fields:**
- `recurrent layers`: Number of GDN layers in the model (60 for Qwen3.6-27B)
- `max tokens`: Maximum draft tokens (`--spec-draft-n-max`)
- `S_k`: Key state dimension (ssm_d_state, typically 128)
- `H_k`: Key head count (ssm_n_group, typically 16 for fused GDN)
- `S_v`: Value state dimension (same as S_k)
- `H_v`: Value head count (ssm_dt_rank, typically 48 for Qwen3.6)
- `conv_ch`: Convolution channels (d_inner + 2 * n_group * d_state)

**Source:** [`server-dflash-custom.cpp:161-212`](../common/server-dflash-custom.cpp:161)

#### Tape Allocation Failure

```
[dflash-custom] Failed to read GDN hparams from model.
[dflash-custom] Tape allocation failed. Custom mode disabled.
```

**Cause:** Model is not a DFlash model, or hparams cannot be read.

**Source:** [`server-dflash-custom.cpp:68`](../common/server-dflash-custom.cpp:68)

#### No Recurrent Layers

```
[dflash-custom] No recurrent layers found. Tape not needed.
```

**Cause:** Model has no GDN layers (not a recurrent model).

**Source:** [`server-dflash-custom.cpp:83`](../common/server-dflash-custom.cpp:83)

#### Device Allocation Failure

```
[dflash-custom] No device found for layer N.
[dflash-custom] Failed to allocate tape buffer for layer N on device CUDA0.
```

**Cause:** GPU device not available for a model layer, or VRAM insufficient.

**Source:** [`server-dflash-custom.cpp:113`](../common/server-dflash-custom.cpp:113)

---

### 2. Backup/Restore Logs

Produced by `dflash_custom_backup()` and `dflash_custom_restore()`.

#### Backup Skipped (R0)

```
[dflash-custom] backup skipped (R0): mem=0x7f..., n_cells=4, n_backup_cells=0
```

**Reason Code:** R0

**Meaning:** Backup cells are insufficient. Either:
- `n_backup_cells < n_cells` (not enough backup rows allocated)
- `mem` pointer is null
- The n_backup_cells fix may not be active in this build

**Severity:** ERROR - Indicates backup infrastructure is not properly configured.

**Source:** [`server-dflash-custom.cpp:253-259`](../common/server-dflash-custom.cpp:253)

#### Restore Skipped (R0)

```
[dflash-custom] restore skipped (R0): mem=0x7f..., n_cells=4, n_backup_cells=0
```

**Reason Code:** R0

**Meaning:** Same as backup skipped - backup cells insufficient for restore operation.

**Source:** [`server-dflash-custom.cpp:274-280`](../common/server-dflash-custom.cpp:274)

#### Normal Operation

No log is produced when backup/restore succeeds. Silence = success for these operations.

---

### 3. Replay Skip Logs (R1-R9)

Produced by `dflash_custom_replay()` when preconditions fail.

#### R1 - Invalid Preconditions

```
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x7f..., ctx=0x7f..., n_accepted=0)
```

**Reason Code:** R1

**Meaning:** One or more of:
- `state` pointer is null
- `ctx` pointer is null
- `n_accepted <= 0` (zero or negative accepted tokens)

**Expected When:** Verification rejects all draft tokens (n_accepted=0 is valid but means nothing to replay).

**Severity:** INFO when n_accepted=0, ERROR if state/ctx is null.

**Source:** [`server-dflash-custom.cpp:315-320`](../common/server-dflash-custom.cpp:315)

#### R2 - Custom Mode Not Enabled

```
[dflash-custom] replay skipped (R2): custom mode not enabled
```

**Reason Code:** R2

**Meaning:** `dflash_custom_is_enabled()` returned false. Either:
- `state->enabled` is false
- `state->tape` is null
- The `--beefix-dflash-custom` flag was not processed

**Severity:** WARNING - Indicates flag processing issue or init failure.

**Source:** [`server-dflash-custom.cpp:322-326`](../common/server-dflash-custom.cpp:322)

#### R3 - n_accepted Exceeds Captured Tokens

```
[dflash-custom] replay skipped (R3): n_accepted=8 > tokens_captured=6
```

**Reason Code:** R3

**Meaning:** More tokens were accepted than were captured during the draft forward pass. This indicates a mismatch between draft token count and verification results.

**Severity:** WARNING - Indicates draft/capture synchronization issue.

**Source:** [`server-dflash-custom.cpp:328-334`](../common/server-dflash-custom.cpp:328)

#### R4 - No Tape Allocated

```
[dflash-custom] replay skipped (R4): no tape allocated
```

**Reason Code:** R4

**Meaning:** `state->tape` is null. Tape allocation failed during initialization.

**Severity:** ERROR - Indicates critical init failure.

**Source:** [`server-dflash-custom.cpp:336-340`](../common/server-dflash-custom.cpp:336)

#### R5 - No Recurrent Memory

```
[dflash-custom] replay skipped (R5): no recurrent memory
```

**Reason Code:** R5

**Meaning:** `llama_get_memory(ctx)` returned null. The context has no memory structure.

**Severity:** ERROR - Model memory not properly initialized.

**Source:** [`server-dflash-custom.cpp:346-350`](../common/server-dflash-custom.cpp:346)

#### R6 - No Recurrent Memory Component

```
[dflash-custom] replay skipped (R6): no recurrent memory component
```

**Reason Code:** R6

**Meaning:** The memory is not a hybrid memory, or `get_mem_recr()` returned null.

**Severity:** ERROR - Memory type mismatch.

**Source:** [`server-dflash-custom.cpp:355-359`](../common/server-dflash-custom.cpp:355)

#### R7 - No Backup Cells

```
[dflash-custom] replay skipped (R7): n_backup_cells=0 (n_backup_cells fix may not be active)
```

**Reason Code:** R7

**Meaning:** `mem->n_backup_cells` is zero. The backup cell allocation fix may not be compiled in, or backup cells were not allocated.

**Severity:** ERROR - Backup infrastructure missing.

**Source:** [`server-dflash-custom.cpp:365-369`](../common/server-dflash-custom.cpp:365)

#### R8 - No Scheduler

```
[dflash-custom] replay skipped (R8): no scheduler available
```

**Reason Code:** R8

**Meaning:** `ctx->get_sched()` returned null. No compute scheduler available for replay graph execution.

**Severity:** ERROR - Context scheduler not initialized.

**Source:** [`server-dflash-custom.cpp:409-415`](../common/server-dflash-custom.cpp:409)

#### R9 - Conv Rebuild Skipped (No Conv)

```
[dflash-custom] conv rebuild skipped (R9): conv_window=0 (model has no conv)
```

**Reason Code:** R9

**Meaning:** The model has no convolution state (conv_window = 0). This is expected for models without conv layers.

**Severity:** INFO - Normal for non-conv models.

**Source:** [`server-dflash-custom.cpp:588`](../common/server-dflash-custom.cpp:588)

---

### 4. Replay Execution Logs

Produced during successful replay graph construction and execution.

#### Replay Execute Start

```
[dflash-custom] replay execute: n_accepted=5, n_cells=1, layers=60
```

**Fields:**
- `n_accepted`: Number of accepted draft tokens to replay
- `n_cells`: Number of backup cells (typically 1 for single parallel cell)
- `layers`: Number of recurrent layers in the tape

**Source:** [`server-dflash-custom.cpp:517-519`](../common/server-dflash-custom.cpp:517)

#### Conv Rebuild Start

```
[dflash-custom] conv rebuild: 60 layers, conv_window=4, channels=2560, n_accepted=5
```

**Fields:**
- `layers`: Number of tape layers being rebuilt
- `conv_window`: Convolution window size (n_embd_r / conv_channels)
- `channels`: Number of convolution channels
- `n_accepted`: Number of accepted tokens

**Source:** [`server-dflash-custom.cpp:597-599`](../common/server-dflash-custom.cpp:597)

#### Replay Success

```
[dflash-custom] replay success: n_accepted=5, n_cells=1, S_k=128, S_v=128, H_k=16, H_v=48
```

**Fields:**
- `n_accepted`: Number of tokens successfully replayed
- `n_cells`: Number of cells used
- `S_k`, `S_v`, `H_k`, `H_v`: Tape dimensions (validation that dimensions are correct)

**Source:** [`server-dflash-custom.cpp:562-564`](../common/server-dflash-custom.cpp:562)

#### Graph Execution Failure

```
[dflash-custom] Replay graph execution failed.
```

**Cause:** `ggml_backend_sched_graph_compute()` returned false. GPU execution error, CUDA error, or compute graph issue.

**Severity:** ERROR - Indicates GPU/graph issue.

**Source:** [`server-dflash-custom.cpp:521`](../common/server-dflash-custom.cpp:521)

#### Conv QKV Tape Null

```
[dflash-custom] Layer N: qkv tape tensor is null. Conv state rebuild skipped. This indicates a capture failure.
```

**Cause:** The qkv tape tensor for layer N is null. Graph-embedded capture did not write to this tensor.

**Severity:** WARNING - Indicates capture path issue for this layer.

**Source:** [`server-dflash-custom.cpp:626`](../common/server-dflash-custom.cpp:626)

---

### 5. Server-Level Warning Logs

Produced by the server integration code in `server-context.cpp`.

#### Custom Mode Initialized (Slot Info)

```
SLT_INF: dflash custom mode initialized: n_draft_max=8, n_layers=60, S_k=128, S_v=128, H_k=16, H_v=48, conv_channels=2560
```

**Source:** [`server-context.cpp:1464`](../tools/server/server-context.cpp:1464)

#### Replay Failed - Fallback to Checkpoint

```
SLT_WRN: dflash custom replay failed - falling back to checkpoint
```

**Cause:** `dflash_custom_replay()` returned false, or threw an exception.

**Behavior:** Server falls back to stock checkpoint rollback for this cycle.

**Source:** [`server-context.cpp:4320`](../tools/server/server-context.cpp:4320)

#### Replay Exception

```
SLT_WRN: dflash custom replay failed: exception_message - falling back to checkpoint
```

**Cause:** `dflash_custom_replay()` threw a C++ exception.

**Behavior:** Failure counter incremented, fallback to checkpoint.

**Source:** [`server-context.cpp:4320`](../tools/server/server-context.cpp:4320)

#### Permanent Disable

```
SLT_WRN: dflash custom replay permanently disabled after 3 consecutive failures
```

**Cause:** `fail_count >= 3` consecutive failures.

**Behavior:** `replay_failed` flag set to true. Custom mode will not be attempted for this slot until server restart.

**Source:** [`server-context.cpp:4315`](../tools/server/server-context.cpp:4315)

---

## Reason Code Quick Reference

| Code | Function | Condition | Severity | Action |
|------|----------|-----------|----------|--------|
| R0 | backup/restore | `n_backup_cells < n_cells` or null mem | ERROR | Check backup cell allocation |
| R1 | replay | null state/ctx or `n_accepted <= 0` | INFO/ERROR | Normal for n_accepted=0 |
| R2 | replay | custom mode not enabled | WARNING | Verify `--beefix-dflash-custom` flag |
| R3 | replay | `n_accepted > tokens_captured` | WARNING | Check draft/capture sync |
| R4 | replay | no tape allocated | ERROR | Check init success |
| R5 | replay | no recurrent memory | ERROR | Check model memory |
| R6 | replay | no recurrent memory component | ERROR | Check memory type |
| R7 | replay | `n_backup_cells = 0` | ERROR | Check n_backup_cells fix |
| R8 | replay | no scheduler | ERROR | Check context scheduler |
| R9 | replay | `conv_window = 0` | INFO | Normal for non-conv models |

---

## Log Pattern Matching for Automated Tests

### Grep Patterns

```bash
# All custom DFlash messages
grep '\[dflash-custom\]' server.log

# Replay successes only
grep '\[dflash-custom\] replay success:' server.log

# Replay skips with reason code
grep '\[dflash-custom\] replay skipped (R[0-9])' server.log

# Backup failures only
grep '\[dflash-custom\] backup skipped (R0)' server.log

# Conv rebuild messages
grep '\[dflash-custom\] conv rebuild:' server.log

# Server-level warnings
grep 'dflash custom' server.log | grep -i 'warn\|fail\|disabled'

# Count each reason code
for code in R0 R1 R2 R3 R4 R5 R6 R7 R8 R9; do
    count=$(grep -c "skipped ($code)" server.log 2>/dev/null || echo 0)
    echo "$code: $count"
done
```

### Expected Log Sequence for Successful Cycle

```
# 1. Backup (no log on success)
# 2. Draft forward pass (upstream logs)
# 3. Verification (upstream logs)
# 4. Custom replay
[dflash-custom] replay execute: n_accepted=N, n_cells=M, layers=L
[dflash-custom] conv rebuild: L layers, conv_window=W, channels=C, n_accepted=N
[dflash-custom] replay success: n_accepted=N, n_cells=M, S_k=128, S_v=128, H_k=16, H_v=48
```

### Expected Log Sequence for Failed Cycle

```
# 1. Backup (no log on success)
# 2. Draft forward pass (upstream logs)
# 3. Verification (upstream logs)
# 4. Custom replay attempt
[dflash-custom] replay skipped (R1): invalid preconditions (state=0x..., ctx=0x..., n_accepted=0)
# 5. Fallback
SLT_WRN: dflash custom replay failed - falling back to checkpoint
```

---

## Diagnostic Checklist

When troubleshooting, check these log patterns in order:

1. **Is custom mode initialized?**
   - Look for: `[dflash-custom] Initialized:`
   - Missing means: Flag not passed, init failed, or model not DFlash

2. **Is the tape allocated?**
   - Look for: `[dflash-custom] GPU tape allocated:`
   - Missing means: Tape allocation failed

3. **Are backup cells available?**
   - Look for: Absence of R0 messages
   - Present R0 means: Backup cells not configured

4. **Does replay execute?**
   - Look for: `[dflash-custom] replay execute:`
   - Missing means: Replay was skipped (check R1-R8 codes)

5. **Does replay succeed?**
   - Look for: `[dflash-custom] replay success:`
   - Missing means: Graph execution failed or preconditions not met

6. **Is conv state rebuilt?**
   - Look for: `[dflash-custom] conv rebuild:`
   - Missing means: No conv (R9) or qkv tape null

7. **Are there fallbacks?**
   - Look for: `falling back to checkpoint`
   - Present means: Custom replay failed, stock path used

8. **Was custom mode disabled?**
   - Look for: `permanently disabled`
   - Present means: 3+ consecutive failures occurred
