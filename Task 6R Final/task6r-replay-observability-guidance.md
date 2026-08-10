# DFlash Custom Replay — Replay Observability Logging Guidance

**Date:** 2026-08-10
**Priority:** P1 (Runtime validation of P0 correctness fixes)
**Target:** Code mode implementation

---

## Disclaimer

This document provides logging/diagnostics guidance only. It does not contain implementation code. Each logging point specifies the exact file, line context, message format, and conditional logic. The implementing developer should use these as specifications, not copy-paste templates.

---

## 1. Objective

After the P0 correctness fixes (`n_backup_cells` population and convolution-state replay rebuild), there is no runtime visibility into whether:

1. Custom replay actually executes (vs. checkpoint rollback fallback).
2. Backup/restore operations succeed.
3. Convolution state is correctly rebuilt.
4. Edge cases (zero acceptance, first cycle, failure accumulation) behave as expected.

This guidance defines the minimal set of log statements needed to establish runtime confidence in both correctness fixes without introducing excessive verbosity.

---

## 2. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Minimal** | Only log at decision boundaries and state transitions. Avoid per-token or per-layer logs. |
| **Distinguishable** | Logs must clearly indicate whether custom replay or checkpoint rollback was used. |
| **Non-intrusive** | Use existing `SLT_*` macros. Do not add new logging infrastructure. |
| **Conditional** | Respect existing `trace > 0` guards. Most logs should be trace-gated; critical failures are always logged. |
| **Actionable** | Each log should help diagnose a specific failure mode. |

---

## 3. Current Logging State (Baseline)

The existing code already has these log points:

| Location | Current Log | Level |
|----------|-------------|-------|
| [`server-context.cpp:4310`](tools/server/server-context.cpp:4310) | `"accepted %2zu/%2zu draft tokens (dflash custom replay)"` | `SLT_INF` (trace-gated) |
| [`server-context.cpp:4316`](tools/server/server-context.cpp:4316) | `"dflash custom replay permanently disabled after 3 consecutive failures"` | `SLT_WRN` |
| [`server-context.cpp:4320`](tools/server/server-context.cpp:4320) | `"dflash custom replay failed: %s - falling back to checkpoint"` | `SLT_WRN` |
| [`server-context.cpp:4331`](tools/server/server-context.cpp:4331) | `"accepted %2zu/%2zu draft tokens (restore checkpoint)"` | `SLT_INF` (trace-gated) |
| [`server-dflash-custom.cpp:379`](common/server-dflash-custom.cpp:379) | `"Failed to create replay graph context."` | `fprintf(stderr)` |
| [`server-dflash-custom.cpp:494`](common/server-dflash-custom.cpp:494) | `"Replay graph execution failed."` | `fprintf(stderr)` |
| [`server-dflash-custom.cpp:593-595`](common/server-dflash-custom.cpp:593) | `"Layer %d: qkv tape tensor is null. Conv state rebuild skipped."` | `fprintf(stderr)` |

**Gap:** The existing logs do not indicate:
- Whether backup actually executed (pre-draft).
- Whether replay was attempted but skipped due to guards.
- Whether convolution state rebuild ran.
- Any metadata about replay dimensions (n_accepted, n_cells, conv_channels).

---

## 4. Recommended Log Points

### 4.1 Pre-Draft Backup Execution

**Purpose:** Confirm backup cells are populated and backup actually executes.

**Location:** [`tools/server/server-context.cpp:3308-3319`](tools/server/server-context.cpp:3308), after the `dflash_custom_backup()` call.

**Insertion point:** After line 3316 (`break;`), before the closing `}` of the backup block.

**Log statement:**
```
Format: "[dflash-custom] backup: n_cells=%u, backup_offset=%u (slot %d)"
Parameters: mem->n_backup_cells, mem->backup_offset(), slot->id
Level: SLT_DBG (debug-level, always emitted regardless of trace flag)
```

**Rationale:** This confirms:
- `n_backup_cells` is non-zero (validates the `n_backup_cells` fix).
- `backup_offset()` returns the expected value (`mem_size`).
- Backup executes each speculative cycle.

**Conditional logic:** None — always log. This is a debug-level statement that runs once per speculative cycle.

---

### 4.2 Replay Entry — Attempt Decision

**Purpose:** Confirm replay is attempted (not skipped by guards).

**Location:** [`tools/server/server-context.cpp:4302-4303`](tools/server/server-context.cpp:4302), at the start of the `if (dflash_custom_is_enabled(...) && !slot.dflash_custom->replay_failed)` block.

**Insertion point:** After line 4303, before the `try {` block.

**Log statement:**
```
Format: "[dflash-custom] replay attempt: n_accepted=%d, n_draft=%zu, fail_count=%u"
Parameters: n_accepted, slot.spec_draft.size(), slot.dflash_custom->fail_count
Level: SLT_INF (trace-gated, trace > 0)
```

**Rationale:** This confirms:
- Replay was attempted (not skipped by `replay_failed` or `dflash_custom_is_enabled` guards).
- The number of accepted tokens.
- Whether this is a retry after previous failures.

**Conditional logic:** Only when `trace > 0`. This fires every speculative cycle where replay is attempted.

---

### 4.3 Replay Success — Path Confirmation

**Purpose:** Clearly distinguish custom replay success from checkpoint rollback.

**Location:** [`tools/server/server-context.cpp:4307-4311`](tools/server/server-context.cpp:4307), after `replay_succeeded = true`.

**Current log (line 4310):**
```
"accepted %2zu/%2zu draft tokens (dflash custom replay)"
```

**Recommended enhancement:** Change the existing log to include more metadata:

```
Format: "[dflash-custom] replay SUCCESS: accepted %2zu/%2zu tokens, layers_replayed=%u, conv_rebuilt=%u"
Parameters: accepted.size() - 1, slot.spec_draft.size(), number of GDN layers replayed, number of conv layers rebuilt
Level: SLT_INF (trace > 0)
```

**Rationale:** The current log only shows accepted/draft counts. Adding `layers_replayed` and `conv_rebuilt` counts confirms:
- GDN replay executed across expected layers.
- Convolution state rebuild ran (validates the conv fix).

**Implementation note:** The `dflash_custom_replay()` function currently returns `bool`. To report layer counts, either:
- Option A: Add output parameters to `dflash_custom_replay()` (e.g., `int * out_layers_replayed, int * out_layers_conv`).
- Option B: Track counts in `server_dflash_custom_state` and read them after replay.
- Option C: Keep the log simple and just log `"accepted %2zu/%2zu draft tokens (dflash custom replay)"` — the mere fact of reaching this line confirms replay succeeded.

**Recommendation:** Option C for minimal invasiveness. The success of `dflash_custom_replay()` returning `true` implicitly confirms all internal steps (GDN replay + conv rebuild) completed. A separate conv rebuild log (see 4.5) provides the detail.

---

### 4.4 Checkpoint Fallback — Path Confirmation

**Purpose:** Confirm when checkpoint rollback is used instead of custom replay.

**Location:** [`tools/server/server-context.cpp:4329-4332`](tools/server/server-context.cpp:4329), in the `if (!replay_succeeded)` block.

**Current log (line 4331):**
```
"accepted %2zu/%2zu draft tokens (restore checkpoint)"
```

**Recommended enhancement:** Add a reason code to distinguish fallback causes:

```
Format: "[dflash-custom] checkpoint fallback: accepted %2zu/%2zu tokens, reason=%s"
Parameters: accepted.size() - 1, slot.spec_draft.size(), reason string
Level: SLT_INF (trace > 0)
```

**Reason codes:**
| Reason | When |
|--------|------|
| `"replay_disabled"` | `slot.dflash_custom->replay_failed` is true (permanent disable after 3 failures) |
| `"replay_not_enabled"` | `dflash_custom_is_enabled()` returned false (custom mode not active) |
| `"replay_failed"` | `dflash_custom_replay()` returned false (precondition guard failed) |
| `"replay_exception"` | `dflash_custom_replay()` threw an exception |
| `"no_custom_mode"` | `slot.dflash_custom` is null or not initialized |

**Rationale:** Without reason codes, the log `"restore checkpoint"` does not indicate why replay was not used. This makes it impossible to distinguish "replay works but checkpoint was used this cycle" from "replay is permanently disabled."

**Implementation note:** This requires tracking the reason at the call site. A local variable `const char * fallback_reason = nullptr;` set at each failure point would suffice.

---

### 4.5 Convolution State Rebuild — Internal Confirmation

**Purpose:** Confirm convolution state rebuild executes and report dimensions.

**Location:** [`common/server-dflash-custom.cpp:620-622`](common/server-dflash-custom.cpp:620), after the `ggml_backend_tensor_set()` call that writes the updated conv state back.

**Insertion point:** After the per-layer conv rebuild loop completes (after line 622, inside the `for (size_t ti = ...)` loop but after the `ggml_backend_tensor_set` call).

**Log statement:**
```
Format: "[dflash-custom] conv rebuild layer %d: conv_window=%u, channels=%u, n_accepted=%d"
Parameters: il, conv_window, conv_channels, n_accepted
Level: fprintf(stderr) — unconditional, but only fires once per layer per replay
```

**Rationale:** This is the key diagnostic for the convolution state fix. It confirms:
- Conv rebuild actually ran (not skipped by `conv_channels == 0` or `conv_window == 0`).
- The dimensions match expectations.
- The correct number of accepted tokens was applied.

**Conditional logic:** None — always log. This fires once per recurrent layer per replay cycle (e.g., 28 layers for Qwen3.6). To avoid excessive output, consider gating on a debug flag or logging only the first and last layer.

**Alternative (reduced verbosity):** Log only once at the start of the conv rebuild block (before the layer loop, around line 554):

```
Format: "[dflash-custom] conv rebuild: %u layers, conv_window=%u, channels=%u, n_accepted=%d"
Parameters: tape_layers.size(), conv_window, conv_channels, n_accepted
Level: fprintf(stderr) — unconditional, fires once per replay
```

**Recommendation:** Use the alternative (single log before the layer loop). This provides the key metadata without per-layer verbosity.

---

### 4.6 Replay Guard Diagnostics — Skip Reasons

**Purpose:** Log when replay is skipped due to precondition guards (helps diagnose why replay returns `false`).

**Location:** [`common/server-dflash-custom.cpp:307-324`](common/server-dflash-custom.cpp:307), at each early-return guard in `dflash_custom_replay()`.

**Current guards and recommended logs:**

| Guard | Line | Current Behavior | Recommended Log |
|-------|------|------------------|-----------------|
| `!state \|\| !ctx \|\| n_accepted <= 0` | 309 | Silent return `false` | `"[dflash-custom] replay skipped: invalid preconditions (state=%p, ctx=%p, n_accepted=%d)"` |
| `!dflash_custom_is_enabled(state)` | 313 | Silent return `false` | `"[dflash-custom] replay skipped: custom mode not enabled"` |
| `n_accepted > tokens_captured` | 317 | Silent return `false` | `"[dflash-custom] replay skipped: n_accepted=%d > tokens_captured=%u"` |
| `!state->tape` | 322 | Silent return `false` | `"[dflash-custom] replay skipped: no tape allocated"` |
| `!mem_base` | 330 | Silent return `false` | `"[dflash-custom] replay skipped: no recurrent memory"` |
| `!mem` | 337 | Silent return `false` | `"[dflash-custom] replay skipped: no recurrent memory component"` |
| `n_cells == 0` | 345 | Silent return `false` | `"[dflash-custom] replay skipped: n_backup_cells=0 (n_backup_cells fix may not be active)"` |

**Level:** `fprintf(stderr)` — unconditional.

**Rationale:** These guards currently return `false` silently. When replay fails, the caller has no way to know which guard triggered. This makes debugging impossible without source-level debugging.

**Recommendation:** Add logs to ALL guards. These are rare events (should only fire during misconfiguration or edge cases), so verbosity is not a concern.

---

### 4.7 Backup/Restore Execution Confirmation

**Purpose:** Confirm backup and restore operations actually copy data (not early-returning due to guards).

**Location:** [`common/server-dflash-custom.cpp:253-263`](common/server-dflash-custom.cpp:253) for backup, [`common/server-dflash-custom.cpp:271-281`](common/server-dflash-custom.cpp:271) for restore.

**Recommended logs:**

**Backup (`dflash_custom_backup`):**
```
Insertion: After the for loop (after line 262), before the closing `}`.
Format: "[dflash-custom] backup complete: %u cells copied"
Parameters: n_cells
Level: fprintf(stderr) — unconditional
```

**Restore (`dflash_custom_restore`):**
```
Insertion: After the for loop (after line 280), before the closing `}`.
Format: "[dflash-custom] restore complete: %u cells restored"
Parameters: n_cells
Level: fprintf(stderr) — unconditional
```

**Rationale:** These confirm that backup/restore actually iterate (not early-returning due to `n_backup_cells < n_cells` guard). After the `n_backup_cells` fix, these should fire every cycle.

**Alternative (reduced verbosity):** Only log if the guard triggers (early return):
```
Insertion: Inside the `if (!mem || n_cells == 0 || mem->n_backup_cells < n_cells)` block.
Format: "[dflash-custom] backup skipped: mem=%p, n_cells=%u, n_backup_cells=%u"
Parameters: (void*)mem, n_cells, mem ? mem->n_backup_cells : 0
Level: fprintf(stderr) — unconditional
```

**Recommendation:** Use the alternative. Logging on the skip path is more valuable than logging on the success path, since the skip path indicates a problem.

---

## 5. Summary Table

| # | Location | Purpose | Format | Level | Conditional |
|---|----------|---------|--------|-------|-------------|
| L1 | [`server-context.cpp:3316`](tools/server/server-context.cpp:3316) | Backup execution | `"backup: n_cells=%u, offset=%u"` | `SLT_DBG` | Always |
| L2 | [`server-context.cpp:4303`](tools/server/server-context.cpp:4303) | Replay attempt | `"replay attempt: n_accepted=%d, n_draft=%zu"` | `SLT_INF` | `trace > 0` |
| L3 | [`server-context.cpp:4307`](tools/server/server-context.cpp:4307) | Replay success | `"replay SUCCESS: accepted %zu/%zu tokens"` | `SLT_INF` | `trace > 0` (existing, enhanced) |
| L4 | [`server-context.cpp:4329`](tools/server/server-context.cpp:4329) | Checkpoint fallback with reason | `"checkpoint fallback: reason=%s"` | `SLT_INF` | `trace > 0` |
| L5 | [`server-dflash-custom.cpp:554`](common/server-dflash-custom.cpp:554) | Conv rebuild start | `"conv rebuild: %u layers, window=%u, channels=%u"` | `fprintf` | Always |
| L6 | [`server-dflash-custom.cpp:309-345`](common/server-dflash-custom.cpp:309) | Replay guard diagnostics | `"replay skipped: [reason]"` | `fprintf` | Always |
| L7 | [`server-dflash-custom.cpp:254`](common/server-dflash-custom.cpp:254) | Backup skip diagnostic | `"backup skipped: n_cells=%u, n_backup_cells=%u"` | `fprintf` | On skip only |
| L8 | [`server-dflash-custom.cpp:272`](common/server-dflash-custom.cpp:272) | Restore skip diagnostic | `"restore skipped: n_cells=%u, n_backup_cells=%u"` | `fprintf` | On skip only |

---

## 6. Expected Log Output — Normal Replay Cycle

With all log points active and `trace > 0`, a successful replay cycle should produce:

```
[dflash-custom] backup: n_cells=4, backup_offset=8192 (slot 0)
... (draft forward pass) ...
[dflash-custom] replay attempt: n_accepted=3, n_draft=8, fail_count=0
[dflash-custom] restore complete: 4 cells restored
[dflash-custom] conv rebuild: 28 layers, conv_window=2, channels=2176, n_accepted=3
[dflash-custom] replay SUCCESS: accepted  3/ 8 tokens
```

## 7. Expected Log Output — Checkpoint Fallback

When replay fails and checkpoint rollback is used:

```
[dflash-custom] backup: n_cells=4, backup_offset=8192 (slot 0)
... (draft forward pass) ...
[dflash-custom] replay attempt: n_accepted=3, n_draft=8, fail_count=0
[dflash-custom] replay skipped: n_accepted=3 > tokens_captured=0
[dflash-custom] checkpoint fallback: reason=replay_failed
accepted  3/ 8 draft tokens (restore checkpoint)
```

## 8. Expected Log Output — Backup Skip (n_backup_cells Fix Not Active)

If the `n_backup_cells` fix is not applied or not taking effect:

```
[dflash-custom] backup skipped: n_cells=4, n_backup_cells=0
... (draft forward pass) ...
[dflash-custom] replay skipped: n_backup_cells=0 (n_backup_cells fix may not be active)
[dflash-custom] checkpoint fallback: reason=replay_failed
```

---

## 9. Implementation Order

| Priority | Log Point | Effort | Impact |
|----------|-----------|--------|--------|
| **P1** | L6 (replay guard diagnostics) | ~7 lines | Critical — identifies which guard causes replay failure |
| **P1** | L7/L8 (backup/restore skip diagnostics) | ~4 lines | Critical — confirms n_backup_cells fix is active |
| **P2** | L5 (conv rebuild confirmation) | ~2 lines | Validates conv state rebuild executes |
| **P2** | L4 (checkpoint fallback with reason) | ~10 lines | Distinguishes fallback causes |
| **P3** | L1 (backup execution) | ~2 lines | Confirms backup runs each cycle |
| **P3** | L2 (replay attempt) | ~2 lines | Confirms replay is attempted |
| **P3** | L3 (replay success enhancement) | ~2 lines | Enhances existing success log |

---

## 10. Warnings and Constraints

1. **Do not add logging that changes behavior.** All logs should be observational only. Do not add logging that modifies state, timing, or control flow.

2. **Respect existing `trace` gating.** The `SLT_INF` macro already respects `trace > 0`. Do not change this behavior.

3. **Avoid `fprintf` in hot paths.** The `fprintf(stderr)` calls in L5-L8 fire once per replay cycle, which is acceptable. Do not add `fprintf` calls inside per-token or per-layer loops.

4. **Log format consistency.** All `[dflash-custom]` prefixed logs should use the same prefix for grep-ability.

5. **Memory overhead.** The log strings themselves are small and transient. No additional memory allocation is needed.

---

*End of guidance document.*
