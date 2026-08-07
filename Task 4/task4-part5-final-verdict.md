# Task 4.5: Final Minimal Patch Architecture and Verdict

**Author:** Architect mode investigation
**Date:** 2026-08-07
**Related:** task4-part1 through task4-part4 (see task4-backup-restore-feasibility.md)

---

## SECTION 10 — Minimal Patch Architecture

### 10.0 Design Goal

The smallest realistic patch for:
```
DFlash + n_rs_seq=0 + recurrent backup cell
```

This design eliminates the ~5 GB RS snapshot tensor allocation (currently `n_rs_seq = draft.n_max ≈ 14` rows per cell) by setting `n_rs_seq=0` and replacing snapshot-based rollback with a pre-draft backup cell copy. The accepted recurrent state is recovered through re-decode in the next speculative cycle (current Option B behavior), accepting a compute-for-VRAM trade-off.

### 10.1 Concrete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STARTUP                                                     │
│ ───────────────────────────────────────────────────────────  │
│ 1. Allocate n_parallel * 2 recurrent cells                  │
│    - Cells 0..n_parallel-1: normal (user slots)             │
│    - Cells n_parallel..2*n_parallel-1: backup               │
│ 2. R/S tensors sized: mem_size * (1 + 0) = mem_size rows    │
│    (NO snapshot columns — ~5 GB saved vs current)           │
│ 3. Server detects DFlash mode; overrides seq_rm type        │
│    to TYPE_RS to prevent routine checkpoint serialization   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ BEFORE DRAFT (each speculative cycle)                       │
│ ───────────────────────────────────────────────────────────  │
│ 4. cell_copy(working_cell, backup_cell)                     │
│    - Copies R/S tensor rows from working cell to backup     │
│    - Backup now holds S0 (pre-draft recurrent state)        │
│    - O(n_layers * n_embd) — fast GPU D2D copy              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ DRAFT                                                       │
│ ───────────────────────────────────────────────────────────  │
│ 5. Current DFlash draft() implementation (UNCHANGED)        │
│    - ctx_dft decodes noise tokens                           │
│    - Greedy sample produces predicted tokens                │
│    - Draft recurrent state modified in ctx_dft (discarded)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ VERIFY                                                      │
│ ───────────────────────────────────────────────────────────  │
│ 6. Current DFlash verify (UNCHANGED)                        │
│    - Target runs forward pass on all draft tokens           │
│    - Target recurrent state advances S0 → S1 → ... → SN    │
│    - NO snapshot columns written (n_rs_seq = 0)             │
│    - Returns accepted[] vector (K tokens accepted)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ AFTER ACCEPTANCE                                            │
│ ───────────────────────────────────────────────────────────  │
│ 7. If n_rollback == 0 (full acceptance):                    │
│    - Target recurrent state already correct (SN)            │
│    - No action needed                                       │
│                                                             │
│ 8. If n_rollback > 0 (partial acceptance, K < N):           │
│    a. cell_copy(backup_cell, working_cell)                  │
│       - Restores S0 to working recurrent cell               │
│    b. common_context_seq_rm(ctx_tgt, slot.id,               │
│         P + K + 1, P + N + 1)                               │
│       - Removes rejected KV positions from attention cache  │
│       - Accepted KV positions P+1..P+K preserved            │
│    c. Next cycle: re-decode K accepted tokens               │
│       - Target forward pass advances S0 → SK                │
│       - Batched with new draft verification                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ NEXT CYCLE                                                  │
│ ───────────────────────────────────────────────────────────  │
│ 9. Continue from step 4 with updated state SK               │
│    - Accepted tokens treated as "previous partial draft"    │
│    - New draft extends from accepted tokens                 │
│    - Full verification batch includes accepted + new draft  │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 File/Function Change List

| File | Function/Location | Classification | Change Description |
|------|-------------------|----------------|-------------------|
| [`src/llama-model.cpp`](src/llama-model.cpp:2198) | `llama_model::init()` recurrent mem_size | **MUST CHANGE** | Pass `mem_size = n_seq_max * 2` for DFlash target context to allocate backup cells |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | `cell_copy()` declaration | **MUST CHANGE** | Add public method: `void cell_copy(uint32_t src, uint32_t dst)` for R/S tensor row copy |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:1330) | `cell_copy()` implementation | **MUST CHANGE** | Implement R/S tensor row copy using `ggml_backend_tensor_set()` per layer |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp:35) | Constructor `rs_idx` sizing | **MUST CHANGE** | Enlarge `rs_idx` vector to `n_seq_max + n_backup_count` (or skip `set_rs_idx` for backup cells) |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp:1389) | `server_context_impl::init()` | **MUST CHANGE** | Detect DFlash mode; override `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_RS` to prevent checkpoint every cycle |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp:3306) | Draft phase — backup copy | **MUST CHANGE** | Before verification: `cell_copy(slot.id, backup_seq_id)` to save S0 |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp:4221) | Verification phase — rollback | **MUST CHANGE** | Add backup-cell restore path: `cell_copy(backup_seq_id, slot.id)` + KV `seq_rm` cleanup |
| [`common/common.h`](common/common.h:417) | `need_n_rs_seq()` | **OPTIONAL** | Exclude DFlash from `n_rs_seq` request (or leave as-is with server override) |
| [`common/common.cpp`](common/common.cpp) | `common_context_cell_copy()` | **LIKELY CHANGE** | Add public wrapper API for server to call `cell_copy()` on target context |
| [`src/llama-context.cpp`](src/llama-context.cpp:264) | Arch support check | **LIKELY CHANGE** | Ensure DFlash target arch (Qwen3.6, Gemma 4) is in `llm_arch_supports_rs_rollback()` whitelist, OR ensure `n_rs_seq=0` path is valid |
| [`include/llama.h`](include/llama.h) | Public API declarations | **OPTIONAL** | Expose `cell_copy` through C API if external tools need it |
| [`common/speculative.cpp`](common/speculative.cpp:905) | DFlash impl | **NOT NEEDED** | DFlash `draft()`, `verify()`, `accept()` are unchanged |
| [`src/llama-memory-hybrid.cpp`](src/llama-memory-hybrid.cpp) | Hybrid memory | **NOT NEEDED** | No changes — hybrid delegates to recurrent sub-component |
| [`src/llama-kv-cache.cpp`](src/llama-kv-cache.cpp) | KV cache | **NOT NEEDED** | Standard `seq_rm` handles accepted/rejected KV cleanup |
| [`src/llama-kv-cache-kvarn.cpp`](src/llama-kv-cache-kvarn.cpp) | KVarN cache | **NOT NEEDED** | K changes — accepted KV preserved, rejected KV removed via seq_rm |

### 10.3 Change Classification Summary

| Classification | Count | Files |
|----------------|-------|-------|
| **MUST CHANGE** | 7 | `llama-model.cpp`, `llama-memory-recurrent.h/.cpp` (x3), `server-context.cpp` (x3) |
| **LIKELY CHANGE** | 1 | `common/common.cpp` (public wrapper) |
| **OPTIONAL** | 3 | `common/common.h`, `llama-context.cpp`, `include/llama.h` |
| **NOT NEEDED** | 4 | `speculative.cpp`, `llama-memory-hybrid.cpp`, `llama-kv-cache.cpp`, `llama-kv-cache-kvarn.cpp` |

**Total files that MUST change: 5 unique files** (llama-model.cpp, llama-memory-recurrent.h, llama-memory-recurrent.cpp, server-context.cpp, common/common.cpp)

### 10.4 Key Design Decisions

1. **Static allocation over dynamic.** Backup cells are pre-allocated at construction. No dynamic expansion/shrinking machinery. This eliminates the old deferred expansion system entirely.

2. **Re-decode over tape replay.** Accepted recurrent state is recovered by re-decoding K accepted tokens in the next speculative cycle (batched with new draft verification). This avoids reintroducing tape replay machinery.

3. **Server-level override over memory-layer changes.** The server detects DFlash and overrides `ctx_tgt_seq_rm_type` to prevent routine checkpoint serialization. The memory layer remains unaware of DFlash semantics.

4. **Backup cell copy over checkpoint serialization.** `cell_copy()` performs O(n_layers * n_embd) GPU D2D tensor row copy — orders of magnitude faster than full checkpoint serialization which includes attention KV.

---

## SECTION 11 — Compare Against Old Implementation

### 11.1 Old Mechanism Assessment

| Old Mechanism | Required? | Why? |
|---|---|---|
| DFlash excluded from `need_n_rs_seq()` | **Optional** | Server-level override achieves the same effect (preventing checkpoint serialization). Excluding DFlash from `need_n_rs_seq()` is cleaner but requires touching the generic function. Server override is more localized. |
| Extra backup cells | **YES — Required** | Core of the design. Backup cells hold S0 for rollback. Allocated statically at `n_parallel * 2`. Without these, there is no backup to restore from. |
| Deferred expansion | **NO** | Old dynamic cell expansion/shrinking machinery. Current design uses static allocation — cells exist from startup and are reused every cycle. No dynamic machinery needed. |
| Recurrent-only backup copy | **YES — Required** | `cell_copy()` copies R/S tensor rows from working to backup before draft. This is the backup mechanism. Without it, backup cells are empty and cannot restore state. |
| Async/no-sync copy | **NO** | Old async copy was a performance optimization (non-blocking GPU D2D). Current `cell_copy()` can use synchronous `ggml_backend_tensor_set()`. Async copy could be added later if benchmark shows copy is on critical path. |
| `dflash_rollback()` | **Partially** | The old `dflash_rollback()` function triggered backup-cell restore + KV cleanup. Current design integrates this logic into the server's existing rollback path at `server-context.cpp:4221`. A separate `dflash_rollback()` function is not needed — the logic is inline in the server. |
| Tape replay | **NO** | Old tape replay recorded intermediate R/S states during verification and replayed S0→SK after restore. Current design accepts re-decode overhead instead — K accepted tokens are re-verified in the next cycle. Tape replay would eliminate re-decode but requires significant new code (tape buffer, capture logic, replay logic). |
| DDTree tape buffers | **NO** | DDTree was the old tree-structured speculative decoding system. Current DFlash is linear (single draft chain, no branches). DDTree tape buffers are irrelevant. |
| `dflash_prepare_branch()` | **NO** | Old function prepared branch-specific state for DDTree tree speculation. Linear DFlash has no branches. Not needed. |
| DFlash `accept()` no-op | **YES — Keep** | Current `common_speculative_impl_draft_dflash::accept()` is correctly a no-op. DFlash re-encodes from target embeddings each cycle — no state needs to be carried forward from verification to the next draft. |

### 11.2 Summary: What Survives from Old Implementation

| Category | Old Mechanism | Current Status |
|---|---|---|
| **Survives** | Extra backup cells | Required — core of design |
| **Survives** | Recurrent-only backup copy | Required — `cell_copy()` implementation |
| **Survives** | DFlash `accept()` no-op | Already correct in current code |
| **Replaced** | Deferred expansion | Static allocation (simpler) |
| **Replaced** | `dflash_rollback()` | Inline server logic (cleaner) |
| **Replaced** | `need_n_rs_seq()` exclusion | Server-level override (more localized) |
| **Dropped** | Tape replay | Re-decode instead (simpler, more code) |
| **Dropped** | DDTree tape buffers | Not needed (linear DFlash) |
| **Dropped** | `dflash_prepare_branch()` | Not needed (no branches) |
| **Dropped** | Async/no-sync copy | Synchronous copy sufficient |

### 11.3 Net Result

The minimal patch reintroduces **2 mechanisms** from the old implementation:
1. Extra backup cells (static allocation)
2. Recurrent-only backup copy (`cell_copy()`)

Everything else from the old system (tape replay, DDTree, deferred expansion, branch machinery) is replaced with simpler alternatives or dropped entirely.

---

## SECTION 12 — Final Verdict

### 12.A — Can current upstream DFlash use `n_rs_seq=0` without routine checkpoint serialization?

**YES, with a server-level override.**

Current upstream DFlash with `n_rs_seq=0` will trigger `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` because recurrent memory reports `suffix_rollback_tokens = 0`. This causes `server_speculative_rollback_requires_checkpoint()` to return `true` for any `n_rollback > 0`, resulting in full checkpoint serialization every cycle.

The fix is to override `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_RS` at [`server-context.cpp:1389`](tools/server/server-context.cpp:1389) when DFlash is active, and provide a DFlash-specific `max_rollback` value equal to `draft.n_max`. This tells the server that DFlash has a lightweight backup mechanism and does not need routine checkpoint serialization.

**Estimated code:** ~20 lines in `server-context.cpp`.

### 12.B — Can current recurrent memory provide backup cell without old dynamic expansion?

**YES, with static allocation.**

The recurrent memory constructor takes `mem_size` (total cell count) at [`llama-memory-recurrent.cpp:20`](src/llama-memory-recurrent.cpp:20). By passing `mem_size = n_seq_max * 2` for DFlash contexts, backup cells are allocated alongside normal cells at construction time. No dynamic expansion/shrinking machinery is required.

R/S tensors are sized to `mem_size * (1 + n_rs_seq)` rows. With `n_rs_seq = 0`, this becomes `mem_size * 1` — exactly the number of cells, with no snapshot columns. Backup cells occupy the same tensor space as normal cells (rows `n_parallel` through `2*n_parallel - 1`).

**VRAM cost:** Backup cells double the R/S tensor allocation. However, this is significantly less than the current `n_rs_seq = 14` snapshot allocation (which multiplies by `1 + 14 = 15`).

### 12.C — Can recurrent-only backup/restore use existing upstream APIs?

**NO — new `cell_copy()` API required.**

Existing APIs are insufficient:
- `seq_cp()` only copies cell metadata (reference copy), not R/S tensor data.
- `state_write()`/`state_read()` serialize through `llama_memory_hybrid`, which includes attention KV.
- No public API exposes raw R/S tensor pointers for manual copying.

A new `llama_memory_recurrent::cell_copy(src_cell, dst_cell)` method is required to copy R/S tensor rows between cells within the same context. This method uses `ggml_backend_tensor_set()` for GPU D2D tensor row copy.

**Estimated code:** ~30 lines in `llama-memory-recurrent.cpp`, plus wrapper in `common/common.cpp`.

### 12.D — After restoring backup, how is accepted recurrent state reconstructed?

**Through re-decode in the next speculative cycle.**

After partial acceptance (K < N tokens accepted):
1. `cell_copy(backup_cell, working_cell)` restores S0 to working recurrent cell.
2. `seq_rm()` removes rejected KV positions from attention cache.
3. Next speculative cycle: accepted tokens are re-included in the verification batch.
4. Target forward pass re-processes K accepted tokens, advancing state S0 → SK.
5. New draft tokens extend from SK, and the full batch (accepted + new draft) is verified together.

This is the current Option B behavior — re-decode overhead is proportional to K accepted tokens per cycle. For typical DFlash acceptance rates (70-80%), this means 10-12 tokens re-verified per cycle, batched with new draft verification.

**Alternative:** Option A-refined from Part 3 would use `n_rs_seq` snapshots to select SK directly, eliminating re-decode. However, this requires `n_rs_seq > 0`, which defeats the goal of eliminating snapshot rows.

### 12.E — Is old tape replay required?

**NO.**

Tape replay was designed to record intermediate R/S states during verification and replay S0 → SK without re-decode. The current design accepts re-decode overhead instead, which is simpler and requires no new tape machinery.

Tape replay would require:
- Tape buffer allocation (~16 MB/seq for attention KV of accepted tokens)
- Capture logic during verification forward pass
- Replay logic to advance S0 → SK from tape records
- Tape cleanup after each cycle

The re-decode approach eliminates all of this at the cost of compute (re-verifying K tokens per cycle).

### 12.F — If tape replay required, what minimum subset?

**N/A — tape replay is not required.** See 12.E.

If tape replay were desired in the future, the minimum subset for linear DFlash would be:
1. R/S state at position K (already available in `n_rs_seq` snapshot if `n_rs_seq > 0`)
2. Attention KV for positions P+1 through P+K (already in the cache, lost on checkpoint restore)
3. A KV range-copy mechanism to preserve accepted KV entries

This is Option C from Part 3, estimated at ~16 MB/seq additional VRAM and high code complexity.

### 12.G — Can linear DFlash be VRAM-efficient without DDTree?

**YES.**

Linear DFlash (single draft chain, no branches) does not require DDTree machinery. The minimal patch achieves VRAM efficiency through:
1. `n_rs_seq = 0` — eliminates snapshot columns from R/S tensors.
2. Static backup cells — O(1) extra cells per slot, no dynamic allocation.
3. Standard `seq_rm` for KV cleanup — accepted KV preserved, rejected KV removed.

DDTree was designed for tree-structured speculation with multiple branches. Linear DFlash has no branches, so DDTree tape buffers, branch tracking, and tree-aware replay are not needed.

### 12.H — What is the smallest realistic patch?

**Five files, ~120 lines of new code:**

| File | Lines | Description |
|------|-------|-------------|
| [`src/llama-model.cpp`](src/llama-model.cpp:2198) | ~5 | Pass doubled `mem_size` for DFlash |
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | ~3 | Declare `cell_copy()` method |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | ~35 | Implement `cell_copy()`, enlarge `rs_idx` |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | ~50 | Server override + backup copy before draft + restore after partial accept |
| [`common/common.cpp`](common/common.cpp) | ~25 | Public `common_context_cell_copy()` wrapper |

**Total: ~118 lines across 5 files.**

This patch:
- Eliminates ~5 GB RS snapshot allocation (current `n_rs_seq = 14` → `n_rs_seq = 0`).
- Adds ~256 MB backup cells (doubled R/S tensor for `n_parallel` slots).
- Net VRAM savings: approximately 4.7 GB.
- Accepts re-decode overhead of K tokens per cycle (compute-for-VRAM trade-off).

### 12.I — Approximate VRAM savings vs current DFlash?

| Component | Current (n_rs_seq=14) | Patch (n_rs_seq=0 + backup) | Delta |
|---|---|---|---|
| R/S tensor rows | `mem_size * (1 + 14)` = `15 * mem_size` | `mem_size * (1 + 0)` = `1 * mem_size` | -14× snapshot rows |
| Backup cells | 0 | `mem_size` additional rows | +1× cell rows |
| Net R/S rows | `15 * mem_size` | `2 * mem_size` | -87% |
| Example (Qwen3.6 3B) | ~5.4 GB | ~0.7 GB | **~4.7 GB saved** |

**Net VRAM savings: approximately 87% of current R/S tensor allocation.**

For reference, the current `n_rs_seq = 14` allocation for Qwen3.6 3B is approximately 5.4 GB. The patched allocation with `n_rs_seq = 0` and 2× backup cells is approximately 0.7 GB.

### 12.J — Expected performance penalty?

**Re-decode overhead: K tokens re-verified per speculative cycle.**

For typical DFlash acceptance rates:

| Scenario | Accepted K | Rejected N-K | Re-decode overhead |
|---|---|---|---|
| Full acceptance | 14-15 | 0-1 | Minimal (K tokens verified once) |
| Typical (70-80%) | 10-12 | 3-5 | Moderate (10-12 tokens re-verified) |
| Poor (<50%) | 3-7 | 8-12 | Low (3-7 tokens re-verified) |

**Key factors:**
1. Re-decode is batched with new draft verification — accepted tokens and new draft tokens are processed in a single forward pass.
2. Attention computation for accepted tokens may overlap with new token computation on GPU.
3. The actual overhead depends on whether the backend can pipeline the work.

**Worst-case estimate:** For K=12 accepted tokens, the re-decode represents approximately 80% of the draft batch size being re-processed. However, since these tokens are batched with new draft verification (additional 3-5 tokens), the total batch is 15-17 tokens vs the current 15 tokens — the overhead is the re-computation of 10-12 attention operations, not additional decode calls.

**Benchmark recommendation:** Measure actual throughput before and after the patch to determine if the re-decode overhead is acceptable for the target workload. If DFlash drafting is the bottleneck (not target verification), the overhead may be negligible.

---

## SECTION 13 — Risk Assessment and Recommendations

### 13.1 Risk Matrix

| Risk | Severity | Mitigation |
|---|---|---|
| Re-decode overhead unacceptable | Medium | Benchmark before committing. If overhead >20%, consider Option A-refined (use `n_rs_seq` snapshots with reduced count). |
| `cell_copy()` GPU D2D not supported | Low | All major backends (CUDA, HIP, Metal, Vulkan) support `ggml_backend_tensor_set()`. Fallback to CPU copy if needed. |
| Server override breaks other speculative types | Low | Override is gated on DFlash detection (`ctx_dft && spec && is_dflash(spec)`). Other types unaffected. |
| Backup cells exhaust VRAM on large n_parallel | Low | Backup cells are `n_parallel` additional cells. For `n_parallel = 4`, this is ~0.7 GB. For `n_parallel = 32`, this is ~5.6 GB — consider limiting backup cells to active DFlash slots only. |
| Arch whitelist blocks `n_rs_seq=0` | Low | Verify Qwen3.6/Gemma 4 are in `llm_arch_supports_rs_rollback()`. If not, add them or use server override. |

### 13.2 Recommendations

1. **Implement the minimal patch as described.** Five files, ~120 lines, ~4.7 GB VRAM savings.

2. **Benchmark re-decode overhead.** Measure tokens-per-second before and after the patch. If overhead exceeds 20%, consider reducing `n_rs_seq` to 1-2 instead of 0 (67-78% savings per Task 3, with snapshot-based rollback).

3. **Add DFlash-specific capability to `seq_rm_capability`.** Long-term, add a `bool lightweight_backup_rollback = false` field to signal that DFlash uses backup-cell rollback without requiring RS snapshots. This allows the server to stay generic.

4. **Consider async `cell_copy()` if benchmark shows copy is on critical path.** The synchronous `ggml_backend_tensor_set()` call blocks until the copy completes. If backup copy becomes a bottleneck, switch to async GPU D2D with event synchronization.

5. **Keep `n_rs_seq` configurable.** Allow users to choose between `n_rs_seq = 0` (maximum VRAM savings, re-decode overhead) and `n_rs_seq = 1-2` (moderate VRAM savings, snapshot-based rollback). This provides flexibility for different workloads.

---

*End of document.*
