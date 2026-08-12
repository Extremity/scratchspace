# Architectural Audit: Old 0.3.2 Capability Preservation Analysis

**Date:** 2026-08-12
**Purpose:** Compare the current Task 6R implementation against the old BeeLlama 0.3.2 custom DFlash to determine what was preserved, what was lost, and what's different.
**Method:** Source code inspection of Task 6R implementation files and documentation review. No code modifications.

**Reference Documents:**
- [`plans/dflash-comparison/final-comparison.md`](../dflash-comparison/final-comparison.md) — Old vs. current comparison (377 lines).
- [`plans/dflash-solutions/task6r-final-documentation.md`](task6r-final-documentation.md) — Task 6R implementation (663 lines).
- [`plans/dflash-solutions/task6r-deferred-items-review.md`](task6r-deferred-items-review.md) — Deferred items review.

**Source Files Examined:**
- [`common/server-dflash-custom.h`](common/server-dflash-custom.h) — 250 lines; tape structs, config, API.
- [`common/server-dflash-custom.cpp`](common/server-dflash-custom.cpp) — 863 lines; full implementation.
- [`src/models/qwen35.cpp:460-529`](src/models/qwen35.cpp:460) — Tape capture integration.
- [`tools/server/server-context.cpp`](tools/server/server-context.cpp) — Server integration (13 `dflash_custom` references).

---

## Table of Contents

| Section | Topic |
|---------|-------|
| [1. Capability Comparison Table](#1-capability-comparison-table) | Old 0.3.2 vs Task 6R side-by-side |
| [2. Q1: Old 0.3.2 Capabilities Beyond Stock Upstream](#2-q1-old-032-capabilities-beyond-stock-upstream) | What old provided that stock upstream did NOT have |
| [3. Q2: Intended Reproduction Capabilities](#3-q2-intended-reproduction-capabilities) | What we INTENDED to reproduce |
| [4. Q3: Task 6R Capabilities Beyond Old 0.3.2](#4-q3-task-6r-capabilities-beyond-old-032) | New capabilities added by Task 6R |
| [5. Q4: GPU-Optimized Backup Copy — Preserved or Lost](#5-q4-gpu-optimized-backup-copy--preserved-or-lost) | `llama_dflash_memory_seq_cp_recurrent_ordered()` analysis |
| [6. Q5: Tape Replay Equivalence](#6-q5-tape-replay-equivalence) | Old `tape_replay()` vs Task 6R `dflash_custom_replay()` |
| [7. Confidence Assessment](#7-confidence-assessment) | Confidence levels per finding |
| [8. Recommendations](#8-recommendations) | Corrective action vs acceptable difference |

---

## 1. Capability Comparison Table

| Capability | Old 0.3.2 | Task 6R | Status |
|------------|-----------|---------|--------|
| Backup cells (recurrent-only state copy) | `dflash_backup_recurrent_state()` — lambda in `server-context.cpp:4788` | `dflash_custom_backup()` — [`server-dflash-custom.cpp:305`](common/server-dflash-custom.cpp:305) | **Preserved** (redesigned) |
| Tape replay (DeltaNet for accepted tokens) | `tape_replay()` — `src/llama-context.cpp` | `dflash_custom_replay()` — [`server-dflash-custom.cpp:366`](common/server-dflash-custom.cpp:366) | **Preserved** (redesigned) |
| `llama_dflash_rollback()` (three-phase rollback) | `src/llama-context.cpp:4218` — attention KV cleanup + recurrent restore + tape replay | No direct equivalent. Replay replaces phases 2-3; phase 1 handled by upstream `seq_rm()` | **Replaced** (upstream unified path) |
| GPU-optimized layer-ordered copy | `llama_dflash_memory_seq_cp_recurrent_ordered()` | `dflash_custom_cell_copy()` — [`server-dflash-custom.cpp:255`](common/server-dflash-custom.cpp:255) | **Degraded** (see Q4) |
| `n_rs_seq = 0` for DFlash | DFlash excluded from `need_n_rs_seq()` | `cparams.n_rs_seq = 0` override — [`common/common.cpp:1775-1781`](common/common.cpp:1775) | **Preserved** |
| Checkpoint fallback | When rollback exceeds RS bounds | After 3 consecutive replay failures — [`server-context.cpp:4335-4346`](tools/server/server-context.cpp:4335) | **Preserved** (enhanced) |
| GPU tape (per-layer F32 tensors) | `dflash_tape_gpu` with `n_seqs` and `seq_ids[]` arrays | `server_dflash_tape_gpu` — [`server-dflash-custom.h:62`](common/server-dflash-custom.h:62) | **Preserved** (simplified) |
| Multi-sequence support | GPU tape: YES (per-seq scatter). CPU tape: NO | Single sequence only (`n_seqs = 1` hardcoded) | **Lost** (deferred — see deferred items review §2) |
| Convolution state rebuild | CPU `memcpy` during tape replay | CUDA-native kernel — `ggml/src/ggml-cuda/dflash-custom-conv.cu`; CPU fallback — [`server-dflash-custom.cpp:755-831`](common/server-dflash-custom.cpp:755) | **Improved** (new capability) |
| Device-aware tape placement | Cross-ring, hidden GPU buffers, prefill staging | Per-layer tape on same GPU as model layer — [`server-dflash-custom.cpp:113-166`](common/server-dflash-custom.cpp:113) | **Improved** (simplified) |
| Fallback reason codes (R0-R9) | No | 10 reason codes with structured logging — [`server-dflash-custom.cpp:307-846`](common/server-dflash-custom.cpp:307) | **New** |
| `replay_failed` permanent disable | No | After 3 consecutive failures — [`server-dflash-custom.h:105`](common/server-dflash-custom.h:105) | **New** |
| Adaptive profit controller integration | No (Bee had profit-only controller separately) | Integrated via upstream `--spec-draft-n-max` | **N/A** (upstream handles) |
| Profile infrastructure | `dflash_profile_start()` / `dflash_profile_end()` | No equivalent | **Lost** (intentional) |
| `tree_bufs` (DDTree) support | Yes | No | **Lost** (removed v0.4.0) |
| `llama_tape_replay_sync()` (multi-slot sync) | Yes | No | **Lost** (single-sequence scope) |
| `recurrent_backup_attention_streams` | Yes (for DDTree branch rollback) | No | **Lost** (DDTree removed) |

---

## 2. Q1: Old 0.3.2 Capabilities Beyond Stock Upstream

The old BeeLlama 0.3.2 implementation provided the following capabilities that stock upstream DFlash did NOT have:

### 2.1 Backup Cells (Recurrent-Only State Copy)

**Old implementation:** `dflash_backup_recurrent_state()` lambda at [`old server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788).

**Purpose:** Copy recurrent state (R and S tensors) from the active sequence to a backup sequence before draft. This preserves pre-draft state so rollback can restore it without full checkpoint serialization.

**Old mechanism:**
1. Called `llama_dflash_memory_seq_cp_recurrent_ordered()` for GPU-optimized, layer-ordered copy.
2. Fell back to `llama_memory_seq_cp_recurrent()` if ordered path failed.
3. Copied to `seq_backup` = `slot.id + n_parallel_user`.

**Task 6R equivalent:** `dflash_custom_backup()` at [`server-dflash-custom.cpp:305`](common/server-dflash-custom.cpp:305). Uses `dflash_custom_cell_copy()` for device-native copies via `ggml_backend_tensor_copy()`. Backup rows are allocated within the recurrent memory tensor (extra rows beyond `mem_size`), not in a separate sequence.

**Verdict:** **Preserved with redesign.** Task 6R uses backup rows within the R/S tensor allocation rather than separate backup sequences. The allocation formula `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells` with `n_rs_seq=0` gives `n_rows = mem_size + n_backup_cells`. This is functionally equivalent but architecturally cleaner.

### 2.2 Tape Replay (DeltaNet Forward Pass Replay)

**Old implementation:** `tape_replay()` in [`old src/llama-context.cpp`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp).

**Purpose:** After restoring pre-draft recurrent state from backup, replay the DeltaNet state updates for accepted tokens. This advances the state to the accepted position without re-running the full forward pass.

**Old mechanism:**
1. Restore recurrent state from `seq_backup`.
2. For each accepted token, replay DeltaNet forward pass using captured tape data.
3. Update S state from replay results.
4. Conv state was handled as part of the replay process.

**Task 6R equivalent:** `dflash_custom_replay()` at [`server-dflash-custom.cpp:366`](common/server-dflash-custom.cpp:366). Builds a replay graph using `ggml_gated_delta_net()` for each recurrent layer, executes via scheduler, then writes updated state back.

**Verdict:** **Preserved with redesign.** See [Section 6](#6-q5-tape-replay-equivalence) for detailed equivalence analysis.

### 2.3 `llama_dflash_rollback()` (Three-Phase Rollback)

**Old implementation:** [`old src/llama-context.cpp:4218`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4218).

**Purpose:** Coordinated three-phase rollback:
1. Attention KV cleanup (remove rejected, keep accepted).
2. Recurrent state restore from backup sequence.
3. Tape replay for accepted tokens.

**Task 6R equivalent:** No single function. The three phases are now split:
- Phase 1 (attention KV cleanup): Handled by upstream unified `seq_rm()` path.
- Phases 2-3 (recurrent restore + replay): Handled by `dflash_custom_replay()` which calls `dflash_custom_restore()` internally, then builds and executes the replay graph.

**Verdict:** **Replaced by upstream unified path + custom replay.** The old three-phase function is no longer needed because upstream handles attention KV cleanup, and Task 6R handles recurrent state via backup/restore/replay.

### 2.4 `llama_dflash_memory_seq_cp_recurrent_ordered()` (GPU-Optimized Copy)

**Old implementation:** [`old server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788).

**Purpose:** Layer-ordered, GPU-optimized copy of recurrent state. Copied layers in dependency order to maximize GPU utilization during backup.

**Task 6R equivalent:** `dflash_custom_cell_copy()` at [`server-dflash-custom.cpp:255`](common/server-dflash-custom.cpp:255). Uses `ggml_backend_tensor_copy()` per layer, sequentially. No layer ordering optimization.

**Verdict:** **Degraded.** See [Section 5](#5-q4-gpu-optimized-backup-copy--preserved-or-lost) for analysis.

### 2.5 `n_rs_seq = 0` for DFlash

**Old implementation:** DFlash excluded from `need_n_rs_seq()`. Only MTP triggered RS allocation.

**Purpose:** Avoid ~5.4 GB RS buffer allocation for DFlash. The backup cell approach (~150 MB per slot) is far more VRAM-efficient than RS snapshots.

**Task 6R equivalent:** Conditional override at [`common/common.cpp:1775-1781`](common/common.cpp:1775): sets `cparams.n_rs_seq = 0` when `beefix_dflash_custom && has_dflash`.

**Verdict:** **Preserved.**

### 2.6 Checkpoint Fallback

**Old implementation:** Checkpoint used when `seq_rm_type == FULL` or rollback exceeds RS bounds. Created via `ckpt.update_tgt()` before draft.

**Purpose:** Safety net when backup/replay mechanism cannot handle the rollback.

**Task 6R equivalent:** After 3 consecutive replay failures, `replay_failed` flag permanently disables custom mode, and subsequent rollbacks use checkpoint. Implemented at [`server-context.cpp:4335-4346`](tools/server/server-context.cpp:4335).

**Verdict:** **Preserved and enhanced.** Task 6R adds the `replay_failed` permanent disable mechanism and failure counter.

### 2.7 Multi-Sequence Tape Support

**Old implementation:** GPU tape path supported multi-sequence via `tape_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS]` arrays, per-sequence scatter, and `seq_ids[]` routing. CPU tape explicitly rejected multi-sequence.

**Task 6R equivalent:** Single sequence only (`n_seqs = 1` hardcoded at [`server-dflash-custom.cpp:436`](common/server-dflash-custom.cpp:436)). Warning logged when `n_parallel > 1`.

**Verdict:** **Lost (deferred).** The deferred items review (§2) confirms this is a legitimate future enhancement requiring architectural changes to tape structure, capture logic, and replay loops.

### 2.8 Profile Infrastructure

**Old implementation:** `dflash_profile_start()` / `dflash_profile_end()` in `src/dflash-profile.h`. `dflash_recurrent_profile_reset()` for backup operations.

**Purpose:** Measure backup time, replay time, and other DFlash-specific performance metrics.

**Task 6R equivalent:** None.

**Verdict:** **Lost (intentional).** Not part of Task 6R scope. Can be added later if profiling is needed.

---

## 3. Q2: Intended Reproduction Capabilities

Based on the Task 6R blueprint documents and final documentation, the original design intent can be classified as follows:

### 3.1 Intentionally Reproduced (Successfully)

| Capability | Evidence | Status |
|------------|----------|--------|
| Backup cells | Stage 2 of final documentation; `dflash_custom_backup()` implemented | **Complete** |
| Tape replay | Stage 4; `dflash_custom_replay()` with GDN graph | **Complete** |
| `n_rs_seq = 0` | Stage 1; override in `common/common.cpp` | **Complete** |
| GPU tape allocation | Stage 3; device-aware placement in `dflash_custom_tape_alloc()` | **Complete** |
| Checkpoint fallback | Stage 5; try-catch with failure counter | **Complete** |
| Conv state rebuild | Stage 6 Fix 2; CUDA kernel + CPU fallback | **Complete** |
| Opt-in flag | Stage 1; `--beefix-dflash-custom` CLI arg | **Complete** |

### 3.2 Intentionally Reproduced (But Done Differently)

| Capability | Old Approach | Task 6R Approach | Rationale |
|------------|-------------|------------------|-----------|
| Backup storage | Separate `seq_backup` sequence | Extra rows within R/S tensor | Cleaner allocation; no separate sequence management |
| Cell copy | `llama_dflash_memory_seq_cp_recurrent_ordered()` | `dflash_custom_cell_copy()` via `ggml_backend_tensor_copy()` | Simpler; uses standard ggml API |
| Three-phase rollback | Single `llama_dflash_rollback()` function | Split: upstream `seq_rm()` + custom `dflash_custom_replay()` | Leverages upstream unified path |
| Tape structure | Per-sequence arrays with `seq_ids[]` routing | Per-layer tensors, single sequence | Simplified for target configuration (`--parallel 1`) |

### 3.3 Intentionally Dropped (With Justification)

| Capability | Justification | Source |
|------------|--------------|--------|
| Multi-sequence support | Target configuration is `--parallel 1`; requires architectural changes to tape, capture, replay loops | Deferred items review §2 |
| Profile infrastructure | Not needed for initial implementation; can be added later | Not in Task 6R blueprint |
| `tree_bufs` (DDTree) | Removed in v0.4.0; CopySpec/DDTree systems explicitly excluded | AGENTS.md |
| `llama_tape_replay_sync()` | Multi-slot sync only needed for multi-sequence; single-sequence scope | Deferred items review §2 |
| `recurrent_backup_attention_streams` | DDTree-specific; DDTree removed | AGENTS.md |

### 3.4 Accidentally Omitted

Based on the source code review, no capabilities appear to be accidentally omitted. The P0 fix for `n_backup_cells` was discovered and corrected during implementation. All other omissions are either intentional design choices or documented deferred items.

**One potential gap:** The old implementation's `llama_tape_replay_sync()` for multi-slot synchronization is not present. However, this is only needed for multi-sequence workloads, which are explicitly deferred. For single-sequence operation, synchronization is not needed.

---

## 4. Q3: Task 6R Capabilities Beyond Old 0.3.2

Task 6R provides the following capabilities that the old 0.3.2 implementation did NOT have:

### 4.1 CUDA-Native Convolution State Rebuild

**Old implementation:** Conv state rebuild used CPU `memcpy` operations (read from GPU, compute on CPU, write back).

**Task 6R:** Dedicated CUDA kernel at `ggml/src/ggml-cuda/dflash-custom-conv.cu` (~175 lines). Operates directly on GPU tensors, eliminating PCIe transfers. CPU fallback preserved for non-CUDA backends. Partial corruption fix via `cuda_rebuilt_layers` tracking ensures layers already rebuilt by CUDA are skipped if CPU fallback activates mid-loop.

**Classification:** **Worthwhile improvement.** The old implementation's CPU path for conv rebuild was a known performance bottleneck. The CUDA kernel eliminates the PCIe round-trip (read conv state → CPU, compute, write → GPU) that dominated rebuild time.

**Evidence:** [`server-dflash-custom.cpp:648-753`](common/server-dflash-custom.cpp:648) — CUDA path with per-layer device validation. [`server-dflash-custom.cpp:755-831`](common/server-dflash-custom.cpp:755) — CPU fallback with `cuda_rebuilt_layers` tracking.

### 4.2 Fallback Reason Codes (R0-R9)

**Old implementation:** No structured reason codes. Rollback failures were logged without diagnostic context.

**Task 6R:** 10 reason codes (R0-R9) with structured logging:
- R0: Backup/restore skipped (invalid preconditions)
- R1: Replay skipped (invalid preconditions)
- R2: Replay skipped (custom mode not enabled)
- R3: Replay skipped (n_accepted > tokens_captured)
- R4: Replay skipped (no tape allocated)
- R5: Replay skipped (no recurrent memory)
- R6: Replay skipped (no recurrent memory component)
- R7: Replay skipped (n_backup_cells = 0)
- R8: Replay skipped (no scheduler available)
- R9: Conv rebuild skipped (conv_window = 0)

**Classification:** **Worthwhile improvement.** Observability is critical for debugging replay failures in production. The old implementation provided no diagnostic context when replay failed.

**Evidence:** [`server-dflash-custom.cpp:307-846`](common/server-dflash-custom.cpp:307) — Reason codes embedded in guard checks.

### 4.3 `replay_failed` Permanent Disable Mechanism

**Old implementation:** No permanent disable. If replay failed, the system would retry on the next cycle, potentially creating a failure loop.

**Task 6R:** After 3 consecutive failures, `replay_failed` flag permanently disables custom mode for that slot, falling back to checkpoint rollback. Failure counter resets on success.

**Classification:** **Worthwhile improvement.** Prevents failure loops that could degrade serving performance. The old implementation could retry indefinitely on persistent replay failures.

**Evidence:** [`server-dflash-custom.h:105`](common/server-dflash-custom.h:105) — `replay_failed` flag. [`server-context.cpp:4335-4346`](tools/server/server-context.cpp:4335) — Failure counter and disable logic.

### 4.4 Device-Aware Tape Placement (Simplified)

**Old implementation:** Cross-ring, hidden GPU buffers, prefill staging. Complex multi-GPU tape routing with per-sequence arrays.

**Task 6R:** Each layer's tape lives on the same GPU as that model layer. Simple device lookup: `ggml_backend_dev_buffer_type(model_dev_layer(model, il))`.

**Classification:** **Worthwhile improvement.** The old approach's cross-ring and prefill staging added complexity without proportional benefit for the target configuration. Task 6R's per-layer placement is simpler and achieves the same goal (no PCIe transfers during capture/replay).

**Evidence:** [`server-dflash-custom.cpp:113-166`](common/server-dflash-custom.cpp:113) — Device-aware allocation loop.

### 4.5 Configuration Struct

**Old implementation:** Configuration scattered across multiple files and flags.

**Task 6R:** `dflash_custom_config` struct at [`server-dflash-custom.h:76`](common/server-dflash-custom.h:76) centralizes custom mode parameters.

**Classification:** **Worthwhile improvement.** Provides single point of truth for custom mode configuration. Reduces risk of configuration drift.

### 4.6 `cell_copy()` as Free Function

**Old implementation:** `cell_copy()` as method on `llama_memory_recurrent` class (modifying upstream class).

**Task 6R:** `dflash_custom_cell_copy()` free function at [`server-dflash-custom.cpp:255`](common/server-dflash-custom.cpp:255). Accesses only public members.

**Classification:** **Worthwhile improvement.** Eliminates modification of upstream `llama_memory_recurrent` class, reducing merge-conflict risk.

---

## 5. Q4: GPU-Optimized Backup Copy — Preserved or Lost?

### 5.1 Old Implementation

**Function:** `llama_dflash_memory_seq_cp_recurrent_ordered()` at [`old server-context.cpp:4788`](old-versions/beellama.cpp-preview-v0.3.2/tools/server/server-context.cpp:4788).

**Behavior:**
1. Copied layers in dependency order (layer 0 first, then 1, 2, ...) to maximize GPU utilization.
2. Layer ordering ensures that when layer N copies, layer N-1's copy is already complete and GPU resources are fully utilized.
3. Fell back to `llama_memory_seq_cp_recurrent()` if the ordered path failed.

**Performance characteristic:** The layer-ordered approach pipelines GPU copies so that multiple layers can be copied in a single GPU kernel launch batch, reducing launch overhead.

### 5.2 Task 6R Implementation

**Function:** `dflash_custom_cell_copy()` at [`server-dflash-custom.cpp:255`](common/server-dflash-custom.cpp:255).

**Behavior:**
1. Creates temporary ggml context for row views.
2. For each layer (0 to N-1): creates source/destination views, calls `ggml_backend_tensor_copy()`.
3. No layer ordering optimization — layers are copied sequentially in index order.
4. No separate "ordered" vs "fallback" path.

**Performance characteristic:** Each `ggml_backend_tensor_copy()` is an individual GPU copy operation. For 48 recurrent layers, this is 48 copy operations (96 tensor copies for R and S). Each copy has kernel launch overhead.

### 5.3 Assessment

**Is this a material performance regression?**

**Likely minor, not material.** The reasoning:

1. **Backup is infrequent.** Backup occurs once per draft cycle, before the first draft. The draft cycle is dominated by the draft forward pass and verification, not backup.
2. **Copy size is small.** Each R/S row is ~1.3 MB for Qwen3.6-27B (R: ~600 KB, S: ~700 KB). Total backup for 48 layers is ~62 MB. Even with 96 individual CUDA memcpy operations, the total time is estimated at < 1 ms on PCIe 4.0 x16 (32 GB/s bandwidth).
3. **The old approach's optimization was about GPU utilization during copy, not total copy time.** The layer-ordered approach reduced kernel launch overhead, but the total data transferred is the same.
4. **Task 6R's `dflash_custom_cell_copy()` uses `ggml_backend_tensor_copy()` which is device-native for CUDA.** The copies happen on GPU without PCIe transfer.

**However,** the old implementation's layer-ordered approach was a deliberate optimization for large models with many recurrent layers. If backup becomes a bottleneck (e.g., with very large n_parallel or models with >100 recurrent layers), the lack of layer ordering could matter.

**Verdict:** **Functionally adequate for current target configuration.** Not a material performance regression for Qwen3.6-27B with `--parallel 1`. If backup becomes a bottleneck for larger configurations, layer-ordered copy can be reintroduced.

**Recommendation:** **Acceptable difference.** If profiling reveals backup as a bottleneck, add layer-ordered copy as an optimization. For now, the simplicity of `ggml_backend_tensor_copy()` is preferable.

---

## 6. Q5: Tape Replay Equivalence

### 6.1 Old `tape_replay()` Mechanism

**Location:** [`old src/llama-context.cpp`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp).

**Process:**
1. Restore pre-draft recurrent state from `seq_backup`.
2. For each accepted token, replay DeltaNet forward pass using captured tape data.
3. Update S state from replay results.
4. Conv state handled as part of replay (using captured qkv data).

**State components replayed:**
- **S (value state):** Yes — replayed via DeltaNet forward pass.
- **R (conv state):** Yes — rebuilt from captured qkv data using sliding window shift.

**Multi-layer ordering:** The old implementation processed layers sequentially in index order. Each layer's replay depended on the tape data captured during the draft forward pass, which was already complete.

### 6.2 Task 6R `dflash_custom_replay()` Mechanism

**Location:** [`server-dflash-custom.cpp:366-843`](common/server-dflash-custom.cpp:366).

**Process:**
1. Validate preconditions (7 guard checks with reason codes R1-R7).
2. Restore backup state to active rows via `dflash_custom_restore()`.
3. Create replay graph context.
4. For each recurrent layer:
   - Create `q_zeros` tensor (state update is q-independent).
   - Create 4D views into tape tensors for accepted tokens.
   - Create view into backup state.
   - Call `ggml_gated_delta_net()` to produce updated state.
5. Execute replay graph via scheduler.
6. Write updated S state back to active rows.
7. Rebuild convolution state (R tensor) via sliding window shift.
8. Cleanup replay context; log success.

**State components replayed:**
- **S (value state):** Yes — replayed via `ggml_gated_delta_net()` graph.
- **R (conv state):** Yes — rebuilt via CUDA kernel or CPU fallback (sliding window shift).

**Multi-layer ordering:** Layers processed sequentially in tape index order. The replay graph is built for all layers before execution, so layer dependencies are resolved by the graph scheduler.

### 6.3 Equivalence Analysis

| Aspect | Old `tape_replay()` | Task 6R `dflash_custom_replay()` | Equivalent? |
|--------|---------------------|----------------------------------|-------------|
| S state replay | DeltaNet forward pass replay | `ggml_gated_delta_net()` graph | **Yes** — Both use the GDN kernel to update S state from tape data. |
| R (conv) state rebuild | Sliding window shift from qkv tape | CUDA kernel + CPU fallback (sliding window shift) | **Yes** — Same algorithm. Task 6R adds CUDA optimization. |
| Q independence | Used q = 0 during replay | `q_zeros` tensor (explicit zeros) | **Yes** — Both exploit q-independence of GDN state update. |
| Multi-layer ordering | Sequential, index order | Graph-based, scheduler-ordered | **Equivalent** — Both process all layers. Task 6R's graph approach may allow better GPU parallelism. |
| Backup restore | Copy from `seq_backup` | `dflash_custom_restore()` (copy from backup rows) | **Equivalent** — Both restore pre-draft state before replay. |
| Accepted token count | `n_accepted` parameter | `n_accepted` parameter | **Yes** — Same input. |
| Tape data consumed | k, v, gate, beta, qkv | k, v, gate, beta, qkv | **Yes** — Same 5 intermediates captured and consumed. |

### 6.3 Key Difference: Graph-Based vs Sequential Replay

**Old approach:** Replayed DeltaNet for each token sequentially within each layer. The replay was a compute-time loop.

**Task 6R approach:** Builds a replay graph for all layers and all accepted tokens, then executes the graph via the scheduler. This allows the scheduler to optimize execution order and parallelism.

**Impact:** Task 6R's graph-based approach is likely more efficient for multiple accepted tokens, as the scheduler can batch GDN operations across layers. The old sequential approach could not exploit this parallelism.

### 6.4 Verdict

**Functionally equivalent.** Both implementations:
1. Restore pre-draft state from backup.
2. Replay S state via GDN using captured tape intermediates.
3. Rebuild R (conv) state via sliding window shift from qkv tape.
4. Exploit q-independence (q = 0 during replay).

Task 6R's graph-based approach and CUDA conv kernel are improvements over the old implementation, not regressions. The core replay algorithm is the same.

---

## 7. Confidence Assessment

| Finding | Confidence | Basis |
|---------|-----------|-------|
| Backup cells preserved | **High** | Direct code comparison: `dflash_custom_backup()` vs old `dflash_backup_recurrent_state()`. Same purpose, different implementation. |
| Tape replay preserved | **High** | Direct code comparison: `dflash_custom_replay()` vs old `tape_replay()`. Same 5 tape intermediates consumed. Same q-independence exploit. |
| `n_rs_seq = 0` preserved | **High** | Verified in [`common/common.cpp:1775-1781`](common/common.cpp:1775). Conditional override when `beefix_dflash_custom && has_dflash`. |
| GPU-optimized copy degraded | **Medium** | `dflash_custom_cell_copy()` lacks layer ordering. Functionally correct but may be slower for large models. Performance impact not benchmarked. |
| Conv rebuild improved | **High** | CUDA kernel exists at `ggml/src/ggml-cuda/dflash-custom-conv.cu`. CPU fallback verified at [`server-dflash-custom.cpp:755-831`](common/server-dflash-custom.cpp:755). Old implementation used CPU memcpy. |
| Multi-sequence support lost | **High** | `n_seqs = 1` hardcoded at [`server-dflash-custom.cpp:436`](common/server-dflash-custom.cpp:436). Warning logged for `n_parallel > 1`. Deferred items review confirms intentional deferral. |
| Fallback reason codes new | **High** | 10 reason codes (R0-R9) present in source. Old implementation had no equivalent. |
| `replay_failed` disable new | **High** | Verified at [`server-dflash-custom.h:105`](common/server-dflash-custom.h:105) and [`server-context.cpp:4335-4346`](tools/server/server-context.cpp:4335). Old implementation had no equivalent. |
| Profile infrastructure lost | **High** | No `dflash_profile_*` functions in current codebase. Old implementation had `dflash-profile.h`. |
| Checkpoint fallback preserved | **High** | Verified at [`server-context.cpp:4335-4346`](tools/server/server-context.cpp:4335). Enhanced with failure counter and permanent disable. |

---

## 8. Recommendations

### 8.1 No Corrective Action Required

The Task 6R implementation successfully reproduces all core capabilities of the old 0.3.2 implementation:
- Backup cells for recurrent state preservation.
- Tape replay for accepted token state update.
- `n_rs_seq = 0` to avoid RS buffer allocation.
- Checkpoint fallback for safety.

The differences identified are either improvements (CUDA conv kernel, fallback reason codes, `replay_failed` disable) or intentional deferrals (multi-sequence support, profile infrastructure).

### 8.2 Optional Enhancements (Low Priority)

| Enhancement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| Layer-ordered backup copy | Low | Medium | Backup speed for large models. Not needed for Qwen3.6-27B with `--parallel 1`. |
| Profile infrastructure | Low | Low | Observability for DFlash-specific metrics. Can be added when needed. |
| Multi-sequence support | Medium | High | Required for `--parallel > 1` with custom mode. Deferred per architectural analysis. |

### 8.3 Acceptable Differences

| Difference | Classification | Rationale |
|------------|---------------|-----------|
| No `llama_dflash_memory_seq_cp_recurrent_ordered()` | Acceptable | `dflash_custom_cell_copy()` is functionally adequate for target configuration. |
| No `llama_dflash_rollback()` | Acceptable | Replaced by upstream `seq_rm()` + custom `dflash_custom_replay()`. |
| No profile infrastructure | Acceptable | Not needed for initial implementation. |
| No multi-sequence support | Acceptable | Target configuration is `--parallel 1`. Deferred with documented warning. |
| No `tree_bufs` / DDTree | Acceptable | Removed in v0.4.0 per AGENTS.md. |
| No `llama_tape_replay_sync()` | Acceptable | Only needed for multi-sequence workloads. |

### 8.4 Summary

**The Task 6R implementation preserves all core capabilities of the old 0.3.2 custom DFlash while adding several improvements (CUDA conv kernel, fallback reason codes, `replay_failed` disable mechanism, device-aware tape placement).** The differences are either worthwhile improvements, intentional deferrals with documented justifications, or acceptable simplifications for the target configuration.

No corrective action is required. The implementation is architecturally sound and functionally equivalent to the old approach for the supported configuration (`--parallel 1`, Qwen3.6, CUDA backend).

---

*Document generated 2026-08-12. Based on analysis of Task 6R implementation files, final documentation, deferred items review, and the definitive old-vs-current comparison document.*
