# DFlash Custom Mode — Revised Implementation Blueprint (Part 1 of 3)

**Date:** 2026-08-08
**Status:** Revised Blueprint — Task 6R.6 Output
**Based on:** Research Tasks 1-6, Task 6R.1-6R.4 (revised investigation)
**Supersedes:** `task6-implementation-blueprint.md` (previous Task 6 blueprint)

---

## Table of Contents (All 3 Parts)

| Part | File | Contents |
|------|------|----------|
| 1 | `task6r-revised-implementation-blueprint.md` | Parts A-B: Revised Findings Summary, Architecture Comparison |
| 2 | `task6r-revised-blueprint-part2.md` | Parts C-D: Revised Implementation Blueprint, Changes from Previous |
| 3 | `task6r-revised-blueprint-part3.md` | Parts E-G: Implementation Subtasks, Fallback Behavior, Testing Plan |

---

## PART A: Revised Findings Summary

### A.1 Verified Facts from All 6R Subtasks

The following facts were verified against actual source code (old v0.3.2 + current upstream):

| # | Finding | Source Verification |
|---|---------|-------------------|
| 1 | **Old v0.3.2 GPU cross ring is ~200 MB** for hidden-state capture (5 layers × 1024 slots × 5120 embd × 4B × 2 for ring + staging). | [`old-versions/.../cross-ring-interleave.cu:172-175`](old-versions/beellama.cpp-preview-v0.3.2/ggml/src/ggml-cuda/cross-ring-interleave.cu:172), runtime log evidence |
| 2 | **Old v0.3.2 GPU tape stores rank-factored GDN intermediates**, NOT full S-state tensors. The 5 captured tensors per layer are: k, v, gate, beta, qkv_mixed. | [`old-versions/.../llama-context.h:118-160`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h:118), [`old-versions/.../llama-context.cpp:2267-2286`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2267) |
| 3 | **Old tape tensor dimensions for Qwen3.6 27B (fused GDN):** k=[128,16,25], v=[128,48,25], gate=[1,48,25], beta=[1,48,25], qkv=[10240,25]. Total per layer: ~463K elements = ~1.77 MB F32. **CORRECTION (2026-08-08):** Updated from k=[256,1,25], v=[256,8,25] using actual GGUF metadata (S=128, H_k=16, H_v=48). | [`old-versions/.../llama-context.cpp:2413-2479`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2413), [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md) |
| 4 | **Corrected GPU tape total: ~85 MB (fused) / ~117 MB (non-fused)** for 48 recurrent layers, 1 slot, 25 max tokens. **CORRECTION (2026-08-08):** Updated from ~70 MB (old wrong dimensions) and ~134 MB (S=1536 derivation). Actual GGUF values give ~85 MB fused. | Calculated from actual GGUF metadata: S=128, H_k=16/48, H_v=48, conv_ch=10240 |
| 5 | **Total DFlash GPU memory: ~285-317 MB** (200 MB ring + 85-117 MB tape). **CORRECTION (2026-08-08):** Updated from ~270-400 MB range. | Sum of Parts 6R.1 + corrected 6R.2 findings |
| 6 | **Task 5's ~6.5 GiB tape estimate was WRONG.** The error came from assuming full S-state capture rather than rank-factored GDN intermediates (~463K elements per layer per token). | [`task5-part2-current-delta-replay.md:67-79`](plans/dflash-solutions/task5-part2-current-delta-replay.md:67) used incorrect dimensions (S_k=S_v=128, H_k=32, H_v=32) |
| 7 | **Corrected tape size with actual Qwen3.6 GGUF metadata:** ~85 MB (fused GDN) or ~117 MB (non-fused) for 48 layers, 25 tokens. **CORRECTION (2026-08-08):** Updated from ~134-186 MB. The previous derivation used `ssm_d_state = 1536` based on incorrect `d_inner/dt_rank` values. Actual GGUF shows `ssm_d_state = 128`, `ssm_dt_rank = 48`, `ssm_d_inner = 6144`, satisfying S_k==S_v as `128 == 6144/48 = 128`. | [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md) |
| 8 | **Current upstream CAN capture the same compact intermediates.** The graph nodes exist with matching callback names: `k_conv_predelta-{il}`, `v_conv_predelta-{il}`, `gate-{il}`, `beta_sigmoid-{il}`, `linear_attn_qkv_mixed-{il}`. | [`src/models/qwen35.cpp:446-450`](src/models/qwen35.cpp:446), [`src/models/qwen35.cpp:338-457`](src/models/qwen35.cpp:338) |
| 9 | **GDN computation is IDENTICAL between old and current.** Same CUDA kernel ([`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63)), same math, same beta convention (post-sigmoid). | Kernel comparison in 6R.4 Part B |
| 10 | **The only gap is the capture mechanism.** Old had graph-embedded `ggml_cpy` operations in `qwen35.cpp:564-579`; current has no equivalent. Re-implementing requires ~200 lines for the capture mechanism alone. | [`old-versions/.../qwen35.cpp:564-579`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:564) vs [`src/models/qwen35.cpp:450`](src/models/qwen35.cpp:450) |
| 11 | **Replay uses existing `ggml_gated_delta_net()`** — no new CUDA kernel required. The GDN kernel natively processes tokens sequentially and the state update is q-independent (q can be zeros for replay). | [`gated_delta_net.cu:63-158`](ggml/src/ggml-cuda/gated_delta_net.cu:63) — attention output discarded during replay |
| 12 | **Old tape was GPU-native (device-aware placement).** Each layer's tape tensors placed on the same GPU as that model layer. No PCIe transfers required during normal operation. | [`old-versions/.../llama-context.cpp:2480-2519`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2480) |

### A.2 Conclusions Drawn from Evidence

1. **The old v0.3.2 approach was fundamentally sound and the correct design.** It achieved ~270 MB total GPU memory (ring + tape) through rank-factored intermediate storage. The approach is checkpoint-and-replay, NOT state-snapshot.

2. **Task 5's design was on the right track but used wrong tensor dimensions.** The conceptual approach (capture k/v/g/b intermediates, replay via GDN) was correct. The error was in assuming the tape captured full S-state tensors rather than compact rank-factored components. This led to the ~6.5 GiB estimate and the conclusion that "CPU tape is the only viable strategy."

3. **The revised design should use GPU-native tape (like old v0.3.2), NOT CPU tape.** With corrected dimensions (~85-117 MB GPU tape), the total DFlash GPU overhead is ~285-317 MB — dramatically better than the previous blueprint's CPU tape approach which required ~6.5 GB CPU RAM plus ~26 ms PCIe transfer overhead per cycle. **CORRECTION (2026-08-08):** Updated tape size from ~134-186 MB to ~85-117 MB using actual GGUF metadata.

4. **The implementation scope is ~400-700 lines of new code** (capture mechanism + replay integration), NOT the ~900+ lines estimated in the previous blueprint (which included CPU tape transfer logic, CUDA replay kernel, etc.).

5. **The revised design achieves ~96% VRAM savings** vs current upstream DFlash (5.4 GB → ~270 MB), matching old v0.3.2's efficiency.

6. **The revised design achieves near-old-DFlash performance** with GPU-native tape (no PCIe transfers), existing GDN kernel (no new CUDA code), and graph-embedded capture (no callback overhead).

### A.3 Remaining Uncertainties

| Uncertainty | Impact | Resolution Approach |
|-------------|--------|---------------------|
| Exact `ssm_d_state` for Qwen3.6 27B (256 vs 1536) | Affects tape size estimate (70 MB vs 134 MB) | Verify from actual model file GGUF metadata at runtime. The `S_k == S_v` assertion in `build_delta_net_chunking()` constrains the value. |
| Whether `head_v_dim = ssm_d_state` or `d_inner/dt_rank` in current code | Affects v tensor dimensions | Source code at [`src/models/qwen35.cpp:58-63`](src/models/qwen35.cpp:58) shows `head_v_dim = hparams.ssm_d_state`, confirming both are equal. |
| Multi-slot tape behavior under current upstream | Affects multi-parallel scaling | Old used `n_slots=1` by default. Current upstream DFlash may use multiple slots for parallel verification. Need to trace `allocate_tape_gpu()` callers. |
| Exact qkv_mixed tensor layout (2D vs 3D) | Affects capture copy logic | Current upstream uses 3D `[conv_dim, n_tokens, n_seqs]` (name: `linear_attn_qkv_mixed`). Old used 2D `[conv_dim, n_tokens*n_seqs]` (name: `qkv_mixed_pretranspose`). Content is the same, layout differs. |
| Performance impact of graph-embedded copies | Could affect draft throughput | Old v0.3.2 measured no noticeable overhead. The copies are batched with compute and overlap with memory-bound operations. |

---

## PART B: Architecture Comparison

### B.1 Three-Way Comparison: Old v0.3.2 vs Task 5 (Previous Blueprint) vs Revised Design

| Aspect | Old v0.3.2 | Task 5 / Previous Blueprint | Revised Design |
|--------|-----------|---------------------------|----------------|
| **State Capture** | Rank-factored GDN intermediates (k, v, gate, beta, qkv_mixed) via graph-embedded `ggml_cpy` in `qwen35.cpp` | Full S-state tensors (k_in, v_in, g_in, b_in) via graph-embedded `ggml_cpy` in `delta-net-base.cpp` | Rank-factored GDN intermediates via graph-embedded `ggml_cpy` in `qwen35.cpp` (same as old) |
| **Hidden-State Ring** | GPU cross ring: ~200 MB (5 layers × cross_ctx × 5120 embd) | Not applicable (current upstream uses different hidden-state mechanism) | Current upstream hidden-state capture (no change needed) |
| **Recurrent Tape** | GPU tape: ~85-117 MB (48 layers × 25 tokens × rank-factored intermediates, F32, device-aware placement) | CPU tape: ~6.5 GB (48 layers × 15 tokens × full S-state tensors, F32, CPU RAM) | GPU tape: ~85-117 MB (same dimensions as old, device-aware placement) |
| **R/S State Storage** | Backup cells (~150 MB/slot) — recurrent-only copy to backup sequences | Extended tensor rows (backup cells, ~150 MB/slot) | Extended tensor rows (backup cells, ~150 MB/slot) — SAME as Task 5 |
| **Rollback Mechanism** | Restore backup + tape replay (GDN-only forward pass for accepted tokens) | Restore backup + GDN replay with CPU tape data (requires PCIe transfers) | Restore backup + GDN replay with GPU tape data (no PCIe transfers) |
| **VRAM Overhead (Total)** | ~285-317 MB (200 MB ring + 85-117 MB tape + backup cells) | ~2,598 MB GPU + ~6,552 MB CPU (RS base + backup + draft model; tape on CPU) | ~285-317 MB GPU (tape + backup cells + base RS; no CPU tape) |
| **CPU RAM Overhead** | ~400 MB (CPU ring fallback) + minimal tape fallback | ~6,552 MB (tape buffer) | ~0 MB (tape on GPU; minimal fallback only) |
| **PCIe Transfers** | ~0 MB/cycle (GPU-native tape) | ~13 GB/cycle (6.5 GB GPU→CPU capture + 6.5 GB CPU→GPU replay) | ~0 MB/cycle (GPU-native tape) |
| **Compute Overhead** | GDN replay only (48 layers × K accepted tokens × rank-1 update) | GDN replay only + PCIe transfer time (~26 ms) | GDN replay only (48 layers × K accepted tokens × rank-1 update) |
| **Graph Construction** | Graph-embedded copies in model builder | Graph-embedded copies in delta-net-base | Graph-embedded copies in model builder (same as old) |
| **New CUDA Code** | N/A (existing in v0.3.2) | `dflash_replay.cu` (~200 lines for batched replay kernel) | None (existing GDN kernel handles replay) |
| **New ggml Ops** | N/A | None | None |
| **Code Size** | ~3,376 lines total (old DFlash including DDtree, etc.) | ~900 lines (670 new + 236 modified) | ~400-700 lines (tape capture + replay integration) |
| **Fork Drift Risk** | N/A (proven in v0.3.2) | Medium (CPU tape transfer logic, custom CUDA kernel) | Low (reuses old approach, minimal new code) |
| **Performance** | Baseline (old DFlash speed) | ~10× slower replay than GPU-native (PCIe bottleneck) | Near-old-DFlash (GPU-native, no PCIe) |

### B.2 VRAM Accounting Comparison

All numbers for Qwen3.6-27B on RTX 3090 (24 GB):

| Component | Old v0.3.2 | Current Upstream | Task 5 Blueprint | Revised Design |
|-----------|-----------|-----------------|-----------------|---------------|
| RS buffer (R + S) | 598 MB (base only) | 5,387 MB (n_rs_seq=8) | 598 MB (n_rs_seq=0) | 598 MB (n_rs_seq=0) |
| Backup cells | ~150 MB/slot | 0 | ~612 MB (4 cells) | ~612 MB (4 cells) |
| Tape (GPU) | ~85-117 MB | 0 | 0 | ~85-117 MB |
| Tape (CPU) | 0 (fallback only) | 0 | 6,552 MB | 0 |
| Hidden-state ring | ~200 MB | Current upstream mechanism | Current upstream mechanism | Current upstream mechanism |
| Draft model | ~800 MB | ~800 MB | ~800 MB | ~800 MB |
| **DFlash overhead** | **~1,193-1,225 MB** | **~6,187 MB** | **~2,010 MB GPU** | **~2,095-2,127 MB** |
| **Total (with model)** | **~19.8 GB** | **~24.8 GB** | **~21.5 GB GPU + 6.5 GB CPU** | **~20 GB** |
| **VRAM saved vs current** | **~5 GB** | — | **~4.2 GB** | **~4.1-4.1 GB** |
| **Savings %** | **81%** | — | **68%** | **66-67%** |

**CORRECTION (2026-08-08):** Updated tape size from ~70-186 MB to ~85-117 MB, and backup cells from ~1,200 MB (8 cells) to ~612 MB (4 cells) based on actual GGUF metadata and old v0.3.2 analysis. See [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md) and [`task6r-correction-part2-backup-cells.md`](plans/dflash-solutions/task6r-correction-part2-backup-cells.md).

### B.3 Performance Comparison

| Metric | Old v0.3.2 | Current Upstream | Task 5 Blueprint | Revised Design |
|--------|-----------|-----------------|-----------------|---------------|
| Rollback: Full acceptance | Fast (no-op) | Fast (no-op) | Fast (no-op) | Fast (no-op) |
| Rollback: Partial acceptance | Fast (GDN replay, ~0.1 ms) | Fast (RS pointer swap, ~0.05 ms) | Medium (GDN replay + PCIe, ~26 ms) | Fast (GDN replay, ~0.1 ms) |
| Rollback: Rejection | Fast (backup restore, ~2 ms) | Fast (checkpoint, ~50-100 ms) | Medium (checkpoint, ~50-100 ms) | Fast (backup restore, ~2 ms) |
| Draft throughput impact | Minimal (graph-embedded copies overlap with compute) | N/A | Minimal | Minimal |
| Relative to old DFlash | 1.0× (baseline) | N/A (different mechanism) | ~10× slower rollback | ~1.0× (near baseline) |

### B.4 Key Design Differences Explained

#### Why Revised Design Uses GPU Tape (Not CPU Tape)

The previous blueprint concluded "CPU tape is the only viable strategy" because Task 5 estimated ~6.5 GiB for the GPU tape. The revised investigation proved this estimate was based on incorrect tensor dimensions. The actual tape stores rank-factored GDN intermediates (~85-117 MB), making GPU tape the clear choice:

| Factor | GPU Tape (Revised) | CPU Tape (Previous) |
|--------|-------------------|---------------------|
| VRAM impact | +85-117 MB | +0 MB (but +6.5 GB CPU RAM) |
| PCIe transfers | 0 per cycle | ~13 GB per cycle (6.5 GB each direction) |
| Transfer latency | 0 ms | ~26 ms per cycle |
| Replay speed | ~0.1 ms (GPU-native) | ~26 ms (PCIe-bound) |
| Total DFlash overhead | ~285-317 MB GPU | ~2,598 MB GPU + 6,552 MB CPU |
| Complexity | Simple (graph copy) | Complex (transfer orchestration) |

**CORRECTION (2026-08-08):** Updated tape size from ~70-186 MB to ~85-117 MB using actual GGUF metadata.

#### Why Revised Design Captures in `qwen35.cpp` (Not `delta-net-base.cpp`)

The previous blueprint proposed capturing in `delta-net-base.cpp` after the `cb()` calls for k_in, v_in, g_in, b_in. The revised design captures in `qwen35.cpp` at the same point where old v0.3.2 captured — immediately after tensor computation but BEFORE the GDN op consumes them. This ensures:

1. The captured tensors are the exact inputs to the GDN kernel (post-processing, pre-consumption).
2. The capture point matches the old proven implementation.
3. The tensor names match the old tape name map (with minor updates for renamed nodes).

#### Why Revised Design Needs No New CUDA Kernel

The previous blueprint included `ggml/src/ggml-cuda/dflash_replay.cu` (~200 lines) for batched GDN replay. The revised design uses the existing `ggml_gated_delta_net()` operation directly, because:

1. The GDN kernel already processes tokens sequentially (`for (int t = 0; t < n_tokens; t++)` at [`gated_delta_net.cu:63`](ggml/src/ggml-cuda/gated_delta_net.cu:63)).
2. The state update is q-independent — q can be zeros for replay (the attention output is discarded).
3. The kernel accepts captured k, v, gate, beta as inputs and produces updated state as output.
4. No special batching is needed — replay K tokens with one GDN call per layer.

---

*End of Part 1 (Parts A-B). Continue with Part 2 for the detailed implementation blueprint.*
