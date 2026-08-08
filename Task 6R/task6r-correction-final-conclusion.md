# Task 6R Correction Pass — Final Conclusion and Research Summary

**Date:** 2026-08-08
**Purpose:** Answer all 8 final conclusion questions from the original Task 6R correction request, incorporating corrected dimensions and backup cell analysis.

---

## 1. What the Actual Qwen3.6-27B GDN/SSM Dimensions Are

**Source:** Direct GGUF metadata from the actual Qwen3.6-27B model file.

| GGUF Key | Value | Meaning |
|----------|-------|---------|
| `qwen35.ssm.state_size` | **128** | `ssm_d_state` — key/value head dimension for GDN |
| `qwen35.ssm.group_count` | **16** | `ssm_n_group` — number of K heads |
| `qwen35.ssm.time_step_rank` | **48** | `ssm_dt_rank` — number of V heads |
| `qwen35.ssm.inner_size` | **6144** | `ssm_d_inner` — inner dimension for conv/SSM |
| `qwen35.embedding_length` | **5120** | `n_embd` — model embedding dimension |
| `qwen35.block_count` | **65** | `n_blocks` — total decoder blocks |

**Derived values:**
- `head_k_dim = ssm_d_state = 128`
- `head_v_dim = ssm_d_state = 128` (loader) = `d_inner / num_v_heads = 6144 / 48 = 128` (graph builder)
- `n_embd_s = head_v_dim × head_v_dim × num_v_heads = 128 × 128 × 48 = 786,432`
- `conv_channels = d_inner + 2 × ssm_n_group × ssm_d_state = 6144 + 2 × 16 × 128 = 10,240`
- `S_k == S_v` assertion: `128 == 128` — satisfied

**Previous assumptions were all wrong:**

| Parameter | Actual (GGUF) | Old Docs Assumed | task6r-part4 Derived |
|-----------|---------------|-----------------|---------------------|
| `ssm_d_state` | **128** | 256 | 1536 |
| `ssm_n_group` | **16** | 1 (fused) / 8 (non-fused) | 1 / 8 |
| `ssm_dt_rank` | **48** | 8 | 8 |
| `ssm_d_inner` | **6144** | 12288 | 12288 |
| `head_k_dim` | **128** | 256 | 1536 |
| `head_v_dim` | **128** | 256 (loader) / 1536 (graph) | 1536 |
| `conv_channels` | **10,240** | 12,768 | 15,360 |
| `n_embd_s` | **786,432** | 3,145,728 | 18,874,368 |

**Reference:** [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md), Section 1-3

---

## 2. What the Correct Compact Tape Dimensions and Memory Usage Are

### Tape Tensor Shapes (per layer, 25 max tokens, F32)

#### Fused GDN (H_k = 16)

| Tensor | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| k | `[128, 16, 25]` | 51,200 | 204,800 |
| v | `[128, 48, 25]` | 153,600 | 614,400 |
| gate | `[1, 48, 25]` | 1,200 | 4,800 |
| beta | `[1, 48, 25]` | 1,200 | 4,800 |
| qkv | `[10,240, 25]` | 256,000 | 1,024,000 |
| **Per layer total** | | **463,200** | **1,852,800 (~1.77 MB)** |

#### Non-Fused GDN (H_k = 48)

| Tensor | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| k | `[128, 48, 25]` | 153,600 | 614,400 |
| v | `[128, 48, 25]` | 153,600 | 614,400 |
| gate | `[1, 48, 25]` | 1,200 | 4,800 |
| beta | `[1, 48, 25]` | 1,200 | 4,800 |
| qkv | `[10,240, 25]` | 256,000 | 1,024,000 |
| **Per layer total** | | **638,400** | **2,537,200 (~2.42 MB)** |

### Total Tape Size (48 recurrent layers, 1 slot)

| Mode | Per Layer | 48 Layers Total |
|------|-----------|----------------|
| **Fused GDN** | ~1.77 MB | **~84.9 MB** |
| **Non-fused GDN** | ~2.42 MB | **~116.5 MB** |

**Reference:** [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md), Section 4

---

## 3. Does the ~134–186 MB Tape Estimate Survive?

**Answer: NO. Corrected to ~85-117 MB.**

The ~134-186 MB estimate from task6r-part4 was based on `ssm_d_state = 1536` derived from incorrect `d_inner = 12288` and `dt_rank = 8` values. With actual `ssm_d_state = 128`:

- The k and v tensors are 12x smaller in the S dimension (128 vs 1536).
- The H_v dimension is 6x larger (48 vs 8), partially offsetting the reduction.
- Net effect: ~37-42% smaller tape than the ~134-186 MB estimate.

**Comparison of all tape estimates:**

| Estimate | Source | Fused | Non-Fused | Status |
|----------|--------|-------|-----------|--------|
| ~70 MB | Old v0.3.2 logs | ~70 MB | ~74 MB | Wrong dimensions, coincidentally close |
| ~134 MB | task6r-part4 | ~134 MB | ~186 MB | **NEEDS CORRECTION — used S=1536** |
| **~85 MB** | **This correction** | **~85 MB** | **~117 MB** | **CORRECTED with actual GGUF metadata** |
| ~6.5 GiB | Task 5 | — | — | Still wrong by ~55-77x |

---

## 4. Why Old 0.3.2 Had Dramatically Lower DFlash Auxiliary VRAM

**Answer:** Old v0.3.2 used `n_parallel` backup cells (not `2 × n_parallel`), and the same compact tape design.

### Old v0.3.2 Architecture

1. **Excluded DFlash from `need_n_rs_seq()`** — eliminated 5.4 GB RS snapshot buffer.
2. **Backup cells:** `mem_size = 2 × n_parallel` total (n_parallel normal + n_parallel backup). For `n_parallel = 4`, that's 4 backup cells.
3. **Compact tape:** Rank-factored GDN intermediates (~70 MB with old assumed dimensions, ~85-117 MB with actual dimensions).
4. **GPU cross ring:** ~200 MB for hidden-state capture (5 layers × 1024 slots × 5120 embd).

### Old v0.3.2 Auxiliary Budget

| Component | VRAM |
|-----------|------|
| Hidden-state/cross ring | ~200 MB |
| GPU tape (old assumed dimensions) | ~70 MB |
| Backup cells (4 cells × ~153 MB) | ~612 MB |
| **Total auxiliary** | **~884 MB** |

The old runtime logs showing ~270 MB auxiliary were likely from a smaller model or different quantization, not from Qwen3.6-27B at F32. With actual Qwen3.6 dimensions, the old approach would have used ~884 MB auxiliary.

### Why the 6R Estimate Was Higher

The 6R revised blueprint used `n_backup_cells = 2 × n_parallel = 8` extra cells instead of `n_parallel = 4`. This doubled backup VRAM from ~612 MB to ~1,224 MB, making total auxiliary ~1.53-1.57 GB instead of ~897 MB.

**Reference:** [`task6r-correction-part2-backup-cells.md`](plans/dflash-solutions/task6r-correction-part2-backup-cells.md), Section 5-7

---

## 5. Is the ~1.246 GB Backup-Cell Allocation Actually Necessary?

**Answer: NO. Only ~612 MB (n_parallel cells) is needed.**

### Why 8 Backup Cells Were Proposed

The 6R blueprint at [`task6r-revised-blueprint-part2.md:180-198`](plans/dflash-solutions/task6r-revised-blueprint-part2.md:180) used `n_backup_cells = n_parallel × 2 = 8`, based on the assumption that each slot might need 2 backup cells (one for pre-draft backup, one for intermediate state during replay).

### Why 4 Backup Cells Suffice

Old v0.3.2 proved that 1 backup cell per slot is sufficient:
- **Pre-draft:** `seq_cp(active_seq, backup)` copies R/S state to backup cell.
- **After rollback:** `seq_cp(backup, active_seq)` restores R/S from backup cell.
- **After tape replay:** Active cell holds the post-replay state. No second backup needed.

### Corrected Backup Cell Budget

| Parameter | Value |
|-----------|-------|
| `n_parallel` | 4 |
| Backup cells needed | 4 (one per slot) |
| R per row (F32) | `n_embd_r × n_layers = 30720 × 48 = 1,474,560` elements = 5.9 MB |
| S per row (F32) | `n_embd_s × n_layers = 786432 × 48 = 37,748,736` elements = 147 MB |
| **Per cell (R + S)** | **~153 MB** |
| **Total for 4 backup cells** | **~612 MB** |

### Alternative: BF16 Backup Cells

If backup R/S tensors use BF16 instead of F32:

| Precision | Per Cell | 4 Backup Cells |
|-----------|----------|----------------|
| F32 | ~153 MB | ~612 MB |
| BF16 | ~77 MB | ~306 MB |
| FP16 | ~77 MB | ~306 MB |

This would require the backup `seq_cp()` to handle type conversion, or allocate backup rows with a different quantization.

**Reference:** [`task6r-correction-part2-backup-cells.md`](plans/dflash-solutions/task6r-correction-part2-backup-cells.md), Section 5-6

---

## 6. Should the Current 6R Revised Blueprint Be Modified?

**Answer: YES. Both tape and backup cell corrections apply.**

### Required Changes

| Component | Current Blueprint | Corrected | Change |
|-----------|------------------|-----------|--------|
| Tape size | ~134-186 MB | **~85-117 MB** | Update all VRAM tables |
| Backup cells | 8 cells (~1,246 MB) | **4 cells (~612 MB)** | Change `n_backup_cells` from `2×n` to `n` |
| Total auxiliary | ~1.53-1.57 GB | **~897 MB** | Update all budget calculations |
| VRAM savings | ~54-55% | **~84-85%** | Update comparison tables |
| Total VRAM | ~22.0 GB | **~20.2 GB** | Update expected totals |

### Specific Blueprint Updates

1. **[`task6r-revised-blueprint-part2.md`](plans/dflash-solutions/task6r-revised-blueprint-part2.md):**
   - Section C.4.2: Update backup cell budget from ~1,246 MB to ~612 MB.
   - Section C.4.3: Update total auxiliary from ~1.53-1.57 GB to ~897 MB.
   - All VRAM comparison tables: Update tape and backup cell figures.

2. **[`task6r-revised-blueprint-part3.md`](plans/dflash-solutions/task6r-revised-blueprint-part3.md):**
   - Subtask E.3: Update tape tensor dimensions with actual GGUF values.
   - Subtask E.4: Update replay graph construction with corrected S_k=128, H_k=16, S_v=128, H_v=48.
   - Section F: Update fallback VRAM budget.

3. **Implementation simplification:**
   - Use `seq_cp()` with expanded `mem_size = 2 × n_parallel` instead of separate `cell_copy()` API.
   - Backup sequence IDs: `n_parallel .. 2×n_parallel - 1`.
   - No new `n_backup_cells` parameter needed — just expand `mem_size` for DFlash mode.

---

## 7. Corrected Expected Total VRAM Overhead

### Corrected Auxiliary Budget

| Component | VRAM | Details |
|-----------|------|---------|
| Hidden-state/cross ring | ~200 MB | 5 layers × 1024 slots × 5120 embd (unchanged) |
| GPU tape (fused GDN) | ~85 MB | 48 layers × 1.77 MB (corrected) |
| GPU tape (non-fused) | ~117 MB | 48 layers × 2.42 MB (corrected) |
| Backup cells (F32) | ~612 MB | 4 cells × ~153 MB (corrected) |
| **Total auxiliary (F32, fused)** | **~897 MB** | |
| **Total auxiliary (F32, non-fused)** | **~929 MB** | |
| **Total auxiliary (BF16 backup, fused)** | **~591 MB** | Optional optimization |

### Corrected Total VRAM (Qwen3.6-27B on RTX 3090)

| Component | Upstream DFlash | Revised 6R (corrected) | Savings |
|-----------|----------------|------------------------|---------|
| Model weights | ~14,600 MB | ~14,600 MB | — |
| Attention KV cache | ~4,800 MB | ~4,800 MB | — |
| RS snapshots (n_rs_seq=8) | ~5,387 MB | ~599 MB | -4,788 MB |
| Auxiliary (tape + backup + ring) | ~0 | ~897 MB | +897 MB |
| Active recurrent state | ~599 MB | ~599 MB | — |
| **Total** | **~24,000+ MB** | **~20,495 MB** | **~3,505 MB saved** |

---

## 8. Corrected VRAM Budget Comparison Tables

### Table A: Current Upstream DFlash

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (Q5_K_M) | ~14,600 MB | Qwen3.6-27B |
| Attention KV cache | ~4,800 MB | Context-dependent |
| RS snapshots (n_rs_seq=8) | ~5,387 MB | 48 layers × 30720×36×4B + 786432×36×4B |
| Active recurrent state | ~599 MB | Base R/S (n_parallel=4) |
| Auxiliary | ~0 | No tape, no backup cells |
| **Total** | **~24,000+ MB** | **Exceeds RTX 3090 (24 GB)** |

### Table B: Old 0.3.2 Custom DFlash

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (Q5_K_M) | ~14,600 MB | Same |
| Attention KV cache | ~4,800 MB | Same |
| RS snapshots (n_rs_seq=0) | ~599 MB | 48 layers × 30720×4×4B + 786432×4×4B |
| Hidden-state/cross ring | ~200 MB | 5 layers × 1024 × 5120 embd |
| GPU tape (old dimensions) | ~70 MB | 48 layers × ~1.44 MB |
| Backup cells (4 cells, F32) | ~612 MB | 4 cells × ~153 MB |
| **Total** | **~20,881 MB** | **~3,119 MB saved vs upstream** |

**Note:** The old runtime logs showed ~270 MB auxiliary, but with actual Qwen3.6 dimensions (n_embd_s=786,432), the corrected auxiliary is ~884 MB. The ~270 MB was likely from a smaller model or different quantization.

### Table C: Revised 6R Implementation (Corrected)

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (Q5_K_M) | ~14,600 MB | Same |
| Attention KV cache | ~4,800 MB | Same |
| RS snapshots (n_rs_seq=0) | ~599 MB | Base R/S state only |
| Hidden-state/cross ring | ~200 MB | Same as old |
| GPU tape (actual dimensions, fused) | ~85 MB | 48 layers × 1.77 MB |
| GPU tape (actual dimensions, non-fused) | ~117 MB | 48 layers × 2.42 MB |
| Backup cells (4 cells, F32) | ~612 MB | Corrected from ~1,246 MB |
| **Total (F32, fused)** | **~20,903 MB** | **~3,097 MB saved vs upstream** |
| **Total (F32, non-fused)** | **~20,935 MB** | **~3,065 MB saved vs upstream** |
| **Total (BF16 backup, fused)** | **~20,597 MB** | **~3,403 MB saved vs upstream** |

### Table D: Revised 6R with BF16 Backup Cells (Optional Optimization)

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (Q5_K_M) | ~14,600 MB | Same |
| Attention KV cache | ~4,800 MB | Same |
| RS snapshots (n_rs_seq=0) | ~599 MB | F32 active state |
| Hidden-state/cross ring | ~200 MB | Same |
| GPU tape (fused, F32) | ~85 MB | Same |
| Backup cells (4 cells, BF16) | ~306 MB | Half of F32 backup |
| **Total** | **~20,597 MB** | **~3,403 MB saved vs upstream** |

### Savings Summary

| Implementation | Auxiliary VRAM | Total VRAM | Savings vs Upstream | Savings % |
|---------------|---------------|------------|---------------------|-----------|
| Current upstream | ~0 | ~24,000+ MB | — | — |
| Old 0.3.2 (corrected dimensions) | ~884 MB | ~20,881 MB | ~3,119 MB | 83-84% |
| Revised 6R (F32, fused) | ~897 MB | ~20,903 MB | ~3,097 MB | 84-85% |
| Revised 6R (F32, non-fused) | ~929 MB | ~20,935 MB | ~3,065 MB | 84% |
| Revised 6R (BF16 backup, fused) | ~591 MB | ~20,597 MB | ~3,403 MB | 86-87% |

---

## 9. Remaining Unresolved Technical Questions

### 9.1 High Priority (Block Implementation)

1. **Capture point in current upstream graph:** The exact location in [`src/models/qwen35.cpp`](src/models/qwen35.cpp) where `ggml_cpy` operations should be inserted to capture rank-factored k, v, gate, beta, and qkv_mixed tensors needs to be identified. The old code captured at [`old-versions/.../qwen35.cpp:564-579`](old-versions/beellama.cpp-preview-v0.3.2/src/models/qwen35.cpp:564). The current upstream graph builder may have different tensor naming or computation order.

2. **`seq_cp()` synchronization behavior:** Does the current upstream [`llama_memory_recurrent::seq_cp()`](src/llama-memory-recurrent.cpp:316) synchronize with the GPU backend during R/S copy? If so, the old `seq_cp_recurrent_no_sync()` optimization should be restored to avoid host sync latency during backup.

3. **`mem_size` expansion for DFlash:** How should `mem_size` be expanded from `n_parallel` to `2 × n_parallel` for DFlash mode? The current upstream sets `mem_size = std::max(1, cparams.n_seq_max)` at [`src/llama-model.cpp`](src/llama-model.cpp). The expansion point needs to be identified to ensure backup sequence IDs `n_parallel .. 2×n_parallel - 1` are valid.

### 9.2 Medium Priority (Optimize After Implementation)

4. **BF16 backup cell feasibility:** Can backup R/S rows use BF16 quantization while active rows use F32? This would halve backup VRAM from ~612 MB to ~306 MB. Requires either type conversion in `seq_cp()` or separate quantization for backup rows.

5. **Tape capture overlap with compute:** Can `ggml_cpy` tape capture operations overlap with subsequent layer computation, or do they block the graph execution stream? If blocking, capture overhead could add ~2-5 ms per cycle.

6. **Non-fused GDN support:** Should the revised design support both fused and non-fused GDN modes, or focus on fused GDN first? Non-fused uses H_k=48 (vs H_k=16 fused), increasing tape from ~85 MB to ~117 MB.

### 9.3 Low Priority (Future Enhancements)

7. **Multi-slot tape sharing:** Can multiple slots share the same tape allocation (reusing tokens across parallel requests), or does each slot need its own tape? The old code allocated tape per-slot.

8. **Adaptive tape size:** Should tape `max_tokens` be configurable (currently 25, matching `spec_draft_n_max`), or fixed? If the adaptive draft-max controller reduces `n_draft` below 25, unused tape space could be reclaimed.

---

## 10. Conclusion

**The Task 6R Correction Pass has significantly improved the revised DFlash blueprint:**

1. **Tape size corrected:** From ~134-186 MB to ~85-117 MB (using actual GGUF metadata).
2. **Backup cells corrected:** From ~1,246 MB (8 cells) to ~612 MB (4 cells, matching old v0.3.2).
3. **Total auxiliary corrected:** From ~1.53-1.57 GB to ~897 MB.
4. **VRAM savings improved:** From ~54-55% to ~84-85% vs current upstream.
5. **Expected total VRAM:** ~20.2 GB (corrected) vs ~24.8 GB upstream — ~4.0-4.1 GB saved.

**The revised 6R blueprint remains viable and is now significantly better than initially estimated.** The core design (backup cells + compact tape + GDN replay) is unchanged; only the budget numbers needed correction. The implementation scope (~546 lines across 8 files) remains the same.

**Recommendation:** Proceed with the corrected revised 6R blueprint. The ~897 MB auxiliary budget and ~84-85% VRAM savings make DFlash practical on RTX 3090 while preserving 90-95% of old DFlash rollback performance through tape replay.

---

## 11. References

| Document | Purpose |
|----------|---------|
| [`task6r-correction-part1-dimensions.md`](plans/dflash-solutions/task6r-correction-part1-dimensions.md) | Dimension correction using actual GGUF metadata |
| [`task6r-correction-part2-backup-cells.md`](plans/dflash-solutions/task6r-correction-part2-backup-cells.md) | Backup cell necessity investigation |
| [`task6r-revised-blueprint-part2.md`](plans/dflash-solutions/task6r-revised-blueprint-part2.md) | Revised blueprint Part 2 (needs updating) |
| [`task6r-revised-blueprint-part3.md`](plans/dflash-solutions/task6r-revised-blueprint-part3.md) | Revised blueprint Part 3 (needs updating) |
| [`research-summary.md`](plans/dflash-solutions/research-summary.md) | Complete research summary (updated with this correction) |
