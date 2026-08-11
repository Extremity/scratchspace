# Task 6R — Targeted Review of Deferred Items

**Date:** 2026-08-11
**Author:** Roo (Code Mode)
**Purpose:** Supplemental technical assessment of three deferred items to justify scope classifications.
**Method:** Source code inspection only. No code modifications. No speculative changes.

---

## 1. Compile-Time `BEE_DFLASH_CUSTOM` Guards

### 1.1 What the guards would do

Wrap custom additions in upstream files with `#ifdef BEE_DFLASH_CUSTOM` / `#endif` blocks, so custom code is absent from the compiled binary when the flag is not defined.

### 1.2 Files requiring modification

| File | Custom Addition | Guard Complexity |
|------|----------------|------------------|
| `src/llama-cparams.h` | Forward declaration (line 11), `n_backup_cells` (line 21), `tape_gpu` (line 96) | Low — simple field guards |
| `src/llama-context.h` | `set_tape_gpu()` method (line 67) | Low — single method guard |
| `src/llama-context.cpp` | `set_tape_gpu()` implementation, `n_backup_cells` validation | Low — guards around existing `#ifdef GGML_CUDA` patterns |
| `src/llama-memory-recurrent.h` | `n_backup_cells`, `backup_offset()`, `mem_size` | Medium — class members and methods |
| `src/llama-memory-recurrent.cpp` | Constructor param, allocation formula, backup logging | Medium — constructor changes require careful guard placement |
| `src/llama-model.cpp` | `n_backup_cells` forwarding (4 call sites) | Medium — call sites scattered across file |
| `src/llama-memory-hybrid.h/cpp` | `n_backup_cells` forwarding | Low — single constructor param |
| `src/llama-memory-hybrid-iswa.h/cpp` | `n_backup_cells` forwarding | Low — single constructor param |
| `src/models/qwen35.cpp` | Capture block (~70 lines at line 460) | Low — already gated by `cparams.tape_gpu != nullptr` |
| `include/llama.h` | `n_backup_cells` in `llama_context_params` | Low — single field |
| `CMakeLists.txt` | Add `BEE_DFLASH_CUSTOM` define | Required — otherwise guards remove all custom code |

### 1.3 Concrete technical assessment

**Runtime benefit:** **None.** The `--beefix-dflash-custom` flag already gates all behavior at runtime. When the flag is not present:
- `n_rs_seq` is NOT overridden to 0 (stock value used)
- `n_backup_cells` is NOT set (defaults to 0, no backup rows allocated)
- `tape_gpu` is NULL (capture block in qwen35.cpp is skipped)
- Server integration code checks `dflash_custom_is_enabled()` which returns false
- No custom code path executes

**Correctness benefit:** **None.** The current implementation is correct with or without compile-time guards.

**VRAM benefit:** **Negligible.** The custom code in upstream files consists primarily of struct fields (a few bytes each) and conditional checks. The compiled code size impact is minimal compared to the overall binary.

**Performance benefit:** **None.** The conditional checks (`tape_gpu != nullptr`, `n_backup_cells > 0`) are branch-predicted and effectively free. They do not add measurable overhead.

**Primary benefit:** **Modularity / merge hygiene only.** Compile-time guards would:
- Produce cleaner upstream diffs (custom fields absent when flag undefined)
- Make custom code disappear from binary symbol tables
- Signal intent more clearly to reviewers

**Risk:** Moderate. Adding `#ifdef` blocks around constructor parameters and class members in files that upstream actively modifies increases the risk of:
- Mismatched `#ifdef`/`#endif` pairs causing build failures
- Guards wrapping the wrong scope (e.g., missing a related field)
- CMake configuration errors on platforms where `BEE_DFLASH_CUSTOM` is not added

### 1.4 Classification

**Legitimate future enhancement.** The guards provide zero runtime, correctness, VRAM, or performance benefit. They are purely a modularity/merge-hygiene improvement. Deferring is reasonable and well-justified.

---

## 2. Multi-Sequence / `--parallel > 1`

### 2.1 What the current implementation assumes

**Exact locations of single-sequence assumptions:**

1. **Replay graph `n_seqs = 1`** — [`server-dflash-custom.cpp:436`](common/server-dflash-custom.cpp:436)
   ```cpp
   const int n_seqs = 1;
   ```
   This is hardcoded. All replay tensor shapes use `n_seqs = 1`:
   - `q_zeros`: `[S_k, H_k, n_accepted, n_seqs=1]` (line 492-493)
   - Tape views: reshape to `[..., max_tokens, 1]` then view `[..., n_accepted, n_seqs=1]` (lines 502-539)
   - Backup state: reshape to `[S_v, S_v, H_v, n_seqs=1]` (lines 551-552)
   - GDN output: `K=1` with `n_seqs=1` (line 559)

2. **Backup state view uses cell 0 only** — [`server-dflash-custom.cpp:546-549`](common/server-dflash-custom.cpp:546)
   ```cpp
   ggml_tensor * s_backup_2d = ggml_view_2d(replay_ctx, mem->s_l[il],
       (int64_t)hp.n_embd_s(), 1,
       mem->s_l[il]->nb[1],
       mem->backup_offset() * mem->s_l[il]->nb[1]);
   ```
   This views backup row `backup_offset() + 0` (cell 0). Cell 1, 2, 3 are not replayed.

3. **State writeback writes to active row 0** — [`server-dflash-custom.cpp:608-611`](common/server-dflash-custom.cpp:608)
   ```cpp
   ggml_tensor * s_dst = ggml_view_2d(replay_ctx, mem->s_l[il],
       (int64_t)hp.n_embd_s(), 1,
       mem->s_l[il]->nb[1],
       0);  // offset 0 = active row 0
   ```
   Updated state is written to row 0 only.

4. **Conv rebuild operates on active row 0** — [`server-dflash-custom.cpp:727-729`](common/server-dflash-custom.cpp:727)
   ```cpp
   const float * old_conv_ptr = static_cast<const float *>(r_tensor->data);
   const float * qkv_ptr = static_cast<const float *>(tl.qkv->data);
   float * dst_ptr = static_cast<float *>(r_tensor->data);
   ```
   Both read and write use `r_tensor->data` (offset 0 = row 0).

### 2.2 What backup/restore does with `n_parallel`

Backup and restore handle ALL `n_parallel` cells:
- [`dflash_custom_backup()`](common/server-dflash-custom.cpp:305): Copies cells `0..n_cells-1` from active to backup rows.
- [`dflash_custom_restore()`](common/server-dflash-custom.cpp:327): Copies cells `0..n_cells-1` from backup to active rows.

The server calls backup with `mem->n_backup_cells` (equal to `n_parallel`):
```cpp
// server-context.cpp:3321
dflash_custom_backup(mem, mem->n_backup_cells);
```

**Result:** All `n_parallel` cells are backed up and restored. But only cell 0 is replayed. Cells 1..N-1 have restored-but-not-replayed state (pre-draft state, not advanced by `n_accepted` tokens).

### 2.3 What would be required to support `n_parallel > 1`

Supporting multiple parallel sequences would require:

1. **Per-cell replay loop:** Instead of replaying only cell 0, loop over `0..n_cells-1`:
   - Each cell has its own backup row
   - Each cell needs its own tape (or tape must store per-cell data)
   - GDN replay must produce output for each cell independently

2. **Tape restructuring:** Current tape captures single-sequence data. Multi-sequence would require:
   - Tape tensors shaped `[dim0, dim1, max_tokens, n_seqs]` instead of `[dim0, dim1, max_tokens]`
   - Capture block in qwen35.cpp would need to capture all sequences, not just the first
   - Replay views would need per-cell offsets

3. **Conv rebuild per cell:** Each cell needs its own conv state rebuilt from its own tape data.

4. **Server integration changes:** The replay call would need to know which cells belong to which slot, and replay each cell independently.

**Assessment:** This is **NOT a small, straightforward change**. It requires:
- Restructuring tape tensors from 3D to 4D
- Restructuring capture logic in qwen35.cpp
- Adding per-cell replay loops with independent tape views
- Restructuring conv rebuild to handle per-cell state
- Modifying server integration to track per-cell acceptance

This is a meaningful architectural change to the backup/replay state management, not a simple parameter adjustment.

### 2.4 Is the current warning sufficient?

**Yes.** The warning at [`server-context.cpp:1473-1475`](tools/server/server-context.cpp:1473):
```cpp
if (params_base.n_parallel > 1) {
    SLT_WRN(slot, "dflash custom mode: n_parallel=%d > 1 — replay assumes single sequence per slot; results may be incorrect for multi-sequence workloads\n",
            params_base.n_parallel);
}
```

This warning:
- Is logged at startup (before any inference begins)
- Uses `SLT_WRN` (warning level, visible in logs)
- States the limitation clearly ("replay assumes single sequence")
- States the consequence ("results may be incorrect")

The backup/restore path remains safe (all cells backed up and restored correctly). Only the replay path is incomplete. If replay fails, checkpoint rollback is the fallback, which is correct regardless of `n_parallel`.

### 2.5 Historical evidence from v0.3.2

**Critical finding from `old-versions/beellama.cpp-preview-v0.3.2/`:**

The old v0.3.2 implementation had extensive multi-sequence support:

- **Per-sequence tape arrays:** `tape_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS]` and `tape_gpu_n_seqs` in [`llama-cparams.h:100-103`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-cparams.h:100)
- **Tape struct with `n_seqs` and `seq_ids[]`:** `dflash_tape_gpu` contained `int n_seqs` and `llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS]` ([`llama-context.h:128-129`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.h:128))
- **Per-sequence QKV scatter:** Replay code scattered QKV tape data per-sequence using `seq_ids` lookups ([`llama-context.cpp:3777-3794`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3777), [`3951-3954`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3951))
- **Multi-sequence verify batching:** `tape_gpu_seqs` populated before each `process_ubatch()` ([`llama-cparams.h:99-103`](old-versions/beellama.cpp-preview-v0.3.2/src/llama-cparams.h:99))

**However, the old v0.3.2 CPU tape fallback explicitly rejected multi-sequence:**
```cpp
// llama-context.cpp:1874-1876
// CPU tape fallback: no multi-seq support
if (n_seqs_unq > 1) {
    return false;
}
```

**Key insight:** The old v0.3.2 GPU tape path supported multi-sequence, but the CPU path did not. The current Task 6R implementation is a simplified redesign that starts with single-sequence support and defers multi-sequence for later. The old implementation's multi-sequence support was complex (per-seq scatter, tape routing, verify batching) and was part of a fundamentally different architecture (cross-ring, hidden GPU buffers, prefill staging) that Task 6R intentionally replaced with the simpler backup-cell-and-tape approach.

This confirms that multi-sequence support in Task 6R is a legitimate future enhancement. The old v0.3.2 architecture had multi-sequence because it was designed around GPU cross-ring and per-sequence tape arrays. Task 6R's backup-cell approach would need equivalent per-cell tape and per-cell replay loops, which is the architectural change identified in §2.3.>>>>>>> REPLACE

### 2.6 Classification

**Legitimate future enhancement that would disproportionately expand current scope.** Multi-sequence replay requires meaningful architectural changes to tape structure, capture logic, replay loops, and conv rebuild. The target configuration is `--parallel 1`, and the warning makes the limitation explicit and safe. Deferring is justified.

---

## 3. Model-Specific Capture / K=1 GDN Limitation

### 3.1 Current capture implementation

The capture block in [`qwen35.cpp:460-529`](src/models/qwen35.cpp:460) captures:
- `k_conv` → tape `k` (key after l2_norm)
- `v_conv` → tape `v` (value after conv/silu)
- `gate` → tape `gate` (pre-exp, after reshape)
- `beta` → tape `beta` (post-sigmoid)
- `qkv_mixed` → tape `qkv` (raw QKV for conv rebuild)

**Source views capture the first sequence only:**
```cpp
// qwen35.cpp:478-480
ggml_tensor * k_src = ggml_view_3d(ctx0, k_conv,
    k_conv->ne[0], k_conv->ne[1], n_seq_tokens,
    k_conv->nb[1], k_conv->nb[2], 0);  // offset 0 = first sequence
```

The source tensors have shape `[dim0, dim1, n_seq_tokens, n_seqs]`. The view captures `[dim0, dim1, n_seq_tokens]` from offset 0, which is the first sequence. This is correct for `n_seqs = 1` (single sequence per slot).

### 3.2 K=1 GDN output

The replay calls `ggml_gated_delta_net()` with `K=1`:
```cpp
// server-dflash-custom.cpp:559
ggml_tensor * gdn_out = ggml_gated_delta_net(replay_ctx,
    q_zeros, k_view, v_view, g_view, b_view, s_backup, /* K= */ 1);
```

**K=1 means the GDN outputs state only** (no RS snapshots). The output layout is:
```
[attention_scores | updated_state]
[S_v*H_v * n_accepted*n_seqs | S_v*S_v*H_v*n_seqs]
```

With `K=1` and `n_seqs=1`:
- Attention scores: `S_v * H_v * n_accepted` elements
- Updated state: `S_v * S_v * H_v` elements

This is correct for replay because:
1. The replay purpose is to rebuild state for accepted tokens, not produce RS snapshots
2. The stock DFlash checkpoint system handles RS snapshots separately
3. K=1 is the minimum K value supported by `ggml_gated_delta_net()`

### 3.3 Internal correctness for K=1 GDN/Qwen path

**Verified correct:**
- Tape tensors capture all 5 GDN intermediates (k, v, gate, beta, qkv)
- Replay graph creates correct 4D views with proper stride calculations
- GDN output state extraction uses correct byte offset (`S_v * H_v * n_accepted * n_seqs * sizeof(float)`)
- State writeback copies to correct active row
- Conv rebuild advances conv state by `n_accepted` tokens

### 3.4 What would change to support other model arrangements

To support non-Qwen SSM models or K>1 output:

1. **Different capture tensors:** Other SSM architectures may have different intermediate names or shapes. The capture block is Qwen-specific (uses `k_conv`, `v_conv`, `gate`, `beta`, `qkv_mixed` from qwen35.cpp).

2. **Different conv formula:** `conv_channels = d_inner + 2 * H_k * S_k` is Qwen-specific. Other architectures may project differently.

3. **K>1 output:** If RS snapshots are needed during replay, K would need to be >1, changing the output layout and state extraction logic.

4. **Virtual capture interface:** A model-agnostic approach would require:
   - Virtual method on model graph builder: `capture_gdn_intermediates(tape, layer_index)`
   - Each model architecture implements its own capture logic
   - Replay remains model-agnostic (uses tape data, doesn't know source)

### 3.5 Is this an intentional scope boundary?

**Yes.** The Task 6R design principles explicitly state "Model-generic: Dimensions derived from runtime hparams; no hardcoded model-specific values." The capture block in qwen35.cpp is the one model-specific component, and it's necessary because different SSM architectures produce different intermediates.

However, the capture block is:
- Gated on `cparams.tape_gpu != nullptr` (strictly opt-in)
- Limited to the graph builder for one model (qwen35.cpp)
- Does not affect other model architectures
- Does not affect stock DFlash behavior

### 3.6 Classification

**Legitimate future enhancement.** The current implementation is internally correct for K=1 GDN/Qwen. Supporting other architectures requires:
- Adding capture blocks to other model graph builders (model-specific work)
- Potentially a virtual capture interface (architectural change)
- Testing with each new architecture

This is out of scope for Task 6R, which targets Qwen3.6 specifically. The implementation is correct for the supported path.

---

## 4. Summary Table

| Deferred Item | Correctness Issue? | Performance Improvement? | Scope Impact | Classification |
|---------------|-------------------|-------------------------|--------------|----------------|
| Compile-time guards | No | No | Moderate (11+ files, CMake changes, build risk) | **Merge hygiene only** — defer |
| Multi-sequence replay | No (warning + safe fallback) | No (not achievable without architectural changes) | High (tape restructure, capture changes, per-cell loops) | **Future enhancement** — defer |
| Model-specific capture | No (correct for K=1 GDN/Qwen) | No (correct for supported path) | Moderate (virtual interface, per-model capture blocks) | **Future enhancement** — defer |

**Conclusion:** All three deferred items are correctly classified. None represent unresolved correctness problems. None represent meaningful performance or functionality improvements that are reasonably achievable within the current Task 6R scope. All three are future enhancements that would require disproportionate effort relative to the current target configuration (`--parallel 1`, Qwen3.6, CUDA backend).

---

*End of Targeted Deferred Items Review.*