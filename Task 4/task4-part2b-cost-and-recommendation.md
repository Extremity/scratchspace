# Task 4.2b: Backup Cell Allocation — Cost Analysis and Recommendation

*(Continuation of [`task4-part2-allocation-and-rollback.md`](task4-part2-allocation-and-rollback.md))*

---

## SECTION 5: VRAM COST ANALYSIS

### 5.1 Recurrent Memory VRAM per Cell

From [`src/llama-memory-recurrent.cpp:99-105`](src/llama-memory-recurrent.cpp:99):
```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

For Qwen3.6 (typical DFlash target):
- `n_embd_r()` = model hidden size (e.g., 4096 for 7B, 5120 for 14B)
- `n_embd_s()` = same as `n_embd_r()` for Mamba-style models
- `type_r = type_s = GGML_TYPE_F32` (4 bytes per element)
- `n_rs_seq` = depends on `params.speculative.need_n_rs_seq()`

**Per-cell VRAM formula:**
```
per_cell_bytes = (1 + n_rs_seq) * n_recurrent_layers * (n_embd_r + n_embd_s) * 4
```

**Example — Qwen3.6 7B with ~16 recurrent layers, n_embd=4096:**
```
per_cell_bytes = (1 + n_rs_seq) * 16 * (4096 + 4096) * 4
               = (1 + n_rs_seq) * 16 * 8192 * 4
               = (1 + n_rs_seq) * 524,288 bytes
               = (1 + n_rs_seq) * 0.5 MiB per cell
```

For `n_rs_seq = 8` (typical for DFlash with `n_max = 8`):
```
per_cell = 9 * 0.5 MiB = 4.5 MiB
```

**Cost of doubling cells (4 → 8, n_parallel = 4):**
```
Additional cells = 4
Additional VRAM = 4 * 4.5 MiB = 18 MiB
```

This is modest for Qwen3.6 7B. For larger models:

**Example — Qwen3.6 32B with ~40 recurrent layers, n_embd=5120:**
```
per_cell_bytes = (1 + 8) * 40 * (5120 + 5120) * 4
               = 9 * 40 * 10240 * 4
               = 9 * 1.6 MiB = 14.4 MiB
Additional VRAM for 4 backup cells = 57.6 MiB
```

Still modest compared to total model VRAM (tens of GB).

### 5.2 Comparison with Current Checkpoint Approach

The current checkpoint approach saves the **entire context state** (KV cache + recurrent + sampler) to CPU memory before verification. For a typical DFlash session:

| Approach | Memory Overhead | Restore Speed |
|----------|----------------|---------------|
| Current checkpoint | Full context clone (GB range) | Full state restore (slow) |
| Current rs_idx | `(1 + n_rs_seq)` tensor rows per cell | Graph-native (fast) |
| **Proposed backup cells** | `n_parallel` extra cells (~tens of MB) | Direct tensor copy (fast) |

The backup cell approach is:
- **Lower memory** than checkpoint (tens of MB vs GB)
- **Faster restore** than checkpoint (GPU tensor copy vs full state load)
- **Unlimited rollback depth** (unlike rs_idx which is bounded by `n_rs_seq`)

### 5.3 Impact on Non-DFlash Models

The cell count increase would only affect models with recurrent memory (hybrid architectures). Pure attention models (Llama, Mistral, etc.) use `llama_kv_cache` which is unaffected.

For non-hybrid recurrent-only models (Mamba, RWKV), the same `llama_memory_recurrent` constructor is used. Doubling cells for all recurrent models would add overhead even when speculative decoding is not used. A conditional approach (only double when DFlash is active) is preferred.

---

## SECTION 6: IMPLEMENTATION RECOMMENDATION

### 6.1 Preferred Approach: Static Extra Cells with New `cell_copy` API

Based on the analysis, the recommended implementation is:

**Phase 1 — Infrastructure (minimal code changes):**

1. **Extend cell allocation** in [`src/llama-model.cpp`](src/llama-model.cpp):
   - Detect DFlash mode and pass `mem_size = n_seq_max * 2` to `llama_memory_recurrent`
   - Enlarge `rs_idx` to match (or make `rs_idx` sized by `size` instead of `n_seq_max`)

2. **Add `cell_copy` method** to [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp):
   ```cpp
   void llama_memory_recurrent::cell_copy(uint32_t src_cell, uint32_t dst_cell) {
       // Copy R/S tensor rows using ggml_backend_tensor_set
       // Copy cell metadata (pos, src, seq_id set)
   }
   ```

3. **Expose through public API** in [`include/llama.h`](include/llama.h):
   ```cpp
   LLAMA_API void llama_memory_cell_copy(llama_context * ctx,
                                          llama_seq_id src_seq, llama_seq_id dst_seq);
   ```

**Phase 2 — Server integration:**

4. **Modify [`tools/server/server-context.cpp`](tools/server/server-context.cpp):**
   - Before verification: copy working cell to backup
   - After rejection: copy backup cell to working (instead of rs_idx rollback or checkpoint)

5. **Add server config option:**
   - `--spec-backup-cells on/off` to enable the feature
   - Default to `off` to preserve current behavior

### 6.2 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| VRAM increase for non-DFlash models | Medium | Only double cells when DFlash active |
| `rs_idx` bounds violation | High | Enlarge `rs_idx` or skip backup seq_ids |
| `seq_cp` metadata-only limitation | High | Use `cell_copy` with direct tensor access |
| Graph builder src/src0 confusion | Medium | Ensure backup cells have valid `src` values |
| Sampler array size mismatch | Medium | Enlarge `samplers` in `common.cpp` |

### 6.3 Alternative Approaches Considered and Rejected

**Approach A — Dynamic cell expansion:**
- Rejected: No `resize()` method exists; would require significant restructuring.
- Static allocation is simpler and more predictable.

**Approach B — Per-token R/S snapshots in separate buffer:**
- Rejected: Would require separate tensor allocation and management outside `llama_memory_recurrent`.
- Backup cells reuse existing tensor infrastructure.

**Approach C — Extend `n_rs_seq` to cover draft window:**
- Rejected: `n_rs_seq` directly multiplies tensor rows. For `n_max = 32`, this would add 32× the recurrent VRAM.
- Backup cells add a fixed 2× overhead regardless of draft window.

**Approach D — Use existing `state_write`/`state_read` for backup:**
- Rejected: These are internal-only APIs with no public exposure. The serialization format includes metadata that would need to match exactly.
- Direct tensor copy (`cell_copy`) is cleaner and faster.

### 6.4 Summary

| Aspect | Finding |
|--------|---------|
| Cell allocation | Fixed at construction, can be doubled |
| VRAM cost | ~2× recurrent memory (typically < 100 MB total) |
| Rollback extension point | [`server-context.cpp:4225`](tools/server/server-context.cpp:4225) — between checkpoint and rs_idx paths |
| New API needed | `cell_copy()` for immediate R/S data transfer |
| DFlash algorithm impact | None — rollback is orthogonal to draft/verify logic |
| Server functions to modify | 5 functions across 3 files |
| Risk level | Medium — requires careful bounds checking |

### 6.5 Files Summary

| File | Role in Backup Cell Approach |
|------|-------|------|
| [`src/llama-memory-recurrent.h`](src/llama-memory-recurrent.h) | Header — add `cell_copy()` declaration |
| [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp) | Implementation — add `cell_copy()`, enlarge `rs_idx` |
| [`src/llama-model.cpp`](src/llama-model.cpp) | Model init — pass doubled `mem_size` for DFlash |
| [`common/common.cpp`](common/common.cpp) | Context params — enlarge `samplers` array |
| [`include/llama.h`](include/llama.h) | Public API — expose `llama_memory_cell_copy` |
| [`tools/server/server-context.cpp`](tools/server/server-context.cpp) | Server — add backup copy before/after verification |
| [`tools/server/server-task.h`](tools/server/server-task.h) | Server — checkpoint decision logic (reference only) |
| [`common/speculative.cpp`](common/speculative.cpp) | DFlash impl — no changes needed |

---

*End of Task 4.2 analysis.*
