# Research Task 6.2 — Source Verification: Recurrent State Representation and Backup Cell Implementation

**Date:** 2026-08-08
**Status:** Verification Complete

---

## 1. Recurrent State Representation

### 1.1 R State Tensor — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:99-105`](src/llama-memory-recurrent.cpp:99)

**Claim:** R state: `n_embd_r × n_rows` at line 99.

**VERIFIED — Exact match.**

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
ggml_format_name(r, "cache_r_l%d", i);
ggml_format_name(s, "cache_s_l%d", i);
r_l[i] = r;
s_l[i] = s;
```

**Tensor details:**

| Property | Value | Source |
|----------|-------|--------|
| Name | `cache_r_l{il}` | [`llama-memory-recurrent.cpp:102`](src/llama-memory-recurrent.cpp:102) |
| Shape | `{hparams.n_embd_r(), n_rows}` (2D) | [`llama-memory-recurrent.cpp:100`](src/llama-memory-recurrent.cpp:100) |
| Datatype | `type_r` (always `GGML_TYPE_F32`) | [`llama-memory-recurrent.cpp:22`](src/llama-memory-recurrent.cpp:22) |
| `n_rows` formula | `mem_size * (1 + n_rs_seq)` | [`llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) |
| Storage | `std::vector<ggml_tensor *> r_l` per layer | [`llama-memory-recurrent.h:118`](src/llama-memory-recurrent.h:118) |

### 1.2 S State Tensor — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:99-105`](src/llama-memory-recurrent.cpp:99)

**Claim:** S state: `n_embd_s × n_rows` at line 99.

**VERIFIED — Exact match.**

| Property | Value | Source |
|----------|-------|--------|
| Name | `cache_s_l{il}` | [`llama-memory-recurrent.cpp:103`](src/llama-memory-recurrent.cpp:103) |
| Shape | `{hparams.n_embd_s(), n_rows}` (2D) | [`llama-memory-recurrent.cpp:101`](src/llama-memory-recurrent.cpp:101) |
| Datatype | `type_s` (always `GGML_TYPE_F32`) | [`llama-memory-recurrent.cpp:23`](src/llama-memory-recurrent.cpp:23) |
| `n_rows` formula | `mem_size * (1 + n_rs_seq)` | [`llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99) |
| Storage | `std::vector<ggml_tensor *> s_l` per layer | [`llama-memory-recurrent.h:119`](src/llama-memory-recurrent.h:119) |

### 1.3 `mem_size` and `n_rs_seq` Mapping to Tensor Dimensions — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99)

The tensor has `n_rows = mem_size * (1 + n_rs_seq)` rows. Each row corresponds to one cell at one snapshot index.

**Cell structure within the tensor:**

The tensor is organized as groups of `mem_size` rows, where each group represents one snapshot:

```
Row 0          = cell 0, snapshot 0 (active)
Row 1          = cell 1, snapshot 0 (active)
...
Row mem_size-1 = cell mem_size-1, snapshot 0 (active)
Row mem_size   = cell 0, snapshot 1
Row mem_size+1 = cell 1, snapshot 1
...
Row 2*mem_size-1 = cell mem_size-1, snapshot 1
...
Row n_rs_seq * mem_size + cell_idx = cell cell_idx, snapshot n_rs_seq
```

**Cell indexing formula:** `row = snapshot_idx * mem_size + cell_idx`

This is verified by [`s_copy()`](src/llama-memory-recurrent.cpp:1330) (see section 2.1):

```cpp
int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
```

The return value `(idx * mem->size) + src0` confirms: `row = snapshot_idx * mem_size + cell_idx`.

### 1.4 Cell Count with Different `n_rs_seq` Values

| `n_rs_seq` | Snapshot groups | Total rows | Active cells | Snapshot cells |
|------------|----------------|------------|-------------|----------------|
| 0 | 1 (active only) | `mem_size` | `mem_size` | 0 |
| 8 | 9 (1 active + 8 snapshots) | `9 * mem_size` | `mem_size` | `8 * mem_size` |

**VRAM impact for Qwen3.6-27B** (verified from research-summary.md):
- `n_embd_r = 30720`, `n_embd_s = 786432`, 48 recurrent layers
- With `n_rs_seq = 8`: `n_rows = 4 * 9 = 36`
  - R: `30720 * 36 * 4B * 48 = 202.5 MiB`
  - S: `786432 * 36 * 4B * 48 = 5,184 MiB`
  - Total: ~5,386 MiB
- With `n_rs_seq = 0`: `n_rows = 4 * 1 = 4`
  - R: `30720 * 4 * 4B * 48 = 22.5 MiB`
  - S: `786432 * 4 * 4B * 48 = 576 MiB`
  - Total: ~598 MiB

---

## 2. Existing Copy Primitives

### 2.1 `s_copy()` — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:1330-1348`](src/llama-memory-recurrent.cpp:1330)
**Declaration:** [`src/llama-memory-recurrent.h:183`](src/llama-memory-recurrent.h:183)

```cpp
int32_t llama_memory_recurrent_context::s_copy(int i) const;
```

**Claim:** `s_copy()` exists at line 1330.

**VERIFIED — Exact match.**

**What it does:** `s_copy()` is a **row index calculator**, NOT a data copy function. It returns the source row index within the R/S tensor for cell `i + head`, considering the rollback snapshot index.

- When `n_rs_seq == 0`: returns `src0` (the active cell row).
- When `n_rs_seq > 0`: returns `(rs_idx[seq] * mem_size) + src0` (the snapshot cell row).

**Usage:** Called during graph build at [`src/llama-graph.cpp:398`](src/llama-graph.cpp:398) and [`src/llama-graph.cpp:1319`](src/llama-graph.cpp:1319) to populate the `s_copy` input tensor that drives `ggml_get_rows` operations in `build_rs()`.

**Critical finding:** `s_copy()` does NOT copy R/S tensor data. It computes row indices for the graph's `ggml_get_rows` operation, which performs the actual data read during graph execution.

### 2.2 `seq_cp()` — VERIFIED (Metadata Only)

**Location:** [`src/llama-memory-recurrent.cpp:316-351`](src/llama-memory-recurrent.cpp:316)

**Claim:** `seq_cp()` copies metadata only, not R/S data.

**VERIFIED — Exact match.**

```cpp
void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) { p0 = 0; }
    if (p1 < 0) { p1 = std::numeric_limits<llama_pos>::max(); }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            auto & cell_dst = cells[tail_dst.tail];
            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                    cell_dst.pos = -1;
                    cell_dst.src = -1;
                    used -= 1;
                }
            }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];
            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}
```

**What it does:**
- Copies sequence ID membership from source cell to destination cell.
- Updates `tail` pointer in destination to point to source's tail cell.
- Does NOT copy R/S tensor data.
- Does NOT allocate new R/S rows.
- The destination sequence now shares the same physical cell (and thus the same R/S tensor rows) as the source.

**What it does NOT do:**
- No `ggml_cpy`, no `ggml_backend_tensor_copy`, no memory operations on R/S tensors.
- No new tensor rows are created.

### 2.3 `ggml_backend_tensor_copy()` — EXISTS as Public API

**Location:** [`ggml/include/ggml-backend.h:71`](ggml/include/ggml-backend.h:71)
**Implementation:** [`ggml/src/ggml-backend.cpp:477-498`](ggml/src/ggml-backend.cpp:477)

```cpp
GGML_API void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);
```

**Behavior:**
- Requires `ggml_are_same_layout(src, dst)` (same shape and type).
- If source is host memory: uses `ggml_backend_tensor_set()` to write directly to destination.
- If destination is host memory: uses `ggml_backend_tensor_get()` to read from source.
- If same backend: calls `ggml_backend_buffer_copy_tensor()` (device-native copy).
- Otherwise: falls back to slow host-mediated copy (get to host, then set).

**Internal `ggml_backend_buffer_copy_tensor()`** ([`ggml/src/ggml-backend.cpp:205-211`](ggml/src/ggml-backend.cpp:205)):
- Calls `dst_buf->iface.cpy_tensor(dst_buf, src, dst)` if the backend implements `cpy_tensor`.
- CUDA implements this at [`ggml/src/ggml-cuda/ggml-cuda.cu:820`](ggml/src/ggml-cuda/ggml-cuda.cu:820) — uses `cudaMemcpy` for same-GPU copies and `cudaMemcpyPeer` for cross-GPU copies.
- Returns `false` if not implemented, triggering slow host fallback.

**Applicability to R/S tensors:**
- `ggml_backend_tensor_copy()` operates on **entire tensors**, not individual rows.
- To copy individual cells (single rows), you would need to create **views** into the source and destination tensors and copy those views.
- View-to-view copy is supported: `ggml_backend_tensor_copy()` resolves `dst->view_src->buffer` for views ([`ggml/src/ggml-backend.cpp:206`](ggml/src/ggml-backend.cpp:206)).

### 2.4 Recommended `cell_copy()` Implementation Approach

Based on source verification, a `cell_copy()` function needs to copy R and S data for a single cell across all recurrent layers. Two approaches:

**Approach A — Tensor views + `ggml_backend_tensor_copy()` (Recommended):**

For each recurrent layer `il`:
1. Create source view: `ggml_view_1d()` into `r_l[il]` at offset `src_cell * row_stride`
- Create destination view: `ggml_view_1d()` into `r_l[il]` at offset `dst_cell * row_stride`
- Call `ggml_backend_tensor_copy(src_view, dst_view)`
- Repeat for `s_l[il]`

**Pros:** Device-native copy (no host round-trip), uses existing API, handles cross-GPU via `cudaMemcpyPeer`.
**Cons:** Requires allocating temporary view tensors (no allocation cost — views are metadata-only).

**Approach B — Graph-based copy via `ggml_cpy()`:**

For each layer: build `ggml_cpy()` operations into the compute graph.
**Pros:** Integrated with graph execution.
**Cons:** Requires graph context, cannot be done outside inference cycle.

**Verdict:** Approach A is correct. The views are metadata-only (no memory allocation), and `ggml_backend_tensor_copy()` handles same-GPU and cross-GPU efficiently. A new `cell_copy(uint32_t src, uint32_t dst)` method on `llama_memory_recurrent` would iterate over all layers and copy R/S rows using views.

---

## 3. Backup Cell Allocation Strategy

### 3.1 Current Tensor Allocation

**Location:** [`src/llama-memory-recurrent.cpp:99-106`](src/llama-memory-recurrent.cpp:99)

Current allocation formula: `n_rows = mem_size * (1 + n_rs_seq)`

With `n_rs_seq = 0`: `n_rows = mem_size * 1 = mem_size`

The tensor has exactly `mem_size` rows — one per active cell. No extra rows exist for backup cells.

### 3.2 Backup Cell Requirements

Research claims: `n_parallel × 2` backup cells needed.

**Verification:** The `mem_cell` struct at [`llama-memory-recurrent.h:94-113`](src/llama-memory-recurrent.h:94) stores metadata (`pos`, `src`, `src0`, `tail`, `seq_id`). The cells vector is sized to `mem_size`:

```cpp
cells.clear();
cells.resize(mem_size);  // line 38-39
```

Backup cells would need:
1. **Metadata:** Entries in the `cells` vector (or a separate backup cells vector).
2. **R/S data:** Extra rows in the R/S tensors.

### 3.3 Allocation Options

**Option A — Extend existing tensor allocation:**

Modify `n_rows` formula to include backup cells:
```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells;
```

With `n_rs_seq = 0` and `n_backup = n_parallel * 2`:
```cpp
const uint32_t n_rows = mem_size + n_parallel * 2;
```

**Pros:** Single contiguous allocation per layer. Backup rows reside on same backend buffer.
**Cons:** Changes the row indexing formula — existing code assumes `n_rows = mem_size * (1 + n_rs_seq)`.

**Option B — Separate backup tensor:**

Add separate `r_l_backup` and `s_l_backup` tensors:
```cpp
ggml_tensor * r_backup = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_backup_cells);
ggml_tensor * s_backup = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_backup_cells);
```

**Pros:** No changes to existing row indexing. Backup cells are clearly separated.
**Cons:** Additional tensors to manage. `cell_copy()` needs to handle cross-tensor copies.

**Option C — Extend `mem_size` at construction:**

Pass `mem_size + n_backup` to the constructor and treat rows beyond `mem_size` as backup cells.

**Pros:** Minimal code change — the constructor already handles `mem_size * (1 + n_rs_seq)` rows.
**Cons:** The `cells` vector would need to be larger, and `find_slot()` logic needs to exclude backup rows from normal allocation.

### 3.4 Recommended Approach

**Option A with explicit backup offset** is recommended:

1. Store `n_backup_cells` as a member variable.
2. Compute `n_rows = mem_size * (1 + n_rs_seq) + n_backup_cells`.
3. Define `backup_offset = mem_size * (1 + n_rs_seq)` as the starting row for backup cells.
4. Backup cell `b` maps to row `backup_offset + b`.
5. The `cells` vector remains `mem_size` entries (backup cells don't need `mem_cell` metadata since they are ephemeral copies).

**Memory layout:**
```
Rows [0, mem_size-1]:              Active cells (snapshot 0)
Rows [mem_size, 2*mem_size-1]:     Snapshot 1 (if n_rs_seq >= 1)
...
Rows [n_rs_seq*mem_size, ...-1]:   Snapshot n_rs_seq
Rows [..., n_rows-1]:              Backup cells (n_backup_cells rows)
```

**VRAM impact for Qwen3.6-27B** with `n_parallel = 4`, `n_backup = 8`:
- R: `30720 * 8 * 4B * 48 = 45 MiB`
- S: `786432 * 8 * 4B * 48 = 1,152 MiB`
- Total backup overhead: ~1,197 MiB

This is significantly more than the research estimate of ~100 MB. The research may have assumed only R state backup or a smaller S state. Verify: with `n_embd_s = 786432`, S state dominates. Even with just 2 backup cells: S alone is ~295 MiB × 48 layers = ~141 MiB. Re-checking: `786432 * 2 * 4 * 48 = 300,256,512` bytes = 286 MiB. So 2 backup cells = ~288 MiB total. The research estimate of 100 MB may have used fewer backup cells or a smaller model.

**DISCREPANCY:** Research claimed ~100 MB backup cell overhead. Actual calculation shows ~1,197 MiB for `n_parallel × 2 = 8` backup cells. The S state (`n_embd_s = 786432`) is 25× larger than R state. Even with just 2 backup cells: S alone is ~286 MiB.

---

## 4. Recurrent Layer Identification

### 4.1 Arch Support Check — VERIFIED

**Location:** [`src/llama-arch.cpp:967-975`](src/llama-arch.cpp:967)

```cpp
bool llm_arch_supports_rs_rollback(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}
```

**Finding:** Only Qwen3.5 and Qwen3.5 MoE architectures support RS rollback. Qwen3.6 uses the same `LLM_ARCH_QWEN35` arch identifier (verified by [`src/models/qwen35.cpp`](src/models/qwen35.cpp) which handles both Qwen3.5 and Qwen3.6).

### 4.2 Per-Layer Recurrent Flag — VERIFIED

**Location:** [`src/models/qwen35.cpp:21-27`](src/models/qwen35.cpp:21)

```cpp
if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
    }
}
```

**Behavior:**
1. First tries to read `LLM_KV_ATTENTION_RECURRENT_LAYERS` from GGUF metadata.
2. If not present, computes from `full_attn_interval`: layer `i` is recurrent if `(i + 1) % full_attn_interval != 0`.
3. MTP layers (beyond `n_layer()`) are marked non-recurrent.

**For Qwen3.6-27B** (`n_layer = 64`, `full_attn_interval = 4`):
- Recurrent layers: 0,1,2, 4,5,6, 8,9,10, ... (every layer except 3,7,11,15,... i.e. layers where `(i+1) % 4 == 0`)
- Full attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
- Recurrent layer count: `64 - 16 = 48` (matches research claim of 48 recurrent layers)

### 4.3 Layer Filter in Memory Constructor

**Location:** [`src/llama-memory-recurrent.cpp:76-79`](src/llama-memory-recurrent.cpp:76)

```cpp
for (int i = 0; i < n_layer; i++) {
    if (filter && !filter(i)) {
        LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
        continue;
    }
```

The `filter` parameter determines which layers get R/S tensors. Non-recurrent layers skip tensor allocation (`r_l[i] = nullptr`, `s_l[i] = nullptr`).

### 4.4 Recurrent Layer Access Pattern

The `r_l` and `s_l` vectors are sized to `n_layer` (total layers), not just recurrent layers. Non-recurrent layers have `nullptr` entries. Code that accesses these vectors must check for `nullptr`:

```cpp
// From size_r_bytes() at llama-memory-recurrent.cpp:801
for (const auto & r : r_l) {
    if (r != nullptr) {
        size_r_bytes += ggml_nbytes(r);
    }
}
```

---

## 5. Summary of Discrepancies

| Claim | Status | Detail |
|-------|--------|--------|
| R state: `n_embd_r × n_rows` | VERIFIED | [`llama-memory-recurrent.cpp:100`](src/llama-memory-recurrent.cpp:100) |
| S state: `n_embd_s × n_rows` | VERIFIED | [`llama-memory-recurrent.cpp:101`](src/llama-memory-recurrent.cpp:101) |
| `s_copy()` at line 1330 | VERIFIED | Exists and computes row indices |
| `s_copy()` copies data | DISCREPANCY | It computes indices; `ggml_get_rows` in the graph does the actual data read |
| `seq_cp()` copies metadata only | VERIFIED | Only updates `seq_id` and `tail` fields |
| Backup cell overhead ~100 MB | DISCREPANCY | S state dominates: ~286 MB for 2 cells, ~1,197 MB for 8 cells (Qwen3.6-27B) |
| 48 recurrent layers for Qwen3.6 | VERIFIED | 64 total - 16 full-attention = 48 recurrent |

---

## 6. Key Findings for Implementation

1. **`cell_copy()` is required.** No existing primitive copies R/S tensor data between cells. `seq_cp()` only copies metadata.
2. **Use `ggml_backend_tensor_copy()` with views.** Create 1D views into source/destination rows and copy. This is device-native and handles same-GPU efficiently.
3. **Backup cells need extra tensor rows.** The current allocation `mem_size * (1 + n_rs_seq)` does not include backup cells. Extend the formula.
4. **S state dominates backup VRAM.** Each backup cell costs `n_embd_s * element_size * n_recurrent_layers` for S alone. For Qwen3.6: ~144 MiB per backup cell.
5. **Recurrent layers are filter-controlled.** Non-recurrent layers have `nullptr` in `r_l[]/s_l[]`. The `cell_copy()` function must skip `nullptr` entries.
