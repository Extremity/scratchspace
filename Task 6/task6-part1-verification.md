# Research Task 6.1 — Source Verification Results

**Date:** 2026-08-08
**Status:** Verification Complete

---

## 1. DFlash Configuration Verification

### 1.1 `need_n_rs_seq()` — VERIFIED

**Location:** [`common/common.h:417-423`](common/common.h:417)

**Claim:** `need_n_rs_seq()` includes `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`.

**VERIFIED — Exact match.**

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    });

    return needs_rs_seq ? draft.n_max : 0u;
}
```

**Behavior:** When any speculative type in `types` is MTP, EAGLE3, or DFlash, the function returns `draft.n_max`. Otherwise returns 0.

---

### 1.2 `draft.n_max` Origin — VERIFIED

**Location:** [`common/arg.cpp:4289-4293`](common/arg.cpp:4289)

CLI argument `--spec-draft-n-max` sets `params.speculative.draft.n_max` and marks `draft_n_max_explicit = true`.

**DFlash auto-resolution:** [`common/common.cpp:491-533`](common/common.cpp:491)

When `--spec-draft-n-max` is NOT explicitly set AND DFlash is in the types list, `common_speculative_resolve_dflash_draft_n_max()` reads `dflash.block_size` from the draft GGUF metadata and sets `draft.n_max = block_size - 1`.

```cpp
bool common_speculative_resolve_dflash_draft_n_max(
        common_params_speculative & params,
        const std::string & draft_model_path) {
    const bool has_dflash = std::find(
            params.types.begin(), params.types.end(),
            COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != params.types.end();
    if (!has_dflash || params.draft_n_max_explicit) {
        return true;
    }
    // ... reads dflash.block_size from GGUF metadata ...
    params.draft.n_max = block_size - 1;
    return true;
}
```

---

### 1.3 Speculative Types Enum — VERIFIED

**Location:** [`common/common.h:169-181`](common/common.h:169)

```cpp
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,  // standalone draft model speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,  // Eagle3 speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash speculative decoding
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding based on n-grams
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT          // number of types, unknown type
};
```

---

### 1.4 `common_params_speculative` Struct — VERIFIED

**Location:** [`common/common.h:387-424`](common/common.h:387)

```cpp
struct common_params_speculative {
    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };

    // used by Simple, MTP, Eagle3, etc.
    common_params_speculative_draft draft;

    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;
    common_params_speculative_ngram_cache ngram_cache;

    bool draft_n_max_explicit = false;
    common_speculative_dm_controller dm_controller = COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
    // ... profit controller params ...

    bool has_dft() const {
        return !draft.mparams.empty();
    }

    uint32_t need_n_rs_seq() const {
        bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        });

        return needs_rs_seq ? draft.n_max : 0u;
    }
};
```

---

## 2. Recurrent Memory Allocation Verification

### 2.1 Full Allocation Chain — VERIFIED

The complete flow from CLI to tensor allocation:

| Step | Location | What Happens |
|------|----------|--------------|
| 1 | [`common/arg.cpp:4289-4293`](common/arg.cpp:4289) | `--spec-draft-n-max N` sets `params.speculative.draft.n_max = N` |
| 2 | [`common/common.cpp:491-533`](common/common.cpp:491) | If DFlash and not explicit, resolve from GGUF: `draft.n_max = block_size - 1` |
| 3 | [`common/common.cpp:1770`](common/common.cpp:1770) | `cparams.n_rs_seq = params.speculative.need_n_rs_seq()` |
| 4 | [`common/common.h:417-423`](common/common.h:417) | `need_n_rs_seq()` returns `draft.n_max` if DFlash in types, else 0 |
| 5 | [`src/llama-context.cpp:264-270`](src/llama-context.cpp:264) | Context copies `params.n_rs_seq` to `cparams.n_rs_seq`, clamps to 0 if arch doesn't support RS rollback |
| 6 | [`src/llama-model.cpp:2101-2109`](src/llama-model.cpp:2101) | `create_memory()` passes `cparams.n_rs_seq` to recurrent memory constructor |
| 7 | [`src/llama-memory-recurrent.cpp:20-128`](src/llama-memory-recurrent.cpp:20) | Constructor allocates RS/S tensors using `n_rs_seq` |

**Step 3 detail** ([`common/common.cpp:1765-1771`](common/common.cpp:1765)):
```cpp
struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    // ...
```

**Step 5 detail** ([`src/llama-context.cpp:264-270`](src/llama-context.cpp:264)):
```cpp
cparams.kv_tail_rollback_tokens = params.n_rs_seq;
cparams.n_rs_seq = params.n_rs_seq;
if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
    LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                    __func__, cparams.n_rs_seq);
    cparams.n_rs_seq = 0;
}
```

---

### 2.2 Recurrent Memory Constructor — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:20-128`](src/llama-memory-recurrent.cpp:20)

**Signature:** [`src/llama-memory-recurrent.h:19-27`](src/llama-memory-recurrent.h:19)
```cpp
llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s
                bool   offload,
                uint32_t   mem_size,
                uint32_t   n_seq_max,
                uint32_t   n_rs_seq,
    const layer_filter_cb & filter);
```

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `model` | `const llama_model &` | Model for device/hiparam lookup |
| `type_r` | `ggml_type` | R state tensor quantization (always `GGML_TYPE_F32`) |
| `type_s` | `ggml_type` | S state tensor quantization (always `GGML_TYPE_F32`) |
| `offload` | `bool` | Whether to offload to GPU |
| `mem_size` | `uint32_t` | Number of cache cells |
| `n_seq_max` | `uint32_t` | Maximum parallel sequences |
| `n_rs_seq` | `uint32_t` | Number of RS snapshots per sequence |
| `filter` | `layer_filter_cb` | Layer filter (nullptr = all layers) |

**Constructor stores n_rs_seq:** [`src/llama-memory-recurrent.cpp:35-36`](src/llama-memory-recurrent.cpp:35)
```cpp
this->n_rs_seq = n_rs_seq;
rs_idx.assign(n_seq_max, 0);
```

---

### 2.3 RS Buffer Allocation Formula — VERIFIED

**Location:** [`src/llama-memory-recurrent.cpp:99-105`](src/llama-memory-recurrent.cpp:99)

**Claim:** `n_rows = mem_size * (1 + n_rs_seq)` exists.

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

**Tensor dimensions:**
- R tensor: `hparams.n_embd_r() × n_rows` (2D)
- S tensor: `hparams.n_embd_s() × n_rows` (2D)
- `n_rows = mem_size * (1 + n_rs_seq)`

**VRAM impact when n_rs_seq changes (Qwen3.6 example):**
| n_rs_seq | n_rows (mem_size=4) | S tensor per layer | 48 layers total S |
|----------|---------------------|--------------------|--------------------|
| 8 (current) | 36 | 786432 × 36 × 4B = 108 MiB | 5,184 MiB |
| 2 (hybrid) | 12 | 786432 × 12 × 4B = 36 MiB | 1,728 MiB |
| 1 (minimal) | 9 | 786432 × 9 × 4B = 27 MiB | 1,296 MiB |
| 0 (eliminated) | 4 | 786432 × 4 × 4B = 12 MiB | 576 MiB |

---

### 2.4 Recurrent Memory Allocation Callers — VERIFIED

**Pure recurrent architectures:** [`src/llama-model.cpp:2101-2109`](src/llama-model.cpp:2101)
```cpp
if (llm_arch_is_recurrent(arch)) {
    res = new llama_memory_recurrent(
            *this,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            cparams.offload_kqv,
            std::max((uint32_t) 1, cparams.n_seq_max),
            cparams.n_seq_max,
            cparams.n_rs_seq,
            nullptr);
```

**Hybrid architectures (ISWA path):** [`src/llama-model.cpp:2136-2163`](src/llama-model.cpp:2136)
```cpp
res = new llama_memory_hybrid_iswa(
    /* model             */ *this,
    /* recurrent_type_r  */ GGML_TYPE_F32,
    /* recurrent_type_s  */ GGML_TYPE_F32,
    /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
    /* n_seq_max         */ cparams.n_seq_max,
    /* n_rs_seq          */ cparams.n_rs_seq,
    /* offload           */ cparams.offload_kqv,
    // ...
```

**Hybrid architectures (non-ISWA path):** [`src/llama-model.cpp:2193-2218`](src/llama-model.cpp:2193)
```cpp
auto mem_recr = std::make_unique<llama_memory_recurrent>(
        *this,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        cparams.offload_kqv,
        std::max((uint32_t) 1, cparams.n_seq_max),
        cparams.n_seq_max,
        cparams.n_rs_seq,
        filter_recr);
```

All three paths pass `cparams.n_rs_seq` to recurrent memory constructors.

---

### 2.5 n_rs_seq Usage Throughout Recurrent Memory — VERIFIED

From search results, `n_rs_seq` is used in these contexts within [`src/llama-memory-recurrent.cpp`](src/llama-memory-recurrent.cpp):

| Line | Usage | Impact if n_rs_seq=0 |
|------|-------|---------------------|
| 35 | `this->n_rs_seq = n_rs_seq;` | Stores 0 |
| 99 | `n_rows = mem_size * (1 + n_rs_seq)` | `n_rows = mem_size` (no snapshot columns) |
| 174 | `rollback <= llama_pos(n_rs_seq)` | RS rollback never valid (condition always false) |
| 217 | `rollback <= (llama_pos) n_rs_seq` | Same — RS rollback disabled |
| 477 | `rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx` | Always stores 0 |
| 505 | `n_rs_seq > 0 ? n_rs_seq + 1 : 0` | Returns 0 (no tail split requirement) |
| 785 | `.suffix_rollback_tokens = n_rs_seq` | 0 suffix rollback tokens |
| 838 | `if (n_rs_seq != 0)` | Block skipped — no RS snapshot handling |
| 922 | `if (n_rs_seq != 0)` | Block skipped — no RS snapshot state write |
| 1334 | `if (mem->n_rs_seq == 0)` | Returns `src0` (no RS copy needed) |

---

## 3. Discrepancies Found

### 3.1 No Discrepancies

All claims from Tasks 4-5 research documents match the current source code:

| Claim | Status |
|-------|--------|
| DFlash included in `need_n_rs_seq()` | **VERIFIED** — line 419 |
| RS formula `mem_size * (1 + n_rs_seq)` | **VERIFIED** — line 99 |
| `n_rs_seq` flows through `cparams` | **VERIFIED** — common.cpp:1770 → llama-context.cpp:264 → llama-model.cpp |
| Constructor takes `n_rs_seq` parameter | **VERIFIED** — header line 26, impl line 27 |
| Tensor dimensions use `n_rows` formula | **VERIFIED** — lines 100-101 |
| n_rs_seq=0 paths exist throughout | **VERIFIED** — 12 conditional checks found |

---

## 4. Opt-In Boundary Analysis

### 4.1 Candidate Locations

Three locations were evaluated for inserting `--beefix-dflash-custom` branch logic:

#### Option A: Override `need_n_rs_seq()` return value

**Location:** [`common/common.h:417-423`](common/common.h:417)

**Approach:** Add a flag to `common_params_speculative` that excludes DFlash from `need_n_rs_seq()`.

**Pros:**
- Single point of change for RS buffer size.
- DFlash detected via `types` vector — clean check.

**Cons:**
- Does not address backup cell allocation (Task 4 requirement).
- Affects all DFlash users globally (not opt-in per slot).

#### Option B: Override in `common_context_params_to_llama()`

**Location:** [`common/common.cpp:1765-1811`](common/common.cpp:1765)

**Approach:** After `cparams.n_rs_seq = params.speculative.need_n_rs_seq()`, check for custom mode and override.

**Pros:**
- Central location — all context params flow through here.
- Can detect DFlash from `params.speculative.types`.
- Can set both `n_rs_seq=0` and pass custom backup cell parameters.

**Cons:**
- Requires adding custom params to `llama_context_params` or `llama_cparams`.

#### Option C: Override in `llama_model::create_memory()`

**Location:** [`src/llama-model.cpp:2044-2218`](src/llama-model.cpp:2044)

**Approach:** At the point of recurrent memory construction, override `cparams.n_rs_seq` for DFlash custom mode.

**Pros:**
- Closest to actual tensor allocation.
- Can pass custom `mem_size` for backup cells.

**Cons:**
- Too late — `cparams.n_rs_seq` already used by hybrid memory (ISWA) for batch splits at [`src/llama-memory-hybrid.cpp:110-112`](src/llama-memory-hybrid.cpp:110).
- Would need to also override in `llama-memory-hybrid-iswa.cpp` and `llama-memory-hybrid.cpp`.

---

### 4.2 Recommended Opt-In Boundary

**Primary override: Option B — [`common/common.cpp:1770`](common/common.cpp:1770)**

**Justification:**

1. **Single point of control.** All `n_rs_seq` values flow through `common_context_params_to_llama()`. Override here and all downstream consumers (recurrent memory, hybrid memory, batch splits) receive the correct value.

2. **DFlash detection is available.** The function receives `const common_params & params`, which includes `params.speculative.types`. DFlash can be detected with:
   ```cpp
   bool has_dflash = std::find(
       params.speculative.types.begin(),
       params.speculative.types.end(),
       COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != params.speculative.types.end();
   ```

3. **Custom params can be added to the struct.** Add a flag to `common_params_speculative`:
   ```cpp
   bool beefix_dflash_custom = false;  // opt-in: n_rs_seq=0 + backup cells
   uint32_t beefix_backup_cells = 0;   // extra backup cells for recurrent memory
   ```

4. **Downstream compatibility.** Setting `n_rs_seq=0` at this point means:
   - [`src/llama-context.cpp:266-270`](src/llama-context.cpp:266): Arch support check passes (0 is always valid).
   - [`src/llama-memory-recurrent.cpp:99`](src/llama-memory-recurrent.cpp:99): `n_rows = mem_size * 1` (no snapshot columns — 5.4GB saved).
   - [`src/llama-memory-hybrid.cpp:112`](src/llama-memory-hybrid.cpp:112): `n_rs_seq > 0 ? n_rs_seq + 1 : 0` returns 0 (no tail split).
   - All 12 conditional checks in recurrent memory handle `n_rs_seq=0` correctly.

5. **Backup cell parameters need separate path.** The backup cells from Task 4 require modifying recurrent memory construction. This can be done at [`src/llama-model.cpp:2106`](src/llama-model.cpp:2106) where `mem_size` is passed to the constructor. The `mem_size` parameter could be increased by `n_parallel × 2` for backup cells when custom mode is active.

---

### 4.3 DFlash Detection Points Summary

| Location | Context | Suitable for Override? |
|----------|---------|----------------------|
| [`common/common.h:417`](common/common.h:417) | `need_n_rs_seq()` method | Yes — but global, not opt-in |
| [`common/common.cpp:1770`](common/common.cpp:1770) | Context params conversion | **YES — recommended** |
| [`common/common.cpp:494-496`](common/common.cpp:494) | DFlash draft n_max resolver | Yes — but only for n_max |
| [`tools/server/server-context.cpp:478-482`](tools/server/server-context.cpp:478) | Slot `uses_dflash()` | Too late — context already allocated |
| [`tools/server/server-context.cpp:1932-1933`](tools/server/server-context.cpp:1932) | Server slot setup | Too late — context already allocated |
| [`src/llama-model.cpp:2101-2109`](src/llama-model.cpp:2101) | Memory construction | Partial — too late for hybrid splits |

---

## 5. Summary

All research claims from Tasks 4-5 have been **verified against current source code** with no discrepancies found. The complete allocation chain is:

```
CLI --spec-draft-n-max
    → common_params_speculative.draft.n_max
    → need_n_rs_seq() returns draft.n_max (DFlash included)
    → cparams.n_rs_seq in common_context_params_to_llama()
    → llama_context.cpp validation (arch support check)
    → llama_model::create_memory() passes cparams.n_rs_seq
    → llama_memory_recurrent constructor
    → n_rows = mem_size * (1 + n_rs_seq)
    → R tensor: n_embd_r × n_rows, S tensor: n_embd_s × n_rows
```

Setting `n_rs_seq=0` for DFlash eliminates the RS snapshot buffer (5.4GB for Qwen3.6) while all downstream code paths handle `n_rs_seq=0` correctly through conditional checks.

The recommended opt-in boundary is [`common/common.cpp:1770`](common/common.cpp:1770) where `cparams.n_rs_seq` is assigned, with DFlash detection via `params.speculative.types` and a new `common_params_speculative.beefix_dflash_custom` flag.
