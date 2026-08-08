# Task 6R.2 Part 3: Tape Replay, Rollback, and Task 5 Comparison

## Overview

This document continues the investigation from Part 2, analyzing how the tape is consumed during DFlash verification (replay and rollback mechanics), and comparing the v0.3.2 tape footprint with our Task 5 design estimates.

**Source files analyzed (continuation):**
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:2900-3200` — GPU tape replay launch
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3260-3400` — GPU tape replay execution
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3600-3800` — GPU conv replay
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:3860-4020` — CPU conv replay fallback
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:4140-4220` — CPU recurrent replay fallback
- `old-versions/beellama.cpp-preview-v0.3.2/src/llama-context.cpp:5500-5730` — Tree commit with tape replay

---

## 1. Tape Relationship to Rollback and Replay

### 1.1 The DFlash Verification Flow

DFlash verification with recurrent models requires:
1. **Backup** the S-state (recurrent state) before verification.
2. **Verify** the draft tokens by running the target model forward pass.
3. **Record** the GDN intermediate tensors (k, v, gate, beta, qkv) during verification — this is the tape.
4. **Accept or reject** based on verification results.
5. If accepted: **replay** the tape to reconstruct the S-state for accepted tokens.
6. If rejected: **rollback** to the backup S-state.

### 1.2 Why Tape is Needed for Rollback

Recurrent models maintain state (S-state) that evolves with each token. During DFlash verification:
- The target model processes draft tokens, updating its S-state.
- If verification fails, we need to restore the S-state to pre-verification state.
- Simple backup/restore works for rejection.
- But for **partial acceptance** (accept first N of M draft tokens), we need to reconstruct the S-state at the exact point of the Nth accepted token.
- The tape enables this: restore backup + replay tape for accepted tokens = correct S-state at acceptance point.

### 1.3 Tape Replay Mechanics

From the GPU replay code (llama-context.cpp:3260-3400, function `dflash_replay_recurrent_gpu`):

```cpp
// Pseudocode from actual implementation:
bool llama_context::dflash_replay_recurrent_gpu(...) {
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const uint32_t n_embd_s = model.hparams.n_embd_s();

    for each recurrent layer:
        // Get S-state tensor and cell offset
        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        size_t s_offset = cell_idx * n_embd_s * element_size;

        // Get tape data (from GPU tape or CPU tape)
        // Replay GDN update: S_new = g*S + k⊗delta
        // The tape provides k, v, gate, beta for the GDN recomputation
    }
}
```

The replay consumes tape data to recompute the GDN forward pass for accepted tokens, advancing the S-state from the backup point to the acceptance point.

### 1.4 Tape vs Hidden-State Ring

| Aspect | GPU Tape | Hidden-State Ring |
|--------|----------|-------------------|
| Purpose | Recurrent rollback/replay | Cross-attention for drafter |
| Data | GDN intermediates (k, v, gate, beta, qkv) | Target model hidden states |
| Layers | All recurrent layers (48 for Qwen3.6 27B) | Selected capture layers (5 typical) |
| Tokens | Verify batch (max 25) | Full context (RING_SIZE = 4096) |
| Precision | F32 | F32 |
| Location | GPU (device-aware, per-layer placement) | GPU cross ring + CPU fallback |
| Size | ~70 MB | ~200 MB |

These are completely separate systems with different purposes.

---

## 2. Tape Consumption During Replay

### 2.1 What Replay Does

The GDN replay recomputes the S-state update using tape data. From `delta-net-base.cpp`, the GDN math is:

```
S_new = g * S + k ⊗ delta
```

Where:
- `S` is the current S-state (from backup or previous replay step)
- `g = exp(gate)` — the gate value (pre-exp stored in tape)
- `k` — the normalized K vector (stored in tape)
- `delta` — derived from v, beta, and the GDN computation
- `⊗` — outer product (rank-1 update)

The replay doesn't need the full S-state in the tape because it starts from the backup S-state and applies rank-factored updates using the tape intermediates.

### 2.2 Replay is Per-Verify, Not Per-Token

The tape is recorded once per verify pass (which processes multiple draft tokens), and replay processes the accepted tokens from that verify batch. The tape `n_tokens` field tracks actual tokens recorded, which is the verify batch size (up to `LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25`).

---

## 3. Task 5 Design Estimate vs v0.3.2 Reality

### 3.1 Task 5 Estimate Breakdown

Our Task 5 design estimated ~6.5 GiB for the replay tape. This estimate likely assumed:

| Assumption | Task 5 Estimate | v0.3.2 Reality |
|------------|----------------|----------------|
| Data stored | Full S-state per layer per token | GDN intermediates (rank-factored) |
| Elements per layer per token | `n_embd_s = 3,145,728` | ~377,200 elements (8.4x smaller) |
| Precision | Possibly F16 or F32 | F32 |
| Max tokens | Full context? (e.g., 8192) | 25 (verify batch only) |
| Layers | 48 recurrent layers | 48 recurrent layers |
| Slots | Multiple? (e.g., 8) | 1 |

### 3.2 If Task 5 Assumed Full S-State Storage

If the Task 5 design assumed storing full S-state (`n_embd_s = 3,145,728` elements) per layer per token:

| Scenario | Calculation | Size |
|----------|-------------|------|
| Full S-state, F32, 48 layers, 8192 tokens, 1 slot | 3,145,728 * 4 * 48 * 8192 | ~485 GiB (absurd) |
| Full S-state, F16, 48 layers, 8192 tokens, 1 slot | 3,145,728 * 2 * 48 * 8192 | ~242 GiB (absurd) |
| Full S-state, F32, 48 layers, 25 tokens, 1 slot | 3,145,728 * 4 * 48 * 25 | ~15 GiB |
| Full S-state, F16, 48 layers, 25 tokens, 1 slot | 3,145,728 * 2 * 48 * 25 | ~7.5 GiB |

The ~6.5 GiB estimate is closest to "full S-state, F16, 48 layers, 25 tokens, 1 slot" at ~7.5 GiB.

### 3.3 The Critical Insight

The v0.3.2 tape stores **GDN intermediates** (rank-factored k, v, gate, beta, qkv), NOT the full S-state. This provides:

- **8.4x reduction** in elements per layer per token (376,778 vs 3,145,728)
- The tape enables **recomputation** of the S-state update rather than **storage** of the S-state itself
- Replay = restore backup S-state + apply tape intermediates = correct S-state at acceptance point

This is a **checkpoint-and-replay** pattern, not a **state-snapshot** pattern.

### 3.4 Implications for Task 5 Design

If our Task 5 design stores full S-state snapshots instead of GDN intermediates, we are over-allocating by ~8.4x. The correct approach would be:

1. Store GDN intermediates (k, v, gate, beta, qkv) — same as v0.3.2 tape.
2. Size for verify batch, not full context.
3. Use 1 slot for single-seq verify, or multiple slots for multi-seq verify.

The v0.3.2 approach is:
- **~70 MB GPU tape** for 48 layers * 25 tokens * GDN intermediates (F32)
- **~200 MB hidden-state ring** for cross-attention
- **~270 MB total** for DFlash GPU memory

---

## 4. Multi-Slot Tape Allocation

### 4.1 When Multiple Slots are Used

From `llama_dflash_allocate_slots()` (llama-context.cpp:8968-8970):

```cpp
void llama_dflash_allocate_slots(llama_context * ctx, int n_slots) {
    ctx->allocate_tape_gpu(n_slots, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
}
```

Multiple slots are allocated for multi-sequence verification. Each slot has its own independent tape. For `n_slots = 8`:
- **Total tape = 8 * 70 MB = ~560 MB**

### 4.2 Device-Aware Placement

The `allocate_tape_gpu()` function places each layer's tape tensors on the same GPU as that model layer:

```cpp
ggml_backend_dev_t layer_dev = model.dev_layer(il);
const dflash_capture_backend capture_backend = dflash_capture_backend_for_layer(backends, layer_dev);
// ... allocate on capture_backend.backend ...
```

For multi-GPU models, the tape is distributed across GPUs proportional to layer placement.

---

## 5. Summary of Findings

### 5.1 Memory Footprint Breakdown (Qwen3.6 27B, single slot)

| Component | Size | Details |
|-----------|------|---------|
| Hidden-state ring (GPU cross) | ~200 MB | 5 capture layers * cross_ctx slots * 5120 embd * F32 |
| GPU tape | ~70 MB | 48 recurrent layers * 25 tokens * GDN intermediates * F32 |
| **Total DFlash GPU memory** | **~270 MB** | |

### 5.2 Why v0.3.2 is So Efficient

1. **Rank-factored storage:** GDN intermediates are 8.4x smaller than full S-state.
2. **Verify-batch sizing:** Tape sized for 25 verify tokens, not full context.
3. **Single slot by default:** Multi-slot only for multi-seq verify.
4. **Device-aware placement:** Tape follows model layers to minimize cross-GPU transfers.
5. **Graph-embedded copies:** GPU tape populated by graph operations, not CPU callback.

### 5.3 Key Design Lesson for Task 5

The v0.3.2 approach uses **checkpoint-and-replay** with rank-factored GDN intermediates, NOT **state-snapshot** with full S-state storage. This is the fundamental reason v0.3.2 uses ~270 MB total while Task 5 estimated ~6.5 GiB.

If Task 5 can adopt the same rank-factored intermediate storage approach, the tape size should drop from ~6.5 GiB to ~70 MB (for single slot, verify-batch sizing).

---

## 6. Appendix: Tape Tensor Size Formula

For a model with:
- `S = ssm_d_state` (typically 256)
- `H_k = ssm_n_group` for fused GDN (typically 1) or `ssm_dt_rank` for non-fused (typically 8)
- `H_v = ssm_dt_rank` (typically 8)
- `conv_ch = ssm_d_inner + 2 * ssm_n_group * ssm_d_state` (typically 12,768)
- `max_tokens = LLAMA_DFLASH_MAX_VERIFY_TOKENS` (25)
- `n_rec = number of recurrent layers` (48 for Qwen3.6 27B)
- `n_slots` (1 by default)

**Per layer per slot:**
```
size = (S * H_k * max_tokens +        // k
        S * H_v * max_tokens +        // v
        1 * H_v * max_tokens +        // gate
        1 * H_v * max_tokens +        // beta
        conv_ch * max_tokens) * 4     // qkv (F32)
```

For Qwen3.6 27B (fused GDN):
```
size = (256*1*25 + 256*8*25 + 1*8*25 + 1*8*25 + 12768*25) * 4
     = (6,400 + 51,200 + 200 + 200 + 319,200) * 4
     = 377,200 * 4
     = 1,508,800 bytes ≈ 1.44 MB
```

**Total tape:**
```
total = n_layers * n_slots * per_layer_size
      = 48 * 1 * 1.44 MB
      = 69.5 MB
```
