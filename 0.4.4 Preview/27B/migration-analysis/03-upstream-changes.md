# 4. X → Z — Upstream Architectural Changes (0.4.1 Preview → 0.4.4 Preview)

430 commits, zero Extremity authorship (CONFIRMED). This section covers only
the subsystems the local fork touches or depends on. Line counts are from
`git diff --stat X..Z`.

## 4.1 Speculative decoding

- `common/speculative.cpp` +595, `common/speculative.h` +29;
  `src/models/dflash.cpp` +404; `src/models/eagle3.cpp` +15.
- **New:** `DSpark` (`COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`) — DFlash
  backbone + semi-autoregressive Markov head; a strictly newer draft method
  for Qwen targets (docs/speculative.md). `need_n_rs_seq()` now includes
  DSpark *and* DFlash (Z `common/common.h:409-412`).
- **New:** multi-output backend sampling for speculative decoding (0.4.4
  changelog); speculative metrics; draft-sidecar discovery (0.4.2).
- **Kept:** upstream-owned DFlash, MTP, EAGLE-3, n-gram family. Bee's server
  controls (adaptive draft-max, loop guard) remain "Bee-added around
  upstream paths" per docs/speculative.md.
- **Relevance to fork:** The speculative init path
  (`common_speculative_init_result`) was reworked — F1/F2 plumbing must be
  re-verified there. DSpark gives the project a stronger default draft path
  that the 0.4.1 fork cannot use.

## 4.2 KV cache / KVarN

- `src/llama-kv-cache.cpp` +1,570 / `llama-kv-cache.h` +115;
  `llama-kv-cache-kvarn.cpp` +924 / `.h` +60; `llama-kv-cache-tail.cpp` +369;
  `llama-kv-cache-dsv4.cpp` +495; `llama-kv-cache-dsa.cpp` +27;
  `llama-kv-cache-iswa.cpp` +67; `llama-kv-cache-msa.cpp/.h` **new** (+458/+171,
  MiniMax-M3 sparse attention); `llama-kv-cache-placement.cpp/.h` **new**
  (+117/+41); `llama-kv-tail-request.cpp/.h` **new** (+194/+61);
  `llama-kvarn.cpp` +106 / `.h` +39.
- **Prompt-cache reuse rework** (0.4.3 changelog): one safe-prefix planner +
  transactional target/draft/speculative restore across standard, recurrent,
  and KVarN state; self-contained sequence-selective checkpoints; unified
  KVarN slots borrowing shared capacity; compact sparse reads.
- **Exact/bounded KVarN tail fitting** (0.4.3): immutable tail request shared
  by CLI/fit/final context; `llama-kv-tail-request.*` is the new primitive;
  `common/fit-kvarn-tail.{cpp,h}` new.
- **CUDA KVarN route policy** (0.4.2/0.4.3): versioned backend capability
  record; pre-Turing portable route; `fattn-kvarn-route-policy.h` +206;
  fattn-kvarn-dispatch +367.
- **Relevance to fork:** This is the highest-churn zone and exactly where F3
  (device-chain) is plumbed. The constructor signatures and planning sites
  that F3 threads through have all moved. The new `placement`/`tail-request`
  primitives are the upstream hooks a future device-chain re-implementation
  would attach to.

## 4.3 Multi-GPU / device placement

- **New `--split-mode tensor`** (0.4.2, experimental): tensor-parallel split
  of weights AND KV via a "meta device" abstraction; KV components (standard
  K/V, tails, KVarN records/stages) split at complete-head boundaries via
  `llama_kv_cache_component_from_name` + `llama_tensor_split_counts`
  (cumulative-ratio convention, `src/llama-kv-cache-placement.cpp`).
- `docs/multi-gpu.md` updated; NCCL/RCCL, `GGML_CUDA_P2P` opt-in;
  `--fit` **not supported** in tensor mode.
- **Arch support:** `llm_arch_supports_sm_tensor` (Z `src/llama-arch.cpp:1010`)
  is a deny-list; `LLM_ARCH_QWEN35`/`QWEN35MOE` are **not** on it → tensor
  split is *allowed* for the Qwen3.5/3.6 family. Deny-listed: DeepSeek2/3.2/4,
  GLM-DSA, Nemotron-H(-MoE), Mamba/Jamba/Falcon-H1, Kimi-Linear, LFM2,
  Minimax-M2/M3, Mistral4, Granite-Hybrid, etc.
- **Validation status (CONFIRMED, from 0.4.4's own docs/multi-gpu.md):**
  "Local Qwen3.6-27B checks cover q4_0 plus a 1024-token BF16 tail at 1,1 and
  KVarN4 plus a 1024-token F16 tail at 3,1 using two logical CUDA0 shards...
  These are not physical two-GPU or peer-transfer results. Keep tensor KVarN
  and precision tails labeled experimental until the external two-GPU
  checklist is completed on two distinct device IDs."
- **Relevance to fork:** The conceptual successor to F3, but (a) different
  model (head-granularity TP vs layer-granularity spill), (b) no `--fit`,
  (c) unvalidated on physical multi-GPU, (d) requires `-fa on`. See
  `04-kv-kvarn-multigpu.md`.

## 4.4 Recurrent / hybrid models

- `src/llama-memory-recurrent.cpp` +93; `src/llama-memory-hybrid.cpp` +33;
  `src/models/qwen35.cpp` +31 (MTP tensor-flag handling only — CONFIRMED the
  GDN intermediate tensors used by F4's tape hook are unchanged in kind);
  `src/models/qwen3next.cpp` +324; `ggml/src/ggml-cuda/ssm-scan.cu` new +481.
- `llm_arch_supports_rs_rollback` (Z `src/llama-arch.cpp:997`) — QWEN35,
  QWEN35MOE, DEEPSEEK4 return true: upstream now explicitly models RS
  rollback support per-arch (relevant context for F4).
- **Relevance to fork:** F4's `llama_memory_recurrent` backup-cell API and
  qwen35.cpp hook point must be re-validated against these changes.

## 4.5 Server / common infrastructure

- `tools/server/server-context.cpp` +773, `server-task.cpp` +406,
  `server-models.cpp` +562, `server.cpp` +58; new `server-mcp.*` (+820/+176).
- 0.4.4 changelog: media-aware server slot save/restore, router LRU
  scheduling (0.4.3), speculative metrics, `--load-mode auto` default policy
  (avoids mmap on integrated GPUs), hardened GGUF loading, semantic versioning.
- **Relevance to fork:** F2's server-side validation messaging and F4's
  per-slot lifecycle live in `server-context.cpp` — the file with the second-
  largest local delta; re-verification required.

## 4.6 Backends (CUDA/Vulkan) — context only

- Large CUDA churn (mmq configs for RDNA, rope.cu +235, quantize +323,
  `dsv4-hc.cu` +294, ssm-scan +481); Vulkan +2,158 (flash_attn_tail.comp new,
  kvarn shaders reworked). The fork's only CUDA-local code (F4's
  `dflash-custom-conv.cu`) is a standalone kernel — unaffected by the
  upstream CUDA churn except for build-system inclusion.

## 4.7 What upstream did NOT change that the fork needs

- No independent draft-context-size knob (F1 still needed).
- No draft VRAM reservation/measurement (F2 still needed).
- DFlash still in `need_n_rs_seq()` (F4's RS-buffer problem persists).
- No layer-granularity KV spill (F3 still needed; tensor-split is a
  different, experimental mechanism).
