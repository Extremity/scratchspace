# 8. Feature-by-Feature Architecture Assessment, 9. Forced vs Optional, 17. Evidence

## 8. Feature-by-feature architecture assessment

| Feature | Architecturally sound? | Independent or coupled? | Verdict |
|---|---|---|---|
| F1 draft-ctx | **Sound.** A pure parameter gap in upstream's draft-context init; the fix is the minimal possible (propagate `draft.n_ctx`). No state, no lifecycle, no failure mode beyond "unset = upstream behavior." | Independent | Keep, port as-is. The single best value-per-line feature in the fork. |
| F2 VRAM reservation/measure | **Sound.** Reservation is a budget-input correction (subtract draft footprint before KV planning) — the correct layer to fix the OOM/undersize bug. Measurement is a diagnostic that reuses the real init path, so it cannot drift from production sizing. | Coupled to F3 (feeds its budgets) but degrades gracefully without it | Keep, port as-is. |
| F3 KV device-chain | **Sound, with one design smell.** The planner (first-fit over an ordered chain with per-device safety margins) is the right model for heterogeneous-VRAM spill and is cleanly isolated in `spill.h`. The smell: it is plumbed as a `const char *` through ~15 constructor signatures and a public-API field, which is invasive relative to the algorithm's size. 0.4.4's component taxonomy (§7.4) is the better long-term home. | Coupled to F2 (budgets), to KV constructor chain (plumbing), independent of F1/F4 | Keep, port in two phases (re-thread now, re-express later). |
| F4 custom DFlash | **Sound, with the highest coupling in the fork.** The tape-capture + backup-cell + replay design is the correct VRAM answer for recurrent-target DFlash (it replaces per-token RS snapshots with bounded backup cells + a bounded F32 tape). Its cost: it reaches into qwen35.cpp graph internals and `llama_memory_recurrent` layout, so it is the feature most exposed to upstream churn. The opt-in flag + `replay_failed` permanent-disable fallback are the right safety design. | Coupled to qwen35.cpp graph, `llama_memory_recurrent`, server slot lifecycle; independent of F1–F3 | Keep, port with restore-semantics validation as the acceptance gate. |
| F5 logging | Sound (diagnostic only). | Incidental | Keep. |
| F6 test infra | Sound; encodes validated scenarios. | Standalone | Keep, copy as-is. |
| F7 docs | Sound. | Standalone | Keep, update. |

## 9. Forced vs optional/superseded differences (consolidated)

| # | Difference (Y vs Z) | Class |
|---|---|---|
| 1 | KV spill-by-layer (F3) vs tensor-parallel head-split | **B** design choice (different objective; neither forced nor redundant) |
| 2 | Device-chain budget/margin vs `--fit` | **B** design choice |
| 3 | Draft VRAM reservation (F2) | **B/D-adjacent** — not superseded; no 0.4.4 equivalent |
| 4 | Draft-ctx knob (F1) | **B/D-adjacent** — not superseded; no 0.4.4 equivalent |
| 5 | KVarN record/stage/tail component model | **A** forced — F3's KVarN size computation must follow the new descriptors |
| 6 | Transactional prompt-cache/restore rework (0.4.3) | **A** forced — F4 must be re-validated against the new restore semantics |
| 7 | `llama-kv-cache-placement` meta-device machinery | **C** for upstream (TP integration convenience) / **E** for the fork (may become F3's home) |
| 8 | DFlash RS snapshots (`need_n_rs_seq` incl. DFlash+DSpark) | **D (not superseded)** — F4 still needed |
| 9 | DSpark / multi-output sampling / MTP arch coverage | **D (upstream superset)** — inherited free by migration; the main reason not to stay |
| 10 | `llm_arch_supports_rs_rollback` per-arch flag | **A** forced context — upstream now models RS-rollback capability explicitly; F4's premise (recurrent targets need special rollback handling) is now an upstream-recognized arch property |
| 11 | Qwen3.5 MTP tensor-flag loading (Z qwen35.cpp) | **A** forced — F4's hook re-target must sit inside the new loading/graph structure |
| 12 | `--split-mode tensor` deny-list excluding QWEN35 | **E** — allowed but unvalidated on physical 2-GPU; performance vs device-chain UNKNOWN |

## 17. Evidence register

All facts below were established by direct Git inspection in the two
repositories (root repo = fork history; `other-versions/beellama_0.4.4-preview`
= full upstream clone containing X, Y, `589fc8b89` as identical objects).

| # | Fact | Evidence | Class |
|---|---|---|---|
| E1 | `589fc8b89` parent is X; single squashed local commit | `git log -1 --format=%P 589fc8b89` → `ca155ad07...` | CONFIRMED |
| E2 | `176c1a16a` is a 2-parent Anbeeld merge | `git log -1 --format=%P 176c1a16a` → `dd53db764... 5e5f09968...`; `--format=fuller` author Anbeeld | CONFIRMED |
| E3 | `deeda007d` merges local + 0.4.1-actual | parents `589fc8b89... 176c1a16a...`; author Extremity | CONFIRMED |
| E4 | X is ancestor of 0.4.1-actual | `git merge-base --is-ancestor X 176c1a16a` → 0 | CONFIRMED |
| E5 | Post-merge local commits = 9 (all Extremity) | `git log --author=Extremity deeda007d..Y` | CONFIRMED |
| E6 | X→Z = 430 commits, zero Extremity | `git rev-list --count X..Z`; `git log --author=Extremity X..Z` empty | CONFIRMED |
| E7 | Pre-merge local delta = 21 files, +680/−66 | `git diff --stat X..589fc8b89` | CONFIRMED |
| E8 | Post-merge local delta = 37 files, +4,828/−96 | `git diff --stat deeda007d..Y` | CONFIRMED |
| E9 | Y→Z delta = 1,239 stat lines; local files appear as deletions | `git diff --stat Y..Z` (e.g. `server-dflash-custom.cpp | 862 -`) | CONFIRMED |
| E10 | F1 mechanism: arg + 1 propagation line | Y `common/arg.cpp:4262-4270`; Y `common/speculative.cpp:2238` | CONFIRMED |
| E11 | Z has no draft-ctx knob | `grep draft.n_ctx` in Z `common/speculative.cpp` → 0; no arg in Z `common/arg.cpp` | CONFIRMED |
| E12 | Z `need_n_rs_seq()` includes DFlash **and** DSpark | Z `common/common.h:409-412` | CONFIRMED |
| E13 | Z `n_rs_seq = 0` is draft-context-only | Z `common/speculative.cpp:2406` inside `common_speculative_init_result` ctor | CONFIRMED |
| E14 | Z has zero `beefix` symbols | `git grep beefix 0b035b3a2 -- common/ src/ tools/ include/` → 0 | CONFIRMED |
| E15 | F3 algorithm isolated in `spill.h` (276 lines) | Y `src/llama-kv-cache-spill.h` (full file read) | CONFIRMED |
| E16 | F3 plumbing = 83 `kv_device_chain` refs across 15 files | `git grep -c kv_device_chain Y` (file list in §3) | CONFIRMED |
| E17 | F4 = self-contained modules + 1 qwen35.cpp hook block | Y `common/server-dflash-custom.{h,cpp}` (249+862 lines); Y→Z `qwen35.cpp` diff shows the tape block is the only local hunk | CONFIRMED |
| E18 | Z tensor-split: experimental, no `--fit`, `-fa on` required | Z `docs/multi-gpu.md` (split-modes table + tensor section) | CONFIRMED |
| E19 | Z tensor-split Qwen3.6-27B checks not physical 2-GPU | Z `docs/multi-gpu.md` "Local Qwen3.6-27B check" note | CONFIRMED |
| E20 | QWEN35 allowed for tensor split (not deny-listed) | Z `src/llama-arch.cpp:1010-1041` `llm_arch_supports_sm_tensor` deny-list lacks QWEN35/QWEN35MOE | CONFIRMED |
| E21 | KV cache files are the highest-churn zone X→Z | `git diff --stat X..Z`: `llama-kv-cache.cpp` +1,570, `kvarn` +924, `tail` +369, `dsv4` +495, `msa` new | CONFIRMED |
| E22 | Speculative rework X→Z | `speculative.cpp` +595, `dflash.cpp` +404, DSpark type added | CONFIRMED |
| E23 | `llm_arch_supports_rs_rollback` true for QWEN35/QWEN35MOE/DEEPSEEK4 | Z `src/llama-arch.cpp:997-1006` | CONFIRMED |
| E24 | 0.4.3 restore rework (one safe-prefix planner, transactional restore) | Z `CHANGELOG.md` 0.4.3 prompt-cache section | CONFIRMED |
| E25 | 5.4 GB RS buffer at 32k ctx, `n_rs_seq=8` | Prior investigation (Memory MCP: "DFlash VRAM Issue Root Cause", "DFlash RS Buffer Issue") | STRONG INFERENCE (re-derivable from Z `llama-memory-recurrent.cpp` sizing) |
| E26 | Tensor-split vs device-chain performance on the actual 3090+3080 box | No physical 2-GPU data exists (0.4.4's own docs concede this) | UNKNOWN |
| E27 | F4 invariants under 0.4.3 transactional restore | Requires code-level validation during port | UNKNOWN |
| E28 | DSpark tape-capture feasibility | Post-migration design question | UNKNOWN |

### Cross-check notes

- The X→Y raw diff (100 files, +26,419/−15,632) deliberately was **not**
  used as the local inventory; the merge-resolution slice
  (`176c1a16a..deeda007d`, +16,229/−15,613) is dominated by
  re-baselining churn, and the semantic local content in it equals the
  pre-merge slice (E7) plus the post-merge slice (E8).
- Every "0.4.4 does not provide X" claim (E11, E12, E13, E14) was checked by
  direct grep/read of Z objects, not by absence in diff output.
- No source repository was modified; all reads were via `git show`/`git
  grep`/`git diff` on committed objects. Scratch files used during the
  investigation are listed in `roo-temp/` (see README of this directory).
