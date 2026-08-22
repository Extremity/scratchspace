# 6. Speculative Decoding / DFlash / DSpark Implications

## 6.1 Speculative landscape in each state

| Method | X (0.4.1 Preview) | Y (local fork) | Z (0.4.4 Preview) |
|---|---|---|---|
| Draft (simple) | yes | yes | yes |
| EAGLE-3 | yes | yes | yes |
| MTP | yes | yes | yes (+Qwen3-Next, DeepSeek, GLM, Nemotron targets) |
| DFlash | yes (upstream) | upstream + **custom Task 6R mode** | upstream (reworked, +404 in dflash.cpp) |
| **DSpark** | no | no | **yes (new)** — DFlash + Markov head |
| n-gram family | yes | yes | yes (expanded: map-k4v, mod) |
| Multi-output backend sampling | no | no | **yes (new, 0.4.4)** |

## 6.2 Does DSpark change the value of the DFlash-specific local work?

**No — it raises it.**

- DSpark is a *superset* of DFlash (same block-diffusion backbone, plus a
  semi-autoregressive Markov head). It is recurrent-target-compatible in the
  same way DFlash is: the draft runs against a target whose recurrent (GDN)
  state must be handled on accept/reject.
- The local DFlash-specific work (F1 draft-ctx, F2 reservation, F4 custom
  tape/backup replay) is **target-architecture-driven, not draft-method-
  driven**: the 5.4 GB RS-buffer problem and the draft-KV-sizing problem
  arise from (a) the *target* being recurrent with `n_rs_seq` snapshots and
  (b) the *draft* context inheriting the target's `n_ctx`. Both apply
  identically when the draft method is DSpark — CONFIRMED by Z's
  `need_n_rs_seq()` including `DRAFT_DSPARK` alongside `DRAFT_DFLASH`
  (`common/common.h:409-412`).
- Therefore the DFlash-specific local features are *not* obsolete; they are
  required by DSpark too. If anything, DSpark makes F1/F2/F4 more valuable,
  because DSpark drafts are longer/more accepted and the VRAM math matters
  more.
- The custom Task 6R DFlash (F4) is DFlash-specific in its tape capture
  (GDN intermediates during the *draft* forward pass). A DSpark variant of
  the tape would need the Markov-head intermediates captured too — an
  *extension*, not an invalidation. For migration, F4 ports as-is for
  DFlash; DSpark custom-tape is future work, not a migration gate.

## 6.3 The RS-buffer question (F4's core problem) in Z

- Z still computes `need_n_rs_seq()` including DFlash **and** DSpark
  (CONFIRMED, `common/common.h:409-412`). The target-context RS snapshot
  allocation (`mem_size * (1 + n_rs_seq)` rows) therefore still happens for
  Qwen3.5/3.6 DFlash/DSpark serving — the ~5.4 GB overhead at 32k context
  with `n_rs_seq=8` persists.
- Z's `common/speculative.cpp:2406` `cparams.n_rs_seq = 0` applies only to
  the **draft** context init, not the target. (CONFIRMED by reading the
  function: it is inside `common_speculative_init_result`'s constructor for
  the draft `llama_context_params`.)
- `llm_arch_supports_rs_rollback` (Z `src/llama-arch.cpp:997`) now returns
  true for QWEN35/QWEN35MOE/DEEPSEEK4 — upstream *recognizes* these archs
  need RS rollback, which is why the snapshots are allocated. The local
  custom DFlash sidesteps this by using backup cells + tape replay instead
  of per-token snapshots.
- **Consequence:** F4 is still needed on 0.4.4. The alternative "just set
  `n_rs_seq=0`" (prior Solution 2) trades the 5.4 GB for checkpoint-based
  rollback (slower rejects); F4 is the higher-performance answer and is the
  reason the custom mode exists.

## 6.4 Draft context sizing (F1) in Z

- CONFIRMED absent: Z's `common_speculative_params_to_llama` has no
  `draft.n_ctx` propagation (0 matches for `draft.n_ctx` in Z
  `common/speculative.cpp`), and no CLI arg for it. The draft context
  inherits the target's `n_ctx` exactly as in X.
- This is the feature the project owner flagged as crucial. It ports in
  ~2 lines (arg + propagation) and is the single highest-value-per-line
  feature in the inventory.

## 6.5 Speculative init path rework (port risk for F1/F2)

- Z's `common_speculative.cpp` changed 595 lines; the
  `common_speculative_init_result` constructor (Z ~line 2387) still has the
  same shape (mparams/cparams from `common_params`, draft model load via
  `llama_model_load_from_file`, draft context via `llama_init_from_model`),
  so F1's one-line propagation and F2's reservation plumbing attach to the
  same structural points. Risk: LOW–MEDIUM (re-verify, don't redesign).
- Z's `common/speculative.h` changed 29 lines — the draft params struct
  (where `n_ctx` and the new `beefix_spec_draft_res` field would live) must
  be checked for field renames; expected to be additive.

## 6.6 Server-side speculative controls

- Bee's server controls (adaptive draft-max/profit controller, reasoning-
  loop guard) live in `tools/server/server-context.cpp` (+773 in Z) and
  `server-task.cpp` (+406 in Z). These are Bee-maintained around upstream
  paths (docs/speculative.md) and are *not* part of the Extremity local
  inventory (they predate the fork's local line — they are in X already).
  They are unaffected by the fork's local features except that F4's
  per-slot tape lifecycle and F2's validation messaging also live in
  `server-context.cpp` — the file is a shared edit surface for the port.

## 6.7 Net speculative conclusion

- The speculative subsystem is the *second* highest-churn zone after KV.
- None of the local speculative features are superseded; all four (F1, F2,
  F4, plus F4's DSpark-extension potential) remain necessary.
- DSpark is a strict benefit of 0.4.4 for the Qwen workload and is a
  concrete reason not to stay on 0.4.1.
- Port risk is concentrated in re-verifying F1/F2 against the reworked
  speculative init and F4 against the reworked qwen35.cpp graph +
  `llama_memory_recurrent`.
