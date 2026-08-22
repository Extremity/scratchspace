# 10–16. Option Analyses, Effort/Risk, Final Recommendation

## 10. Option 1 — Stay on the current fork (0.4.1-based Y)

**Can the fork be finished, fixed, optimized, and maintained?** Yes — it is
stable and its local features are coherent. That is the honest baseline
assessment.

**What would be missed from 0.4.4 (CONFIRMED gaps):**
- DSpark (new draft method; strictly stronger for Qwen targets).
- Multi-output backend sampling for speculative decoding.
- The 0.4.3 prompt-cache/transactional-restore rework (one safe-prefix
  planner across standard/recurrent/KVarN; self-contained checkpoints) —
  directly relevant to DFlash/Draft rollback correctness and multi-slot
  serving.
- Exact/bounded KVarN tail fitting (`llama-kv-tail-request`) — the fork's
  KVarN tail sizing is the older, looser model.
- CUDA KVarN route-policy hardening (versioned capability record,
  pre-Turing portable route) — the 3080 (Ampere, Turing-adjacent) benefits.
- MTP for Qwen3-Next/DeepSeek/GLM/Nemotron; MiniMax-M3 sparse attention;
  server media-aware slot save/restore; router LRU scheduling; `--load-mode
  auto` (mmap avoidance on integrated GPUs); hardened GGUF loading.

**Do those matter to the workload?** Yes, in three concrete ways:
1. DSpark + multi-output sampling are the speculative-decoding quality
   improvements the project exists to exploit.
2. The prompt-cache/restore rework is the correctness substrate under the
   local DFlash work — staying means the local F4 sits on a rollback model
   that upstream has since revised twice.
3. KVarN tail fitting + route policy directly serve the constrained-VRAM
   KVarN serving that is the fork's bread and butter.

**Are current architectural problems untenable?** No. The fork is not
broken; it is *behind*. The problems are (a) the 5.4 GB RS-buffer
overhead (solved locally by F4), (b) draft KV sizing (solved locally by
F1), (c) multi-GPU KV fit (solved locally by F3). All three have local
solutions *today*.

**Does staying save meaningful engineering effort?** It saves the one-time
migration cost (see §14) in exchange for:
- **Permanent merge debt.** Upstream's change velocity is concentrated in
  exactly the files the fork extends: `llama-kv-cache*.cpp` (+1,570/+924/
  +369/+495 between 0.4.1 and 0.4.4 alone), `speculative.cpp` (+595),
  `server-context.cpp` (+773). Every future release (0.4.5, 0.5.x) will
  re-conflict in the same files. The fork's local hunks in those files
  (device-chain plumbing, reservation budgeting, tape hooks) will need
  manual re-merge *every release*, forever, with no upstream release
  absorbing the conflicts.
- **Correctness drift.** Upstream has already changed the rollback/restore
  semantics the local DFlash depends on. Staying means the fork
  re-implements — and re-verifies — those changes itself.
- **Feature starvation.** New speculative methods (DSpark), new archs, and
  new backends arrive upstream-first; each is a mini-project to backport.

**Verdict on Option 1:** Rational *today*, but it converts a bounded
one-time migration into an unbounded recurring cost, and it forgoes the
specific improvements (DSpark, restore rework, KVarN tail fitting) that the
workload most needs. **Rejected** — see §16.

## 11. Option 2 — Migrate the existing fork to 0.4.4 (RECOMMENDED)

**What can be preserved?** All of it, at three tiers:

| Tier | Features | Port shape |
|---|---|---|
| Copy as-is | F6 test infra, F4's standalone modules (`server-dflash-custom.*`, `dflash-custom-conv.*`), F3's `llama-kv-cache-spill.h` algorithm, F7 docs | verbatim + build-system inclusion |
| Re-attach (same hunks, new anchors) | F1 (arg + 1 propagation line), F2 (budget/margin blocks at new planning sites, server validation), F5 logging | mechanical re-anchoring against 0.4.4 |
| Re-target (hook moved, mechanism same) | F3 plumbing (83 `kv_device_chain` references re-threaded through new constructor signatures; KVarN size computation follows new descriptors), F4's qwen35.cpp tape hook + `llama_memory_recurrent` backup-cell API | bounded re-verification |

**Where do upstream conflicts occur?** Only where local hunks and upstream
churn share a file: `llama-kv-cache.cpp` / `llama-kv-cache-kvarn.cpp`
(F3+F2), `speculative.cpp` (F1+F2), `server-context.cpp` (F2+F4),
`qwen35.cpp` (F4), `llama-memory-recurrent.cpp` (F4), `llama-model.cpp`
(F3 call sites), `arg.cpp`/`common.cpp`/`common.h` (all flags). These are
all *additive* local hunks (new parameters, new opt-in blocks) — conflicts
are "both sides added nearby," not "both sides rewrote the same logic,"
which makes them low-ambiguity to resolve.

**Which local assumptions no longer hold?**
1. KV constructor signatures (F3) — changed; re-thread.
2. KVarN size model (F3) — new descriptors/tail-request; recompute.
3. Speculative init flow (F1/F2) — reworked but same shape; re-verify.
4. qwen35.cpp GDN graph (F4) — MTP-flag rework around the hook; the GDN
   intermediate tensors (k_conv, v_conv, gate, beta, qkv_mixed) are intact.
5. `llama_memory_recurrent` layout (F4) — +93 lines; backup-cell API to
   re-verify.
6. Checkpoint/restore semantics (F4) — 0.4.3 transactional restore; the
   tape/backup design must be validated against it (the one genuine
   *semantic* risk in the port).

**What becomes easier / redundant / harder?**
- Easier: DSpark and multi-output sampling come free; KVarN tail fitting
  gives F3's KVarN path a better sizing primitive; the component taxonomy
  (`llama_kv_cache_component_from_name`) offers a cleaner long-term home
  for the device-chain planner (§7.4).
- Redundant: nothing (all local features still needed — §4.7).
- Harder: nothing structurally; the port is the same work as backporting
  0.4.4 to the fork, minus the debt it leaves behind.

**Verdict on Option 2:** Bounded, enumerable, low-ambiguity; preserves all
validated behavior; captures all upstream benefit. **Recommended.**

## 12. Option 3 — Start from pristine 0.4.4

**Which local features must be reintroduced?** All of F1–F7 (none are
superseded — §4.7). **Which should be abandoned?** None identified.
**Which redesigned?** Optionally F3 (component-taxonomy re-expression, §7.4).

**Does a clean start create a genuinely better architectural boundary?**
Marginal. The local features are already isolated and opt-in; a pristine
start does not change their shape, only the order of operations (apply to
Z directly instead of migrating Y onto Z). The "cleaner boundary" argument
only materializes if the fork's *local* code were entangled with upstream
in a way that a rewrite would fix — the inventory shows it is not: the
entire local surface is additive parameters, opt-in modules, and isolated
algorithms.

**Does it make future upstream updates materially easier?** No. A
pristine-start fork and a migrated fork both carry the same beefix layer
against the same upstream; future-release friction is identical. The only
difference is that the migrated fork has a *working reference* (Y) to
diff against during re-verification, which the pristine start lacks.

**Is rebuilding actually cleaner than migration?** No — it is the same
work (re-attach F1–F7 to Z) with strictly less information (no validated
runtime reference, no merge history documenting why each hunk exists). The
one scenario where Option 3 wins is if the local code were found, during
migration, to be so entangled that re-derivation is cheaper than
disentanglement; the inventory (≈5,500 functional lines, all additive/opt-
in) indicates that scenario does not exist.

**Verdict on Option 3:** Strictly dominated by Option 2 for this codebase.
**Rejected** — see §16.

## 13. Maintenance / future-upstream implications

- **All three options** carry the same long-term shape: a small beefix
  layer on top of upstream. The difference is the *starting point* of that
  layer and the debt accumulated before it.
- **Option 1** accumulates the most: each upstream release re-conflicts in
  the KV/speculative/server files; the fork's KVarN/rollback substrate
  stays two generations behind the correctness fixes.
- **Options 2 and 3** reset the debt to zero at migration time; thereafter
  per-release friction is the same for both (re-verify the beefix hunks
  against new upstream hunks in the same ~10 files).
- **Recommended guardrail for Option 2:** keep the beefix layer as
  *additive, opt-in, flag-gated* code (as it already is) so that each
  future merge is "apply N small hunks," and track the anchor points in a
  short `docs/beellama-port-anchors.md` (file + symbol per feature) so
  future re-anchoring is mechanical.
- **DSpark watch:** if DSpark becomes the default Qwen draft path, F4's
  custom-tape should be extended (not replaced) to DSpark; this is
  post-migration work.

## 14. Effort / risk comparison

No calendar estimates (per project rules); relative engineering effort only.

| Work item | Option 1 (stay) | Option 2 (migrate) | Option 3 (fresh) |
|---|---|---|---|
| Restore current behavior | 0 | **High** — F3 re-thread (83 refs, 8 constructor chains, KVarN size recompute) | High (same, minus reference) |
| Re-verify F1/F2 | 0 | **Low–Med** (re-anchored hunks + speculative-init re-check) | Low–Med (same, no diff base) |
| Re-target F4 | 0 | **Med** (qwen35 hook + recurrent API + restore-semantics validation) | Med (same, no diff base) |
| Capture 0.4.4 benefits | **Unbounded** (per-release backports: DSpark, restore rework, KVarN tail fitting, archs, server) | 0 (inherited) | 0 (inherited) |
| Per-release maintenance (steady state) | **High** (re-conflict in KV/spec/server every release) | Low–Med | Low–Med (identical to O2) |
| Testing effort | 0 now | **Med** — re-run `test-runner.py` + `dflash-custom-test.py` suite (already written, copy as-is) against the migrated build | Med (same suite, but no prior passing baseline to compare) |
| Debugging risk | Low now, **rising** (drift from upstream rollback semantics) | Medium, concentrated in F3 re-thread + F4 restore-semantics | Medium, same, with worse diagnostics (no reference tree) |
| Migration complexity | 0 | **Bounded** (enumerated file set, additive hunks) | Bounded (same) + reference loss |
| Future maintenance burden | **Highest** | Lowest of the three | Equal to O2 |

**Dominance summary:** Option 2 ≤ Option 3 on every axis (strictly better
on reference/diagnostics). Option 1 is cheapest *now* but strictly worse on
every future axis. The decision is a standard one-time-cost vs perpetual-
cost tradeoff, and the one-time cost is bounded and small relative to the
perpetual cost.

## 15. Final Recommendation

**Migrate the existing fork to 0.4.4 Preview (Option 2), carrying the local
beefix layer forward as additive, opt-in code.**

**Decisive technical reasons (each CONFIRMED unless marked):**
1. *No local feature is superseded* — verified per-feature against Z
   (§4.7, §6.2–6.4): F1, F2, F3 (spill semantics), F4 all remain necessary.
   So migration is "port + inherit," not "port + discard."
2. *The local surface is small and additive* (~5,500 functional lines, all
   opt-in; §3) — the port is re-anchoring, not redesign.
3. *The conflict set is enumerable and low-ambiguity* (~10 files, additive
   hunks on both sides; §11).
4. *0.4.4's benefit is concentrated in the fork's core workload* — DSpark,
   multi-output sampling, transactional restore rework, KVarN tail fitting,
   CUDA KVarN route policy (§4.1, §4.2, §4.5).
5. *0.4.4's tensor-split does not displace the device-chain* for
   heterogeneous PCIe consumer hardware (§7.2) — so the fork's most
   substantial feature survives migration intact, and no "replace F3 with
   upstream" decision is forced.
6. *Staying forfeits the correctness substrate* — upstream revised the
   rollback/restore model twice (0.4.2/0.4.3) under the local DFlash work;
   staying means re-implementing those revisions locally, forever (§10).

**Major risks (and mitigations):**
- **F3 re-thread correctness** — mitigate by porting in two phases
  (§7.4): first restore exact current behavior via parameter re-threading,
  validated by the existing test suite; second, optionally re-express on
  the component taxonomy.
- **F4 vs 0.4.3 transactional restore** — the tape/backup-cell design must
  be validated against the new restore semantics before enabling the flag;
  mitigate by keeping the flag opt-in and defaulting to stock DFlash until
  the `dflash-custom-test.py` suite passes on the migrated build.
- **Speculative-init rework** — F1/F2 hunks re-verified against Z's
  `common_speculative_init_result` (same structural shape, CONFIRMED).
- **No build validation in this investigation** (research-only; builds are
  performed by the user) — the port plan must include a full
  `test-runner.py` pass on the target hardware as the acceptance gate.

**Assumptions that would change the recommendation:**
- 0.4.4's tensor-split validated on *physical* two-GPU with KVarN+DFlash
  and matching device-chain performance → F3 could be dropped (shrinks the
  port; does not change the migrate/stay choice).
- Upstream 0.4.5+ adds an independent draft-ctx knob or removes DFlash from
  `need_n_rs_seq()` → F1/F4 shrink or drop; migration still preferred.
- The target hardware moves to NVLink/PCIe x16 homogeneous GPUs → tensor-
  split becomes competitive; F3's value drops (still not zero, since
  spill-by-layer remains the right model for *heterogeneous* boxes).
- A future release rewrites the KV cache again in a way that breaks the
  additive-hunk property (e.g., removes constructor parameter threading) →
  re-evaluate at that release; the port-anchors doc keeps this cheap.

**Unresolved questions (marked UNKNOWN, not blocking the decision):**
1. *Performance:* relative throughput of 0.4.4 tensor-split vs the device-
   chain on the actual 3090+3080 box (needs a physical two-GPU run; 0.4.4
   itself has not done one).
2. *F4 + transactional restore:* whether the 0.4.3 restore rework changes
   any invariant the tape/backup design relies on (needs code-level
   validation during the port; not resolvable by diff alone).
3. *DSpark + custom tape:* whether the Markov-head intermediates are
   capturable by the same tape mechanism (post-migration design question).
4. *KVarN descriptor sizes:* exact mapping from Y's record/stage/tail size
   computation to Z's `llama-kv-tail-request` model (mechanical, resolved
   during F3 port).
