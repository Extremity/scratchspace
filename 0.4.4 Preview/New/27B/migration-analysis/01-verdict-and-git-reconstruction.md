# 1. Executive Verdict + 2. Historical Git Reconstruction

## 1. Executive Verdict

**Preferred option: Option 2 — Migrate the existing fork to 0.4.4 Preview, carrying
forward the local features as a small, isolated, opt-in "beefix" layer.**

The decisive technical reasons (all CONFIRMED from source/Git):

1. **The local feature set is small and cleanly isolated.** The complete
   Extremity-attributed delta is ~5,500 lines across ~40 files, organized into
   four self-contained subsystems (draft-ctx, VRAM reservation/measure, KV
   device-chain, custom DFlash) plus logging and test infra. None of them
   require a structural rewrite of upstream code; they are parameter
   plumbing, opt-in modules, and additive algorithms.
2. **0.4.4 does NOT supersede any of the crucial local features.** Verified
   directly: 0.4.4 has no independent draft-context-size knob (the crucial
   `--beefix-spec-draft-ctx` capability), still includes DFlash in
   `need_n_rs_seq()` (the 5.4 GB RS-buffer problem persists), and contains
   zero `beefix` VRAM-reservation/measurement code. These features remain
   necessary on 0.4.4.
3. **0.4.4's new tensor-split KV placement does not cleanly replace the local
   KV device-chain** for this workload: it is marked EXPERIMENTAL, disables
   `--fit`, and is explicitly unvalidated on physical multi-GPU ("not physical
   two-GPU or peer-transfer results... Keep tensor KVarN and precision tails
   labeled experimental until the external two-GPU checklist is completed").
   The local device-chain is already runtime-validated on the target hardware.
4. **The migration cost is bounded and concentrated.** The local features
   touch a known, enumerable set of files; the high-churn conflict zone
   (KV cache files) is where the device-chain is *plumbed in*, not where its
   *algorithm* lives (`llama-kv-cache-spill.h`, 276 lines, self-contained and
   portable as-is). Re-plumbing is mechanical work against 0.4.4's new
   constructor signatures.
5. **Staying on 0.4.1 forfeits material upstream gains** that directly serve
   the stated workload: DSpark (a newer, stronger draft method for Qwen
   targets), MTP for Qwen3-Next/DeepSeek, KVarN prompt-cache reuse rework,
   exact/bounded KVarN tail fitting, CUDA KVarN route-policy hardening, and
   the multi-output backend sampling for speculative decoding.

**Why the other two options were rejected** (full detail in
`05-options-and-recommendation.md`):

- **Option 1 (stay) rejected** because the fork is now 3 releases behind a
  fast-moving upstream whose changes concentrate exactly in the subsystems the
  fork extends (KV cache, speculative, server). Every future upstream fix
  (KVarN correctness, DFlash, server) would have to be manually re-merged
  into a tree that already diverges in those same files. Staying converts a
  one-time bounded migration into a permanent, unbounded merge debt. The
  current fork is *stable enough* to keep using, which is precisely why
  migration is affordable — but the direction of travel (0.4.4's KVarN/spec
  improvements) is toward the features this project depends on.
- **Option 3 (pristine 0.4.4 restart) rejected** because it buys nothing
  over Option 2: the local features are small, well-understood, and
  isolated, so "recreating" them on a pristine tree is the same work as
  "porting" them, minus the benefit of a working reference implementation
  and validated runtime behavior to diff against. A pristine start also
  discards the merge history that documents *why* each local change exists,
  and it provides no reduction in future-upstream friction relative to a
  migrated fork — in both cases the fork carries the same beefix layer.

**Major risks of the recommended option:**
- Re-plumbing the KV device-chain through 0.4.4's rewritten KV cache
  constructors (standard, KVarN, ISWA, hybrid, DSV4) — the largest single
  work item; mitigated by the fact that the algorithm is isolated and the
  integration is a bounded, enumerable set of call sites.
- The custom DFlash depends on Qwen3.5 GDN graph internals that changed
  between Y and Z (the `build_layer_attn_linear` signature/flow). The hook
  point moved but the mechanism (tape capture via `ggml_cpy` in the graph)
  is unchanged in kind; requires re-validation against the new graph.
- 0.4.4's speculative.cpp was heavily reworked (595 lines changed); the
  draft-ctx and reservation plumbing must be re-verified against the new
  `common_speculative_init_result` flow.

**Assumptions that would change the recommendation:**
- If 0.4.4's tensor-split were validated on physical two-GPU hardware with
  KVarN + DFlash and matched/exceeded device-chain performance, the
  device-chain could be dropped in favor of upstream, shrinking the port.
- If the project's model set moves to architectures where tensor-split is
  mature and `--fit`-free operation is acceptable, Option 3 becomes more
  attractive (less local surface to port over time).
- If 0.4.5+ upstreams an independent draft-context-size knob or removes
  DFlash from `need_n_rs_seq()`, those two local features become redundant
  (they are the two smallest to drop).

---

## 2. Historical Git Reconstruction

### 2.1 Verified commit graph

```
X = ca155ad07  (Anbeeld)  "kvarn: fix portable prefill and Vulkan decode"
 |                     [0.4.1 Preview upstream]
 |
 +--> 589fc8b89  (Extremity)  "Confirmed good pre-merge local state."
 |        |        [parent == X, CONFIRMED: single squashed local commit]
 |        |
 |        +----> deeda007d  (Extremity)  "Post-merge (0.4.1 actual), first working build"
 |                 |        [merge: parent1 = 589fc8b89, parent2 = 176c1a16a]
 |                 |
 |                 +--> d0e945434 (Extremity) "Before adding spec draft model reservations"
 |                 +--> cc25aa90b (Extremity) "Complete speculative VRAM reservation feature..."
 |                 +--> aa7367659 (Extremity) "Add Beefix verbose debug logging..."
 |                 +--> 17f53f8cd (Extremity) "Add Python test runner for fork feature test suite"
 |                 +--> 033ed8a0a (Extremity) "Fix include paths for common/log.h..."
 |                 +--> e67fdb054 (Extremity) "Fix verbose logging: LLAMA_LOG_DEBUG..."
 |                 +--> b572baab6 (Extremity) "Fix test runner: --spec-draft-model..."
 |                 +--> dc867534c (Extremity) "Task 6R custom DFlash - implementation milestone 1"
 |                 +--> Y = 75ebe5454 (Extremity) "LAST LOCAL COPY BEFORE 0.4.4 PREVIEW COMPARISONS"
 |
176c1a16a  (Anbeeld)  "Merge branch 'v0.4.1'"   [0.4.1 actual]
 |        [merge: parent1 = dd53db764, parent2 = 5e5f09968]
 |
 +----> (merged into deeda007d above)

Z = 0b035b3a2  (Anbeeld)  "Fix ARM64 all-variants fatal warning"   [0.4.4 Preview]
```

Verified facts:
- `git log -1 --format=%P 589fc8b89` → `ca155ad07...` (X is the direct parent).
- `git log -1 --format=%P 176c1a16a` → `dd53db764... 5e5f09968...` (two Anbeeld parents).
- `git log -1 --format=%P deeda007d` → `589fc8b89... 176c1a16a...` (our merge).
- `git merge-base --is-ancestor X 176c1a16a` → YES (X is inside the 0.4.1-actual line).
- Post-merge local commits (Extremity) from `deeda007d` to Y: exactly 9 commits
  (`git log --author=Extremity deeda007d..Y`).
- The 0.4.4-preview repo contains X, Y, and `589fc8b89` as identical objects
  (`cat-file -t` = commit for all three), so cross-state diffs are object-accurate.
- `git rev-list --count X..Z` = 430 commits; `git log --author=Extremity X..Z`
  = **zero** commits → the X→Z delta is pure upstream evolution.

### 2.2 Attribution of the X→Y source delta

The raw `git diff X..Y` (100 files, +26,419/−15,632) is **not** "our local
changes." It decomposes as:

| Slice | Command | Files | +/− | Attribution |
|---|---|---|---|---|
| Pre-merge local work | `diff X..589fc8b89` | 21 | +680/−66 | 100% Extremity (device-chain + draft-ctx + ISWA/KVarN plumbing) |
| Upstream 0.4.1 Preview→actual | (inside `176c1a16a` line) | many | large | Anbeeld/upstream — NOT local |
| Merge resolution | `diff 176c1a16a..deeda007d` | 21 | +16,229/−15,613 | Mostly line-ending/reformat churn from the merge; semantically = local work re-applied onto 0.4.1-actual |
| Post-merge local work | `diff deeda007d..Y` | 37 | +4,828/−96 | 100% Extremity (custom DFlash, VRAM reservation, logging, tests) |

The merge-resolution slice shows huge churn (e.g. `llama-kv-cache.cpp`
+13,002/−...) because the three-way merge re-baselined those files onto the
0.4.1-actual version while preserving the local hunks; the *semantic* local
content in that slice is identical to the pre-merge slice's content.

**Therefore the true local modification inventory is the union of:**
- `diff X..589fc8b89` (pre-merge), and
- `diff deeda007d..Y` (post-merge),
which is itemized in `02-local-inventory.md`.

### 2.3 Consequence for Y→Z analysis

Because Y already contains 0.4.1-actual (via the merge), the Y→Z delta
(1,239 stat lines) is: (upstream 0.4.1-actual → 0.4.4) + (local features
being removed/replaced). The local features appear in Y→Z as *deletions*
(e.g. `common/server-dflash-custom.cpp | 862 -`, `src/llama-kv-cache-spill.h |
276 -`, `test-runner.py | 1913 -`), which is the correct signal that they
would be dropped by a naive migration and must be consciously re-applied.
