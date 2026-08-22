# Independent Architectural Migration Analysis

**Scope:** Decide the future base of the BeeLlama/llama.cpp fork — stay on the
0.4.1-based local fork (Y), migrate it forward to 0.4.4 Preview (Z), or start
from pristine 0.4.4 and selectively recreate local functionality.

**Research-only.** No source was modified. All Git objects were read in place.

**State definitions (verified by commit metadata, not subject lines):**

| State | Commit | Author | Role |
|---|---|---|---|
| X | `ca155ad078c5d42cd2e350c3a6409d05f4e3da43` | Anbeeld | Original 0.4.1 Preview upstream |
| pre-merge local | `589fc8b8958e58d4ed256627e24c42e14405ac3c` | Extremity | Local work on top of X (parent == X, single squashed commit) |
| 0.4.1 actual merge | `176c1a16a54f955e5a803b948c746e0a4f58b447` | Anbeeld | Merge `dd53db764` + `5e5f09968` ("Merge branch 'v0.4.1'") |
| post-merge local | `deeda007d872f68bf3d8241863a9f6ef73501af6` | Extremity | Merge of `589fc8b89` + `176c1a16a` |
| **Y** | `75ebe54544c15d0dbd7b3a15884c939654d1ce86` | Extremity | LAST LOCAL COPY BEFORE 0.4.4 PREVIEW COMPARISONS |
| **Z** | `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` | Anbeeld | 0.4.4 Preview (repo: `other-versions/beellama_0.4.4-preview`) |

All commit authorship was verified with `git show -s --format=fuller`. The
0.4.4-preview repo is a full upstream clone: X, Y, and the pre-merge local
commit all exist there as identical objects, so all cross-state diffs below
were run from that single repository.
