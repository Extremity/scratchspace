# Git Commit State Model

This project uses X/Y/Z/F variables to represent important repository states.

These values are the default baseline. If a task prompt or instruction explicitly provides different commit values, those values override this file.

## Project Origins Overview

This section provides historical context only. It explains why the repository history, merge plans, and documentation may reference unusual states or commits. Do not treat this section as defining the current project state; use the actual repository state and task instructions for decisions.

- The project originally began from "Beellama 0.4.1 Preview," an upstream GitHub project that is itself a fork of "llama.cpp." This is why many filenames contain "llama" references.
- The initial source was obtained as a ZIP archive rather than a git clone, meaning the project initially had no git history.
- Git was introduced later during development, but initially only as a local repository tracking our own work.
- While development continued, upstream Beellama released "0.4.1 actual" (the full release, not another preview version).
- We initially attempted to merge our local work with upstream 0.4.1 actual after connecting our local repository to upstream history. This produced a difficult merge situation because our local history and upstream history did not share a true common ancestor.
- We changed approaches by obtaining a fresh git clone of Beellama 0.4.1 Preview, restoring the missing upstream history.
- Our local modifications were then reapplied to this new clone. Using the commit model below, this effectively reconstructed our local delta (`Y - X`) and applied it to the new upstream-backed history.
- After this reconstruction, the project had proper shared ancestry: the upstream history, the reconstructed local fork state, and a valid merge base.
- We then performed a three-way merge with upstream Beellama 0.4.1 actual, resolved conflicts, corrected merge issues, and verified that the resulting codebase built successfully.
- This document describes the origin of the repository structure only. Additional changes may have occurred after this point; do not assume this represents the current endpoint of development.

Documentation related to the merge can be found at `plans/merge-plans/` with file-specific merge analysis at `plans/merge-plans/file-specific/`. DO NOT READ THESE FILES JUST BECAUSE THEY ARE REFERENCED HERE. They are only listed here for reference purposes and should only be utilized if you need context specifically related from the first project merge with upstream Beellama 0.4.1 Actual.

## Commit Definitions

### X - Original Fork Point
Commit: ca155ad07

- Common ancestor between our fork and upstream.
- Original BeeLlama 0.4.1 Preview codebase.
- Contains no intentional local modifications.

### Y - Local Fork State
Commit: 589fc8b89

- Our fork state before merging upstream 0.4.1.
- Represents X plus intentional local BeeLlama changes.

Conceptually:
- Y = X + (local fork changes)

### Z - Upstream Release State
Commit: 176c1a16a

- Upstream llama.cpp 0.4.1 release.
- Represents upstream changes after X.
- Contains no BeeLlama-specific modifications.

### F - Final Merge State
Commit: deeda007d

- Final repository state after merging Y and Z.
- True three-way merge using X as the merge base.
- Also contains conflict resolutions, manual fixes, and adjustments required to produce the working, final merge state.

Conceptually:
- F = merge(Y, Z, base=X)
- F = Z + (Y - X)
- Note: The formulas above describe the merge intent, not a literal patch operation.

### C - Current Working State
Commit: None (floating reference)

- Represents the current, live state of the working tree at any given moment. Unlike X/Y/Z/F, C has no fixed commit — it moves as changes are made. Used conceptually to compare current files against historical states.
- If a task needs a pinned reference point, C can be temporarily assigned to a specific commit (e.g., the latest commit) for concrete diff operations. Once that task is complete, C returns to representing the current working tree.

# Commit Comparison Rules

The following comparisons describe conceptual diffs between states.

## Fork Changes

(Y - X)

Meaning:
- All changes introduced by our fork after branching from X.
- The original BeeLlama-specific delta.

## Upstream Changes

(Z - X)

Meaning:
- All upstream changes introduced after X.
- The llama.cpp 0.4.1 release delta.

## Total Changes Since Fork Point

(F - X)

Meaning:
- Everything different in the final merged state compared to X.

## Changes Added By Merge

(F - Y)

Meaning:
- Upstream changes incorporated into our fork.

## Remaining Fork Delta

(F - Z)

Meaning:
- Changes still present in F that differ from upstream.
- The final BeeLlama-specific delta after merge.

## Fork Preservation Check

(Y - X) vs (F - Z)

Meaning:
- Compare original fork changes against the final fork-specific delta.
- Useful for verifying that local changes survived the merge.

## Upstream Preservation Check

(Z - X) vs (F - Y)

Meaning:
- Compare upstream changes against the changes incorporated into the final merge.
- Useful for verifying that upstream changes were preserved.

## Examples of 'C' usage

(C - F)

Represents:
- All differences between the current state of the codebase and the state immediately post-merge.

(C - Z)

Represents:
- Everything in the current workspace that differs from the upstream release of 0.4.1 Actual.
- Includes both surviving merge deltas and any new post-merge changes.
- Use this to answer "what makes our codebase different from the release we merged in to?" instead of "what makes our codebase different from the final merge result?"

((C - F) vs (Y - X))

Represents:
- Asking: "Of the changes between current state and post-merge, which ones are NEW (not part of the original fork delta)?"
- Useful for tracking what was added after the merge without noise from changes that already existed in the fork.

---

# General Rules

- Treat X/Y/Z/F as commit states, not branch names.
- Use these variables whenever comparing project versions or analyzing history.
- Always identify:
  - The commits being compared.
  - The merge base, if applicable.
  - Whether the comparison represents fork changes, upstream changes, surviving changes, or total changes.

These formulas are examples, not an exhaustive list. Construct additional comparisons using normal Git diff reasoning when needed.