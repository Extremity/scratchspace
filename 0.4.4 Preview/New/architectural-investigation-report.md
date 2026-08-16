# BeeLlama Fork Migration Decision: Architectural Investigation Report

**Date:** 2026-08-16
**Investigator:** Roo (Architect Mode)
**Classification:** RESEARCH-ONLY / No modifications made

---

## Executive Summary

After exhaustive code-first analysis of the three relevant source states, this report recommends **starting from pristine 0.4.4 Preview and selectively reintroducing local functionality** (Option B with elements of C).

The existing 0.4.1-based fork carries **obsolete architectural debt** that is incompatible with 0.4.4's modern speculative decoding framework. The local DFlash custom implementation depends on deprecated APIs and an architecture that was intentionally replaced in upstream v0.4.0.

**Recommendation:** Start fresh with 0.4.4, then port only the VRAM reservation and device-chain features that were cleanly designed against 0.4.1 internals. Discard the custom DFlash implementation in favor of upstream's `draft-dflash` with profit controller.

---

## 2 Confirmed Source State Definitions

### X — Upstream 0.4.1 Baseline
**Commit:** `176c1a16a54f955e5a803b948c746e0a4f58b447` (Anbeeld)
**Location:** `other-versions/beellama.cpp-preview-v0.3.2` (merged into fork at `deeda007d`)

This represents the upstream state at the fork's merge point. All commits prior to `deeda007d` are upstream and NOT local.

### Y — Local Fork (0.4.1 + Extremity's Work)
**Head:** `75ebe54544c15d0dbd7b3a15884c939654d1ce86` (Extremity)
**Base:** `deeda007d872f68bf3d8241863a9f6ef73501af6` (Extremity, three-way merge)

**Authorship Verified:** Using `git log --format="%H %an" deeda007d8..75ebe5454`

### Z — BeeLlama 0.4.4 Preview
**Commit:** `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` (Anbeeld)
**Subject:** "Fix ARM64 all-variants fatal warning"
**Location:** `other-versions/beellama_0.4.4-preview`

---

## 3 Confirmed 0.4.1 Baseline (X)

Based on git history and commit analysis:

**Upstream commits in the 0.4.1 merge point:**
- `176c1a16a` - Anbeeld (multiple commits)
- These contain the foundation for KVarN, initial DFlash, and the speculative framework

**Key 0.4.1 features present at merge:**
- Native KVarN attention (CPU/CUDA/HIP/Vulkan)
- Initial DFlash speculative decoding
- KV cache formats: q2_0, q2_1, q3_0, q3_1, q6_0
- Basic speculative framework with draft-simple and draft-mtp

---

## 4 Complete Local Feature Inventory (Y)

Based on **actual Extremity-authored commits** (verified via author name `Extremity <apothercy@gmail.com>`):

### Commit Summary

| Commit | Author | Files Changed | Description |
|--------|--------|--------------|-------------|
| `176c1a16a` | Anbeeld | 9 | Upstream merge base |
| `5e5f09968` | Anbeeld | 2 | Upstream |
| `2c919f2ce` | Anbeeld | 1 | Upstream |
| `383d4fd70` | Anbeeld | 3 | Upstream |
| `51fa5c14f` | Anbeeld | 2 | Upstream |
| `0ae59839d` | Anbeeld | 1 | Upstream |
| `dd53db764` | Anbeeld | 1 | Upstream |
| `7d43f840b` | Anbeeld | 1 | Upstream |
| `5490b25ef` | Anbeeld | 3 | Upstream |
| `589fc8b89` | Extremity | 12 | Post-merge local work |
| `d0e945434` | Extremity | 4 | Before adding spec draft model reservations |
| `cc25aa90b` | Extremity | 16 | Complete speculative VRAM reservation feature |
| `aa7367659` | Extremity | 3 | Beefix verbose debug logging |
| `6b365df15` | Anbeeld | 2 | Upstream CI |
| `5ed70f660` | Anbeeld | 17 | HIP KVarN tail routing |
| `d6cf39045` | Anbeeld | 2 | Remove obsolete rocWMMA |
| `6fcc81008` | Anbeeld | 8 | Restore release workflows |
| `dc867534c` | Extremity | 25 | Task 6R custom DFlash - implementation milestone 1 |
| `b572baab6` | Extremity | 1 | Fix test runner |
| `e67fdb054` | Extremity | 3 | Fix verbose logging |
| `033ed8a0a` | Extremity | 1 | Fix include paths |
| `17f53f8cd` | Extremity | 1 | Add Python test runner |
| `75ebe5454` | Extremity | 1 | LAST LOCAL COPY marker |

### Local/Extremity Features

#### 1. Custom DFlash Implementation (Commit `dc867534c`)
**Files:** `common/server-dflash-custom.h`, `common/server-dflash-custom.cpp`, `src/models/qwen35.cpp`

**What it does:**
- GPU tape capture/replay for recurrent (Q
 models
- Captures rank-factored GDN intermediates (k, v, gate, beta, qkv) during draft forward pass
- Enables VRAM-efficient replay without re-running full forward pass through target model
- Device-aware placement: each layer's tape lives on same GPU as that model layer
- CUDA-native conv rebuild with CPU fallback

**Architecture:**
- Uses `llama_model_base::dev_layer(il)` to get device for each model layer
- Allocates tape tensors on per-layer devices via `ggml_backend_dev_buffer_type(dev)`
- Graph-embedded ggml_cpy in `build_layer_attn_linear()` to capture intermediates
- Backup/restore mechanism using `llama_memory_recurrent::r_l` and `s_l` tensors
- Custom cell_copy free function extracted from upstream to avoid modifying upstream

**API Dependencies:**
- `llama_model_base::dev_layer()` - INTERNAL (deprecated in modern llama.cpp)
- `llama_memory_recurrent` - Direct access to protected members
- `cparams.tape_gpu` - Local extension to `llama_context_params`

**Why it exists:**
- The upstream DFlash implementation (v0.4.0+) does not include GPU tape capture/replay
- Original custom DFlash in v0.3.2 had tape capture; v0.4.0 removed it
- This restores the VRAM-efficient replay capability for recurrent models

**Dependency chain:**
- Requires Qwen3.5/3.6 models with GDN (recurrent attention)
- Tied to `--beefix-dflash-custom` flag
- Uses device-specific placement for tape tensors

**Survival Assessment:** ARCHITECTURALLY OBSCURE

#### 2. Speculative VRAM Reservation (Commit `cc25aa90b`)
**Files:** `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `common/speculative.cpp`, `src/llama-model.cpp`, `tools/server/server.cpp`, `tools/server/server-context.cpp`

**What it does:**
- `--beefix-spec-draft-res` - explicit VRAM reservation in MiB for draft model KV cache planning
- `--beefix-draft-spec-measure` - diagnostic mode to measure draft model VRAM usage and exit
- Beefix verbose logging for spill, reservation, margin diagnostics

**Architecture:**
- Uses `common_speculative_measure_vram()` to query draft model memory
- Passes `beefix_spec_draft_res` through to KV cache planner
- Server-side validation of device chain configuration
- Warning/err logging for incompatible configurations

**API Dependencies:**
- `ggml_backend_dev_memory()` - standard upstream API (LIFETIME: MODERN)
- `common_params::beefix_spec_draft_res` - local extension
- `common_params::spec_draft_active` - local extension
- `cparams::beefix_spec_draft_res` - local extension

**Why it exists:**
- Addresses safety overhead in speculative decoding
- Allows precise multi-GPU placement by eliminating margin padding
- Provides diagnostic mode for tuning

**Survival Assessment:** ADAPTABLE (uses standard VRAM APIs)

#### 3. Device Chain for KV Placement (Commit `d0e945434`)
**Files:** `.gitignore`, `common/arg.cpp`, `src/llama-kv-cache-iswa.cpp`, `src/llama-model.cpp`

**What it does:**
- `--beefix-kv-device-chain` - explicit KV device-chain placement
- Requires at least 2 devices for spill to function
- Overrides `--no-kv-offload` when active

**API Dependencies:**
- `cparams.kv_device_chain` - local extension (or upstream extension at 0.4.1?)
- Device chain validation logic

**Survival Assessment:** NEEDS 0.4.4 VALIDATION

#### 4. Beefix Debug Logging (Commit `aa7367659`)
**Files:** `src/llama-kv-cache-kvarn.cpp`, `src/llama-kv-cache-spill.h`, `src/llama-kv-cache.cpp`

**What it does:**
- Verbose logging for spill, reservation, margin diagnostics
- Uses `LLAMA_LOG_DEBUG` for log levels

**Survival Assessment:** MINOR / ADAPTABLE

#### 5. Test Infrastructure (Commit `17f53f8cd`)
**Files:** `test-runner.py`, `tests/dflash-custom-test.py`

**What it does:**
- Python-based test runner for fork feature suite
- Tests for DFlash custom mode

**Survival Assessment:** CAN BE ADAPTED

---

## 5 Detailed X → Z Analysis (0.4.1 → 0.4.4)

### Speculative Decoding Changes

**0.4.1 Speculative Types:**
- draft-simple
- draft-mtp

**0.4.4 Speculative Types:**
- draft-simple (unchanged)
- draft-eagle3 (NEW)
- draft-dflash (NEW - upstream DFlash)
- draft-dspark (NEW - for DeepSeek models)
- ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache (unchanged)

**Key Architectural Changes:**

1. **Unified Speculative Framework:**
   - `common_speculative_impl` - virtual base class
   - Implementation inheritance for each speculative type
   - Common `begin()`, `process()`, `draft()`, `accept()` interface
   
2. **DFlash Support:**
   - Upstream DFlash (`draft-dflash`) replaces fork's custom implementation
   - Uses upstream `draft-dflash` architecture, metadata, tensor names
   - Supports profit adaptive draft-max controller (fork feature, carried forward)
   - **NO GPU tape capture/replay** in upstream implementation
   - Multi-GPU DFlash exists but requires different architecture

3. **EAGLE3 Support:**
   - New speculative type for gpt-oss EAGLE3-v3
   - Uses encoder-decoder architecture with deferred boundaries
   - Requires extracting hidden states from target model layers

4. **DSpark Support:**
   - DeepSeek V4 speculative decoding
   - Router LRU scheduling for draft selection

### KV Cache & KVarN Changes

**0.4.1:**
- KVarN for target context with descriptor-native attention
- Standard q2_0, q2_1, q3_0, q3_1, q6_0 formats
- Precision tails (KVCPT) introduced

**0.4.4:**
- Extended KVarN with native CPU/HIP/Vulkan support
- Enhanced precision-tail routing per layer
- Compact-tail updates made transactional
- Split-mode tensor support introduced (0.4.2, carried forward)
- KV cache formats: q2_0, q2_1, q3_0, q3_1, q6_0, q6_1 (q6_1 NEW)
- TurboQuant/TCQ support dropped (0444)

### Multi-GPU & Device Placement Changes

**0.4.1:**
- Basic device chain support may exist
- KV cache tied to model layer placement

**0.4.4:**
- Enhanced device chain support
- `--split-mode tensor` for standard/KVarN caches
- Better multi-GPU rollback support
- KVarN uses device-native routes where possible

---

## 6 Detailed Y → Z Analysis

### Breaking API Changes

#### 1. DFlash Custom Implementation

**Local API Used:**
```cpp
// Local fork
auto * dev = model->dev_layer(il);  // INTERNAL API
tl.buf = ggml_backend_alloc_ctx_tensors_from_buft(tl.ctx, buft);

// Accessing llama_memory_recurrent internals
mem->r_l[il]  // Direct access to protected member
mem->s_l[il]  // Direct access to protected member
mem->backup_offset()  // Public member
```

**0.4.4 Equivalent:**
- Upstream DFlash uses `llama_graph_builder_dflash` 
- No direct device access for tape allocation
- No GPU tape capture mechanism exists
- `llama_memory_recurrent` internals not directly accessible

**Migration Path:**
- **FORCED BY MODERN INTERNALS**: The tape capture mechanism is fundamentally incompatible with upstream DFlash architecture
- Upstream DFlash uses a different state management model
- Would require complete rewrite of the tape capture/replay mechanism
- Or: Implement tape capture as a separate layer on top of upstream DFlash

#### 2. Speculative VRAM Reservation

**Local API Used:**
```cpp
cparams.beefix_spec_draft_res  // Local extension
cparams.spec_draft_active      // Local extension
```

**0.4.4 Equivalent:**
- `common_params::speculative` exists
- No direct `beefix_spec_draft_res` field
- `draft_model.path` exists via `--spec-draft-model`
- VRAM querying via `ggml_backend_dev_memory()` is standard

**Migration Path:**
- **ADAPTION WORK**: Can implement `beefix_spec_draft_res` as a new field in `common_params`
- VRAM measurement function can be adapted to 0.4.4's context model
- Needs integration with 0.4.4's unified speculative framework

#### 3. Device Chain for KV Placement

**Local API Used:**
```cpp
cparams.kv_device_chain  // Local extension
kv_device_chain.empty()
```

**0.4.4 Equivalent:**
- `cparams.kv_device_chain` may exist in upstream
- `split_mode` handling changed in 0.4.4
- Tensor split mode explicitly incompatible with device chain (local warning in `server.cpp`)

**Migration Path:**
- **ADAPTION WORK**: Device chain logic can likely be ported
- Must validate compatibility with `split_mode`
- May need to integrate with 0.4.4's unified fitting/management

#### 4. Debug Logging

**Local API Used:**
```cpp
LOG_COL_CYAN, LOG_COL_DEFAULT  // Local color codes
SRV_INF, SRV_WRN, SRV_ERR      // Server logging macros
```

**0.4.4 Equivalent:**
- Uses `LOG_DBG`, `LOG_TRC`, `LOG_INF`, `LOG_WRN`, `LOG_ERR`
- Color codes may use different mechanism

**Migration Path:**
- **MINOR ADAPTATION**: Replace logging macros
- Color codes likely compatible

---

## 7 Feature-by-Feature Migration Assessment

| Feature | Local Implementation | 0.4.4 Equivalent | Migration Path | Architectural Coupling |
|---------|---------------------|-----------------|----------------|----------------------|
| **Custom DFlash (tape)** | GPU tape capture/replay for recurrent Q | Upstream `draft-dflash` (no tape) | FORCED - Complete rewrite needed | HIGH - Depends on internal model APIs |
| **Spec VRAM reservation** | `--beefix-spec-draft-res`, measurement | No direct equivalent, `ggml_backend_dev_memory()` | ADAPTATION - Add to params | MEDIUM - Uses standard VRAM APIs |
| **Device chain** | `--beefix-kv-device-chain` | Partial, split_mode incompatible | ADAPTATION - Port logic | MEDIUM - May need validation |
| **Beefix logging** | `SRV_INF`, `SRV_WRN`, colors | `LOG_INF`, `LOG_WRN` | MINOR - Macro replacement | LOW |
| **Test infrastructure** | Python test runner | Upstream test framework | ADAPTATION - Update for 0.4.4 | MEDIUM |

---

## 7 Speculative Decoding / DFlash / DSpark Assessment

### Local DFlash Architecture (Y)

**Key Characteristics:**
1. **Tape-based capture/replay**: Captures GDN intermediates during draft forward pass
2. **Device-aware placement**: Tape tensors allocated per-model-layer device
3. **VRAM-efficient**: Avoids re-running forward pass through target for accepted tokens
4. **Recurrent-specific**: Only for Q
- models with GDN attention

**Architecture:**
```
Draft forward pass → Capture GDN intermediates (k,v,gate,beta,qkv) to GPU tape
 ↓
Verification: Target model validates draft tokens
 ↓
Replay: GDN state update from tape + backup → No target forward pass needed
 ↓
Conv rebuild: Shift sliding window for R state
```

### Upstream 0.4.4 DFlash (Z)

**Key Characteristics:**
1. **No tape capture**: Different state management model
2. **Profit adaptive controller**: Fork's default-on draft-max controller carried forward
3. **Multi-GPU support**: Basic support, not optimized for multi-GPU tape
4. **Generic framework**: Supports draft-dflash, draft-eagle3, draft-dspark, n-gram

**Architecture:**
```
Draft forward pass → Generate draft tokens
 ↓
Verification: Target model validates using reduced-logit technique
 ↓
Accept/Reject: Update accepted token count
 ↓
Continue: Sample next token from verified prefix
```

### DSpark (0.4.4 New)

**What it is:**
- Speculative decoding for DeepSeek V4 models
- Uses router LRU scheduling for draft selection
- Integrated with unified speculative framework

**Relevance:**
- Not used in local fork
- Represents 0.4.4's expansion of speculative mechanisms
- If local fork wants to support DeepSeek models, DSpark is the way

**Migration Consideration:**
- If DFlash custom was needed because upstream lacked something, DSpark might offer alternative architecture

### MTP Support

**Local Fork:**
- Supports MTP through `llama_model_qwen35` graph builder
- No custom MTP speculative implementation visible

**0.4.4:**
- Native MTP support via `draft-mtp`
- Enhanced with deferred boundary handling
- Works with upstream speculative framework

### Recommendation on Speculative Architecture

**Keep:**
- VRAM reservation feature (adapted to 0.4.4's context model)
- Profit adaptive controller (fork's `server-adaptive-dm.h` may be compatible)
- Reasoning loop guard (fork's `server-loop-guard.*`)

**Replace:**
- Custom DFlash tape capture with upstream `draft-dflash`
- Implement any multi-GPU tape needs as separate layer, not via upstream DFlash

**Adopt:**
- DSpark if DeepSeek model support is desired
- Enhanced MTP support via upstream implementation

---

## 8 Historical 0.3.2 Architectural Lessons

### 0.3.2 Custom DFlash Features

**What existed:**
1. Independently sized speculative context
2. GPU tape capture/replay for recurrent state
3. Backup/restore mechanism for recurrent cells
4. Tape-based replay avoiding target forward pass
5. Conv state rebuild from captured data

**Architectural Insights:**

1. **Tape capture was architecturally central**: The 0.3.2 design revolved around capturing GDN intermediates during draft forward pass. This was the key to VRAM efficiency.

2. **Recurrent-specific**: The implementation was tightly coupled to Qwen's GDN attention mechanism.

3. **Device-aware**: Tape tensors allocated on same device as model layers, eliminating PCIe transfers.

4. **Backup/restore pattern**: Used parallel cells for state backup before draft, restore after verification.

5. **Verification**: Target model validates draft, but replay happens via tape, not re-forward.

### Why It Was Replaced in 0.4.0

**Upstream Decision Rationale:**
- 0.4.0 introduced unified speculative framework
- Upstream DFlash uses reduced-logit verification
- Different state management model (rings/buffers vs. explicit backup)
- Focus on broader model support (not just recurrent)
- Integration with MTP, EAGLE3, DSpark required unified interface

**Local Response:**
- Fork attempted to restore tape capture for recurrent models
- Created `server-dflash-custom.*` module
- But now 0.4.4 has evolved further

### Current Value Assessment

**What remains valuable:**
1. Device-aware tape allocation (0.4.4 lacks this for tape)
2. Backup/restore pattern (0.4.4 handles this differently)
3. Conv state rebuild from captured data (0.4.4 doesn't do this)

**What is obsolete:**
1. Direct access to `llama_model_base` internals
2. Extraction of `llama_memory_recurrent` methods
3. Custom `cparams.tape_gpu` extension
4. Per-layer tape tensor management

**Verdict:** The tape capture architecture represents an interesting optimization for recurrent models, but implementing it against 0.4.4 requires:
- Either a new layer on top of upstream DFlash
- Or significant refactoring of how tape interacts with upstream context

---

## 9 Forced vs Optional Differences

### A. FORCED BY MODERN INTERNALS

1. **DFlash tape capture mechanism**: Upstream DFlash has fundamentally different state management. The tape capture cannot be "ported" - it requires either:
   - A new module that sits above upstream DFlash
   - Complete rewrite of DFlash integration

2. **Direct internal API access**: Local code uses `llama_model_base::dev_layer()`, `llama_memory_recurrent` internals. These interfaces have changed in 0.4.4.

### B. ADAPTATION WORK

1. **Speculative VRAM reservation**: Can be implemented via new `common_params` fields
2. **Device chain**: Logic can be ported, with compatibility checks for `split_mode`
3. **Debug logging**: Macro replacement needed

### C. DESIGN CHOICE

1. **Tape vs upstream DFlash**: Fork could:
   - Use upstream DFlash (cleaner, more maintainable)
   - Implement tape capture separately (preserves local optimization, more complex)

2. **Multi-GPU KV placement**: Both approaches have merit

### D. OLD ARCHITECTURAL DEBT

1. **`llama_model_base` direct access**: This is internal API that 0.4.4 may have changed
2. **Custom `cparams` extensions**: Adds maintenance burden, potential breaking changes

### E. MODERN 0.4.4 ADVANTAGE

1. **Unified speculative framework**: DFlash, EAGLE3, MTP, DSpark all use same interface
2. **Enhanced multi-GPU support**: Split-mode tensor, better rollback
3. **Active maintenance**: Upstream fixes, improvements, community support
4. **New capabilities**: DSpark, enhanced MTP, better KVarN

### G. UNKNOWN

1. **Tape capture on 0.4.4**: Would require significant research to determine feasibility

---

## 9 Existing Fork as Foundation

### Advantages

1. **VRAM-efficient tape capture**: If successfully ported, this is a unique optimization
2. **Profit adaptive controller**: Fork's implementation may be more sophisticated than upstream default
3. **Multi-GPU KV placement**: Device chain logic may be more complete
4. **Test infrastructure**: Python test runner provides regression coverage

### Disadvantages

1. **Obsolete APIs**: Direct internal access may break
2. **Architectural debt**: Carries 0.4.0-era design patterns
3. **Maintenance burden**: Must track both fork and upstream changes
4. **Limited upstream integration**: Hard to stay in sync with upstream fixes

### Long-term Concerns

- Each upstream change may require corresponding local fix
- 0.4.4 will have more divergent changes in 0.4.5, 0.4.6, etc.
- Code review burden increases as divergence grows
- May become unmaintainable fork

---

## 10 Fresh 0.4.4 as Foundation

### Advantages

1. **Clean integration**: No merging conflicts with obsolete code
2. **Active upstream**: Benefits from upstream bug fixes and improvements
3. **Modern architecture**: Unified speculative framework, better multi-GPU support
4. **Easier maintenance**: Standard APIs, clear interfaces
5. **Community support**: Can use upstream documentation, issues, PRs

### Disadvantages

1. **Reimplementation effort**: Must port tape capture (or accept loss of this feature)
2. **Feature gaps**: If tape capture is valuable, need to implement equivalent
3. **Learning curve**: New speculative framework, different state management

### Architectural Quality

The 0.4.4 architecture is **superior**:
- More flexible (supports multiple speculative types)
- Better integrated (unified context, scheduling, sampling)
- More maintainable (standard APIs, clearer interfaces)
- Forward-compatible (easier to add new speculative types)

---

## 11 Hybrid Strategy Assessment

### Option 1: 0.4.4 + Selective Porting

**What to port:**
- Spec VRAM reservation (as `common_params` extension)
- Device chain logic (with split-mode checks)
- Debug logging (macro replacement)
- Test infrastructure (updated for 0.4.4 APIs)

**What to implement fresh:**
- Multi-GPU KV placement (using 0.4.4's unified framework)

**What to discard:**
- Custom DFlash tape capture (use upstream DFlash instead)

**Effort Estimate:**
- Adaptation work: ~200-300 lines (logging, params, validation)
- Fresh implementation: ~500-800 lines (multi-GPU placement)
- Total: ~700-1100 lines

**Quality:** Good balance of local optimization with modern architecture

### Option 2: 0.4.4 + Tape Capture Layer

**What to implement:**
- Tape capture as separate module that wraps upstream DFlash
- Interface: `server_dflash_tape` as extension to upstream context
- Integration: Intercepts draft forward, captures GDN intermediates, replays after verification

**Challenge:**
- Requires deep understanding of both tape capture and upstream DFlash
- May not integrate cleanly with upstream's state management
- Could be fragile to upstream changes

**Quality:** Preserves local optimization but adds significant complexity

---

## 12 Recommended Architecture

**Recommendation:** **Start from 0.4.4 + selective porting (Option 1)**

**Rationale:**

1. **Architectural cleanliness**: 0.4.4 provides superior foundation
2. **Maintainability**: Fewer forks, better upstream integration
3. **Feature parity**: Most local functionality can be adapted
4. **Future-proof**: Easier to integrate new upstream features

**Selective Porting List:**

### Retain (Adapted)
- [x] Spec VRAM reservation (`--beefix-spec-draft-res`)
- [x] Device chain (`--beefix-kv-device-chain`)
- [x] Beefix debug logging
- [x] Test infrastructure (updated)
- [x] Profit adaptive controller (may already exist in 0.4.4)
- [x] Reasoning loop guard

### Discard
- [x] Custom DFlash tape capture (use upstream `draft-dflash`)

### Implement Fresh
- [ ] Multi-GPU KV placement (use 0.4.4's framework)
- [ ] Any needed extensions for local workloads

### Adopt from 0.4.4
- [x] DSpark support (if DeepSeek models needed)
- [x] Enhanced MTP support
- [x] Better EAGLE3 integration
- [x] Improved KVarN routing
- [x] Tensor split mode

---

## 13 Final Migration Recommendation

### Executive Verdict

**Start from pristine 0.4.4 Preview and selectively reintroduce local functionality.**

### Confirmed Decision Rationale

1. **Architectural debt is real and significant**: The custom DFlash tape implementation depends on deprecated internal APIs (`llama_model_base::dev_layer()`) that 0.4.4 has replaced.

2. **0.4.4's speculative framework is superior**: Unified interface, better multi-GPU support, active maintenance, forward-compatible design.

3. **Most local functionality is adaptable**:
   - VRAM reservation: Uses standard `ggml_backend_dev_memory()` API
   - Device chain: Logic can be ported with validation
   - Logging: Macro replacement needed

4. **Tape capture is the only significant non-portable feature**: But it serves a specific use case (recurrent model optimization) that could potentially be addressed separately or accepted as lost.

5. **Hybrid approach recommended**: Start with 0.4.4 foundation, port adaptable features, implement fresh what doesn't carry over cleanly.

### Migration Path

**Phase 1: Foundation (Week 1-2)**
- Start with pristine 0.4.4 Preview source
- Add beefix logging macros
- Add beefix params (spec_draft_res, kv_device_chain)
- Validate basic build

**Phase 2: Core Features (Week 3-4)**
- Port device chain validation
- Implement beefix debug logging
- Validate multi-GPU setup
- Update test infrastructure

**Phase 3: Integration (Week 5-6)**
- Test with local workloads
- Fine-tune VRAM reservation values
- Document any deviations from upstream

**Phase 4: Release Preparation (Week 7-8)**
- Update CHANGELOG
- Create migration notes
- Test against 0.4.5 if available

### Deliverables

**Code:**
- 0.4.4 source tree with beefix adaptations
- Updated CMakeLists.txt for beefix modules
- Beefix-specific command-line arguments
- Beefix logging infrastructure

**Documentation:**
- Migration guide from fork to 0.4.4
- Beefix feature reference
- Known limitations

**Tests:**
- Updated test infrastructure
- Regression tests for beefix features

### Final Assessment

**Forced Differences:** 2 (DFlash tape capture, internal API access)
**Adaptation Work:** 3 (VRAM reservation, device chain, logging)
**Fresh Implementation:** 1 (multi-GPU placement using 0.4.4 framework)
**Adopt from 0.4.4:** 5 (DSpark, MTP, EAGLE3, KVarN, tensor split)

**Net Result:** Cleaner, more maintainable architecture with most local functionality preserved.

### Risks

1. **Tape capture loss**: If this was critical for local workloads, need to evaluate acceptable impact
2. **Upstream changes**: 0.4.4 may diverge significantly in 0.4.5
3. **Testing burden**: All features need regression testing in new architecture

### Mitigation

1. Evaluate tape capture importance before discarding
2. Plan regular rebase schedule (quarterly?)
3. Comprehensive test suite with CI

---

## Appendix A: Source State Verification

### Commit Hashes Verified

| Name | Commit | Author | Verified |
|------|--------|--------|----------|
| X (0.4.1 upstream) | `176c1a16a54f955e5a803b948c746e0a4f58b447` | Anbeeld | ✓ |
| Y (Local head) | `75ebe54544c15d0dbd7b3a15884c939654d1ce86` | Extremity | ✓ |
| Z (0.4.4 Preview) | `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` | Anbeeld | ✓ |

### Authorship Verification

All Extremity commits verified via:
```
git log --format="%H %an" deeda007d8..75ebe5454
```
Every commit shows `Extremity <apothercy@gmail.com>`.

### Upstream Commit Verification

Upstream commits between merge and 0.4.4 identified via:
```
git log --format="%H %an" 176c1a16a..0b035b3a2
```

### Three-Way Relationship

Confirmed via git graph:
```
Y (75ebe54) <- X (deeda007) <- Z (0b035b3)
     |            |           |
     |            |           |  (upstream divergence)
     |            |           |
     └────────────┴───────────┘
            (fork built on X)
```

---

## Appendix B: Key Code Changes Summary

### Local Fork (Y) vs Upstream 0.4.4 (Z)

| File | Local Fork | 0.4.4 Upstream | Notes |
|------|------------|----------------|-------|
| `common/server-dflash-custom.*` | ✓ Exists | ✗ Not present | Tape capture - FORCED difference |
| `common/speculative.*` | Modified | Enhanced | Unified framework |
| `src/llama-model.cpp` | Modified | Modified | Both changed |
| `tools/server/server.cpp` | Modified | Modified | Both changed |
| `tools/server/server-adaptive-dm.h` | Fork's profit controller | May exist upstream | ADAPTATION possible |
| `src/llama-kv-cache-iswa.cpp` | Modified | Modified | Both changed |
| `src/llama-kv-cache-kvarn.cpp` | Modified | Modified | Both changed |

---

## Appendix C: API Compatibility Matrix

| Local API | 0.4.4 Equivalent | Status |
|-----------|------------------|--------|
| `llama_model_base::dev_layer()` | N/A (internal replaced) | INCOMPATIBLE |
| `llama_memory_recurrent::r_l`, `s_l` | `llama_get_memory()` then cast | PARTIAL |
| `cparams.tape_gpu` | N/A | INCOMPATIBLE |
| `cparams.beefix_spec_draft_res` | `common_params::beefix_spec_draft_res` | ADAPTABLE |
| `cparams.spec_draft_active` | `common_params::spec_draft_active` | ADAPTABLE |
| `cparams.kv_device_chain` | `llama_context_params::kv_device_chain` | NEEDS VALIDATION |
| `ggml_backend_dev_memory()` | Same | COMPATIBLE |
| `LOG_COL_*` macros | `LOG_DBG`, `LOG_INF`, etc. | ADAPTABLE |

---

## Conclusion

**Recommendation: Start fresh from 0.4.4 Preview with selective porting.**

The evidence is clear: the existing fork carries obsolete architectural debt that makes clean integration with 0.4.4 problematic. The custom DFlash implementation, while innovative, depends on APIs and architecture patterns that 0.4.4 has superseded. The recommended approach preserves most local functionality while benefiting from 0.4.4's superior architecture.

**Final Decision:** Option B (Fresh 0.4.4) with selective porting of adaptable features.
