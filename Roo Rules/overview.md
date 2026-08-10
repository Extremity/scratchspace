# llama.cpp KV / KVarN Multi-GPU Investigation Notes

# Disclaimer

This overview was created before the merge between our fork and upstream Beellama 0.4.1 Actual. Files may have changed in ways not represented here. Additionally, it may not be an exhaustive list of all features we plan to add. Treat file information and code examples in this document as a supplemental source of knowledge of the original state of the project before the merge, but do not treat it as a source of truth if actual file contents conflict with what is written here, as they may have changed post-merge. The only exception is the "Rules" section which should always be followed and considered up-to-date.

# Rules

### Behavior

- Do not create temporary files alongside actual project files; place them in `roo-temp\`. You do not have file deletion permission; if you create a temporary file please list it at the end of your task so the user can delete it if necessary.
- Place all plans, overviews, analysis, and other various documents created for reference purposes within the `plans\` directory. If you will be creating several of these files within the same context, consider creating a subdirectory first.
- During repair of unresolved Git merge conflicts, if an `apply_diff` SEARCH block must include literal Git conflict markers (`<<<<<<< HEAD`, `=======`, or `>>>>>>> branch`), prefix the marker with a single backslash (for example, `\<<<<<<< HEAD`). This escaping is required by the `apply_diff` parser, not by Git itself.
- The codebase is indexed and semantic search is available through the `codebase_search` tool. Use it for conceptual or intent-based searches when you know what you are trying to find but not necessarily the exact filename, symbol, or wording. Prefer semantic search for questions about how functionality works, where a behavior is implemented, or locating related code across multiple files. Use standard search when you need exact matches for known names, strings, symbols, or parameters.

### Project Memory

- Utilize the Memory MCP tool when attempting to understand project history, reasoning, or intent to help complete a task. Use it to determine whether previous investigations, decisions, explanations, or discoveries already exist before repeating expensive analysis. Do not use it as a replacement for reading the current repository; current code and project files remain the source of truth for current implementation details.
- Update the Memory MCP tool when you discover durable project knowledge that could save future tasks meaningful investigation. Prefer saving:
  - architectural decisions and the reasoning behind them
  - explanations of why changes were made
  - investigations that required significant effort and their conclusions
  - approaches that were attempted and failed, including why they failed
  - important tradeoffs, constraints, limitations, or design assumptions
  - build environment decisions and recurring workflows
  - project conventions or historical context that are not obvious from the code itself
- If you search the Memory MCP for information and no relevant entry exists, but you then investigate the issue yourself and determine an answer that would be useful to a future task, add that conclusion to the Memory MCP.
- Do not update the Memory MCP with information that is easily recovered from the repository or likely to become stale, such as current function signatures, file contents, line numbers, temporary debugging state, or descriptions of the current implementation without broader context.
- Maintain Memory MCP accuracy. If an existing entry is discovered to be stale, incorrect, or misleading, update or remove it rather than allowing outdated information to persist.

# Table of Contents

| Section | Purpose | Approx. Location |
|---|---|---|
| 1. Objective | What we are trying to accomplish | Line ~15 |
| 2. Current Findings Summary | Confirmed facts so far | Line ~35 |
| 3. Current KV Allocation Flow | Where placement currently happens | Line ~75 |
| 4. llama_kv_cache.cpp Findings | Relevant code observations | Line ~130 |
| 5. KVarN Relationship | How KVarN relates to normal KV | Line ~210 |
| 6. Important Unknowns | Questions still requiring investigation | Line ~260 |
| 7. Potential Patch Direction | Current hypothesis only | Line ~300 |
| 8. Investigation History | Why certain paths were checked | Line ~360 |

---

# 1. Objective

We are investigating how to modify llama.cpp/beellama so KV cache can intelligently use multiple GPUs.

Desired behavior:

- Keep the main model placement unchanged.
- Prefer allocating KV cache on the same GPU as the model.
- If that GPU does not have enough memory, allow KV allocations to spill to another GPU.
- Do not move the entire KV cache unless necessary.
- Do not require manually assigning layers to another GPU.
- Preserve compatibility with KVarN and standard KV paths.

The goal is dynamic KV placement/fallback, not normal tensor split.

---

# 2. Current Findings Summary

Confirmed:

- KV placement is currently tied to model layer placement.
- The KV allocation path selects a backend buffer type before allocation occurs.
- The allocator does not appear to make a "try CUDA0, then CUDA1" decision automatically.
- The likely modification point is before the backend buffer type is passed into the KV allocator.

Important discovered code:

llama-kv-cache.cpp

Current flow appears to be:

model layer device
    |
    v
model.dev_layer(il)
    |
    v
ggml_backend_dev_buffer_type(dev)
    |
    v
ctx_for_buft(buft)
    |
    v
ggml_backend_alloc_ctx_tensors_from_buft()

---

# 3. Current KV Allocation Flow

Relevant code:

Inside `llama_kv_cache::llama_kv_cache()`

Each layer determines its buffer type:

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    if (offload) {
        auto * dev = model.dev_layer(il);
        buft = ggml_backend_dev_buffer_type(dev);

        dev_name = ggml_backend_dev_name(dev);
    }

Important:

`offload` is a constructor parameter. It is not decided inside this file.

The caller determines whether KV offloading is enabled.

The important operation is:

    model.dev_layer(il)

This returns the device assigned to the corresponding model layer.

Then:

    ggml_backend_dev_buffer_type(dev)

converts that device into a backend buffer type.

The flow is:

    model layer device
            |
            v
    model.dev_layer(il)
            |
            v
    ggml_backend_dev_buffer_type(dev)
            |
            v
    ctx_for_buft(buft)
            |
            v
    ggml_backend_alloc_ctx_tensors_from_buft()

The allocator is not choosing placement.

The placement decision has already happened before allocation.

---

# 4. llama_kv_cache.cpp Findings

Important structures:

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

This map only stores contexts grouped by backend buffer type.

Example:

    CUDA0 buffer type -> CUDA0 ggml context
    CUDA1 buffer type -> CUDA1 ggml context

It does not decide:

- which GPU receives KV
- whether to spill
- whether another GPU has free memory

Function:

    ctx_for_buft(buft)

Purpose:

Given a selected buffer type:

1. Check whether a context already exists.
2. Create one if needed.
3. Return that context.

It receives the decision. It does not make the decision.

---

# 5. KV Allocation Happens After Placement

After all layers have created their tensors:

    for (auto & [buft, ctx] : ctx_map)

the code allocates buffers:

    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);

At this point the backend has already been selected.

The allocator sees:

"Allocate these tensors on CUDA0."

It does NOT see:

"Try CUDA0. If full, try CUDA1."

Therefore automatic KV spill likely requires changing the buffer assignment before this point.

---

# 6. Current Placement Logic

Current behavior:

    model layer
          |
          v
    assigned device
          |
          v
    KV cache follows that device


Desired behavior:

    model layer
          |
          v
    preferred KV device
          |
          v
    check available memory
          |
          +---- enough space ---> allocate there
          |
          +---- insufficient ---> spill KV allocation elsewhere

The likely patch point is between:

    model.dev_layer(il)

and:

    ctx_for_buft(buft)

because that is where the selected buffer type is created.

---

# 7. KVarN Relationship

Current runtime is using KVarN.

Example log:

    KVarN cache:
    type = kvarn_k6v6_g128

Memory:

    CUDA0 KVarN buffer size = 1016.88 MiB
    CUDA0 KVarN tail buffer size = 96.00 MiB

    Total KVarN = 1112.88 MiB
    Equivalent F16 = 2512 MiB


Important:

KVarN and standard KV are related but not necessarily identical paths.

Need to verify:

- Does KVarN use the same `llama_kv_cache` constructor?
- Does KVarN have its own allocation path?
- Does changing normal KV placement automatically affect KVarN?

Do not assume either direction until traced.

---

# 8. Important Unknowns

Need to investigate:

## Constructor callers

Search:

    new llama_kv_cache

Found locations:

- llama-kv-cache-dsa.cpp
- llama-memory-hybrid-iswa.cpp
- llama-memory-hybrid.cpp
- llama-model.cpp


Need to determine:

- who creates the cache
- when it happens relative to draft/speculative model loading
- whether draft model memory can be known before KV allocation


## KVarN

Search:

    KVarN

Need to find:

- initialization path
- whether it shares placement code
- whether tails are handled separately


## Backend memory APIs

Need to find existing functions for:

- querying free VRAM
- testing allocation size
- fallback allocation

---

# 9. Current Hypothesis

The cleanest patch is probably not modifying the allocator.

Instead:

Before:

    buft = ggml_backend_dev_buffer_type(dev);

add logic:

    preferred device = model layer device

    if preferred device has enough free memory:
        use preferred device

    else:
        select fallback device

Then the existing allocation system should continue working because it already supports multiple buffer types through `ctx_map`.

---

# 10. Investigation Rules

When continuing:

- Use semantic search first.
- Locate symbols before reading large files.
- Trace callers and callees.
- Do not modify code until allocation flow is completely understood.

Important searches:

    new llama_kv_cache

    KVarN

    ggml_backend_alloc_ctx_tensors_from_buft

    ggml_backend_dev_buffer_type

    model.dev_layer
