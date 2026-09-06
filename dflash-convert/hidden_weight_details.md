# DATA SPECIFICATION: Qwen3.8-27B DFlash Alignment Hidden-State Dataset

**File Name:** `dflash_alignment_targets.bin`
**Purpose:** Precomputed Qwen3.8-27B teacher hidden states for Qwen3.8-specific DFlash drafter alignment/fine-tuning
**Teacher Model:** Qwen3.8-27B Q8_0 GGUF
**Hidden Dimension:** `5120`
**Extracted Layers:** `2, 17, 32, 47, 62`
**Prompt/Example Count:** `1,499`
**Total Hidden-State Tokens:** `182,505`
**File Size:** `18,688,559,972` bytes

---

## 1. Purpose

`dflash_alignment_targets.bin` is the precomputed teacher-side hidden-state dataset used for adapting the transplanted Qwen3.6 DFlash drafter to Qwen3.8.

The dataset contains hidden-state representations produced by the **Qwen3.8-27B teacher model** for the 1,499 calibration examples in:

`clean_calibration.txt`

The file is intended to avoid repeatedly running the full Qwen3.8 teacher during DFlash alignment/fine-tuning.

This is **not** the DFlash model itself and does not contain DFlash weights. It contains the teacher hidden-state targets that the DFlash model is trained against.

---

## 2. Important distinction from the previous specification

The previous version of this document incorrectly described this file as a stream of 118 blocks containing hidden states followed by `[64, 248320]` target-logit matrices.

That interpretation is **superseded and incorrect for the actual file**.

The actual `dflash_alignment_targets.bin` on disk was validated directly and contains:

* 1,499 prompt records
* hidden states only
* five hidden-state layers per prompt
* no target-logit section
* no 118-block structure
* no trailing data

The actual binary structure described below is authoritative.

---

## 3. Binary Layout

The file begins with a global prompt-count header:

```
int32 n_prompts_total
```

For this dataset:

```
n_prompts_total = 1499
```

Each prompt record then has the following structure:

```
int32 prompt_idx
int32 n_hidden_tokens
int32 n_layers
int32 layer_idx[5]

float32 hidden_states_layer_0[n_hidden_tokens][5120]
float32 hidden_states_layer_1[n_hidden_tokens][5120]
float32 hidden_states_layer_2[n_hidden_tokens][5120]
float32 hidden_states_layer_3[n_hidden_tokens][5120]
float32 hidden_states_layer_4[n_hidden_tokens][5120]
```

All integer values are little-endian signed 32-bit integers.

All hidden-state values are little-endian IEEE-754 float32 values.

There is no padding between fields.

There is no per-record logits section.

---

## 4. Per-Prompt Header

Every prompt record begins with 8 integer values:

```
prompt_idx
n_hidden_tokens
n_layers
layer_idx[0]
layer_idx[1]
layer_idx[2]
layer_idx[3]
layer_idx[4]
```

That is:

```
8 × 4 bytes = 32 bytes
```

The expected layer indices in every record are:

```
[2, 17, 32, 47, 62]
```

The `prompt_idx` identifies the corresponding example from the calibration dataset.

`n_hidden_tokens` specifies the number of hidden-state vectors stored for that prompt.

`n_layers` is `5`.

---

## 5. Hidden-State Payload

For a prompt containing `N` hidden-state tokens, the payload contains:

```
5 × N × 5120 × 4 bytes
```

of float32 data.

Equivalently:

```
N × 102400 bytes
```

The data is **layer-major**.

The five matrices are stored consecutively:

```
[layer 2:  N × 5120 float32]
[layer 17: N × 5120 float32]
[layer 32: N × 5120 float32]
[layer 47: N × 5120 float32]
[layer 62: N × 5120 float32]
```

The vectors are therefore not interleaved by layer.

A parser should read the five arrays separately.

---

## 6. Exact Size Accounting

The validated dataset contains:

```
Prompt records:       1,499
Total hidden tokens:  182,505
```

The hidden-state payload therefore contains:

```
182,505 × 5 × 5,120 × 4
= 18,688,512,000 bytes
```

The global header occupies:

```
4 bytes
```

The 1,499 per-prompt headers occupy:

```
1,499 × 32
= 47,968 bytes
```

Therefore:

```
4 + 47,968 + 18,688,512,000
= 18,688,559,972 bytes
```

This exactly matches the actual file size:

```
18,688,559,972 bytes
```

There are therefore **zero trailing bytes**.

---

## 7. Validated Dataset Statistics

The actual file was independently validated with the following results:

```
File size:             18,688,559,972 bytes
Prompt count:          1,499
Minimum token count:   1
Maximum token count:   5,179
Total hidden tokens:   182,505

FP32 values:           4,672,128,000
FP32 bytes:            18,688,512,000

NaN values:             0
Inf values:             0
Trailing bytes:         0
```

Every prompt contains exactly:

```
[2, 17, 32, 47, 62]
```

as its five stored layer indices.

The first hidden-state values were also checked for valid finite float32 values.

---

## 8. Reference Python Parser

A minimal parser for the actual format is:

```
import struct
import numpy as np


HEADER = struct.Struct("<iiiiiiii")


def read_dataset(path):
    with open(path, "rb") as f:
        n_prompts_total = struct.unpack("<i", f.read(4))[0]

        for _ in range(n_prompts_total):
            (
                prompt_idx,
                n_hidden_tokens,
                n_layers,
                layer0,
                layer1,
                layer2,
                layer3,
                layer4,
            ) = HEADER.unpack(f.read(HEADER.size))

            layer_indices = [
                layer0,
                layer1,
                layer2,
                layer3,
                layer4,
            ]

            if n_layers != 5:
                raise ValueError(
                    f"Expected 5 layers, got {n_layers}"
                )

            hidden_states = []

            for _ in range(5):
                count = n_hidden_tokens * 5120
                data = np.frombuffer(
                    f.read(count * 4),
                    dtype="<f4",
                    count=count,
                ).reshape(n_hidden_tokens, 5120)

                hidden_states.append(data)

            yield {
                "prompt_idx": prompt_idx,
                "n_hidden_tokens": n_hidden_tokens,
                "layer_indices": layer_indices,
                "hidden_states": hidden_states,
            }
```

For large-scale training, the file should preferably be accessed with memory-mapped or otherwise streaming I/O rather than loading the entire dataset into RAM at once.

---

## 9. Layer-Index Convention

There is an important distinction between the layer indices stored in this binary file and the indices used by the DFlash configuration.

The extraction file uses:

```
[2, 17, 32, 47, 62]
```

The DFlash configuration uses:

```
[1, 16, 31, 46, 61]
```

These are intentional and describe the same five target representations.

The reason is that llama.cpp's extraction layer numbering and the DFlash/Transformers target-layer convention are offset by one.

Therefore:

**Do not change the layer indices in this binary file.**

The correct relationship is:

```
Raw extraction:       [2, 17, 32, 47, 62]
DFlash configuration: [1, 16, 31, 46, 61]
```

---

## 10. What the File Does and Does Not Contain

This file contains:

* Qwen3.8-27B teacher hidden states
* five target layers
* variable-length hidden-state sequences
* prompt/example identifiers
* the layer identifiers required to interpret each record

This file does **not** contain:

* the Qwen3.8 tokenizer
* input token IDs
* response/label token IDs
* assistant-loss masks
* target-model logits
* DFlash model weights
* optimizer state
* training checkpoints

The hidden-state dataset therefore needs to be paired with the corresponding token/label information when constructing the actual DFlash fine-tuning samples.

In particular, the absence of logits is intentional for the conservative alignment baseline. Hard cross-entropy training can use the actual target token IDs as labels without requiring a stored full-vocabulary logit matrix.

---

## 11. Relationship to the Calibration Dataset

The 1,499 records correspond to the examples in:

```
clean_calibration.txt
```

The binary file is the precomputed teacher-side representation of those examples.

The ordering and `prompt_idx` fields allow the hidden-state records to be associated with their corresponding calibration examples.

The `.bin` file should therefore be treated as a completed teacher hidden-state dataset rather than as a prompt-generation input file.

---

## 12. Memory and I/O Considerations

The file is approximately 18.7 GB and should not be loaded into ordinary Python memory as one giant object.

A training data loader should process individual prompt records or selected portions of records incrementally.

`numpy.memmap` or equivalent random-access/streaming mechanisms may be used where appropriate, but the file's variable-length record structure means a parser still needs to respect the per-record headers when locating each payload.

The important invariant is that the parser must advance by:

```
32 + (n_hidden_tokens × 5 × 5120 × 4)
```

bytes for each prompt record.

---

## 13. Integrity Invariants

A valid copy of this dataset should satisfy all of the following:

```
global prompt count = 1499

every record:
    n_layers = 5
    layer_indices = [2, 17, 32, 47, 62]

total hidden tokens = 182,505

total FP32 hidden-state values = 4,672,128,000

total file size = 18,688,559,972 bytes

NaN count = 0
Inf count = 0
trailing bytes = 0
```

If these invariants hold, the binary layout is consistent with the validated dataset.

---

## 14. Summary

`dflash_alignment_targets.bin` is a **precomputed Qwen3.8-27B teacher hidden-state dataset** for the Qwen3.6 → Qwen3.8 DFlash transfer/alignment work.

Its actual structure is:

```
GLOBAL:
    int32 n_prompts_total

FOR EACH OF 1,499 PROMPTS:
    int32 prompt_idx
    int32 n_hidden_tokens
    int32 n_layers          # always 5
    int32 layer_idx[5]      # [2,17,32,47,62]

    float32 [N, 5120]       # layer 2
    float32 [N, 5120]       # layer 17
    float32 [N, 5120]       # layer 32
    float32 [N, 5120]       # layer 47
    float32 [N, 5120]       # layer 62
```

The old 118-block / 64-token-logit interpretation should **not** be used for this file.
