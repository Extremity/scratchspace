import os

# Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

# Use the local Speculators checkout.
sys.path.insert(0, "/home/apoc/clones/speculators/src")

from speculators.models.dflash.core import DFlashDraftModel


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STUDENT_DIR = Path(
#    "/home/apoc/data/drafters/Qwen3.8-dflash-fullpass"
    "/home/apoc/data/drafters/Qwen3.8-dflash-FINAL"
)

VERIFIER_SHIM = Path(
    "/home/apoc/data/drafters/Qwen3.8-verifier-shim"
)

TOKENIZER_DIR = Path(
    "/home/apoc/data/drafters/Qwen3.8-27B-target-meta"
)

CALIBRATION_TXT = Path(
    "/mnt/c/Local AI/data/clean_calibration.txt"
)

TARGET_BIN = Path(
    "/mnt/c/Local AI/data/dflash_alignment_targets-complete.bin"
)

TOKEN_COUNTS_FILE = Path(
    "/home/apoc/data/files/token_counts.txt"
)

OUTPUT_DIR = Path(
    "/home/apoc/data/drafters/Qwen3.8-dflash-3epoch"
)

MIN_TRAINING_TOKENS = 17

# ---------------------------------------------------------------------------
# Alignment settings
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16

TARGET_LAYER_IDS = [2, 17, 32, 47, 62, 64]
DRAFT_LAYER_COUNT = 5
HIDDEN_SIZE = 5120
DRAFT_HIDDEN_SIZE = DRAFT_LAYER_COUNT * HIDDEN_SIZE
VOCAB_SIZE = 248320
MASK_TOKEN_ID = 248070

BLOCK_SIZE = 16
MAX_ANCHORS = 10

# Conservative warm-start alignment.
#LEARNING_RATE = 1e-6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 1324*3
BYPASS_CAP = True  # When set to true, allows MAX_STEPS to be greater than the total number of prompts, ie, allows looping/epochs
GRAD_CLIP = 1.0


SEED = 38
#SEED = 69

#Skip problematic entries
SKIP_RECORDS = [1356]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gib(n):
    return n / (1024 ** 3)


def print_cuda_memory(label):
    allocated = torch.cuda.memory_allocated(DEVICE)
    reserved = torch.cuda.memory_reserved(DEVICE)
    free, total = torch.cuda.mem_get_info(DEVICE)

    print(f"\n===== CUDA memory: {label} =====")
    print(f"allocated: {gib(allocated):.2f} GiB")
    print(f"reserved:  {gib(reserved):.2f} GiB")
    print(f"free:      {gib(free):.2f} GiB")
    print(f"total:     {gib(total):.2f} GiB")


def read_i32(f):
    data = f.read(4)
    if len(data) != 4:
        raise RuntimeError("Unexpected EOF while reading int32")
    return struct.unpack("<i", data)[0]

def load_token_counts(path, expected_count):
    """
    Load the 0-based record-number -> token-count sidecar.
    This exists because some records cannot be used for training,
    and filtering must happen before MAX_STEPS is applied.

    Expected format:

        0:123
        1:87
        2:42
        ...

    The record number must match its zero-based line position.
    """

    token_counts = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record_number_text, token_count_text = line.split(":", 1)
                record_number = int(record_number_text)
                token_count = int(token_count_text)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid token-count entry at line "
                    f"{line_number + 1}: {line!r}"
                ) from exc

            expected_record_number = len(token_counts)

            if record_number != expected_record_number:
                raise RuntimeError(
                    f"Token-count sidecar is not zero-based/sequential "
                    f"at line {line_number + 1}: "
                    f"found record {record_number}, "
                    f"expected {expected_record_number}"
                )

            if token_count < 0:
                raise RuntimeError(
                    f"Invalid negative token count at record "
                    f"{record_number}: {token_count}"
                )

            token_counts.append(token_count)

    if len(token_counts) != expected_count:
        raise RuntimeError(
            f"Token-count sidecar contains {len(token_counts)} records, "
            f"but target binary contains {expected_count}"
        )

    return token_counts

def index_target_bin(path):
    """
    Index the actual dflash_alignment_targets-complete.bin format without
    loading its ~22 GB payload into memory.

    Actual format:

        int32 n_prompts_total

        repeated:
            int32 prompt_idx
            int32 n_hidden_tokens
            int32 n_layers
            int32 layer_ids[n_layers]
            float32 hidden states:
                [n_hidden_tokens, n_layers, 5120]
    """

    entries = []

    with path.open("rb") as f:
        total_prompts = read_i32(f)

        for record_number in range(total_prompts):
            header_offset = f.tell()

            prompt_idx = read_i32(f)
            n_tokens = read_i32(f)
            n_layers = read_i32(f)

            if n_layers <= 0:
                raise RuntimeError(
                    f"Record {record_number}: invalid n_layers={n_layers}"
                )

            layer_bytes = f.read(4 * n_layers)
            if len(layer_bytes) != 4 * n_layers:
                raise RuntimeError(
                    f"Record {record_number}: truncated layer ID list"
                )

            layer_ids = list(
                struct.unpack("<" + "i" * n_layers, layer_bytes)
            )

            if layer_ids != TARGET_LAYER_IDS:
                raise RuntimeError(
                    f"Record {record_number}: unexpected layers "
                    f"{layer_ids}; expected {TARGET_LAYER_IDS}"
                )

            payload_offset = f.tell()
            payload_values = n_tokens * n_layers * HIDDEN_SIZE
            payload_bytes = payload_values * 4

            entries.append(
                {
                    "record_number": record_number,
                    "header_offset": header_offset,
                    "prompt_idx": prompt_idx,
                    "n_tokens": n_tokens,
                    "n_layers": n_layers,
                    "layer_ids": layer_ids,
                    "payload_offset": payload_offset,
                    "payload_bytes": payload_bytes,
                }
            )

            f.seek(payload_bytes, os.SEEK_CUR)

        trailing = f.read(1)
        if trailing:
            raise RuntimeError(
                "Target binary contains unexpected trailing data"
            )

    if len(entries) != total_prompts:
        raise RuntimeError(
            f"Indexed {len(entries)} records but header says "
            f"{total_prompts}"
        )

    return entries


def load_hidden_record(f, entry):
    f.seek(entry["payload_offset"])

    expected_floats = (
        entry["n_tokens"]
        * entry["n_layers"]
        * HIDDEN_SIZE
    )

    raw = f.read(expected_floats * 4)

    if len(raw) != expected_floats * 4:
        raise RuntimeError(
            f"Record {entry['record_number']} payload is truncated"
        )

    # copy() is intentional: frombuffer() would otherwise produce a
    # read-only NumPy array and torch would warn when converting it.
    array = np.frombuffer(
        raw,
        dtype="<f4",
    ).reshape(
        entry["n_tokens"],
        entry["n_layers"],
        HIDDEN_SIZE,
    ).copy()

    if not np.isfinite(array).all():
        raise RuntimeError(
            f"Record {entry['record_number']} contains NaN/Inf"
        )

    return array

def load_verifier_config():
    config_path = STUDENT_DIR / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # The DFlash checkpoint's own config already contains the exact
    # 5-layer Qwen transformer configuration needed by
    # DFlashDraftModel.from_training_args().
    transformer_fields = {
        "hidden_size": raw["hidden_size"],
        "intermediate_size": raw["intermediate_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "head_dim": raw["head_dim"],
        "hidden_act": raw["hidden_act"],
        "max_position_embeddings": raw["max_position_embeddings"],
        "sliding_window": raw["sliding_window"],
        "max_window_layers": raw["max_window_layers"],
        "layer_types": raw["layer_types"],
        "rope_theta": raw["rope_theta"],
        "rope_scaling": raw["rope_scaling"],
        "rms_norm_eps": raw["rms_norm_eps"],
        "attention_bias": raw["attention_bias"],
        "attention_dropout": raw["attention_dropout"],
        "initializer_range": raw["initializer_range"],
        "tie_word_embeddings": raw["tie_word_embeddings"],
        "use_sliding_window": raw["use_sliding_window"],
        "vocab_size": raw["vocab_size"],
    }

    return AutoConfig.for_model(
        "qwen3",
        **transformer_fields,
    )

def validate_calibration(tokenizer, entries):
    with CALIBRATION_TXT.open("r", encoding="utf-8") as f:
        prompts = f.read().splitlines()

    if len(prompts) != len(entries):
        raise RuntimeError(
            f"Calibration prompt count {len(prompts)} != "
            f"binary record count {len(entries)}"
        )

    print(f"Calibration prompts: {len(prompts)}")
    print(f"Binary records:      {len(entries)}")

    token_counts = []

    for i, (prompt, entry) in enumerate(zip(prompts, entries)):
        token_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        actual = len(token_ids)
        expected = entry["n_tokens"]

        if actual != expected:
            raise RuntimeError(
                f"Token count mismatch at record {i}: "
                f"tokenizer={actual}, binary={expected}"
            )

        token_counts.append(actual)

    print(
        f"Token counts verified: min={min(token_counts)}, "
        f"max={max(token_counts)}, total={sum(token_counts)}"
    )

    return prompts


def make_batch(hidden_states, token_ids):
    """
    Convert one raw binary record into the exact tensors expected by the
    actual Speculators DFlash model.

    hidden_states:
        [N, 6, 5120]

    First five layers are DFlash conditioning states.
    Layer 64 is the final verifier hidden state used to construct teacher
    logits internally.
    """

    n_tokens = hidden_states.shape[0]

    draft_hs = torch.from_numpy(
        hidden_states[:, :DRAFT_LAYER_COUNT, :]
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    verifier_last_hs = torch.from_numpy(
        hidden_states[:, 5, :]
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    draft_hs = draft_hs.reshape(
        1,
        n_tokens,
        DRAFT_HIDDEN_SIZE,
    )

    verifier_last_hs = verifier_last_hs.unsqueeze(0)

    loss_mask = torch.ones(
        (1, n_tokens),
        dtype=torch.bool,
        device=DEVICE,
    )

    document_ids = torch.zeros(
        (1, n_tokens),
        dtype=torch.long,
        device=DEVICE,
    )

    position_ids = torch.arange(
        n_tokens,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    return {
        "hidden_states": draft_hs,
        "input_ids": input_ids,
        "verifier_last_hidden_states": verifier_last_hs,
        "loss_mask": loss_mask,
        "document_ids": document_ids,
        "position_ids": position_ids,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Qwen3.8 DFlash warm-start alignment")
    print("------------------------------------")
    print(f"Student:       {STUDENT_DIR}")
    print(f"Verifier shim: {VERIFIER_SHIM}")
    print(f"Tokenizer:     {TOKENIZER_DIR}")
    print(f"Target data:   {TARGET_BIN}")
    print(f"Output:        {OUTPUT_DIR}")
    print()
    print(f"Block size:    {BLOCK_SIZE}")
    print(f"Max anchors:   {MAX_ANCHORS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max steps:     {MAX_STEPS}")
    print(f"Optimizer:     AdamW8bit")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if not STUDENT_DIR.exists():
        raise RuntimeError(f"Missing student directory: {STUDENT_DIR}")

    if not VERIFIER_SHIM.exists():
        raise RuntimeError(f"Missing verifier shim: {VERIFIER_SHIM}")

    if not TOKENIZER_DIR.exists():
        raise RuntimeError(f"Missing tokenizer directory: {TOKENIZER_DIR}")

    if not CALIBRATION_TXT.exists():
        raise RuntimeError(f"Missing calibration file: {CALIBRATION_TXT}")

    if not TARGET_BIN.exists():
        raise RuntimeError(f"Missing target binary: {TARGET_BIN}")

    if not TOKEN_COUNTS_FILE.exists():
        raise RuntimeError(
            f"Missing token-count sidecar: {TOKEN_COUNTS_FILE}"
        )

    # -----------------------------------------------------------------------
    # Index the 22.4 GB target file.
    # -----------------------------------------------------------------------

    print("Indexing target binary...")
    entries = index_target_bin(TARGET_BIN)

    print(f"Indexed {len(entries)} target records.")

    print("\nLoading token-count sidecar...")
    token_counts = load_token_counts(
        TOKEN_COUNTS_FILE,
        len(entries),
    )

    eligible_records = [
        record_index
        for record_index, token_count in enumerate(token_counts)
        if token_count >= MIN_TRAINING_TOKENS
        and record_index not in SKIP_RECORDS
    ]

    skipped_records = [
        record_index
        for record_index, token_count in enumerate(token_counts)
        if token_count < MIN_TRAINING_TOKENS
        or record_index in SKIP_RECORDS
    ]

    print(
        f"Minimum training tokens: {MIN_TRAINING_TOKENS}"
    )
    print(
        f"Eligible records:        {len(eligible_records)}"
    )
    print(
        f"Skipped records:   {len(skipped_records)}"
    )
    if SKIP_RECORDS:
        print(
            f"Manually skipped records: {SKIP_RECORDS}"
        )
    print(
        f"Training steps this run: "
        f"{min(MAX_STEPS, len(eligible_records))}"
    )
    if skipped_records:
        print(
            f"Shortest eligible record: "
            f"{min(token_counts[i] for i in eligible_records)} tokens"
        )

    # -----------------------------------------------------------------------
    # Tokenizer.
    # -----------------------------------------------------------------------

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=True,
    )

    prompts = validate_calibration(tokenizer, entries)

    # -----------------------------------------------------------------------
    # Verifier configuration.
    # -----------------------------------------------------------------------

    print("\nLoading verifier configuration...")
    verifier_config = load_verifier_config()

    print(f"Verifier config class: {type(verifier_config).__name__}")
    print(
        f"Verifier hidden size: "
        f"{getattr(verifier_config, 'hidden_size', 'unknown')}"
    )
    print(
        f"Verifier layers: "
        f"{getattr(verifier_config, 'num_hidden_layers', 'unknown')}"
    )

    # -----------------------------------------------------------------------
    # Build the real DFlash model.
    # -----------------------------------------------------------------------

    print("\nConstructing DFlash model...")

    model = DFlashDraftModel.from_training_args(
        verifier_config=verifier_config,
        draft_vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        target_layer_ids=TARGET_LAYER_IDS[:5],
        mask_token_id=MASK_TOKEN_ID,
        verifier_name_or_path=str(VERIFIER_SHIM),
        speculative_tokens=64,
        loss_fn="ce",
        loss_implementation="eager",
    )

    # Load the 58-tensor bootstrap student checkpoint.
    checkpoint_path = STUDENT_DIR / "model.safetensors"

    print(f"Loading student checkpoint: {checkpoint_path}")

    state_dict = load_file(
        str(checkpoint_path),
        device="cpu",
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Checkpoint tensors: {len(state_dict)}")

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {unexpected}"
        )

    print(f"Expected missing frozen verifier keys: {missing}")

    model = model.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # -----------------------------------------------------------------------
    # Verify gradient configuration.
    # -----------------------------------------------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    frozen_params = [
        p for p in model.parameters()
        if not p.requires_grad
    ]

    trainable_count = sum(
        p.numel() for p in trainable_params
    )

    total_count = sum(
        p.numel() for p in model.parameters()
    )

    print()
    print(f"Total parameters:     {total_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(
        f"Trainable percentage: "
        f"{100.0 * trainable_count / total_count:.2f}%"
    )

    # The four verifier tensors must remain frozen.
    if not frozen_params:
        raise RuntimeError(
            "No frozen parameters found; verifier weights were "
            "unexpectedly made trainable."
        )

    for name, param in model.named_parameters():
        if name in {
            "embed_tokens.weight",
            "lm_head.weight",
            "verifier_lm_head.weight",
            "verifier_norm.weight",
        }:
            if param.requires_grad:
                raise RuntimeError(
                    f"{name} is unexpectedly trainable"
                )

    print("Verifier parameters confirmed frozen.")

    print_cuda_memory("after model load")

    # -----------------------------------------------------------------------
    # DFlash training kwargs.
    #
    # max_anchors=10 is the important memory-control change.
    # -----------------------------------------------------------------------

    train_kwargs, _ = DFlashDraftModel.get_trainer_kwargs(
        loss_fn="ce",
        loss_implementation="eager",
        dflash_decay_gamma=4.0,
        max_anchors=MAX_ANCHORS,
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
    )

    print("\nDFlash training kwargs:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")

    # -----------------------------------------------------------------------
    # Optimizer.
    # -----------------------------------------------------------------------

    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for this alignment run."
        ) from exc

    print("\nCreating PagedAdamW8bit optimizer...")

 #   optimizer = bnb.optim.PagedAdamW8bit(
 #       trainable_params,
 #       lr=LEARNING_RATE,
 #       weight_decay=WEIGHT_DECAY,
 #   )

    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

 #   optimizer = torch.optim.AdamW(
 #       trainable_params,
 #       lr=LEARNING_RATE,
 #       weight_decay=WEIGHT_DECAY,
 #   )


    print("\nOptimizer created successfully")

    print(f"Optimizer type: {type(optimizer)}")
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")

    for i, group in enumerate(optimizer.param_groups):
        print(
            f"  group {i}: "
            f"params={len(group['params'])}, "
            f"lr={group['lr']}"
        )

    print_cuda_memory("after optimizer creation")

    # -----------------------------------------------------------------------
    # Record order.
    #
    # Filter short sequences BEFORE applying MAX_STEPS so that MAX_STEPS
    # means actual training steps rather than raw dataset entries.
    # -----------------------------------------------------------------------

    order = eligible_records.copy()
    random.Random(SEED).shuffle(order)

    if len(order) < MAX_STEPS and not BYPASS_CAP:
        raise RuntimeError(
            f"Only {len(order)} eligible records are available, "
            f"but MAX_STEPS={MAX_STEPS}"
        )

    # -----------------------------------------------------------------------
    # Training.
    # -----------------------------------------------------------------------

    print()
    print("Beginning alignment...")
    print()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()

    with TARGET_BIN.open("rb") as target_file:
        #Modifying loop to allow epochs when toggled
        for step in range(1, MAX_STEPS + 1):
            if BYPASS_CAP:
                relative_idx = (step - 1) % len(order)

                #Check if an epoch just ended, and shuffle the prompt order if so
                if relative_idx == 0 and step > 1:
                    current_epoch = ((step - 1) // len(order)) + 1
                    print(f"\n--- Epoch {current_epoch - 1} Complete! Shuffling data for Epoch {current_epoch} ---")
                    random.Random(SEED + step).shuffle(order)

                record_index = order[relative_idx]
            else:
                record_index = order[step - 1]

            entry = entries[record_index]
            if entry["n_tokens"] < MIN_TRAINING_TOKENS:
                raise RuntimeError(
                    f"Internal filtering error: record {record_index} "
                    f"has only {entry['n_tokens']} tokens"
                )
            prompt = prompts[record_index]

            print(
                f"[{step:03d}/{MAX_STEPS}] "
                f"record={record_index} "
                f"prompt_idx={entry['prompt_idx']} "
                f"tokens={entry['n_tokens']}"
            )
            try:
                hidden_states = load_hidden_record(
                    target_file,
                    entry,
                )

                token_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]

                if len(token_ids) != entry["n_tokens"]:
                    raise RuntimeError(
                        f"Tokenizer count changed unexpectedly: "
                        f"{len(token_ids)} != {entry['n_tokens']}"
                    )

                batch = make_batch(
                    hidden_states,
                    token_ids,
                )

                del hidden_states

                print_cuda_memory(
                    f"before forward step {step}"
                )

                step_start = time.time()

                with torch.autocast(
                    device_type="cuda",
                    dtype=DTYPE,
                ):
                    #ADDEDAPOC
                  #  print("INPUT DEBUG:")
                  #  print("tokens shape:", batch["input_ids"].shape)
                  #  print("loss_mask shape:", batch["loss_mask"].shape)
                  #  print("document_ids shape:", batch["document_ids"].shape)
                  #  print("num anchors:", batch.get("num_anchors"))
                  #  print("tokens:", batch["input_ids"][0, :50])
                  #  print("loss mask:", batch["loss_mask"][0, :50])
                  #  print("document ids:", batch["document_ids"][0, :50])
                  #  print("decoded tokens:")
                  #  print(tokenizer.decode(batch["input_ids"][0].tolist(), skip_special_tokens=False))
                  #  print("individual tokens:")
                  #  for i, tok in enumerate(batch["input_ids"][0].tolist()):
                  #      print(i, tok, repr(tokenizer.decode([tok])))
                  #  print("batch keys:")
                  #  print(batch.keys())
                  #  for k, v in batch.items():
                  #      if hasattr(v, "shape"):
                  #          print(k, v.shape)

                    _draft_tokens, loss, metrics = model(
                        **batch,
                        **train_kwargs,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}: {loss}"
                    )

                print(
                    f"  forward loss = {loss.item():.6f}"
                )

                #ADDEDAPOC
              #  print_cuda_memory(
              #      f"after forward step {step} before backward"
              #  )

                loss.backward()

                #ADDEDAPOC
              #  print("  backward completed.")
              #  print_cuda_memory("after backward")
              #  print("Gradient tensors:")
              #  for i, p in enumerate(trainable_params[:3]):
              #      if p.grad is not None:
              #          print(i, p.shape, p.grad.dtype, gib(p.grad.numel() * p.grad.element_size()))

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    GRAD_CLIP,
                )

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at step {step}: "
                        f"{grad_norm}"
                    )

                #ADDEDAPOC
            #    print("Trainable parameter memory estimate:")
            #    total = sum(p.numel() for p in trainable_params)
            #    print(f"trainable params: {total:,}")
            #    print(f"fp16/bf16 size: {total * 2 / (1024**3):.2f} GiB")
            #    print(f"fp32 size: {total * 4 / (1024**3):.2f} GiB")

                #ADDEDAPOC
            #    print_cuda_memory("before optimizer.step")
            #    print("Calling optimizer.step()")
            #    print("CUDA max allocated:")
            #    print(f"{gib(torch.cuda.max_memory_allocated(DEVICE)):.2f} GiB")
            #    print("CUDA max reserved:")
            #    print(f"{gib(torch.cuda.max_memory_reserved(DEVICE)):.2f} GiB")

                optimizer.step()

                #ADDEDAPOC
            #    print("optimizer.step() completed")
            #    print_cuda_memory("after optimizer.step")
            #    print("Optimizer state entries:")
            #    print(len(optimizer.state))

                optimizer.zero_grad(set_to_none=True)

                elapsed_step = time.time() - step_start
                elapsed_total = time.time() - start_time

                print(
                    f"  grad_norm    = {float(grad_norm):.6f}"
                )
                print(
                    f"  step_time    = {elapsed_step:.2f}s"
                )
                print(
                    f"  elapsed      = {elapsed_total / 60:.1f} min"
                )

                if metrics:
                    print(f"  metrics      = {metrics}")

                del batch
                del loss
                del metrics
                del _draft_tokens

                torch.cuda.empty_cache()

            except Exception as exc:
                print()
                print("TRAINING FAILED")
                print(
                    f"step={step}, record={record_index}, "
                    f"tokens={entry['n_tokens']}"
                )
                print(f"error: {exc}")
                print()
                raise

    # -----------------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------------

    print()
    print("Alignment pass completed.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Saving model to: {OUTPUT_DIR}")

    model.save_pretrained(
        OUTPUT_DIR,
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR,
    )

    print()
    print("Saved aligned model.")
    print(f"Output directory: {OUTPUT_DIR}")
    print_cuda_memory("after save")


if __name__ == "__main__":
    main()
