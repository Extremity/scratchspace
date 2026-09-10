#!/usr/bin/env python3

import struct
from pathlib import Path

# This file starts ordering at 0. We used an "awk" command to fix the numbering, to go from 1 to the final value. Eventually we should fix this to do that...
# ...but this is here so you don't forget.

PATH = "/mnt/c/Local AI/data/extractions/dflash_alignment_10k_code750.bin"
OUT_PATH = Path("token_counts_code.txt")


def read_i32(f, desc):
    data = f.read(4)
    if len(data) != 4:
        raise RuntimeError(f"Unexpected EOF reading {desc}")
    return struct.unpack("<i", data)[0]


def main():
    print(f"Reading: {PATH}")

    with open(PATH, "rb") as f:

        total_prompts = read_i32(
            f,
            "prompt count"
        )

        print(f"Total prompts: {total_prompts}")

        with OUT_PATH.open("w") as out:

            for record in range(total_prompts):

                prompt_idx = read_i32(
                    f,
                    f"prompt {record} index"
                )

                n_tokens = read_i32(
                    f,
                    f"prompt {record} token count"
                )

                n_layers = read_i32(
                    f,
                    f"prompt {record} layer count"
                )

                # Skip layer IDs
                f.seek(
                    n_layers * 4,
                    1
                )

                # Write using stored prompt index so it can map back to txt
                out.write(
                    f"{prompt_idx}:{n_tokens}\n"
                )

                # Skip hidden states
                hidden_values = n_tokens * 5120

                skip_bytes = (
                    n_layers
                    * hidden_values
                    * 4
                )

                f.seek(
                    skip_bytes,
                    1
                )

                if (record + 1) % 100 == 0:
                    print(
                        f"Processed {record + 1}/{total_prompts}"
                    )

    print()
    print(f"Wrote: {OUT_PATH}")
    print("Complete.")


if __name__ == "__main__":
    main()
