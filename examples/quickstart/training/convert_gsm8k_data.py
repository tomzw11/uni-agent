"""Convert the full GSM8K dataset to the uni-agent `qwenpaw` training/val parquet.

The remote training machine keeps the full GSM8K dataset **locally** on disk
(HF-format copy of ``openai/gsm8k``, main split). This script reads that local
path and emits the same row schema as ``prepare_qwenpaw_data.py`` (smoke data),
so the runner / reward / training script work identically for both:

    prompt (system + user)  |  extra_info.tools_kwargs.task.metadata.ground_truth

The ground truth is extracted from GSM8K's ``answer`` field (``... #### <number>``),
matching what `uni_agent.agents.qwenpaw.reward.GSM8KReward` expects from the model.

Usage (on the remote machine, from the uni-agent repo root)::

    python3 examples/quickstart/training/convert_gsm8k_data.py \
        --local-dataset-path /data/datasets/gsm8k \
        --local-save-dir ~/data/qwenpaw_gsm8k

``--local-dataset-path`` must point at a local HF-format copy of ``openai/gsm8k``
(main split). If omitted, the script falls back to downloading ``openai/gsm8k``
from the Hub -- only for local development; the remote run should always use the
local path.
"""

import argparse
import os
import re

import datasets
import pandas as pd

from prepare_qwenpaw_data import build_rows

# GSM8K answers end with a line like "#### <number>"
_HASH_ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:[.,]\d+)?)")


def extract_answer(answer_raw: str) -> str:
    """Extract the final numeric answer from a GSM8K ``answer`` string."""
    match = _HASH_ANSWER_PATTERN.search(answer_raw)
    if match is None:
        raise ValueError(f"GSM8K answer has no `#### <number>`: {answer_raw[:80]!r}")
    return match.group(1).replace(",", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dataset-path",
        default=None,
        help=(
            "Local HF-format copy of `openai/gsm8k` (main split). On the remote "
            "NPU machine the full dataset is kept locally, so pass that path here; "
            "nothing is downloaded at runtime. If omitted, `openai/gsm8k` is "
            "downloaded from the Hub (local development only)."
        ),
    )
    parser.add_argument("--local-save-dir", default="~/data/qwenpaw_gsm8k")
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Subsample the first N train rows (default: all 7473).",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Subsample the first N val rows (default: all 1319).",
    )
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main")

    def rows_for(split: str, limit: int | None) -> list[dict]:
        samples: list[tuple[str, str]] = []
        for example in dataset[split]:
            samples.append((example["question"], extract_answer(example["answer"])))
            if limit is not None and len(samples) >= limit:
                break
        return build_rows(samples)

    train = pd.DataFrame(rows_for("train", args.num_train))
    val = pd.DataFrame(rows_for("test", args.num_val))

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"wrote {len(train)} train rows -> {train_path}")
    print(f"wrote {len(val)} val rows -> {val_path}")


if __name__ == "__main__":
    main()
