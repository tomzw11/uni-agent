"""Prepare the minimal `qwenpaw` training/val parquet for a uni-agent GRPO demo.

Each row asks the QwenPaw ReAct agent (driven through its existing REST APIs by
`uni_agent.agents.qwenpaw.agent.run_qwenpaw`) to solve a toy GSM8K-style math
question. The agent is instructed to end its final answer with a line of the
form ``#### <number>``; the GSM8K reward is 1.0 iff that number matches the
ground truth (see `uni_agent.agents.qwenpaw.reward`).
"""

import argparse

import pandas as pd

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the question step by step and "
    "end your final answer with a line of the form `#### <number>`."
)

# (question, answer)
SAMPLES: list[tuple[str, str]] = [
    ("What is 2 + 3?", "5"),
    ("What is 7 * 6?", "42"),
    ("What is 15 - 9?", "6"),
    ("What is 100 / 4?", "25"),
    ("What is 3 * 8?", "24"),
    ("What is 13 + 29?", "42"),
    ("What is 50 - 17?", "33"),
    ("What is 9 * 9?", "81"),
    ("What is 12 + 34?", "46"),
    ("What is 64 / 8?", "8"),
]


def build_rows(samples: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for question, answer in samples:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        task_config = {
            "name": "qwenpaw",
            "metadata": {"ground_truth": answer},
        }
        rows.append(
            {
                "data_source": "qwenpaw",
                "prompt": prompt,
                "ability": "qwenpaw",
                "extra_info": {"tools_kwargs": {"task": task_config}},
            }
        )
    return rows


def main() -> None:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/qwenpaw")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    train = pd.DataFrame(build_rows(SAMPLES[:8]))
    val = pd.DataFrame(build_rows(SAMPLES[8:10]))

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"wrote {len(train)} train rows -> {train_path}")
    print(f"wrote {len(val)} val rows -> {val_path}")


if __name__ == "__main__":
    main()
