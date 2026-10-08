"""Prepare the minimal `react_simple` training/val parquet for a uni-agent GRPO demo.

Each row asks the react agent to run a shell command in a `local` sandbox; reward
= 1.0 iff the command stdout matches the expected output (see
`uni_agent.tasks.react_simple.task`).
"""

import argparse

import pandas as pd

SYSTEM_PROMPT = (
    "You are a helpful agent that solves simple shell tasks. "
    "Use the shell tool to run commands and report the result."
)

# (question, command, expected stdout)
SAMPLES: list[tuple[str, str, str]] = [
    ("What does `echo hello` print?", "echo hello", "hello"),
    ("What does `echo uni-agent` print?", "echo uni-agent", "uni-agent"),
    ("What is 2 + 2? Use the shell to compute it.", "echo $((2 + 2))", "4"),
    ("What is 3 * 7? Use the shell to compute it.", "echo $((3 * 7))", "21"),
    ("What does `echo GRPO` print?", "echo GRPO", "GRPO"),
    ("What does `echo Megatron` print?", "echo Megatron", "Megatron"),
    ("What does `echo verl` print?", "echo verl", "verl"),
    ("What is 10 - 4? Use the shell to compute it.", "echo $((10 - 4))", "6"),
    ("What is 8 / 2? Use the shell to compute it.", "echo $((8 / 2))", "4"),
    ("What does `echo success` print?", "echo success", "success"),
]


def build_rows(samples: list[tuple[str, str, str]]) -> list[dict]:
    rows = []
    for question, command, expected in samples:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        task_config = {
            "name": "react_simple",
            "sandbox": {"provider": "local"},
            "prompt": prompt,
            "metadata": {"command": command, "expected": expected},
        }
        rows.append(
            {
                "data_source": "react_simple",
                "prompt": prompt,
                "ability": "react_simple",
                "extra_info": {"tools_kwargs": {"task": task_config}},
            }
        )
    return rows


def main() -> None:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/react_simple")
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
