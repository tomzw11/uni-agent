"""finance_kb task: shell agent over the synthetic finance knowledge base.

Per-sample data rides in ``TaskConfig.metadata`` (as written by
``gen_finance_kb.py``): ``ground_truth``, ``answer_type``, ``qid``, ``level``,
``evidence_keys``, ``required_files`` and the mount name ``kb_root`` (default
``kb``). The generated KB directory is configured once via ``kb_source`` in the
task config YAML; it is copied read-only into the session workdir so the
policy's shell sees it at ``kb/`` and cannot alter it.

The session workdir for the ``local`` sandbox provider is the runner process's
cwd (LocalSandbox spawns the stateful tmux shell there), so launch the runner
from a neutral directory -- never from the repo or ``out/`` -- to keep parquet
and generated artifacts out of the policy's view.

Reward: the reply's final ``#### <answer>`` line compared against
``metadata.ground_truth`` -- numbers at two decimals, identifiers exact.
``evidence_keys`` observed in tool outputs and ``required_files`` touched by
shell commands are recorded in ``extra_info`` for retrieval diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


class FinanceKbTaskConfig(TaskConfig):
    name: str = "finance_kb"
    kb_source: str = Field(
        default="",
        description="Absolute path to the generated KB directory copied into the session workdir.",
    )
    workdir: str = Field(
        default="",
        description="Session workdir receiving the read-only kb/ mount; empty means the process cwd.",
    )


def _extract_answer(text: str) -> str | None:
    """Return the value of the last ``#### <answer>`` line, or None."""
    for line in reversed([item.strip() for item in text.strip().splitlines()]):
        if line.startswith("####"):
            return line[4:].strip() or None
    return None


def _is_correct(predicted: str | None, gold: str, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type == "number":
        try:
            return round(float(predicted.replace(",", "")), 2) == round(float(gold), 2)
        except ValueError:
            return False
    return predicted.strip() == str(gold).strip()


def _finish_answer(tool_calls: list[dict[str, Any]]) -> str | None:
    """Answer passed to a ``finish`` tool call, if any."""
    for tool_call in tool_calls:
        if tool_call.get("function", {}).get("name") != "finish":
            continue
        raw = tool_call.get("function", {}).get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            continue
        answer = args.get("answer")
        if answer:
            return _extract_answer(str(answer)) or str(answer).strip()
    return None


@register_task("finance_kb")
class FinanceKbTask(Task):
    name = "finance_kb"
    config_model = FinanceKbTaskConfig

    def _mount_kb(self, kb_name: str) -> Path:
        cfg: FinanceKbTaskConfig = self.config  # type: ignore[assignment]
        if not cfg.kb_source:
            raise ValueError("finance_kb: config.kb_source is not set (path to the generated kb/ directory)")
        source = Path(cfg.kb_source)
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.is_dir():
            raise FileNotFoundError(f"finance_kb: kb_source not found: {source}")

        workdir = Path(cfg.workdir) if cfg.workdir else Path.cwd()
        workdir.mkdir(parents=True, exist_ok=True)
        target = workdir / kb_name
        if not target.exists():
            staging = workdir / f".{kb_name}.staging.{os.getpid()}"
            shutil.copytree(source, staging)
            for path in sorted(staging.rglob("*"), reverse=True):
                path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            staging.chmod(staging.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            try:
                staging.rename(target)
            except OSError:
                shutil.rmtree(staging, ignore_errors=True)  # another rollout won the race
        logger.info("finance_kb kb mounted read-only at %s", target)
        return target

    def _inspect(self, transcript: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        evidence_keys = [str(key) for key in metadata.get("evidence_keys", [])]
        required_files = [str(path) for path in metadata.get("required_files", [])]
        tool_outputs: list[str] = []
        shell_commands: list[str] = []
        for message in transcript:
            for tool_call in message.get("tool_calls") or []:
                name = tool_call.get("function", {}).get("name", "")
                arguments = tool_call.get("function", {}).get("arguments", "")
                if name in ("shell", "stateful_shell"):
                    shell_commands.append(str(arguments))
            if message.get("role") == "tool":
                tool_outputs.append(str(message.get("content", "")))
        joined_outputs = "\n".join(tool_outputs)
        joined_commands = "\n".join(shell_commands)
        return {
            "evidence_hits": {key: key in joined_outputs for key in evidence_keys},
            "files_touched": {path: path in joined_commands for path in required_files},
            "num_shell_commands": len(shell_commands),
        }

    def _predicted(self, transcript: list[dict[str, Any]]) -> str | None:
        for message in reversed(transcript):
            if message.get("role") == "assistant":
                answer = _finish_answer(message.get("tool_calls") or [])
                if answer:
                    return answer
                content = message.get("content")
                if content:
                    answer = _extract_answer(str(content))
                    if answer:
                        return answer
        return None

    async def run(self) -> TaskResult:
        cfg: FinanceKbTaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        ground_truth = str(metadata.get("ground_truth", ""))
        answer_type = str(metadata.get("answer_type", "number"))
        kb_name = str(metadata.get("kb_root", "kb"))

        self._mount_kb(kb_name)
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(sandbox=sandbox, messages=cfg.prompt)

        predicted = self._predicted(agent_result.transcript)
        reward = 1.0 if _is_correct(predicted, ground_truth, answer_type) else 0.0
        extra = {
            "qid": metadata.get("qid"),
            "level": metadata.get("level"),
            "predicted": predicted,
            "ground_truth": ground_truth,
            "resolved": reward == 1.0,
            "eval_completed": True,
            "finished": agent_result.finished,
            "termination_reason": agent_result.info.get("termination_reason")
            or ("finished" if agent_result.finished else "unfinished"),
            "steps": agent_result.info.get("steps"),
            "total_tokens": agent_result.info.get("total_tokens"),
            **self._inspect(agent_result.transcript, metadata),
        }
        logger.info(
            "finance_kb done: qid=%s reward=%s predicted=%r steps=%s",
            metadata.get("qid"), reward, predicted, agent_result.info.get("steps"),
        )
        return TaskResult(reward=reward, accuracy=reward, finished=agent_result.finished, extra_info=extra)
