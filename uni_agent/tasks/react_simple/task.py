"""react_simple task: minimal agent-RL demo task.

The agent is asked to run a shell command and match a ground-truth output.
Reward = 1.0 if stdout matches ``expected`` (after stripping), else 0.0.
Designed for ``sandbox.provider: local`` smoke runs of the uni-agent framework.
"""

from __future__ import annotations

import logging

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


class ReactSimpleTaskConfig(TaskConfig):
    name: str = "react_simple"


@register_task("react_simple")
class ReactSimpleTask(Task):
    name = "react_simple"
    config_model = ReactSimpleTaskConfig

    async def run(self) -> TaskResult:
        cfg: ReactSimpleTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata
        expected = str(sample.get("expected", "")).strip()
        command = str(sample.get("command", "echo ok"))

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(sandbox=sandbox, messages=cfg.prompt)

            # Score by executing the expected command inside the sandbox.
            res = await sandbox.exec(["bash", "-lc", command], timeout=30)
            stdout = (res.stdout or "").strip()
            reward = 1.0 if stdout == expected else 0.0

        logger.info(
            "react_simple done: reward=%s finished=%s command=%r expected=%r got=%r",
            reward, agent_result.finished, command, expected, stdout,
        )
        return TaskResult(reward=reward, accuracy=reward, finished=agent_result.finished)
