"""QwenPaw integration: HTTP client + task_runner for a deployed QwenPaw service.

uni-agent does NOT import AgentScope or manage model clients. It drives a
deployed QwenPaw service through QwenPaw's **existing** REST APIs -- no QwenPaw
code changes are required:

1. Re-point the provider ``base_url`` at the Gateway session URL so every LLM
   call the ReAct loop makes hits the Gateway (trajectory capture).
2. Start a console background chat task to run one ReAct episode.
3. Poll the task until it finishes and extract the final text.

Architecture (RFC #5790 AgentGateway pattern):

    Framework                            QwenPaw Service (remote)
    ─────────                            ────────────────────────
    create_session() → SessionHandle
      │
      │  run_qwenpaw(session=handle)
      │    │
      │    │  PUT /models/{provider_id}/config
      │    │  {base_url: session.base_url, auto_discover: false}
      │    │ ──────────────────────────────>  provider base_url re-pointed
      │    │                                       │
      │    │                              POST {base_url}/chat/completions
      │    │                              ──────> Gateway → tokenize → vLLM → record
      │    │                              <────── OpenAI response
      │    │
      │    │  POST /console/chat/task {input, timeout}
      │    │ ──────────────────────────────>  runs ReAct loop
      │    │  <── {task_id}
      │    │
      │    │  GET /console/chat/task/{task_id}  (poll until finished)
      │    │ <── {status, result}
      │    │
      │    │  POST session.reward_info_url (if report_reward=True)
      │
    finalize_session() → list[Trajectory]

Notes:

- ``session.base_url`` (``.../sessions/{session_id}/v1``) is used verbatim as
  the provider ``base_url``; QwenPaw's OpenAI chat adapter appends
  ``/chat/completions``, which exactly matches the Gateway route.
- The provider ``base_url`` is instance-global, so runner concurrency must be
  capped at ``max_concurrent_sessions: 1`` to avoid switching the endpoint
  under concurrent episodes.
- ``auto_discover: false`` is sent with every re-configuration so QwenPaw does
  not start a background model-discovery task against the Gateway.

YAML configuration::

    agent_runners:
      qwenpaw:
        runner_fqn: uni_agent.agents.qwenpaw.agent.run_qwenpaw
        runner_kwargs:
          qwenpaw_api_url: "http://10.0.0.5:8088"
          provider_id: "openai"
          agent_id: "default"
          auth_token: null
          request_timeout: 600.0
          task_timeout: 600.0
          poll_interval: 2.0
          report_reward: true
        dispatch_mode: inline_async
        max_concurrent_sessions: 1
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel, Field

from uni_agent.agents.qwenpaw.reward import GSM8KReward, RewardFunction
from uni_agent.tasks.base import TaskResult

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class QwenPawConfig(BaseModel):
    """Configuration for connecting to a remote QwenPaw service."""

    qwenpaw_api_url: str = Field(
        default="http://localhost:8088",
        description="Base URL of the deployed QwenPaw service.",
    )
    provider_id: str = Field(
        default="",
        description=(
            "Provider whose base_url is re-pointed at the Gateway session URL "
            "before each episode (PUT /models/{provider_id}/config). "
            "Must be a custom provider (freeze_url=False); built-in providers "
            "silently ignore base_url updates."
        ),
    )
    agent_id: str = Field(
        default="default",
        description="QwenPaw agent ID (X-Agent-Id header).",
    )
    auth_token: str | None = Field(
        default=None,
        description="Bearer token for remote QwenPaw access.",
    )
    request_timeout: float = Field(
        default=600.0,
        description="Total timeout (seconds) for a single QwenPaw API call.",
    )
    connect_timeout: float = Field(
        default=30.0,
        description="Connection timeout (seconds).",
    )
    task_timeout: float = Field(
        default=600.0,
        description=(
            "Server-side timeout (seconds) for each console background chat "
            "task; the client stops polling after this plus a small buffer."
        ),
    )
    poll_interval: float = Field(
        default=2.0,
        description="Seconds between console chat-task status polls.",
    )


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------


class QwenPawClient:
    """Async HTTP client driving a deployed QwenPaw service via its REST APIs.

    No QwenPaw code changes are required: the provider ``base_url`` is
    re-pointed at the Gateway session URL, then a console background chat task
    runs the ReAct episode.

    Usage::

        client = QwenPawClient(
            QwenPawConfig(qwenpaw_api_url="http://10.0.0.5:8088", provider_id="openai")
        )
        result = await client.chat("What is 2+2?", model_endpoint="http://gateway:8000/sessions/x/v1")
        print(result["content"])
    """

    def __init__(self, config: QwenPawConfig) -> None:
        self._cfg = config

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Agent-Id": self._cfg.agent_id,
        }
        if self._cfg.auth_token:
            headers["Authorization"] = f"Bearer {self._cfg.auth_token}"
        return headers

    async def _request(self, method: str, path: str, *, json_body: dict | None = None) -> dict:
        url = f"{self._cfg.qwenpaw_api_url.rstrip('/')}/{path.lstrip('/')}"
        timeout = aiohttp.ClientTimeout(
            total=self._cfg.request_timeout,
            connect=self._cfg.connect_timeout,
        )
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.request(method, url, json=json_body, headers=self._headers()) as resp:
                if resp.status not in (200, 201):
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"QwenPaw API {method} {path} returned {resp.status}: {error_body[:500]}"
                    )
                return await resp.json()

    async def configure_provider(self, provider_id: str, base_url: str) -> None:
        """Re-point a provider's base_url at the Gateway session URL.

        ``auto_discover: false`` prevents QwenPaw from starting a background
        model-discovery task against the Gateway on every re-configuration.
        """
        if not provider_id:
            raise ValueError("QwenPaw provider_id is required for provider reconfiguration")
        await self._request(
            "PUT",
            f"models/{provider_id}/config",
            json_body={"base_url": base_url, "auto_discover": False},
        )

    async def start_chat_task(self, user_message: str, *, session_id: str) -> str:
        """Start a console background chat task and return its task_id."""
        payload: dict[str, Any] = {
            "channel": "console",
            "user_id": "default",
            "session_id": session_id,
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "text", "text": user_message, "status": "created"}],
                }
            ],
            "timeout": self._cfg.task_timeout,
        }
        data = await self._request("POST", "console/chat/task", json_body=payload)
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"QwenPaw chat task returned no task_id: {data}")
        return str(task_id)

    async def poll_task(self, task_id: str) -> dict[str, Any]:
        """Poll a chat task until it finishes; return its result dict.

        The client-side deadline is ``task_timeout`` plus a small buffer so a
        server-side cancellation (which fires at ``task_timeout``) settles
        before the poll gives up.
        """
        deadline = time.monotonic() + self._cfg.task_timeout + 30.0
        while True:
            data = await self._request("GET", f"console/chat/task/{task_id}")
            status = data.get("status")
            if status == "finished":
                result = data.get("result")
                return result if isinstance(result, dict) else {"status": "completed", "raw": result}
            if status in (None, "running"):
                if time.monotonic() >= deadline:
                    return {"status": "timeout", "error": f"chat task {task_id} timed out"}
                await asyncio.sleep(self._cfg.poll_interval)
                continue
            return {"status": "error", "error": f"unexpected task status: {status}"}

    async def chat(
        self,
        user_message: str,
        *,
        model_endpoint: str | None = None,
        session_id: str | None = None,
        provider_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one QwenPaw ReAct episode and return the parsed result.

        Args:
            user_message: The user's message text.
            model_endpoint: Gateway session ``base_url`` used as the provider
                ``base_url`` so every LLM call hits the Gateway. Required.
            session_id: Optional session ID for QwenPaw's session management.
            provider_id: Provider to re-point; defaults to ``QwenPawConfig``.

        Returns:
            Dict with keys: ``content``, ``finish_reason``, ``error`` (optional),
            ``session_id``.
        """
        if session_id is None:
            session_id = f"uniagent-{uuid.uuid4().hex[:12]}"
        if model_endpoint is None:
            return {
                "session_id": session_id,
                "content": "",
                "finish_reason": "error",
                "error": "model_endpoint (Gateway session base_url) is required",
            }

        try:
            await self.configure_provider(provider_id or self._cfg.provider_id, model_endpoint)
            task_id = await self.start_chat_task(user_message, session_id=session_id)
            result = await self.poll_task(task_id)
            if result.get("status") not in (None, "completed") or result.get("error"):
                return {
                    "session_id": session_id,
                    "content": "",
                    "finish_reason": "error",
                    "error": str(result.get("error") or result.get("status")),
                }
            content = _extract_final_text(result)
            finished = bool(content) or result.get("status") == "completed"
            return {
                "session_id": session_id,
                "content": content,
                "finish_reason": "stop" if finished else "error",
            }
        except aiohttp.ClientError as exc:
            logger.exception("QwenPaw HTTP request failed: %s", exc)
            return {
                "session_id": session_id,
                "content": "",
                "finish_reason": "error",
                "error": f"HTTP error: {exc}",
            }
        except Exception as exc:
            logger.exception("QwenPaw client failed: %s", exc)
            return {
                "session_id": session_id,
                "content": "",
                "finish_reason": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }


# ---------------------------------------------------------------------------
# Task Runner (RFC #5790 AgentRunner contract)
# ---------------------------------------------------------------------------


async def run_qwenpaw(
    *,
    session: SessionHandle,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    tools_kwargs: dict[str, Any] | None = None,
    # --- QwenPaw connection (from YAML runner_kwargs) ---
    qwenpaw_api_url: str = "http://localhost:8088",
    provider_id: str = "",
    agent_id: str = "default",
    auth_token: str | None = None,
    request_timeout: float = 600.0,
    connect_timeout: float = 30.0,
    task_timeout: float = 600.0,
    poll_interval: float = 2.0,
    report_reward: bool = False,
    reward_fn: RewardFunction | None = None,
    **_: Any,
) -> TaskResult:
    """Run one QwenPaw episode against a Gateway session.

    Satisfies the framework's ``AgentRunner`` contract. The framework injects
    ``session`` (SessionHandle) directly; all other kwargs come from YAML
    ``runner_kwargs``.

    Args:
        session: Gateway session handle. ``session.base_url`` is used as the
            QwenPaw provider ``base_url`` so every LLM call hits the Gateway.
        raw_prompt: The task prompt (string, message list, or dict).
        sample_index: Sample index for logging.
        tools_kwargs: Task metadata; ``tools_kwargs.task.metadata.ground_truth``
            carries the expected answer for reward computation.
        qwenpaw_api_url: Base URL of the deployed QwenPaw service.
        provider_id: Provider whose base_url is re-pointed at the Gateway
            session URL before the episode. Required.
        agent_id: QwenPaw agent ID (X-Agent-Id header).
        auth_token: Bearer token for remote QwenPaw access.
        request_timeout: Total timeout (seconds).
        connect_timeout: Connection timeout (seconds).
        task_timeout: Server-side timeout (seconds) for the chat task.
        poll_interval: Seconds between chat-task status polls.
        report_reward: If True, POST reward info to ``session.reward_info_url``.
        reward_fn: Scorer for answer-correctness reward; defaults to
            :class:`GSM8KReward`. Used when ``ground_truth`` is available.

    Returns:
        TaskResult with finished status and extra info.
    """
    if session.base_url is None:
        raise ValueError(
            "qwenpaw runner: session.base_url is None. "
            "The Gateway session must provide a base URL."
        )
    if not provider_id:
        raise ValueError(
            "qwenpaw runner: provider_id is required in runner_kwargs "
            "(the provider whose base_url is re-pointed at the Gateway)."
        )

    user_message = _extract_user_message(raw_prompt)
    ground_truth = _extract_ground_truth(tools_kwargs, raw_prompt)

    client = QwenPawClient(
        QwenPawConfig(
            qwenpaw_api_url=qwenpaw_api_url,
            provider_id=provider_id,
            agent_id=agent_id,
            auth_token=auth_token,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            task_timeout=task_timeout,
            poll_interval=poll_interval,
        )
    )

    logger.info(
        "qwenpaw runner start: sample_index=%s qwenpaw_url=%s provider_id=%s",
        sample_index,
        qwenpaw_api_url,
        provider_id,
    )

    result = await client.chat(
        user_message,
        model_endpoint=session.base_url,
    )

    finished = result.get("finish_reason") not in (None, "error")
    content = result.get("content", "")
    error_info = result.get("error")

    # Answer-correctness reward when a ground truth is available; otherwise
    # fall back to a completion reward (chain smoke test only).
    reward = 0.0
    if finished and ground_truth is not None:
        scorer = reward_fn if reward_fn is not None else GSM8KReward()
        reward = await scorer.compute(content, ground_truth)
    elif finished:
        reward = 1.0

    extra_info: dict[str, Any] = {
        "qwenpaw_session_id": result.get("session_id"),
        "agent_id": agent_id,
        "qwenpaw_api_url": qwenpaw_api_url,
        "provider_id": provider_id,
        "content": content,
    }
    if ground_truth is not None:
        extra_info["ground_truth"] = ground_truth
    if error_info:
        extra_info["error"] = error_info

    # Post reward to Gateway session (best-effort)
    reward_posted = False
    if report_reward and session.reward_info_url:
        reward_posted = await _post_reward_info(
            session.reward_info_url,
            reward=reward,
            finished=finished,
            extra_info=extra_info,
        )

    logger.info(
        "qwenpaw runner done: sample_index=%s finished=%s reward=%s reward_posted=%s",
        sample_index,
        finished,
        reward,
        reward_posted,
    )

    return TaskResult(
        reward=reward,
        finished=finished,
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_final_text(value: Any) -> str:
    """Best-effort extraction of the final assistant text from a chat-task result.

    The background chat task returns the last serialized SSE event, which may
    carry the text under ``output``, a literal ``text`` key, or a ``content``
    list of ``{type: text, text}`` blocks. Recurses until the first non-empty
    text is found.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            text = _extract_final_text(item)
            if text:
                return text
    if isinstance(value, dict):
        if "output" in value:
            return _extract_final_text(value["output"])
        raw = value.get("text")
        if isinstance(raw, str) and raw:
            return raw
        content = value.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        return text
        for item in value.values():
            text = _extract_final_text(item)
            if text:
                return text
    return ""


def _extract_ground_truth(tools_kwargs: dict[str, Any] | None, raw_prompt: Any) -> str | None:
    """Extract the expected answer from task metadata or the raw prompt.

    Training data carries it as ``extra_info.tools_kwargs.task.metadata.ground_truth``;
    a raw prompt dict may also carry a top-level ``ground_truth`` field.
    """
    if isinstance(tools_kwargs, dict):
        task = tools_kwargs.get("task")
        if isinstance(task, dict):
            metadata = task.get("metadata")
            if isinstance(metadata, dict) and metadata.get("ground_truth") is not None:
                return str(metadata["ground_truth"])
    if isinstance(raw_prompt, dict):
        ground_truth = raw_prompt.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth)
    return None


def _extract_user_message(raw_prompt: Any) -> str:
    """Extract a plain-text user message from the raw prompt."""
    if isinstance(raw_prompt, str):
        return raw_prompt

    if isinstance(raw_prompt, list):
        parts: list[str] = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("system", "user") and content:
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
        if parts:
            return "\n\n".join(parts)

    if isinstance(raw_prompt, dict):
        text = raw_prompt.get("prompt") or raw_prompt.get("text") or raw_prompt.get("content")
        if isinstance(text, str):
            return text

    raise ValueError(
        f"qwenpaw runner: cannot extract user message from raw_prompt type "
        f"{type(raw_prompt).__name__}"
    )


async def _post_reward_info(
    reward_info_url: str,
    reward: float,
    finished: bool,
    extra_info: dict[str, Any] | None = None,
) -> bool:
    """Best-effort POST of reward metadata to the Gateway session."""
    reward_info: dict[str, Any] = {"reward": reward, "finished": finished}
    if extra_info:
        reward_info["extra_info"] = extra_info

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(
                reward_info_url, json={"reward_info": reward_info}
            ) as response:
                response.raise_for_status()
        logger.debug("posted reward_info to %s", reward_info_url)
        return True
    except Exception as exc:
        logger.warning(
            "failed to post reward_info to %s: %s: %s",
            reward_info_url,
            type(exc).__name__,
            exc,
        )
        return False
