"""Standalone connectivity check for a deployed QwenPaw service.

Validates the preconditions required by ``uni_agent.agents.qwenpaw.agent``
(run_qwenpaw) BEFORE starting RL training on the NPU machine:

1. QwenPaw service is up and ready        -> ``GET /api/healthz``
2. The target agent exists and is enabled -> ``GET /agents``
3. The provider can be re-pointed at the  -> ``GET /models`` (``freeze_url`` /
   Gateway session URL                     ``is_custom`` / ``enabled``)

Optional ``--e2e`` runs one console chat task against a local mock OpenAI
endpoint to prove the whole path: ``PUT /models/{id}/config`` -> ``POST
/console/chat/task`` -> the ReAct loop's LLM call hits the re-pointed
base_url -> task finishes. The provider base_url is restored afterwards.

No QwenPaw code changes are required: the script only reads state and, in
``--e2e`` mode, temporarily re-points the provider base_url (restored in
``finally``).

Usage::

    python3 examples/quickstart/training/check_qwenpaw_connectivity.py \\
        --base-url http://<host>:8088 --provider-id my-training-provider
    python3 examples/quickstart/training/check_qwenpaw_connectivity.py \\
        --provider-id my-training-provider --e2e

Exit code is non-zero if any required check fails.
"""

import argparse
import asyncio
import json
import os
import sys

import aiohttp
from aiohttp import web

HEALTHZ_PATH = "api/healthz"
AGENTS_PATH = "agents"
PROVIDERS_PATH = "models"
PROVIDER_CONFIG_PATH = "models/{provider_id}/config"
CHAT_TASK_PATH = "console/chat/task"

E2E_ANSWER = "The answer is 42. #### 42"


class CheckResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.details: list[str] = []
        self.failures: list[str] = []

    def ok(self, detail: str) -> None:
        self.details.append(detail)

    def fail(self, detail: str) -> None:
        self.failures.append(detail)

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"[{'PASS' if self.passed else 'FAIL'}] {self.name}"]
        lines += [f"    - {d}" for d in self.details]
        lines += [f"    ! {f}" for f in self.failures]
        return "\n".join(lines)


async def http_json(session: aiohttp.ClientSession, method: str, url: str,
                    *, token: str | None = None, agent_id: str | None = None,
                    json_body: dict | None = None,
                    expect_status: tuple[int, ...] = (200,)) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if agent_id:
        headers["X-Agent-Id"] = agent_id
    async with session.request(method, url, json=json_body, headers=headers) as resp:
        body = await resp.json()
        if resp.status not in expect_status:
            raise RuntimeError(f"{method} {url} -> {resp.status}: {body}")
        return resp.status, body


async def check_health(session: aiohttp.ClientSession, base: str, token: str | None) -> CheckResult:
    result = CheckResult(f"GET {HEALTHZ_PATH} (service ready)")
    try:
        status, body = await http_json(session, "GET", f"{base}/{HEALTHZ_PATH}", token=token,
                                       expect_status=(200, 503))
    except Exception as exc:
        result.fail(f"service unreachable: {exc}")
        return result
    if status == 503:
        result.fail("service still starting (startup_ready not set)")
        return result
    if body.get("status") != "ok":
        result.fail(f"unexpected health payload: {body}")
        return result
    result.ok(f"ready, agents_loaded={body.get('agents_loaded')}, uptime={body.get('uptime_seconds')}s")
    return result


async def check_agent(session: aiohttp.ClientSession, base: str, agent_id: str, token: str | None) -> CheckResult:
    result = CheckResult(f"GET {AGENTS_PATH} (agent '{agent_id}' usable)")
    try:
        _, body = await http_json(session, "GET", f"{base}/{AGENTS_PATH}", token=token)
    except Exception as exc:
        result.fail(f"agent list request failed: {exc}")
        return result
    agents = body.get("agents") if isinstance(body, dict) else body
    if not isinstance(agents, list):
        result.fail(f"unexpected agent list payload: {body}")
        return result
    match = next((a for a in agents if isinstance(a, dict) and a.get("id") == agent_id), None)
    if match is None:
        result.fail(f"agent '{agent_id}' not found (X-Agent-Id must resolve to a configured profile)")
        result.details.append(f"available agents: {[a.get('id') if isinstance(a, dict) else a for a in agents]}")
        return result
    if not match.get("enabled", True):
        result.fail(f"agent '{agent_id}' is disabled")
    if match.get("available_in_chat") is False:
        result.fail(f"agent '{agent_id}' is not available in chat (available_in_chat=false)")
    active_model = match.get("active_model")
    if not active_model:
        result.ok("agent exists and enabled")
        result.details.append("active_model is not set; set it to the training provider + any model name "
                              "(Gateway does not validate the model name)")
    else:
        result.ok(f"agent exists and enabled, active_model={active_model}")
    return result


async def check_provider(session: aiohttp.ClientSession, base: str, provider_id: str | None,
                         token: str | None) -> CheckResult:
    name = provider_id or "all providers"
    result = CheckResult(f"GET {PROVIDERS_PATH} (provider '{name}' writable)")
    try:
        _, body = await http_json(session, "GET", f"{base}/{PROVIDERS_PATH}", token=token)
    except Exception as exc:
        result.fail(f"provider list request failed: {exc}")
        return result
    if not isinstance(body, list):
        result.fail(f"unexpected provider list payload: {body}")
        return result

    def verdict(p: dict) -> str:
        if p.get("freeze_url"):
            return "NOT writable (built-in freeze_url=true; base_url updates are silently ignored)"
        if not p.get("is_custom", True) and p.get("freeze_url"):
            return "NOT writable"
        if p.get("enabled", True) is False:
            return "disabled (chat rejected)"
        return "OK"

    if provider_id:
        match = next((p for p in body if p.get("id") == provider_id), None)
        if match is None:
            result.fail(f"provider '{provider_id}' not found; must be a custom provider "
                        f"(POST /models/custom-providers). existing: {[p.get('id') for p in body]}")
            return result
        detail = (f"id={match.get('id')} chat_model={match.get('chat_model')} "
                  f"base_url={match.get('base_url')} is_custom={match.get('is_custom')} "
                  f"freeze_url={match.get('freeze_url')} enabled={match.get('enabled', True)} "
                  f"require_api_key={match.get('require_api_key')}")
        v = verdict(match)
        if v == "OK":
            result.ok(detail)
        else:
            result.fail(f"{detail} -> {v}")
    else:
        for p in body:
            v = verdict(p)
            (result.ok if v == "OK" else result.fail)(f"id={p.get('id')} chat_model={p.get('chat_model')} "
                                                       f"base_url={p.get('base_url')} is_custom={p.get('is_custom')} "
                                                       f"freeze_url={p.get('freeze_url')} enabled={p.get('enabled', True)} -> {v}")
    return result


# ---------------------------------------------------------------------------
# e2e: mock OpenAI-compatible endpoint + one real console chat task
# ---------------------------------------------------------------------------


async def _mock_chat_handler(request: web.Request) -> web.Response:
    body = await request.json()
    content = E2E_ANSWER
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunk = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "model": "mock", "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    return web.json_response({"id": "chatcmpl-mock", "object": "chat.completion", "model": "mock",
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                           "finish_reason": "stop"}]})


async def run_e2e(session: aiohttp.ClientSession, base: str, provider_id: str,
                  agent_id: str, token: str | None, task_timeout: float) -> CheckResult:
    result = CheckResult("e2e console chat task via mock endpoint")
    if not provider_id:
        result.fail("--provider-id is required for --e2e")
        return result

    # current base_url so we can restore it afterwards
    try:
        _, providers = await http_json(session, "GET", f"{base}/{PROVIDERS_PATH}", token=token)
    except Exception as exc:
        result.fail(f"cannot read providers for restore: {exc}")
        return result
    match = next((p for p in providers if p.get("id") == provider_id), None)
    if match is None:
        result.fail(f"provider '{provider_id}' not found")
        return result
    original_base_url = match.get("base_url", "")
    if match.get("freeze_url"):
        result.fail(f"provider '{provider_id}' is freeze_url=true; base_url cannot be re-pointed")
        return result

    runner = web.AppRunner(web.Application())
    app = runner.app
    requests: list[dict] = []

    async def tracking_handler(request: web.Request) -> web.Response:
        requests.append(await request.json())
        return await _mock_chat_handler(request)

    app.router.add_post("/v1/chat/completions", tracking_handler)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    mock_port = site._server.sockets[0].getsockname()[1]
    mock_base = f"http://127.0.0.1:{mock_port}/v1"

    try:
        # 1. re-point provider at the mock endpoint
        try:
            await http_json(session, "PUT", f"{base}/{PROVIDER_CONFIG_PATH.format(provider_id=provider_id)}",
                            token=token, agent_id=agent_id,
                            json_body={"base_url": mock_base, "auto_discover": False})
        except Exception as exc:
            result.fail(f"PUT provider config failed: {exc}")
            return result
        result.ok(f"provider '{provider_id}' re-pointed to mock {mock_base}")

        # 2. start a console chat task
        payload = {
            "channel": "console",
            "user_id": "default",
            "session_id": "e2e-connectivity-check",
            "input": [{"role": "user", "type": "message",
                       "content": [{"type": "text", "text": "What is 2+2?", "status": "created"}]}],
            "timeout": task_timeout,
        }
        try:
            _, started = await http_json(session, "POST", f"{base}/{CHAT_TASK_PATH}",
                                         token=token, agent_id=agent_id,
                                         json_body=payload, expect_status=(200, 201))
        except Exception as exc:
            result.fail(f"POST {CHAT_TASK_PATH} failed: {exc}")
            return result
        task_id = started.get("task_id")
        if not task_id:
            result.fail(f"task API returned no task_id: {started}")
            return result

        # 3. poll until finished
        loop = asyncio.get_running_loop()
        deadline = loop.time() + task_timeout + 30.0
        finished = None
        while loop.time() < deadline:
            _, poll = await http_json(session, "GET", f"{base}/{CHAT_TASK_PATH}/{task_id}",
                                      token=token, agent_id=agent_id)
            if poll.get("status") == "finished":
                finished = poll
                break
            await asyncio.sleep(1.0)
        if finished is None:
            result.fail(f"chat task {task_id} did not finish within {task_timeout + 30:.0f}s")

        # 4. verdicts
        if not requests:
            result.fail("mock endpoint received NO /chat/completions requests; the ReAct loop did not "
                        "reach the re-pointed base_url")
        else:
            result.ok(f"mock received {len(requests)} /chat/completions request(s); "
                      f"model={requests[0].get('model')} stream={requests[0].get('stream')}")
        if finished is not None:
            res = finished.get("result")
            if isinstance(res, dict):
                result.ok(f"task finished; result keys={list(res.keys())}")
                if "output" in res or "text" in res:
                    result.ok(f"final output: {str(res.get('output') or res.get('text'))[:200]}")
            else:
                result.ok(f"task finished; result={str(res)[:200]}")
    finally:
        # restore provider base_url
        try:
            await http_json(session, "PUT", f"{base}/{PROVIDER_CONFIG_PATH.format(provider_id=provider_id)}",
                            token=token, agent_id=agent_id,
                            json_body={"base_url": original_base_url, "auto_discover": False})
            result.ok(f"provider '{provider_id}' base_url restored to {original_base_url!r}")
        except Exception as exc:
            result.fail(f"WARNING: failed to restore provider base_url: {exc}")
        await runner.cleanup()
    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("QWENPAW_API_URL", "http://localhost:8088"))
    parser.add_argument("--provider-id", default=os.environ.get("QWENPAW_PROVIDER_ID", ""))
    parser.add_argument("--agent-id", default="default")
    parser.add_argument("--token", default=None, help="Bearer token if QwenPaw access control is enabled")
    parser.add_argument("--task-timeout", type=float, default=60.0, help="e2e chat task timeout (seconds)")
    parser.add_argument("--e2e", action="store_true", help="run one console chat task against a local mock endpoint "
                        "(temporarily re-points the provider; restored afterwards)")
    args = parser.parse_args()

    print(f"QwenPaw base: {args.base_url}")
    print(f"provider-id: {args.provider_id or '(all)'}  agent-id: {args.agent_id}  e2e: {args.e2e}")

    results: list[CheckResult] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session:
        results.append(await check_health(session, args.base_url, args.token))
        results.append(await check_agent(session, args.base_url, args.agent_id, args.token))
        results.append(await check_provider(session, args.base_url, args.provider_id, args.token))
        if args.e2e:
            results.append(await run_e2e(session, args.base_url, args.provider_id, args.agent_id,
                                         args.token, args.task_timeout))

    print()
    all_ok = True
    for r in results:
        print(r.render())
        all_ok &= r.passed
    print()
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
