from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox


class _LocalShellSession:
    """Host-side ``bash -lc`` session handle honouring the ``open_shell`` contract.

    LocalSandbox has no long-lived pty, so each :meth:`run` spawns one ``bash -lc``
    subprocess with the requested cwd/env -- stateless, but enough for the shell
    tool's exec-style usage without any host package (tmux, root, etc.).
    """

    def __init__(
        self,
        sandbox: LocalSandbox,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self._sandbox = sandbox
        self._cwd = cwd
        self._env = env

    async def run(self, command: str, *, timeout: float = 120.0) -> ExecResult:
        return await self._sandbox.exec(
            ["bash", "-lc", command], timeout=timeout, workdir=self._cwd, env=self._env
        )

    async def close(self) -> None:
        pass


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the host filesystem directly. Constructed with no args,
    so it uses the base :meth:`Sandbox.from_config` (which ignores the config
    fields).
    """

    supports_shell: ClassVar[bool] = True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _LocalShellSession:
        """Return a host-side shell handle (no tmux / root required)."""
        return _LocalShellSession(self, cwd=cwd, env=env)

    async def read_file(self, path: str) -> bytes:
        """Read directly from the host filesystem without base64 transport."""
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write directly to the host filesystem, creating parent directories."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        target = Path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        import os

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={**os.environ, **env} if env else None,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ExecResult(exit_code=-1, stdout="", stderr=f"local exec timed out after {timeout}s")
        return ExecResult(exit_code=proc.returncode or 0, stdout=_to_str(out), stderr=_to_str(err))
