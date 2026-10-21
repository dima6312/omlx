# SPDX-License-Identifier: Apache-2.0
"""Recursive's Flint served by a ``rec serve`` child process.

Rec runs Flint on its own Metal kernels and grammar-constrained tool calls,
which the in-process engines do not have. It speaks the OpenAI chat API, so
this engine reuses Splash's process, streaming and memory handling and only
changes how the server starts and what a request carries. Rec queues
requests itself and cancels a generation whose client disconnects.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from . import splash
from .base import GenerationOutput
from .splash import (
    REQUEST_TIMEOUT_SECONDS,
    SplashEngine,
    _failure_summary,
    _find_launcher,
    _free_port,
    _kv_bytes_per_token,
    _read_config,
    _splash_env,
    _watchdog_command,
    _weight_bytes,
    splash_messages,
)

REC_ENGINE = "rec"
REC_MODEL = "flint"
REC_MAX_CONTEXT = 262144
_RUNTIME_RESERVE_BYTES = 2 * 1024**3


def find_rec_binary() -> Path | None:
    return _find_launcher("rec")


def rec_memory_estimate(model_path: str | Path, context: int | None) -> int:
    """Flint's weights, Rec's scratch, and int8 KV for ``context`` tokens."""
    path = Path(model_path)
    config = _read_config(path) or {}
    context = min(context or REC_MAX_CONTEXT, REC_MAX_CONTEXT)
    kv = _kv_bytes_per_token(config, "int8") * context
    return _weight_bytes(path) + _RUNTIME_RESERVE_BYTES + kv


class RecEngine(SplashEngine):
    """Serves Flint through ``rec serve``."""

    engine_id = REC_ENGINE
    # Rec drops request fields it does not know, response_format among them.
    supports_native_response_format = False

    def __init__(
        self, model_name: str, binary: Path, log_dir: Path, context: int | None
    ) -> None:
        self._path = Path(model_name)
        self._binary = binary
        self._context = min(context or REC_MAX_CONTEXT, REC_MAX_CONTEXT)
        self._repo_id = REC_MODEL
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = ""
        self._api_key = ""
        self._log_path = log_dir / "rec.log"
        self._log_offset = 0
        self._active = 0
        self._requests = 0
        self._live = {}
        self._weights_size: int | None = None
        self._tokenizer: Any = None
        self._model_type: str | None = None
        self._status: dict[str, Any] | None = None
        self._status_task: asyncio.Task | None = None
        self._status_refresh: asyncio.Task | None = None
        self._vision = False

    @property
    def max_context_tokens(self) -> int | None:
        return self._context

    @property
    def runtime_failed_reason(self) -> str | None:
        process = self._process
        if process is None or process.poll() is None:
            return None
        return (
            f"Rec exited with code {process.returncode}: "
            f"{_failure_summary(self._log_path, self._log_offset)} "
            f"(log: {self._log_path})"
        )

    def _serve_command(self, port: int) -> list[str]:
        return [
            str(self._binary),
            "serve",
            "--port",
            str(port),
            "--context",
            str(self._context),
        ]

    def _read_api_key(self) -> str:
        # Rec keeps one key per install and hands it to its server itself.
        result = subprocess.run(
            [str(self._binary), "auth", "token"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_splash_env(None),
        )
        key = result.stdout.strip()
        if result.returncode != 0 or not key:
            raise RuntimeError(f"rec auth token failed: {result.stderr.strip()}")
        return key

    async def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise RuntimeError(self.runtime_failed_reason)
        await asyncio.to_thread(self._load_tokenizer)
        self._api_key = await asyncio.to_thread(self._read_api_key)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        command = self._serve_command(port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        splash.logger.info(
            "Rec: starting %s (log: %s)", " ".join(command), self._log_path
        )
        with open(self._log_path, "ab") as log:
            self._log_offset = log.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_splash_env(None),
                start_new_session=True,
            )
        try:
            await self._wait_until_ready()
        except BaseException:
            await self.stop()
            raise

    async def _serves_model(self) -> bool:
        # /v1/models answers while Flint still loads; /health waits for it.
        try:
            response = await self._client.get("/health", timeout=2)
            return response.json().get("status") == "ready"
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + splash.STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Rec exited with code {self._process.returncode} while "
                    f"loading Flint: "
                    f"{_failure_summary(self._log_path, self._log_offset)} "
                    f"(log: {self._log_path})"
                )
            if await self._serves_model():
                splash.logger.info("Rec: serving Flint at %s", self._base_url)
                return
            await asyncio.sleep(splash.STARTUP_POLL_SECONDS)
        raise RuntimeError(
            f"Rec did not serve Flint within {splash.STARTUP_TIMEOUT_SECONDS}s; "
            f"see {self._log_path}"
        )

    async def _refresh_status(self) -> None:
        # Rec has no /status; nothing to mirror.
        return None

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        # Sampling, effort and budgets stay Rec's: Flint always reasons, Rec
        # sets the release's sampling, and stop strings are rejected while
        # tool grammar is on.
        body: dict[str, Any] = {
            "model": REC_MODEL,
            "max_tokens": max_tokens,
            "messages": splash_messages(messages),
        }
        if tools:
            body["tools"] = tools
        async for output in self._stream(body, kwargs.get("_request_id")):
            yield output

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": REC_ENGINE,
            "model_name": self.model_name,
            "server": self._base_url,
            "log_path": str(self._log_path),
            "running": self._process is not None and self._process.poll() is None,
            "requests": self._requests,
            "active_requests": self._active,
            "process_memory_bytes": self.process_memory_bytes(),
        }

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        return None

    def benchmark_endpoint(self) -> dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Rec engine is not started")
        return {
            "base_url": f"{self._base_url}/v1",
            "api_key": self._api_key,
            "model": REC_MODEL,
            "extra_body": {},
        }
