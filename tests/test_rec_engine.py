# SPDX-License-Identifier: Apache-2.0
"""Tests for the Rec external engine (Flint through ``rec serve``)."""

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from omlx.engine import rec as rec_mod
from omlx.engine import splash as splash_mod
from omlx.engine.rec import RecEngine, rec_memory_estimate
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import ModelBusyError
from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings

GIB = 1024**3
FLINT_CONFIG = {
    "model_type": "flint",
    "head_dim": 128,
    "num_key_value_heads": 4,
    "num_hidden_layers": 4,
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
}


def _write_flint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(FLINT_CONFIG))
    (path / "model-00000.safetensors").write_bytes(b"0" * 1024)
    return path


def _sse(*chunks) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _engine(tmp_path, binary=None) -> RecEngine:
    model = tmp_path / "models" / "RecursiveSI" / "flint"
    if not model.exists():
        _write_flint(model)
    return RecEngine(
        model_name=str(model),
        binary=binary or tmp_path / "rec",
        log_dir=tmp_path / "logs",
        context=None,
    )


class TestServeCommand:
    @pytest.mark.parametrize("sidecar", [False, True])
    def test_bundled_flint_command_is_unchanged(self, tmp_path, sidecar):
        engine = _engine(tmp_path)
        # A sidecar must not change the bundled profile's launch behaviour.
        if sidecar:
            (engine._path / "rec-bundle.json").write_text("{}")

        assert engine._serve_command(1234) == [
            str(tmp_path / "rec"),
            "serve",
            "--port",
            "1234",
            "--context",
            "262144",
        ]

    def test_custom_model_uses_its_own_bundle_through_symlink(self, tmp_path):
        engine = _engine(tmp_path)
        export = _write_flint(tmp_path / "export")
        manifest = export / "rec-bundle.json"
        manifest.write_text("{}")
        link = tmp_path / "models" / "flint-abliterated"
        link.symlink_to(export, target_is_directory=True)
        engine._path = link

        assert engine._serve_command(1234) == [
            str(tmp_path / "rec"),
            "serve",
            "--port",
            "1234",
            "--context",
            "262144",
            "--manifest",
            str(manifest.resolve()),
        ]

    def test_custom_model_without_bundle_fails_closed(self, tmp_path):
        engine = _engine(tmp_path)
        engine._path = _write_flint(tmp_path / "flint-abliterated")
        # The checkpoint manifest is not a Rec release-bundle descriptor.
        (engine._path / "flint-abliterated-int4.json").write_text("{}")

        with pytest.raises(ValueError, match="requires its own rec-bundle.json"):
            engine._serve_command(1234)


class TestStreaming:
    async def test_reasoning_field_and_rec_only_body(self, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=_sse(
                    {"choices": [{"delta": {"reasoning": "hmm"}}]},
                    {
                        "choices": [
                            {"delta": {"content": "Hi"}, "finish_reason": "stop"}
                        ]
                    },
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    },
                ),
            )

        engine = _engine(tmp_path)
        engine._client = httpx.AsyncClient(
            base_url="http://rec", transport=httpx.MockTransport(handler)
        )
        outputs = [
            o
            async for o in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=64,
                temperature=0.2,
                min_p=0.1,
                repetition_penalty=1.1,
                stop=["x"],
                chat_template_kwargs={"reasoning_effort": "high"},
            )
        ]

        assert outputs[-1].text == "<think>hmm</think>Hi"
        assert outputs[-1].prompt_tokens == 5
        body = seen["body"]
        # Rec rejects min_p, repetition_penalty and stop strings; sampling and
        # effort belong to the release.
        assert {k: body[k] for k in ("model", "max_tokens")} == {
            "model": "flint",
            "max_tokens": 64,
        }
        for key in (
            "temperature",
            "min_p",
            "repetition_penalty",
            "stop",
            "reasoning_effort",
        ):
            assert key not in body
        assert body["stream_options"] == {"include_usage": True}


FAKE_REC = textwrap.dedent("""\
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    if args[:2] == ["auth", "token"]:
        print("k123"); sys.exit(0)
    assert args[0] == "serve" and args[args.index("--context") + 1] == "262144"
    port = int(args[args.index("--port") + 1])
    calls = {"n": 0}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer k123":
                self.send_response(401); self.end_headers(); return
            calls["n"] += 1
            status = "ready" if calls["n"] > 2 else "loading"
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode())
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class TestLifecycle:
    async def test_start_waits_for_health_ready_and_stop_ends_the_group(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(splash_mod, "STARTUP_POLL_SECONDS", 0.05)
        monkeypatch.setattr(
            RecEngine,
            "_load_tokenizer",
            lambda self: setattr(self, "_tokenizer", MagicMock()),
        )
        script = tmp_path / "fake_rec.py"
        script.write_text(FAKE_REC)
        binary = tmp_path / "rec"
        binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        engine = _engine(tmp_path, binary)

        await engine.start()
        assert engine._process is not None
        group = engine._process.pid
        try:
            assert engine.get_stats()["running"] is True
            assert engine.get_stats()["engine_type"] == "rec"
            assert engine.max_context_tokens == 262144
        finally:
            await engine.stop()

        assert not _group_alive(group)


class TestEnginePool:
    def _pool(self, tmp_path):
        pool = EnginePool()
        model = _write_flint(tmp_path / "models" / "RecursiveSI" / "flint")
        entry = EngineEntry(
            model_id="flint",
            model_path=str(model),
            model_type="llm",
            engine_type="batched",
            estimated_size=20 * GIB,
            model_context_length=262144,
        )
        pool._entries[entry.model_id] = entry
        return pool, entry

    @pytest.mark.parametrize("state", ["idle", "active", "leased", "pinned", "stop_error"])
    async def test_only_one_rec_process_before_loading_another(
        self, tmp_path, monkeypatch, state
    ):
        global_settings = GlobalSettings(base_path=tmp_path / "omlx")
        monkeypatch.setattr(
            "omlx.engine_pool.get_global_settings", lambda: global_settings
        )
        pool, original = self._pool(tmp_path)
        original.engine = _engine(tmp_path)
        original.engine._tokenizer = MagicMock()
        original.engine._active = int(state == "active")
        original.in_use = int(state == "leased")
        original.is_pinned = state == "pinned"
        events = []

        async def stop():
            events.append("stop flint")
            if state == "stop_error":
                raise RuntimeError("Rec process did not stop")

        original.engine.stop = AsyncMock(side_effect=stop)
        old_engine = original.engine
        model = _write_flint(tmp_path / "models" / "flint-abliterated")
        custom = EngineEntry(
            model_id="flint-abliterated",
            model_path=str(model),
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
        )
        pool._entries[custom.model_id] = custom
        new_engine = RecEngine(str(model), tmp_path / "rec", tmp_path / "logs", None)
        new_engine._tokenizer = MagicMock()

        async def load(model_id, **kwargs):
            events.append("start abliterated")
            pool._entries[model_id].engine = new_engine

        monkeypatch.setattr(pool, "_load_engine", AsyncMock(side_effect=load))
        # Plenty of headroom: exclusivity must not depend on memory pressure.
        monkeypatch.setattr(pool, "_current_ceiling", lambda: 1000 * GIB)
        monkeypatch.setattr(pool, "_admission_soft_target", lambda: 1000 * GIB)
        settings = ModelSettings(external_engine="rec")

        if state in ("active", "leased", "pinned"):
            with pytest.raises(ModelBusyError):
                await pool.get_engine(custom.model_id, runtime_settings=settings)
            assert events == []
            assert original.engine is old_engine
        elif state == "stop_error":
            with pytest.raises(RuntimeError, match="Rec process did not stop"):
                await pool.get_engine(custom.model_id, runtime_settings=settings)
            assert events == ["stop flint"]
            assert original.engine is old_engine
        else:
            loaded = await pool.get_engine(custom.model_id, runtime_settings=settings)
            assert loaded is new_engine
            assert events == ["stop flint", "start abliterated"]
            assert original.engine is None

    def test_rec_setting_builds_a_rec_engine(self, tmp_path, monkeypatch):
        global_settings = GlobalSettings(base_path=tmp_path / "omlx")
        monkeypatch.setattr(
            "omlx.engine_pool.get_global_settings", lambda: global_settings
        )
        monkeypatch.setattr(
            "omlx.engine_pool.find_rec_binary", lambda: tmp_path / "rec"
        )
        pool, entry = self._pool(tmp_path)
        settings = ModelSettings(external_engine="rec", max_context_window=131072)

        engine = pool._external_engine_for(entry, settings, "batched")

        assert isinstance(engine, RecEngine)
        assert engine.max_context_tokens == 131072
        assert engine._serve_command(1234)[-2:] == ["--context", "131072"]

    async def test_missing_rec_explains_why(self, tmp_path, monkeypatch):
        monkeypatch.setattr("omlx.engine_pool.find_rec_binary", lambda: None)
        pool, _ = self._pool(tmp_path)

        problem = await pool.external_engine_problem(
            "flint", ModelSettings(external_engine="rec")
        )

        assert "Rec is not installed" in problem

    def test_resident_size_is_weights_plus_kv(self, tmp_path):
        pool, entry = self._pool(tmp_path)
        settings = ModelSettings(external_engine="rec", max_context_window=1000)

        size = pool._entry_runtime_resident_size(entry, settings)

        # 2 full-attention layers x K,V x 4 heads x 128, int8 plus scales.
        per_token = 2 * 2 * 4 * 128
        per_token += per_token // 64
        assert size == rec_memory_estimate(entry.model_path, 1000)
        assert size == 1024 + rec_mod._RUNTIME_RESERVE_BYTES + per_token * 1000
