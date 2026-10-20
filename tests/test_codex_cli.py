"""Tests for the structured Codex CLI translation adapter."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from daru.translation import codex_cli
from daru.translation.analysis import CodexAnalysisSession
from daru.translation.engine import TranslationEngine


def _fake_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"test")
    return executable


def _command_text(command) -> str:
    return " ".join(str(part) for part in command)


def _completed(command, *, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _stdin_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _argument_path(command, flag: str) -> Path:
    index = [str(part) for part in command].index(flag)
    return Path(command[index + 1])


class _RecordingInput:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, value: bytes) -> int:
        self.data.extend(value)
        return len(value)

    def flush(self) -> None:
        return

    def close(self) -> None:
        self.closed = True


class _FakeAppServerProcess:
    def __init__(self, output: bytes) -> None:
        self.stdin = _RecordingInput()
        self.stdout = io.BytesIO(output)
        self.returncode = 0
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _jsonl(*messages) -> bytes:
    return b"".join(
        (json.dumps(message) + "\n").encode("utf-8") for message in messages
    )


def test_list_codex_models_uses_app_server_picker_catalog(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    source_codex_home = tmp_path / "source-codex-home"
    source_codex_home.mkdir()
    (source_codex_home / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source_codex_home))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    process = _FakeAppServerProcess(
        _jsonl(
            {"id": 0, "result": {"userAgent": "Codex"}},
            {"method": "remoteControl/status/changed", "params": {}},
            {
                "id": 1,
                "result": {
                    "data": [
                        {
                            "id": "gpt-current",
                            "model": "gpt-current",
                            "displayName": "GPT Current",
                            "hidden": False,
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": "low"},
                                {"reasoningEffort": "high"},
                            ],
                            "defaultReasoningEffort": "low",
                            "inputModalities": ["text", "image"],
                            "isDefault": True,
                        },
                        {
                            "id": "hidden-model",
                            "model": "hidden-model",
                            "hidden": True,
                        },
                    ],
                    "nextCursor": "page-2",
                },
            },
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "id": "gpt-next",
                            "displayName": "GPT Next",
                            "supportedReasoningEfforts": [],
                        }
                    ],
                    "nextCursor": None,
                },
            },
        )
    )
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["isolated_auth"] = (
            Path(kwargs["env"]["CODEX_HOME"]) / "auth.json"
        ).is_file()
        return process

    monkeypatch.setattr(codex_cli.subprocess, "Popen", fake_popen)

    models = codex_cli.list_codex_models(str(executable), timeout=5)

    assert [model.model for model in models] == ["gpt-current", "gpt-next"]
    assert models[0].display_name == "GPT Current"
    assert models[0].supported_reasoning_efforts == ("low", "high")
    assert models[0].default_reasoning_effort == "low"
    assert models[0].input_modalities == ("text", "image")
    assert models[0].is_default
    assert models[1].input_modalities == ("text", "image")
    assert captured["command"][-3:] == [
        "app-server",
        "--listen",
        "stdio://",
    ]
    requests = [json.loads(line) for line in process.stdin.data.splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "model/list",
        "model/list",
    ]
    assert requests[2]["params"] == {"limit": 100, "includeHidden": False}
    assert requests[3]["params"]["cursor"] == "page-2"
    assert process.stdin.closed
    assert captured["isolated_auth"]
    assert captured["kwargs"]["env"]["CODEX_HOME"] != str(source_codex_home)
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]


def test_list_codex_models_reports_app_server_errors(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    process = _FakeAppServerProcess(
        _jsonl(
            {"id": 0, "result": {}},
            {"id": 1, "error": {"code": -32601, "message": "Method not found"}},
        )
    )
    monkeypatch.setattr(
        codex_cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(codex_cli.CodexCliError, match="Method not found"):
        codex_cli.list_codex_models(str(executable), timeout=5)


def test_stop_app_server_kills_windows_process_tree(monkeypatch):
    calls = []

    class HangingProcess:
        pid = 321

        def __init__(self):
            self.stdin = _RecordingInput()
            self.stdout = io.BytesIO()
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("codex", timeout)
            return 0

        def terminate(self):
            raise AssertionError("Windows should terminate the whole process tree")

        def kill(self):
            raise AssertionError("taskkill should allow the launcher to exit")

    class FinishedReader:
        def join(self, timeout=None):
            calls.append(("join", timeout))

        def is_alive(self):
            return False

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(command)

    monkeypatch.setattr(codex_cli.os, "name", "nt")
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    process = HangingProcess()

    codex_cli._stop_app_server(process, FinishedReader())

    taskkill = calls[0][0]
    assert taskkill == ["taskkill.exe", "/PID", "321", "/T", "/F"]
    assert calls[1] == ("join", 1)
    assert process.stdin.closed
    assert process.stdout.closed


def test_codex_clients_keep_catalog_reasoning_efforts():
    inspection = {
        "executable": "codex.exe",
        "version": "codex-cli 1.0",
        "supports_ignore_user_config": True,
        "supports_image": True,
    }

    structured = codex_cli.CodexCliStructuredClient(
        model="gpt-next",
        reasoning_effort="max",
        _inspection=inspection,
    )
    translator = codex_cli.CodexCliTranslator(
        model="gpt-next",
        reasoning_effort="none",
        _inspection=inspection,
    )

    assert structured.reasoning_effort == "max"
    assert translator.reasoning_effort == "none"


def test_windows_cmd_launcher_uses_call_for_paths_with_spaces(monkeypatch):
    monkeypatch.setattr(codex_cli.os, "name", "nt")
    monkeypatch.setenv("COMSPEC", "C:/Windows/System32/cmd.exe")

    command = codex_cli._launcher_command(
        "C:/Program Files/nodejs/npm.cmd",
        ["install", "-g", "@openai/codex@latest"],
    )

    assert command[:4] == ["C:/Windows/System32/cmd.exe", "/d", "/c", "call"]
    assert command[4] == "C:/Program Files/nodejs/npm.cmd"
    assert command[5:] == ["install", "-g", "@openai/codex@latest"]


def test_process_output_decodes_windows_oem_text(monkeypatch):
    monkeypatch.setattr(codex_cli.os, "name", "nt")

    message = "не является внутренней или внешней командой"

    assert codex_cli._decode_process_output(message.encode("cp866")) == message


def test_process_file_capture_avoids_pipes_and_reads_output(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write("stdout text".encode("utf-8"))
        kwargs["stderr"].write("stderr text".encode("utf-8"))
        return subprocess.CompletedProcess(command, 0, None, None)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    result = codex_cli._run_process(
        "codex.exe",
        ["exec"],
        cwd=tmp_path,
        timeout=30,
        capture_via_files=True,
    )

    assert "capture_output" not in captured
    assert result.stdout == "stdout text"
    assert result.stderr == "stderr text"


def test_structured_translation_uses_schema_and_sanitized_environment(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    calls = []
    captured = {}

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-leak")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(
                command,
                stdout="--output-schema <FILE>\n--ignore-user-config",
            )

        root = Path(kwargs["cwd"])
        captured["command"] = text
        captured["schema"] = json.loads(
            (root / "translation.schema.json").read_text(encoding="utf-8")
        )
        captured["prompt"] = _stdin_text(kwargs["input"])
        captured["env"] = kwargs["env"]
        captured["command"] = text
        (root / "translation.output.json").write_text(
            json.dumps(
                {
                    "translations": [
                        {"id": "second", "text": "Второй"},
                        {"id": "first", "text": "Первый"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(
        cli_path=str(executable),
        model="gpt-5.4-mini",
        reasoning_effort="low",
        timeout_seconds=45,
    )

    result = translator.translate(
        [
            {"id": "first", "text": "First"},
            {"id": "second", "text": "Second"},
        ],
        instructions="Translate to Russian.",
    )

    assert result == ["Первый", "Второй"]
    assert captured["schema"]["properties"]["translations"]["minItems"] == 2
    item_schema = captured["schema"]["properties"]["translations"]["items"]
    assert item_schema["properties"]["id"]["enum"] == ["first", "second"]
    assert item_schema["additionalProperties"] is False
    assert '"id": "first"' in captured["prompt"]
    assert "--ephemeral" in captured["command"]
    assert "--sandbox read-only" in captured["command"]
    assert "--output-schema" in captured["command"]
    assert "--ignore-user-config" in captured["command"]
    assert "service_tier" not in captured["command"]
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "CODEX_API_KEY" not in captured["env"]
    assert "CODEX_HOME" not in captured["env"]
    assert "stdout" in calls[-1][1]
    assert "stderr" in calls[-1][1]
    assert len(calls) == 3


def test_structured_client_attaches_images_and_parses_schema_output(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        schema_path = _argument_path(command, "--output-schema")
        output_path = _argument_path(command, "--output-last-message")
        captured["cwd"] = Path(kwargs["cwd"])
        captured["artifact_root"] = schema_path.parent
        captured["schema"] = json.loads(
            schema_path.read_text(encoding="utf-8")
        )
        captured["prompt"] = _stdin_text(kwargs["input"])
        captured["env"] = kwargs["env"]
        image_indexes = [
            index
            for index, value in enumerate(command)
            if str(value) == "--image"
        ]
        captured["images"] = [
            Path(command[index + 1]).read_bytes()
            for index in image_indexes
        ]
        output_path.write_text(
            '{"blocks":[{"stable_id":"a"}]}',
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    client = codex_cli.CodexCliStructuredClient(
        model="gpt-5.5",
        reasoning_effort="high",
        timeout_seconds=60,
        _inspection={
            "executable": str(executable),
            "version": "codex-cli 1.0",
            "supports_ignore_user_config": True,
            "supports_image": True,
        },
    )

    result = client.run(
        prompt="Analyze the attached page.",
        schema={
            "type": "object",
            "properties": {"blocks": {"type": "array"}},
            "required": ["blocks"],
            "additionalProperties": False,
        },
        images=[("page.png", b"png-data")],
    )

    assert result == {"blocks": [{"stable_id": "a"}]}
    assert captured["images"] == [b"png-data"]
    assert captured["schema"]["required"] == ["blocks"]
    assert captured["prompt"] == "Analyze the attached page."
    command_text = _command_text(captured["command"])
    assert "--image" in command_text
    assert "--output-schema" in command_text
    assert "--model gpt-5.5" in command_text
    assert "model_reasoning_effort=high" in command_text
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "CODEX_HOME" not in captured["env"]
    assert captured["cwd"] == captured["artifact_root"].parent
    assert not captured["artifact_root"].exists()


def test_structured_client_retries_transient_model_capacity(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    calls = []
    logs = []
    sleeps = []

    def fake_run(command, **_kwargs):
        calls.append(_command_text(command))
        if len(calls) == 1:
            return _completed(
                command,
                returncode=1,
                stderr=(
                    "OpenAI Codex v0.139.0\n"
                    "ERROR: Selected model is at capacity. Please try a different model."
                ),
            )
        _argument_path(command, "--output-last-message").write_text(
            '{"ok":true}',
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_cli.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = codex_cli.CodexCliStructuredClient(
        model="gpt-5.5",
        _inspection={
            "executable": str(executable),
            "version": "codex-cli 1.0",
            "supports_ignore_user_config": True,
            "supports_image": True,
        },
        log=logs.append,
    )

    result = client.run(
        prompt="Return ok.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [codex_cli.CODEX_TRANSIENT_RETRY_DELAYS[0]]
    assert "повтор 2/4" in logs[0]


def test_structured_client_capacity_error_is_compact_after_retries(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    sleeps = []

    def fake_run(command, **_kwargs):
        return _completed(
            command,
            returncode=1,
            stderr=(
                "OpenAI Codex v0.139.0\n"
                "--------\n"
                "workdir: C:\\Temp\n"
                "model: gpt-5.5\n"
                "ERROR: Selected model is at capacity. Please try a different model.\n"
                "ERROR: Selected model is at capacity. Please try a different model."
            ),
        )

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_cli.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = codex_cli.CodexCliStructuredClient(
        model="gpt-5.5",
        _inspection={
            "executable": str(executable),
            "version": "codex-cli 1.0",
            "supports_ignore_user_config": True,
            "supports_image": True,
        },
    )

    with pytest.raises(codex_cli.CodexCliTransientError) as error:
        client.run(
            prompt="Return ok.",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )

    message = str(error.value)
    assert "gpt-5.5" in message
    assert "Selected model is at capacity" in message
    assert "workdir:" not in message
    assert sleeps == list(codex_cli.CODEX_TRANSIENT_RETRY_DELAYS)


def test_structured_client_keeps_valid_response_when_temp_cleanup_is_locked(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    cleanup_attempts = []
    real_rmtree = codex_cli.shutil.rmtree

    def fake_run(command, **_kwargs):
        _argument_path(command, "--output-last-message").write_text(
            '{"ok":true}',
            encoding="utf-8",
        )
        return _completed(command)

    def locked_rmtree(path):
        cleanup_attempts.append(Path(path))
        raise PermissionError(32, "file is in use", str(path))

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_cli.shutil, "rmtree", locked_rmtree)
    monkeypatch.setattr(codex_cli.time, "sleep", lambda _seconds: None)
    client = codex_cli.CodexCliStructuredClient(
        _inspection={
            "executable": str(executable),
            "version": "codex-cli 1.0",
            "supports_ignore_user_config": True,
            "supports_image": True,
        },
    )

    result = client.run(
        prompt="Return ok.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )

    assert result == {"ok": True}
    assert len(cleanup_attempts) == 5
    real_rmtree(cleanup_attempts[0], ignore_errors=True)


def test_structured_document_analysis_returns_compact_profile(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")

        root = Path(kwargs["cwd"])
        captured["command"] = text
        captured["schema"] = json.loads(
            (root / "analysis.schema.json").read_text(encoding="utf-8")
        )
        captured["prompt"] = _stdin_text(kwargs["input"])
        (root / "analysis.output.json").write_text(
            json.dumps(
                {
                    "document_summary": "Elevator maintenance manual.",
                    "translation_guidance": [
                        "Use concise imperative wording.",
                    ],
                    "terminology": [
                        {"source": "safety rope", "target": "предохранительный канат"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    result = translator.analyze_document(
        [{"id": "context-0", "text": "Inspect the safety rope."}],
        instructions="Analyze the document before translation.",
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )

    assert result["document_summary"] == "Elevator maintenance manual."
    assert result["translation_guidance"] == ["Use concise imperative wording."]
    assert result["terminology"] == [
        {"source": "safety rope", "target": "предохранительный канат"}
    ]
    assert captured["schema"]["properties"]["terminology"]["maxItems"] == 24
    assert "representative JSON samples" in captured["prompt"]
    assert '"document_samples"' in captured["prompt"]
    assert "--model gpt-5.4" in captured["command"]
    assert "model_reasoning_effort=xhigh" in captured["command"]


def test_login_status_uses_clean_codex_home(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    commands = []
    environments = []

    def fake_run(command, **kwargs):
        commands.append(_command_text(command))
        environments.append(kwargs["env"])
        text = commands[-1]
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(
            command,
            stdout=(
                "WARNING: proceeding, even though PATH aliases were not created\n"
                "Logged in using ChatGPT"
            ),
        )

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    status = codex_cli.check_codex_cli(str(executable))

    assert status == "codex-cli 1.0\nLogged in using ChatGPT"
    assert "service_tier" not in commands[-1]
    assert commands[-1].endswith("login status")
    assert environments[-1]["CODEX_HOME"].endswith("codex-home")


def test_login_status_stops_when_not_logged_in(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(_command_text(command))
        text = commands[-1]
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(command, stdout="Not logged in")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    with pytest.raises(
        codex_cli.CodexCliAuthenticationError,
        match="не авторизован",
    ):
        codex_cli.check_codex_cli(str(executable), model="gpt-5.5")

    assert len(commands) == 3
    assert commands[-1].endswith("login status")


def test_cli_check_can_require_image_support(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        return _completed(command, stdout="--output-schema <FILE>")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    with pytest.raises(codex_cli.CodexCliError, match="--image"):
        codex_cli.check_codex_cli(
            str(executable),
            require_images=True,
        )


def test_install_or_update_codex_cli_runs_powershell_installer(
    tmp_path,
    monkeypatch,
):
    powershell = tmp_path / "powershell.exe"
    powershell.write_bytes(b"test")
    install_dir = tmp_path / "codex-bin"
    install_dir.mkdir()
    executable = install_dir / "codex.exe"
    executable.write_bytes(b"test")
    calls = []
    environments = []

    monkeypatch.setenv("CODEX_INSTALL_DIR", str(install_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-leak")

    def fake_which(command_name):
        if command_name in {"powershell.exe", "powershell", "pwsh"}:
            return str(powershell)
        return None

    def fake_run(command, **kwargs):
        calls.append(_command_text(command))
        environments.append(kwargs["env"])
        text = calls[-1]
        if codex_cli.CODEX_INSTALLER_URL in text:
            return _completed(command, stdout="Codex CLI installed successfully.")
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(command)

    monkeypatch.setattr(codex_cli.shutil, "which", fake_which)
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    result = codex_cli.install_or_update_codex_cli(timeout=120)

    assert result.executable == str(executable)
    assert result.version == "codex-cli 1.0"
    assert any(
        "-ExecutionPolicy ByPass -c "
        f"{codex_cli.CODEX_INSTALLER_SCRIPT}" in call
        for call in calls
    )
    assert "OPENAI_API_KEY" not in environments[0]
    assert "CODEX_API_KEY" not in environments[0]


def test_install_or_update_codex_cli_requires_powershell(monkeypatch):
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _name: None)

    with pytest.raises(codex_cli.CodexCliError, match="PowerShell не найден"):
        codex_cli.install_or_update_codex_cli()


def test_login_codex_cli_runs_login_and_sanitizes_environment(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)
    captured = {}

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-leak")

    def fake_run(command, **kwargs):
        captured.setdefault("commands", []).append(_command_text(command))
        captured.setdefault("environments", []).append(kwargs["env"])
        return _completed(command, stdout="Successfully logged in")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    message = codex_cli.login_codex_cli(str(executable), timeout=120)

    assert captured["commands"][0].endswith("codex.exe logout")
    assert captured["commands"][1].endswith("codex.exe login")
    assert "OPENAI_API_KEY" not in captured["environments"][0]
    assert "CODEX_API_KEY" not in captured["environments"][0]
    assert "codex login" in message


def test_invalidated_refresh_token_has_concise_authentication_error(
    tmp_path,
    monkeypatch,
):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(
            command,
            returncode=1,
            stderr=(
                "large internal log\n"
                'ERROR: {"code":"refresh_token_invalidated"}\n'
                "Your access token could not be refreshed because your refresh "
                "token was revoked."
            ),
        )

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    with pytest.raises(codex_cli.CodexCliAuthenticationError) as error:
        translator.translate([{"id": "0", "text": "Text"}], instructions="Translate.")

    message = str(error.value)
    assert "codex logout" in message
    assert "codex login" in message
    assert "large internal log" not in message


def test_cli_check_can_probe_selected_model(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(_command_text(command))
        text = commands[-1]
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        if text.endswith("login status"):
            return _completed(command, stdout="Logged in using ChatGPT")
        (Path(kwargs["cwd"]) / "translation.output.json").write_text(
            '{"translations":[{"id":"check","text":"OK"}]}',
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    status = codex_cli.check_codex_cli(
        str(executable),
        model="gpt-5.5",
        reasoning_effort="medium",
    )

    assert status.endswith("Модель gpt-5.5: доступна")
    assert any("--model gpt-5.5" in command for command in commands)
    assert len(commands) == 4


def test_clean_codex_home_copies_file_authentication(tmp_path, monkeypatch):
    source_home = tmp_path / "source-codex"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token":"secret"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source_home))

    environment = codex_cli._isolated_codex_environment(tmp_path / "run")
    isolated_auth = Path(environment["CODEX_HOME"]) / "auth.json"

    assert isolated_auth.read_text(encoding="utf-8") == '{"token":"secret"}'
    assert not (Path(environment["CODEX_HOME"]) / "config.toml").exists()


def test_request_environment_reuses_codex_home_when_config_can_be_ignored(
    tmp_path,
    monkeypatch,
):
    source_home = tmp_path / "source-codex"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))

    environment = codex_cli._codex_request_environment(
        tmp_path / "run",
        supports_ignore_user_config=True,
    )

    assert environment == {}


def test_invalid_json_is_retried_once(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)
    translation_calls = 0

    def fake_run(command, **kwargs):
        nonlocal translation_calls
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        translation_calls += 1
        output = Path(kwargs["cwd"]) / "translation.output.json"
        if translation_calls == 1:
            output.write_text("{broken", encoding="utf-8")
        else:
            output.write_text(
                '{"translations":[{"id":"0","text":"Перевод"}]}',
                encoding="utf-8",
            )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    assert translator.translate(
        [{"id": "0", "text": "Text"}],
        instructions="Translate.",
    ) == ["Перевод"]
    assert translation_calls == 2


@pytest.mark.parametrize("blank", ["", " \t\n"])
def test_blank_seventh_translation_is_retried_once(tmp_path, monkeypatch, blank):
    prompts = []
    items = [
        {"id": f"{index}:0", "text": f"Source text {index + 1}"}
        for index in range(8)
    ]

    def fake_run(command, **kwargs):
        prompts.append(_stdin_text(kwargs["input"]))
        translations = [
            {
                "id": item["id"],
                "text": (
                    blank if index == 6 and len(prompts) == 1 else f"Перевод {index + 1}"
                ),
            }
            for index, item in enumerate(items)
        ]
        # The failing item is identified by its requested position, not response order.
        _argument_path(command, "--output-last-message").write_text(
            json.dumps(
                {"translations": list(reversed(translations))}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(
        _inspection={
            "executable": str(_fake_executable(tmp_path)),
            "version": "codex-cli test",
            "supports_ignore_user_config": True,
        }
    )

    assert translator.translate(items, instructions="Translate to Russian.") == [
        f"Перевод {index + 1}" for index in range(8)
    ]
    assert len(prompts) == 2
    assert "never return empty or whitespace-only text" in prompts[1]
    assert '"id": "6:0"' in prompts[1]
    assert '"text": "Source text 7"' in prompts[1]


@pytest.mark.parametrize("blank", ["", " \t\n"])
def test_repeated_blank_translation_reports_source_after_two_attempts(
    tmp_path, monkeypatch, blank
):
    calls = []
    source = "Inspect the safety circuit. " + "Long detail " * 30

    def fake_run(command, **kwargs):
        calls.append(command)
        _argument_path(command, "--output-last-message").write_text(
            json.dumps({"translations": [{"id": "6:0", "text": blank}]}),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(
        _inspection={
            "executable": str(_fake_executable(tmp_path)),
            "version": "codex-cli test",
            "supports_ignore_user_config": True,
        }
    )

    with pytest.raises(codex_cli.CodexCliStructuredOutputError) as error:
        translator.translate(
            [{"id": "6:0", "text": source}], instructions="Translate to Russian."
        )

    assert len(calls) == 2
    message = str(error.value)
    assert "пустой перевод фрагмента 1" in message
    assert "id: '6:0'" in message
    assert "Исходный текст: 'Inspect the safety circuit." in message
    assert "…" in message
    assert len(message) < 320


def test_blank_sources_allow_blank_translations_without_retry(tmp_path, monkeypatch):
    calls = []
    items = [
        {"id": "empty", "text": ""},
        {"id": "whitespace", "text": " \t\n"},
        {"id": "text", "text": "Inspect the safety circuit."},
    ]
    expected = ["", "\n", "Проверьте цепь безопасности."]

    def fake_run(command, **kwargs):
        calls.append(command)
        _argument_path(command, "--output-last-message").write_text(
            json.dumps(
                {
                    "translations": [
                        {"id": item["id"], "text": value}
                        for item, value in zip(items, expected)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(
        _inspection={
            "executable": str(_fake_executable(tmp_path)),
            "version": "codex-cli test",
            "supports_ignore_user_config": True,
        }
    )

    assert translator.translate(items, instructions="Translate to Russian.") == expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"translations": [{"id": "0", "text": "A"}, {"id": "0", "text": "B"}]},
            "продублировал",
        ),
        ({"translations": []}, "не вернул переводы"),
    ],
)
def test_invalid_id_sets_fail_after_retry(
    tmp_path,
    monkeypatch,
    payload,
    message,
):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        (Path(kwargs["cwd"]) / "translation.output.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return _completed(command)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    with pytest.raises(codex_cli.CodexCliStructuredOutputError, match=message):
        translator.translate([{"id": "0", "text": "Text"}], instructions="Translate.")


def test_missing_executable_fails_without_fallback(tmp_path):
    with pytest.raises(codex_cli.CodexCliError, match="не найден"):
        codex_cli.CodexCliTranslator(cli_path=str(tmp_path / "missing.exe"))


def test_timeout_is_reported(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    with pytest.raises(codex_cli.CodexCliError, match="не завершил"):
        translator.translate([{"id": "0", "text": "Text"}], instructions="Translate.")


def test_nonzero_exit_includes_stderr(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 1.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(command, returncode=1, stderr="authentication failed")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(cli_path=str(executable))

    with pytest.raises(codex_cli.CodexCliError, match="authentication failed"):
        translator.translate([{"id": "0", "text": "Text"}], instructions="Translate.")


def test_old_cli_model_error_has_update_instructions(tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path)

    def fake_run(command, **kwargs):
        text = _command_text(command)
        if "--version" in text:
            return _completed(command, stdout="codex-cli 0.118.0")
        if "exec --help" in text:
            return _completed(command, stdout="--output-schema <FILE>")
        return _completed(
            command,
            returncode=1,
            stderr=(
                "The 'gpt-5.5' model requires a newer version of Codex. "
                "Please upgrade to the latest app or CLI and try again."
            ),
        )

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    translator = codex_cli.CodexCliTranslator(
        cli_path=str(executable),
        model="gpt-5.5",
    )

    with pytest.raises(codex_cli.CodexCliError) as error:
        translator.translate([{"id": "0", "text": "Text"}], instructions="Translate.")

    message = str(error.value)
    assert "codex-cli 0.118.0" in message
    assert "https://chatgpt.com/codex/install.ps1" in message
    assert "gpt-5.4-mini" in message


def test_engine_splits_codex_batches_at_fifty_items(monkeypatch):
    calls = []

    class FakeCodex:
        def __init__(self, **_kwargs):
            pass

        def translate(self, items, *, instructions):
            calls.append((list(items), instructions))
            return [f"translated:{item['text']}" for item in items]

    monkeypatch.setattr("daru.translation.engine.CodexCliTranslator", FakeCodex)
    engine = TranslationEngine(provider="codex")
    values = [f"Source text {index}" for index in range(55)]

    result = engine.translate_many(values)

    assert len(calls) == 2
    assert [len(items) for items, _instructions in calls] == [50, 5]
    assert result[0] == "translated:Source text 0"
    assert result[-1] == "translated:Source text 54"


def test_engine_analyzes_document_once_and_reuses_profile(monkeypatch):
    analysis_calls = []
    translation_calls = []
    init_calls = []
    reviews = []

    class FakeCodex:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def analyze_document(
            self,
            items,
            *,
            instructions,
            model,
            reasoning_effort,
        ):
            analysis_calls.append(
                (list(items), instructions, model, reasoning_effort)
            )
            return {
                "document_summary": "Technical elevator installation manual.",
                "translation_guidance": ["Use imperative wording for procedures."],
                "terminology": [
                    {"source": "main rope", "target": "главный канат"},
                ],
            }

        def translate(self, items, *, instructions):
            translation_calls.append((list(items), instructions))
            return [f"translated:{item['text']}" for item in items]

    monkeypatch.setattr("daru.translation.engine.CodexCliTranslator", FakeCodex)
    session = CodexAnalysisSession(
        lambda review: reviews.append(review) or "Отредактированный анализ"
    )
    engine = TranslationEngine(
        provider="codex",
        source_lang="en",
        target_lang="ru",
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="low",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="high",
        codex_analysis_session=session,
    )
    context = [
        f"Section {index}: inspect the main rope and safety components carefully."
        for index in range(40)
    ]
    engine.set_document_context(context, context_label="DOCX DOCUMENT")

    first = engine.translate_many(["Inspect the main rope."])
    second = engine.translate_many(["Record the inspection result."])

    assert first == ["translated:Inspect the main rope."]
    assert second == ["translated:Record the inspection result."]
    assert len(analysis_calls) == 1
    sampled_items, analysis_instructions, model, effort = analysis_calls[0]
    assert len(sampled_items) <= 120
    assert sum(len(item["text"]) for item in sampled_items) <= 12000
    assert "before translation from Английский (en) to Русский (ru)" in analysis_instructions
    assert "target language (Русский (ru))" in analysis_instructions
    assert model == "gpt-5.5"
    assert effort == "high"
    assert init_calls[0]["model"] == "gpt-5.4-mini"
    assert init_calls[0]["reasoning_effort"] == "low"
    assert len(reviews) == 1
    assert len(translation_calls) == 2
    for _items, instructions in translation_calls:
        assert "[DOCX DOCUMENT PRE-TRANSLATION ANALYSIS]" in instructions
        assert "Отредактированный анализ" in instructions


def test_engine_falls_back_to_raw_context_when_analysis_fails(monkeypatch):
    translation_instructions = []
    reviews = []

    class FakeCodex:
        def __init__(self, **_kwargs):
            pass

        def analyze_document(self, items, **_kwargs):
            raise codex_cli.CodexCliStructuredOutputError("invalid profile")

        def translate(self, items, *, instructions):
            translation_instructions.append(instructions)
            return [item["text"] for item in items]

    monkeypatch.setattr("daru.translation.engine.CodexCliTranslator", FakeCodex)
    session = CodexAnalysisSession(
        lambda review: reviews.append(review) or review.text
    )
    engine = TranslationEngine(provider="codex", codex_analysis_session=session)
    context = [
        f"Long technical context line {index} with installation terminology."
        for index in range(30)
    ]
    engine.set_document_context(context, context_label="DOCUMENT")

    assert engine.translate_many(["Translate this."]) == ["Translate this."]
    assert "[DOCUMENT PRE-TRANSLATION ANALYSIS]" in translation_instructions[0]
    assert "Long technical context line 0" in translation_instructions[0]
    assert reviews[0].used_fallback
    assert "invalid profile" in reviews[0].warning


def test_engine_analyzes_small_documents(monkeypatch):
    analysis_calls = []

    class FakeCodex:
        def __init__(self, **_kwargs):
            pass

        def analyze_document(self, items, **_kwargs):
            analysis_calls.append((items, _kwargs["instructions"]))
            return {
                "document_summary": "Короткий документ.",
                "translation_guidance": [],
                "terminology": [],
            }

        def translate(self, items, *, instructions):
            return [item["text"] for item in items]

    monkeypatch.setattr("daru.translation.engine.CodexCliTranslator", FakeCodex)
    engine = TranslationEngine(provider="codex")
    engine.set_document_context(["Short title", "One short sentence."])

    assert engine.translate_many(["Translate this."]) == ["Translate this."]
    assert len(analysis_calls) == 1


def test_analysis_session_is_reused_by_multiple_engines(monkeypatch):
    analysis_calls = []
    review_calls = []

    class FakeCodex:
        def __init__(self, **_kwargs):
            pass

        def analyze_document(self, items, **_kwargs):
            analysis_calls.append(list(items))
            return {
                "document_summary": "Общий анализ.",
                "translation_guidance": [],
                "terminology": [],
            }

        def translate(self, items, *, instructions):
            assert "Утверждённый анализ" in instructions
            return [item["text"] for item in items]

    monkeypatch.setattr("daru.translation.engine.CodexCliTranslator", FakeCodex)
    session = CodexAnalysisSession(
        lambda review: review_calls.append(review) or "Утверждённый анализ"
    )

    first = TranslationEngine(provider="codex", codex_analysis_session=session)
    first.set_document_context(["Native PDF text"])
    second = TranslationEngine(provider="codex", codex_analysis_session=session)
    second.set_document_context(["OCR fallback text"])

    first.translate_many(["First"])
    second.translate_many(["Second"])

    assert len(analysis_calls) == 1
    assert len(review_calls) == 1
