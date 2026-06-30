"""Tests for the structured Codex CLI translation adapter."""

from __future__ import annotations

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
    assert "before translation from en to ru" in analysis_instructions
    assert "target language (ru)" in analysis_instructions
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
