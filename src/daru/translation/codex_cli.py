"""Non-interactive Codex CLI adapter for structured translation batches."""

from __future__ import annotations

import json
import locale
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

CODEX_INSTALLER_URL = "https://chatgpt.com/codex/install.ps1"
CODEX_INSTALLER_SCRIPT = f"irm {CODEX_INSTALLER_URL} | iex"
CODEX_INSTALLER_COMMAND = (
    f'powershell -ExecutionPolicy ByPass -c "{CODEX_INSTALLER_SCRIPT}"'
)
CODEX_DEFAULT_MODEL = "gpt-5.4-mini"
CODEX_DEFAULT_REASONING_EFFORT = "low"
CODEX_DEFAULT_ANALYSIS_MODEL = "gpt-5.5"
CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT = "high"
CODEX_DEFAULT_TIMEOUT_SECONDS = 300
CODEX_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
CODEX_TRANSIENT_RETRY_DELAYS = (8.0, 20.0, 45.0)

_SENSITIVE_ENV_KEYS = {
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
}


class CodexCliError(RuntimeError):
    """Base error for Codex CLI availability and execution failures."""


class CodexCliAuthenticationError(CodexCliError):
    """Raised when cached Codex CLI credentials are missing or invalid."""


class CodexCliStructuredOutputError(CodexCliError):
    """Raised when Codex returns data that does not match the translation batch."""


class CodexCliTransientError(CodexCliError):
    """Raised when Codex reports a temporary model/service capacity issue."""


@dataclass(frozen=True)
class CodexCliInstallResult:
    executable: str
    version: str
    message: str


def _noop_log(_message: str) -> None:
    return


def resolve_codex_cli(cli_path: str = "") -> str:
    """Resolve an explicit Codex path or find the command on PATH."""

    value = os.path.expandvars(os.path.expanduser((cli_path or "").strip()))
    if value:
        candidate = Path(value)
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(value)
        if resolved:
            return resolved
        raise CodexCliError(f"Codex CLI не найден: {value}")

    command_names = (
        ("codex.cmd", "codex.exe", "codex")
        if os.name == "nt"
        else ("codex",)
    )
    for command_name in command_names:
        resolved = shutil.which(command_name)
        if resolved:
            return resolved
    raise CodexCliError(
        "Codex CLI не найден в PATH. Установите Codex CLI и выполните `codex login`."
    )


def _resolve_powershell() -> str:
    command_names = (
        ("powershell.exe", "powershell")
        if os.name == "nt"
        else ("powershell", "pwsh")
    )
    for command_name in command_names:
        resolved = shutil.which(command_name)
        if resolved:
            return resolved
    raise CodexCliError(
        "PowerShell не найден в PATH. Установите PowerShell или укажите уже "
        "установленный исполняемый файл Codex CLI вручную."
    )


def _standalone_codex_candidates() -> List[Path]:
    configured = os.path.expandvars(
        os.path.expanduser((os.environ.get("CODEX_INSTALL_DIR") or "").strip())
    )
    if configured:
        return [Path(configured) / "codex.exe"]

    local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
    if not local_app_data:
        local_app_data = str(Path.home() / "AppData" / "Local")
    return [
        Path(local_app_data)
        / "Programs"
        / "OpenAI"
        / "Codex"
        / "bin"
        / "codex.exe"
    ]


def _sanitized_environment(
    overrides: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    environment = dict(os.environ)
    for key in _SENSITIVE_ENV_KEYS:
        environment.pop(key, None)
    if overrides:
        environment.update(overrides)
    return environment


def _source_codex_home() -> Path:
    configured = (os.environ.get("CODEX_HOME") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def _isolated_codex_environment(root: Path) -> Dict[str, str]:
    """Create a clean Codex home while retaining file-based CLI authentication."""

    codex_home = root / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    source_auth = _source_codex_home() / "auth.json"
    if source_auth.is_file():
        try:
            shutil.copy2(source_auth, codex_home / "auth.json")
        except OSError as exc:
            raise CodexCliError(
                f"Не удалось подготовить авторизацию Codex CLI: {exc}"
            ) from exc
    return {"CODEX_HOME": str(codex_home)}


def _codex_request_environment(
    root: Path,
    *,
    supports_ignore_user_config: bool,
) -> Dict[str, str]:
    """Reuse Codex caches when user config can be disabled explicitly."""

    if supports_ignore_user_config:
        return {}
    return _isolated_codex_environment(root)


@contextmanager
def _request_temp_directory(prefix: str):
    """Clean request artifacts without masking an already completed Codex call."""

    root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield root
    finally:
        for attempt in range(5):
            try:
                shutil.rmtree(root)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt < 4:
                    time.sleep(0.15 * (attempt + 1))


def _launcher_command(executable: str, arguments: Sequence[str]) -> List[str]:
    suffix = Path(executable).suffix.lower()
    if os.name == "nt" and suffix in {".cmd", ".bat"}:
        return [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            "call",
            executable,
            *arguments,
        ]
    if os.name == "nt" and suffix == ".ps1":
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            executable,
            *arguments,
        ]
    return [executable, *arguments]


def _decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    encodings = ["utf-8-sig", "utf-8"]
    if os.name == "nt":
        encodings.append("cp866")
    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)
    encodings.append("cp1251")

    seen = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return value.decode("utf-8", errors="replace")


def _run_process(
    executable: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
    input_text: str | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    capture_via_files: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        command = _launcher_command(executable, arguments)
        common_kwargs = {
            "cwd": str(cwd),
            "input": input_text.encode("utf-8") if input_text is not None else None,
            "timeout": timeout,
            "check": False,
            "env": _sanitized_environment(environment_overrides),
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        if capture_via_files:
            # Codex may spawn cache/plugin helpers that inherit stdout/stderr. Pipes
            # keep subprocess.run().communicate() waiting after the main process has
            # already written its structured response. Real files avoid that deadlock.
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                result = subprocess.run(
                    command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    **common_kwargs,
                )
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = (
                    result.stdout
                    if result.stdout is not None
                    else stdout_file.read()
                )
                stderr = (
                    result.stderr
                    if result.stderr is not None
                    else stderr_file.read()
                )
        else:
            result = subprocess.run(
                command,
                capture_output=True,
                **common_kwargs,
            )
            stdout = result.stdout
            stderr = result.stderr
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            _decode_process_output(stdout),
            _decode_process_output(stderr),
        )
    except FileNotFoundError as exc:
        raise CodexCliError(f"Не удалось запустить Codex CLI: {executable}") from exc
    except OSError as exc:
        raise CodexCliError(f"Не удалось запустить Codex CLI: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodexCliError(
            f"Codex CLI не завершил операцию за {timeout} с."
        ) from exc


def _process_error(prefix: str, result: subprocess.CompletedProcess[str]) -> CodexCliError:
    details = (result.stderr or result.stdout or "").strip()
    if details:
        return CodexCliError(f"{prefix}:\n{details}")
    return CodexCliError(f"{prefix} (код выхода {result.returncode})")


def _compact_error_details(details: str) -> str:
    lines = [line.strip() for line in details.splitlines() if line.strip()]
    error_lines = [
        line
        for line in lines
        if line.upper().startswith("ERROR") or "capacity" in line.lower()
    ]
    selected = error_lines or lines[-6:]
    compact: List[str] = []
    for line in selected:
        if line not in compact:
            compact.append(line)
    return "\n".join(compact[:8])


def _is_transient_codex_error(details: str) -> bool:
    normalized = details.lower()
    markers = (
        "selected model is at capacity",
        "model is at capacity",
        "at capacity",
        "rate_limit",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "server overloaded",
        "overloaded",
        " 429",
        " 503",
    )
    return any(marker in normalized for marker in markers)


def _translation_process_error(
    result: subprocess.CompletedProcess[str],
    *,
    model: str,
    version: str,
) -> CodexCliError:
    details = (result.stderr or result.stdout or "").strip()
    normalized_details = details.lower()
    authentication_markers = (
        "refresh_token_invalidated",
        "token_invalidated",
        "your session has ended",
        "access token could not be refreshed",
        "refresh token was revoked",
    )
    if any(marker in normalized_details for marker in authentication_markers):
        return CodexCliAuthenticationError(
            "Сессия Codex CLI истекла или была отозвана. Выполните повторный вход: "
            "`codex logout`, затем `codex login`."
        )
    if _is_transient_codex_error(details):
        compact = _compact_error_details(details)
        suffix = f"\n{compact}" if compact else ""
        return CodexCliTransientError(
            f"Временная недоступность модели Codex CLI `{model}`. "
            "Запрос можно повторить позже или выбрать другую Vision-модель."
            f"{suffix}"
        )
    if "requires a newer version of Codex" in details:
        return CodexCliError(
            f"Модель `{model}` не поддерживается установленной версией "
            f"Codex CLI ({version}).\n"
            f"Обновите CLI командой `{CODEX_INSTALLER_COMMAND}` "
            f"или выберите модель `{CODEX_DEFAULT_MODEL}` в настройках.\n\n"
            f"Ответ Codex CLI:\n{details}"
        )
    return _process_error("Codex CLI завершил перевод с ошибкой", result)


def _login_status_text(result: subprocess.CompletedProcess[str]) -> str:
    details = (result.stdout or result.stderr).strip()
    for line in reversed(details.splitlines()):
        normalized = line.strip()
        if normalized.lower().startswith(("logged in", "not logged in")):
            return normalized
    return details or "авторизация активна"


def install_or_update_codex_cli(*, timeout: int = 300) -> CodexCliInstallResult:
    """Install or update Codex CLI with the official PowerShell installer."""

    powershell = _resolve_powershell()
    effective_timeout = max(30, int(timeout))
    with tempfile.TemporaryDirectory(prefix="daru-codex-install-") as temp_dir:
        root = Path(temp_dir)
        result = _run_process(
            powershell,
            [
                "-ExecutionPolicy",
                "ByPass",
                "-c",
                CODEX_INSTALLER_SCRIPT,
            ],
            cwd=root,
            timeout=effective_timeout,
        )
        if result.returncode != 0:
            raise _process_error(
                "Не удалось установить или обновить Codex CLI через официальный "
                "PowerShell installer",
                result,
            )

        install_output = (result.stdout or result.stderr or "").strip()
        info = None
        inspect_error = None
        for candidate in _standalone_codex_candidates():
            if not candidate.is_file():
                continue
            try:
                info = inspect_codex_cli(
                    str(candidate),
                    timeout=min(60, effective_timeout),
                )
                break
            except CodexCliError as exc:
                inspect_error = exc

        if info is None:
            try:
                info = inspect_codex_cli("", timeout=min(60, effective_timeout))
            except CodexCliError as exc:
                inspect_error = exc

        if info is None:
            details = (
                "\n\nВывод installer:\n" + install_output
                if install_output
                else ""
            )
            raise CodexCliError(
                "Codex CLI установлен, но исполняемый файл не найден. Укажите путь "
                "к `codex` вручную или перезапустите приложение после обновления "
                f"PATH.{details}"
            ) from inspect_error

    executable = str(info["executable"])
    version = str(info["version"])
    message = (
        "Codex CLI установлен или обновлён.\n"
        f"{version}\n"
        f"Путь: {executable}"
    )
    return CodexCliInstallResult(
        executable=executable,
        version=version,
        message=message,
    )


def login_codex_cli(cli_path: str = "", *, timeout: int = 300) -> str:
    """Clear cached credentials and run the interactive Codex login flow."""

    executable = resolve_codex_cli(cli_path)
    with tempfile.TemporaryDirectory(prefix="daru-codex-login-") as temp_dir:
        root = Path(temp_dir)
        _run_process(
            executable,
            ["logout"],
            cwd=root,
            timeout=min(60, max(30, int(timeout))),
        )
        result = _run_process(
            executable,
            ["login"],
            cwd=root,
            timeout=max(30, int(timeout)),
        )
    if result.returncode != 0:
        raise _process_error("Не удалось выполнить `codex login`", result)
    return "Авторизация Codex CLI выполнена (`codex login`)."


def inspect_codex_cli(cli_path: str = "", *, timeout: int = 20) -> Dict[str, Any]:
    """Check executable capabilities without starting a model request."""

    executable = resolve_codex_cli(cli_path)
    with tempfile.TemporaryDirectory(prefix="daru-codex-check-") as temp_dir:
        root = Path(temp_dir)
        version_result = _run_process(
            executable,
            ["--version"],
            cwd=root,
            timeout=timeout,
        )
        if version_result.returncode != 0:
            raise _process_error("Не удалось определить версию Codex CLI", version_result)

        help_result = _run_process(
            executable,
            ["exec", "--help"],
            cwd=root,
            timeout=timeout,
        )
        if help_result.returncode != 0:
            raise _process_error("Не удалось прочитать параметры `codex exec`", help_result)

    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    if "--output-schema" not in help_text:
        raise CodexCliError(
            "Установленная версия Codex CLI не поддерживает `codex exec --output-schema`."
        )
    return {
        "executable": executable,
        "version": (version_result.stdout or version_result.stderr).strip(),
        "supports_ignore_user_config": "--ignore-user-config" in help_text,
        "supports_image": "--image" in help_text,
    }


def check_codex_cli(
    cli_path: str = "",
    *,
    timeout: int = 20,
    model: str = "",
    reasoning_effort: str = CODEX_DEFAULT_REASONING_EFFORT,
    require_images: bool = False,
) -> str:
    """Return version, authentication status, and optional model availability."""

    info = inspect_codex_cli(cli_path, timeout=timeout)
    if require_images and not info.get("supports_image", False):
        raise CodexCliError(
            "Установленная версия Codex CLI не поддерживает `codex exec --image`."
        )
    executable = str(info["executable"])
    with tempfile.TemporaryDirectory(prefix="daru-codex-login-") as temp_dir:
        root = Path(temp_dir)
        result = _run_process(
            executable,
            ["login", "status"],
            cwd=root,
            timeout=timeout,
            environment_overrides=_isolated_codex_environment(root),
        )
    if result.returncode != 0:
        raise _process_error(
            f"Codex CLI найден ({info['version']}), но проверка авторизации завершилась ошибкой",
            result,
        )
    status = _login_status_text(result)
    if status.lower().startswith("not logged in"):
        raise CodexCliAuthenticationError(
            "Codex CLI не авторизован. Выполните `codex login`."
        )
    message = f"{info['version']}\n{status}"
    selected_model = (model or "").strip()
    if selected_model:
        translator = CodexCliTranslator(
            cli_path=cli_path,
            model=selected_model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout,
            _inspection=info,
        )
        translator.translate(
            [{"id": "check", "text": "OK"}],
            instructions=(
                "This is a connectivity check. Return the input text exactly unchanged."
            ),
        )
        message += f"\nМодель {selected_model}: доступна"
    return message


def _translation_schema(ids: Sequence[str]) -> Dict[str, Any]:
    count = len(ids)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(ids)},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def _document_analysis_schema() -> Dict[str, Any]:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "document_summary": {"type": "string"},
            "translation_guidance": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string"},
            },
            "terminology": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "document_summary",
            "translation_guidance",
            "terminology",
        ],
        "additionalProperties": False,
    }


def _validate_response(payload: Any, expected_ids: Sequence[str]) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise CodexCliStructuredOutputError("Codex CLI вернул не JSON-объект.")
    translations = payload.get("translations")
    if not isinstance(translations, list):
        raise CodexCliStructuredOutputError(
            "В ответе Codex CLI отсутствует массив `translations`."
        )

    expected = set(expected_ids)
    mapping: Dict[str, str] = {}
    for item in translations:
        if not isinstance(item, dict):
            raise CodexCliStructuredOutputError(
                "Элемент `translations` должен быть JSON-объектом."
            )
        item_id = item.get("id")
        text = item.get("text")
        if not isinstance(item_id, str) or item_id not in expected:
            raise CodexCliStructuredOutputError(
                f"Codex CLI вернул неизвестный идентификатор: {item_id!r}."
            )
        if item_id in mapping:
            raise CodexCliStructuredOutputError(
                f"Codex CLI продублировал идентификатор: {item_id}."
            )
        if not isinstance(text, str):
            raise CodexCliStructuredOutputError(
                f"Перевод для {item_id} должен быть строкой."
            )
        mapping[item_id] = text

    missing = [item_id for item_id in expected_ids if item_id not in mapping]
    if missing:
        raise CodexCliStructuredOutputError(
            "Codex CLI не вернул переводы для: " + ", ".join(missing)
        )
    if len(mapping) != len(expected_ids):
        raise CodexCliStructuredOutputError(
            "Codex CLI вернул неверное количество переводов."
        )
    return mapping


def _validate_document_analysis(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise CodexCliStructuredOutputError(
            "Codex CLI returned an invalid document analysis object."
        )

    summary = payload.get("document_summary")
    guidance = payload.get("translation_guidance")
    terminology = payload.get("terminology")
    if not isinstance(summary, str):
        raise CodexCliStructuredOutputError(
            "Codex CLI document analysis is missing `document_summary`."
        )
    if not isinstance(guidance, list) or not all(
        isinstance(item, str) for item in guidance
    ):
        raise CodexCliStructuredOutputError(
            "Codex CLI document analysis contains invalid `translation_guidance`."
        )
    if not isinstance(terminology, list):
        raise CodexCliStructuredOutputError(
            "Codex CLI document analysis contains invalid `terminology`."
        )

    cleaned_terms: List[Dict[str, str]] = []
    for item in terminology:
        if not isinstance(item, dict):
            raise CodexCliStructuredOutputError(
                "Codex CLI document terminology must contain JSON objects."
            )
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise CodexCliStructuredOutputError(
                "Codex CLI document terminology must contain string values."
            )
        source = source.strip()
        target = target.strip()
        if source and target:
            cleaned_terms.append(
                {"source": source[:200], "target": target[:200]}
            )

    return {
        "document_summary": summary.strip()[:1600],
        "translation_guidance": [
            item.strip()[:400] for item in guidance[:8] if item.strip()
        ],
        "terminology": cleaned_terms[:24],
    }


class CodexCliStructuredClient:
    """Run isolated Codex CLI requests with images and a JSON output schema."""

    def __init__(
        self,
        *,
        cli_path: str = "",
        model: str = CODEX_DEFAULT_ANALYSIS_MODEL,
        reasoning_effort: str = "medium",
        timeout_seconds: int = CODEX_DEFAULT_TIMEOUT_SECONDS,
        _inspection: Mapping[str, Any] | None = None,
        log: Callable[[str], None] = _noop_log,
    ) -> None:
        info = dict(_inspection) if _inspection is not None else inspect_codex_cli(cli_path)
        self.executable = str(info["executable"])
        self.version = str(info["version"])
        self.supports_ignore_user_config = bool(
            info.get("supports_ignore_user_config", False)
        )
        self.supports_image = bool(info.get("supports_image", False))
        self.model = (model or CODEX_DEFAULT_ANALYSIS_MODEL).strip()
        effort = (reasoning_effort or "medium").strip().lower()
        self.reasoning_effort = effort if effort in CODEX_REASONING_EFFORTS else "medium"
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.log = log

    def run(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        images: Sequence[tuple[str, bytes]] = (),
    ) -> Dict[str, Any]:
        attempts = len(CODEX_TRANSIENT_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                return self._run_once(prompt=prompt, schema=schema, images=images)
            except CodexCliTransientError:
                if attempt >= len(CODEX_TRANSIENT_RETRY_DELAYS):
                    raise
                delay = CODEX_TRANSIENT_RETRY_DELAYS[attempt]
                self.log(
                    "Codex CLI: модель временно перегружена, "
                    f"повтор {attempt + 2}/{attempts} через {delay:g} с."
                )
                time.sleep(delay)
        raise CodexCliStructuredOutputError("Codex CLI не вернул структурированный ответ.")

    def _run_once(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        images: Sequence[tuple[str, bytes]] = (),
    ) -> Dict[str, Any]:
        if images and not self.supports_image:
            raise CodexCliError(
                "Установленная версия Codex CLI не поддерживает `codex exec --image`."
            )

        with _request_temp_directory("daru-codex-structured-") as root:
            schema_path = root / "response.schema.json"
            output_path = root / "response.output.json"
            schema_path.write_text(
                json.dumps(dict(schema), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            arguments = ["exec"]
            if self.supports_ignore_user_config:
                arguments.append("--ignore-user-config")
            arguments.extend(
                [
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--disable",
                    "shell_tool",
                    "-c",
                    "web_search=disabled",
                    "-c",
                    "approval_policy=never",
                    "-c",
                    f"model_reasoning_effort={self.reasoning_effort}",
                    "--model",
                    self.model,
                ]
            )
            for index, (name, content) in enumerate(images):
                image_name = Path(name).name or f"image-{index + 1}.png"
                image_path = root / f"{index + 1:02d}-{image_name}"
                image_path.write_bytes(content)
                arguments.extend(["--image", str(image_path)])
            arguments.extend(
                [
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )

            result = _run_process(
                self.executable,
                arguments,
                # Background Codex cache/plugin helpers may briefly outlive the
                # main process. Keeping their cwd outside the request directory
                # lets Windows delete page images and schema/output artifacts.
                cwd=root.parent,
                timeout=self.timeout_seconds,
                input_text=prompt,
                environment_overrides=_codex_request_environment(
                    root,
                    supports_ignore_user_config=self.supports_ignore_user_config,
                ),
                capture_via_files=True,
            )
            if result.returncode != 0:
                raise _translation_process_error(
                    result,
                    model=self.model,
                    version=self.version,
                )
            if not output_path.exists():
                raise CodexCliStructuredOutputError(
                    "Codex CLI не создал файл структурированного ответа."
                )
            raw_output = output_path.read_text(encoding="utf-8").strip()

        try:
            response = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CodexCliStructuredOutputError(
                f"Codex CLI вернул некорректный JSON: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise CodexCliStructuredOutputError(
                "Codex CLI вернул JSON, который не является объектом."
            )
        return response


class CodexCliTranslator:
    """Translate JSON items through isolated `codex exec` calls."""

    def __init__(
        self,
        *,
        cli_path: str = "",
        model: str = CODEX_DEFAULT_MODEL,
        reasoning_effort: str = CODEX_DEFAULT_REASONING_EFFORT,
        timeout_seconds: int = CODEX_DEFAULT_TIMEOUT_SECONDS,
        _inspection: Mapping[str, Any] | None = None,
    ) -> None:
        info = dict(_inspection) if _inspection is not None else inspect_codex_cli(cli_path)
        self.executable = str(info["executable"])
        self.version = str(info["version"])
        self.supports_ignore_user_config = bool(info["supports_ignore_user_config"])
        self.model = (model or CODEX_DEFAULT_MODEL).strip()
        effort = (reasoning_effort or CODEX_DEFAULT_REASONING_EFFORT).strip().lower()
        self.reasoning_effort = (
            effort if effort in CODEX_REASONING_EFFORTS else CODEX_DEFAULT_REASONING_EFFORT
        )
        self.timeout_seconds = max(10, int(timeout_seconds))

    def translate(
        self,
        items: Sequence[Mapping[str, str]],
        *,
        instructions: str,
    ) -> List[str]:
        expected_ids = [str(item["id"]) for item in items]
        last_error: CodexCliStructuredOutputError | None = None
        for attempt in range(2):
            try:
                mapping = self._translate_once(
                    items,
                    instructions=instructions,
                    retry=attempt > 0,
                )
                return [mapping[item_id] for item_id in expected_ids]
            except CodexCliStructuredOutputError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def analyze_document(
        self,
        items: Sequence[Mapping[str, str]],
        *,
        instructions: str,
        model: str = CODEX_DEFAULT_ANALYSIS_MODEL,
        reasoning_effort: str = CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
    ) -> Dict[str, Any]:
        """Build one compact translation profile from representative document text."""

        selected_model = (model or CODEX_DEFAULT_ANALYSIS_MODEL).strip()
        effort = (reasoning_effort or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT).strip().lower()
        selected_effort = (
            effort
            if effort in CODEX_REASONING_EFFORTS
            else CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
        )
        last_error: CodexCliStructuredOutputError | None = None
        for attempt in range(2):
            try:
                return self._analyze_document_once(
                    items,
                    instructions=instructions,
                    model=selected_model,
                    reasoning_effort=selected_effort,
                    retry=attempt > 0,
                )
            except CodexCliStructuredOutputError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _analyze_document_once(
        self,
        items: Sequence[Mapping[str, str]],
        *,
        instructions: str,
        model: str,
        reasoning_effort: str,
        retry: bool,
    ) -> Dict[str, Any]:
        schema = _document_analysis_schema()
        payload = json.dumps(
            {"document_samples": list(items)},
            ensure_ascii=False,
        )
        retry_note = (
            "\n\nThe previous response failed structural validation. Return all required "
            "fields and no additional fields."
            if retry
            else ""
        )
        prompt = (
            f"{instructions}{retry_note}\n\n"
            "Do not use tools, inspect files, or access the network. Analyze only the "
            "representative JSON samples below and return the compact profile required "
            "by the supplied JSON Schema.\n\n"
            f"{payload}"
        )

        with tempfile.TemporaryDirectory(prefix="daru-codex-analysis-") as temp_dir:
            root = Path(temp_dir)
            schema_path = root / "analysis.schema.json"
            output_path = root / "analysis.output.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            arguments = ["exec"]
            if self.supports_ignore_user_config:
                arguments.append("--ignore-user-config")
            arguments.extend(
                [
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--disable",
                    "shell_tool",
                    "-c",
                    "web_search=disabled",
                    "-c",
                    "approval_policy=never",
                    "-c",
                    f"model_reasoning_effort={reasoning_effort}",
                    "--model",
                    model,
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )
            result = _run_process(
                self.executable,
                arguments,
                cwd=root,
                timeout=self.timeout_seconds,
                input_text=prompt,
                environment_overrides=_codex_request_environment(
                    root,
                    supports_ignore_user_config=self.supports_ignore_user_config,
                ),
                capture_via_files=True,
            )
            if result.returncode != 0:
                raise _translation_process_error(
                    result,
                    model=model,
                    version=self.version,
                )
            if not output_path.exists():
                raise CodexCliStructuredOutputError(
                    "Codex CLI did not create a structured document analysis."
                )
            raw_output = output_path.read_text(encoding="utf-8").strip()

        try:
            response = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CodexCliStructuredOutputError(
                f"Codex CLI returned invalid document analysis JSON: {exc}"
            ) from exc
        return _validate_document_analysis(response)

    def _translate_once(
        self,
        items: Sequence[Mapping[str, str]],
        *,
        instructions: str,
        retry: bool,
    ) -> Dict[str, str]:
        expected_ids = [str(item["id"]) for item in items]
        schema = _translation_schema(expected_ids)
        payload = json.dumps({"items": list(items)}, ensure_ascii=False)
        retry_note = (
            "\n\nThe previous response failed structural validation. Return every requested "
            "id exactly once and no additional fields."
            if retry
            else ""
        )
        prompt = (
            f"{instructions}{retry_note}\n\n"
            "Do not use tools, inspect files, or access the network. Translate only the JSON "
            "items below and return the final response required by the supplied JSON Schema.\n\n"
            f"{payload}"
        )

        with tempfile.TemporaryDirectory(prefix="daru-codex-") as temp_dir:
            root = Path(temp_dir)
            schema_path = root / "translation.schema.json"
            output_path = root / "translation.output.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            arguments = ["exec"]
            if self.supports_ignore_user_config:
                arguments.append("--ignore-user-config")
            arguments.extend(
                [
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--disable",
                    "shell_tool",
                    "-c",
                    "web_search=disabled",
                    "-c",
                    "approval_policy=never",
                    "-c",
                    f"model_reasoning_effort={self.reasoning_effort}",
                    "--model",
                    self.model,
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )
            result = _run_process(
                self.executable,
                arguments,
                cwd=root,
                timeout=self.timeout_seconds,
                input_text=prompt,
                environment_overrides=_codex_request_environment(
                    root,
                    supports_ignore_user_config=self.supports_ignore_user_config,
                ),
                capture_via_files=True,
            )
            if result.returncode != 0:
                raise _translation_process_error(
                    result,
                    model=self.model,
                    version=self.version,
                )
            if not output_path.exists():
                raise CodexCliStructuredOutputError(
                    "Codex CLI не создал файл структурированного ответа."
                )
            raw_output = output_path.read_text(encoding="utf-8").strip()

        try:
            response = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CodexCliStructuredOutputError(
                f"Codex CLI вернул некорректный JSON: {exc}"
            ) from exc
        return _validate_response(response, expected_ids)


__all__ = [
    "CODEX_DEFAULT_ANALYSIS_MODEL",
    "CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT",
    "CODEX_DEFAULT_MODEL",
    "CODEX_DEFAULT_REASONING_EFFORT",
    "CODEX_DEFAULT_TIMEOUT_SECONDS",
    "CODEX_INSTALLER_COMMAND",
    "CODEX_INSTALLER_SCRIPT",
    "CODEX_INSTALLER_URL",
    "CODEX_REASONING_EFFORTS",
    "CodexCliError",
    "CodexCliAuthenticationError",
    "CodexCliInstallResult",
    "CodexCliStructuredClient",
    "CodexCliStructuredOutputError",
    "CodexCliTransientError",
    "CodexCliTranslator",
    "check_codex_cli",
    "install_or_update_codex_cli",
    "inspect_codex_cli",
    "login_codex_cli",
    "resolve_codex_cli",
]
