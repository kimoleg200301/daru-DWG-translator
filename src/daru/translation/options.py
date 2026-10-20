"""Shared translation options and credential-free cache identities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from ..config import (
    OPENAI_DEFAULT_MODEL,
    normalize_language_code,
    normalize_openai_base_url,
    normalize_translator_name,
)


@dataclass(frozen=True)
class TranslationOptions:
    provider: str = "google"
    source_lang: str = "en"
    target_lang: str = "ru"
    deepl_auth_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str | None = None
    openai_base_url: str | None = None
    openai_project: str | None = None
    openai_temperature: float = 0.2
    openai_reasoning_effort: str | None = None
    openai_verbosity: str | None = None
    openai_strict_mode: str | None = None
    openai_strict_value: float | None = None
    codex_cli_path: str | None = None
    codex_model: str | None = None
    codex_reasoning_effort: str | None = None
    codex_analysis_model: str | None = None
    codex_analysis_reasoning_effort: str | None = None
    codex_analysis_session: Any = field(default=None, repr=False, compare=False)
    codex_timeout_seconds: int = 300
    system_prompt_template: str | None = None

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], **overrides: Any
    ) -> TranslationOptions:
        aliases = {
            "translator_name": "provider",
            "deepl_key": "deepl_auth_key",
            "openai_key": "openai_api_key",
        }
        known = {item.name for item in fields(cls)}
        selected = {
            aliases.get(key, key): value
            for key, value in values.items()
            if aliases.get(key, key) in known
        }
        selected.update(overrides)
        selected["provider"] = normalize_translator_name(
            selected.get("provider", "google")
        )
        selected["source_lang"] = normalize_language_code(
            selected.get("source_lang"), "en"
        )
        selected["target_lang"] = normalize_language_code(
            selected.get("target_lang"), "ru"
        )
        return cls(**selected)

    def engine_kwargs(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def fingerprint(
        self, job_type: str, *, glossary: Mapping[str, str] | None = None
    ) -> str:
        """Invalidate automatic results when translation semantics change."""
        values: dict[str, Any] = {
            "recipe": 1,
            "job_type": job_type,
            "provider": self.provider,
            "prompt": self.system_prompt_template,
        }
        if glossary:
            values["glossary"] = dict(glossary)
        if self.provider == "chatgpt":
            values.update(
                model=self.openai_model
                or os.environ.get("OPENAI_MODEL")
                or OPENAI_DEFAULT_MODEL,
                base_url=normalize_openai_base_url(
                    self.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
                ),
                temperature=self.openai_temperature,
                effort=self.openai_reasoning_effort
                or os.environ.get("OPENAI_REASONING_EFFORT"),
                verbosity=self.openai_verbosity or os.environ.get("OPENAI_VERBOSITY"),
                strict_mode=self.openai_strict_mode
                or os.environ.get("OPENAI_STRICT_MODE"),
                strict_value=self.openai_strict_value
                if self.openai_strict_value is not None
                else os.environ.get("OPENAI_STRICT_VALUE"),
            )
        elif self.provider == "codex":
            from .codex_cli import (
                CODEX_DEFAULT_MODEL,
                CODEX_DEFAULT_REASONING_EFFORT,
                CODEX_DEFAULT_ANALYSIS_MODEL,
                CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
            )

            values.update(
                model=self.codex_model or CODEX_DEFAULT_MODEL,
                effort=self.codex_reasoning_effort or CODEX_DEFAULT_REASONING_EFFORT,
                analysis_model=self.codex_analysis_model
                or CODEX_DEFAULT_ANALYSIS_MODEL,
                analysis_effort=self.codex_analysis_reasoning_effort
                or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
            )
        encoded = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cache_identity(self, document_hash: str, job_type: str) -> dict[str, str]:
        return {
            "document_sha256": document_hash,
            "job_type": job_type,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "settings_fingerprint": self.fingerprint(job_type),
        }
