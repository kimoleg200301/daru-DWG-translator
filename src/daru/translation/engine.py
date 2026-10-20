"""Translation engine supporting Google, DeepL, ChatGPT and identity backends."""

import os
from typing import Any, Dict, List, Optional, Sequence

from ..config import (
    OPENAI_DEFAULT_MODEL,
    get_openai_model_profile,
    language_prompt_label,
    normalize_language_code,
    normalize_openai_base_url,
    normalize_translator_name,
)
from .codex_cli import (
    CODEX_DEFAULT_ANALYSIS_MODEL,
    CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
    CODEX_DEFAULT_MODEL,
    CODEX_DEFAULT_REASONING_EFFORT,
    CODEX_DEFAULT_TIMEOUT_SECONDS,
    CodexCliError,
    CodexCliTranslator,
)
from .batching import OPENAI_SAFE_TEXT, chunked as chunked, split_payload
from .batching import _group_split_jobs_into_batches as _group_split_jobs_into_batches
from .contracts import BatchProvider, TranslationContext, validate_translations
from .job import check_cancelled
from .options import TranslationOptions
from .providers.openai import OpenAIProvider
from .providers.codex import CodexProvider
from .providers.standard import DeepLProvider, GoogleHttpProvider
from .providers.standard import extract_google_translation
from .analysis import CodexAnalysisReview, CodexAnalysisSession
from .progress import report_progress

DIM_PLACEHOLDER = "__DXF_DIM__"
CODEX_ANALYSIS_MAX_ITEMS = 120
CODEX_ANALYSIS_MAX_CHARS = 12000


def _extract_google_free_translation(payload: Any) -> Optional[str]:
    return extract_google_translation(payload)


def restore_edge_whitespace(original: str, translated: str) -> str:
    if not original.strip():
        return original
    if not translated:
        return translated
    leading = len(original) - len(original.lstrip())
    trailing = len(original) - len(original.rstrip())
    prefix = original[:leading] if leading else ""
    suffix = original[len(original) - trailing :] if trailing else ""
    core = translated.strip() if (leading or trailing) else translated.strip() or translated
    return f"{prefix}{core}{suffix}"


def prepare_for_translation(text: str) -> str:
    if not text:
        return text
    prepared = text.replace("\\P", "\n").replace("<>", DIM_PLACEHOLDER)
    return prepared


def recover_after_translation(original: str, translated: str) -> str:
    if not translated:
        return translated
    restored = translated.replace(DIM_PLACEHOLDER, "<>")
    if "\\P" in original:
        restored = restored.replace("\r\n", "\n").replace("\n", "\\P")
    return restore_edge_whitespace(original, restored)


class TranslationEngine:
    def _provider_context(self) -> TranslationContext:
        return TranslationContext(
            originals=self._last_originals or (), document_text=self._drawing_context,
            label=self._context_label, entity_types=self._entity_types, glossary=self._glossary,
        )

    def _openai_provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            TranslationOptions.from_mapping(vars(self)), self._translator or {}, self._provider_context(),
        )

    def __init__(
        self,
        provider: str = "google",
        source_lang: str = "auto",
        target_lang: str = "ru",
        deepl_auth_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_project: Optional[str] = None,
        openai_temperature: float = 0.2,
        openai_reasoning_effort: Optional[str] = None,
        openai_verbosity: Optional[str] = None,
        openai_strict_mode: Optional[str] = None,
        openai_strict_value: Optional[float] = None,
        codex_cli_path: Optional[str] = None,
        codex_model: Optional[str] = None,
        codex_reasoning_effort: Optional[str] = None,
        codex_analysis_model: Optional[str] = None,
        codex_analysis_reasoning_effort: Optional[str] = None,
        codex_analysis_session: Optional[CodexAnalysisSession] = None,
        codex_timeout_seconds: int = CODEX_DEFAULT_TIMEOUT_SECONDS,
        system_prompt_template: Optional[str] = None,
    ):
        self.provider = normalize_translator_name(provider)
        self.source_lang = normalize_language_code(source_lang, "auto")
        self.target_lang = normalize_language_code(target_lang, "ru")
        self.deepl_auth_key = deepl_auth_key
        self.openai_api_key = (
            None
            if self.provider == "codex"
            else openai_api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.openai_model = openai_model or os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
        self.openai_base_url = normalize_openai_base_url(
            openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
        )
        project_candidate = openai_project or os.environ.get("OPENAI_PROJECT")
        self.openai_project = project_candidate.strip() if project_candidate else None
        self.openai_temperature = openai_temperature
        mode_candidate = (openai_strict_mode or os.environ.get("OPENAI_STRICT_MODE", "")).strip().lower()
        self.openai_strict_mode = mode_candidate if mode_candidate in {"verbosity", "effort"} else "verbosity"
        value_candidate = openai_strict_value
        if value_candidate is None:
            env_value = os.environ.get("OPENAI_STRICT_VALUE")
            if env_value is not None:
                try:
                    value_candidate = float(env_value)
                except ValueError:
                    value_candidate = None
        if value_candidate is None:
            value_candidate = 0.5
        self.openai_strict_value = max(0.0, min(1.0, value_candidate))
        legacy_level = self._strict_descriptor()
        profile = get_openai_model_profile(self.openai_model)
        supported_efforts = tuple(profile["reasoning_efforts"])
        effort_candidate = (
            openai_reasoning_effort
            or os.environ.get("OPENAI_REASONING_EFFORT")
            or (legacy_level if self.openai_strict_mode == "effort" else "")
        ).strip().lower()
        if effort_candidate not in supported_efforts:
            effort_candidate = str(profile["default_reasoning_effort"])
        self.openai_reasoning_effort = effort_candidate or None

        verbosity_candidate = (
            openai_verbosity
            or os.environ.get("OPENAI_VERBOSITY")
            or (legacy_level if self.openai_strict_mode == "verbosity" else "low")
        ).strip().lower()
        self.openai_verbosity = (
            verbosity_candidate if verbosity_candidate in {"low", "medium", "high"} else "low"
        )
        self.codex_cli_path = (codex_cli_path or "").strip()
        self.codex_model = (codex_model or CODEX_DEFAULT_MODEL).strip()
        codex_effort = (
            codex_reasoning_effort or CODEX_DEFAULT_REASONING_EFFORT
        ).strip().lower()
        self.codex_reasoning_effort = codex_effort or CODEX_DEFAULT_REASONING_EFFORT
        self.codex_analysis_model = (
            codex_analysis_model or CODEX_DEFAULT_ANALYSIS_MODEL
        ).strip()
        analysis_effort = (
            codex_analysis_reasoning_effort
            or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
        ).strip().lower()
        self.codex_analysis_reasoning_effort = (
            analysis_effort or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
        )
        self.codex_analysis_session = codex_analysis_session or CodexAnalysisSession()
        self.codex_timeout_seconds = max(10, int(codex_timeout_seconds))
        self.system_prompt_template = system_prompt_template
        self._translator = None
        self._backend = None
        self._translate_batch = None
        self._last_originals: Optional[Sequence[str]] = None
        self._drawing_context: Optional[str] = None
        self._context_label = "DRAWING"
        self._document_context_items: List[Dict[str, str]] = []
        self._document_analysis: Optional[str] = None
        self._document_analysis_attempted = False
        self._entity_types: Optional[Dict[str, str]] = None
        self._glossary: Dict[str, str] = {}
        self._init_translator()

    def _init_translator(self) -> None:
        tried = []

        if self.provider in ("google", "auto"):
            try:
                from deep_translator import GoogleTranslator  # type: ignore

                self._translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                self._backend = "deep-google"
                self._translate_batch = self._deep_translate_batch
                return
            except ImportError:
                tried.append("pip install deep-translator")
            except Exception as exc:
                tried.append(f"deep-translator error: {exc}")

        if self.provider in ("google", "auto", "google_free", "google-free"):
            self._backend = "google-free"
            self._translate_batch = self._google_free_translate_batch
            return

        if self.provider in ("deepl", "auto"):
            auth_key = self.deepl_auth_key or os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
            if auth_key:
                try:
                    import deepl  # type: ignore

                    self._translator = deepl.Translator(auth_key)
                    self._backend = "deepl"
                    self._translate_batch = self._deepl_translate_batch
                    return
                except ImportError:
                    tried.append("pip install deepl")
                except Exception as exc:
                    tried.append(f"deepl error: {exc}")
                    if self.provider != "auto":
                        raise
            elif self.provider == "deepl":
                raise RuntimeError("DEEPL_AUTH_KEY не задан")

        if self.provider == "codex":
            self._translator = CodexCliTranslator(
                cli_path=self.codex_cli_path,
                model=self.codex_model,
                reasoning_effort=self.codex_reasoning_effort,
                timeout_seconds=self.codex_timeout_seconds,
            )
            self._backend = "codex-cli"
            self._translate_batch = self._codex_translate_batch
            return

        if self.provider in ("chatgpt", "gpt", "openai"):
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY не задан")

            try:
                import openai  # type: ignore

                client = None
                mode = "new"
                try:
                    from openai import OpenAI  # type: ignore

                    client_kwargs = {"api_key": self.openai_api_key}
                    if self.openai_base_url:
                        client_kwargs["base_url"] = self.openai_base_url
                    if self.openai_project:
                        client_kwargs["project"] = self.openai_project
                    client = OpenAI(**client_kwargs)
                    if not hasattr(client, "responses"):
                        raise RuntimeError(
                            "Установленный OpenAI SDK не поддерживает Responses API. "
                            "Обновите пакет: pip install --upgrade \"openai>=2.41.1\""
                        )
                except (ImportError, AttributeError, TypeError):
                    mode = "legacy"
                except Exception as exc:
                    tried.append(f"openai error: {exc}")
                    if self.provider != "auto":
                        raise
                    mode = None

                if mode == "legacy":
                    profile = get_openai_model_profile(self.openai_model)
                    if profile["reasoning_efforts"]:
                        raise RuntimeError(
                            "Для моделей GPT-5 требуется openai>=2.41.1 с поддержкой Responses API. "
                            "Обновите пакет: pip install --upgrade \"openai>=2.41.1\""
                        )
                    try:
                        openai.api_key = self.openai_api_key
                        if self.openai_base_url:
                            openai.api_base = self.openai_base_url
                    except Exception as exc:
                        tried.append(f"openai error: {exc}")
                        if self.provider != "auto":
                            raise
                        mode = None

                if mode:
                    self._translator = {
                        "client": client,
                        "module": openai,
                        "model": self.openai_model,
                        "batch_size": 1 if mode == "legacy" else 16,
                        "mode": mode,
                    }
                    self._backend = "chatgpt"
                    self._translate_batch = self._chatgpt_translate_batch
                    return
            except ImportError:
                tried.append("pip install openai")

        if self.provider in ("noop", "identity"):
            self._backend = "identity"
            self._translate_batch = self._identity_translate_batch
            return

        if tried:
            raise RuntimeError("Переводчик недоступен: " + "; ".join(tried))
        raise RuntimeError("Не удалось инициализировать переводчик")

    def backend_name(self) -> str:
        return self._backend or "unknown"

    def set_drawing_context(
        self,
        all_texts: Sequence[str],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """Provide full drawing context for better AI translation.

        Args:
            all_texts: All unique texts from the drawing (for summary).
            entity_types: Optional mapping text -> DXF entity type (TEXT, MTEXT, TABLE, etc.).
        """
        self.set_document_context(
            all_texts,
            context_label="DRAWING",
            entity_types=entity_types,
        )

    def set_document_context(
        self,
        all_texts: Sequence[str],
        *,
        context_label: str = "DOCUMENT",
        entity_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """Provide shared context for consistent document translation."""

        self._entity_types = dict(entity_types) if entity_types else None
        self._glossary = {}
        self._context_label = context_label.strip().upper() or "DOCUMENT"
        self._document_analysis = None
        self._document_analysis_attempted = False
        self._document_context_items = self._sample_document_context(all_texts)
        self._drawing_context = "\n".join(
            (
                f"[{item['type']}] {item['text']}"
                if item.get("type")
                else item["text"]
            )
            for item in self._document_context_items
        )

    def _sample_document_context(
        self,
        all_texts: Sequence[str],
    ) -> List[Dict[str, str]]:
        values = list(dict.fromkeys(str(text).strip() for text in all_texts))
        values = [text for text in values if text]
        if not values:
            return []

        if len(values) <= CODEX_ANALYSIS_MAX_ITEMS:
            selected_indices = list(range(len(values)))
        else:
            last_index = len(values) - 1
            selected_indices = sorted(
                {
                    round(position * last_index / (CODEX_ANALYSIS_MAX_ITEMS - 1))
                    for position in range(CODEX_ANALYSIS_MAX_ITEMS)
                }
            )

        items: List[Dict[str, str]] = []
        used_chars = 0
        for index in selected_indices:
            source_text = values[index]
            remaining = CODEX_ANALYSIS_MAX_CHARS - used_chars
            if remaining <= 0:
                break
            sampled_text = source_text[: min(500, remaining)]
            if not sampled_text:
                continue
            item = {
                "id": f"context-{index}",
                "text": sampled_text,
            }
            if self._entity_types:
                entity_type = self._entity_types.get(source_text)
                if entity_type:
                    item["type"] = entity_type
            items.append(item)
            used_chars += len(sampled_text)
        return items

    def _ensure_codex_document_analysis(self) -> None:
        if self._document_analysis_attempted or self._backend != "codex-cli":
            return
        self._document_analysis_attempted = True
        if not self._document_context_items:
            return

        approved = self.codex_analysis_session.approved_text
        if approved:
            self._document_analysis = approved
            return

        translator: CodexCliTranslator = self._translator
        analyze_document = getattr(translator, "analyze_document", None)
        if not callable(analyze_document):
            return
        source_label = language_prompt_label(
            self.source_lang,
            "auto",
            allow_auto=True,
        )
        target_label = language_prompt_label(self.target_lang, "ru")
        instructions = (
            "Act as a senior technical translation editor. Analyze the representative "
            f"samples from one {self._context_label.lower()} before translation from "
            f"{source_label} to {target_label}. Infer the subject, purpose, audience, "
            "register, recurring abbreviations, and terminology that materially affect "
            "translation choices. Return a compact document summary, no more than six "
            "actionable translation guidance points, and no more than twenty high-value "
            "source-to-target terminology pairs. Write the document summary and all "
            f"translation guidance in the target language ({target_label}). Do not "
            "translate every sample and do not include generic advice."
        )
        used_fallback = False
        warning = ""
        report_progress("analysis")
        try:
            analysis = analyze_document(
                self._document_context_items,
                instructions=instructions,
                model=self.codex_analysis_model,
                reasoning_effort=self.codex_analysis_reasoning_effort,
            )
            analysis_text = self._format_document_analysis(analysis)
        except CodexCliError as exc:
            used_fallback = True
            warning = (
                "Codex CLI не смог сформировать структурированный анализ. "
                f"Показан исходный контекст документа: {exc}"
            )
            analysis_text = self._drawing_context
        if not analysis_text:
            used_fallback = True
            warning = (
                warning
                or "Codex CLI вернул пустой анализ. Показан исходный контекст документа."
            )
            analysis_text = self._drawing_context
        if not analysis_text:
            return
        self._document_analysis = self.codex_analysis_session.resolve(
            CodexAnalysisReview(
                text=analysis_text,
                model=self.codex_analysis_model,
                reasoning_effort=self.codex_analysis_reasoning_effort,
                context_label=self._context_label,
                used_fallback=used_fallback,
                warning=warning,
            )
        )
        report_progress("translate")

    @staticmethod
    def _format_document_analysis(analysis: Dict[str, Any]) -> Optional[str]:
        summary = str(analysis.get("document_summary") or "").strip()
        guidance = analysis.get("translation_guidance") or []
        terminology = analysis.get("terminology") or []
        sections: List[str] = []
        if summary:
            sections.append(f"Document summary: {summary}")
        clean_guidance = [
            str(item).strip() for item in guidance if str(item).strip()
        ][:8]
        if clean_guidance:
            sections.append(
                "Translation guidance:\n"
                + "\n".join(f"- {item}" for item in clean_guidance)
            )
        term_lines = []
        for item in terminology[:24]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            if source and target:
                term_lines.append(f'"{source}" -> "{target}"')
        if term_lines:
            sections.append("Preferred terminology:\n" + "\n".join(term_lines))
        return "\n\n".join(sections) or None

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        check_cancelled()
        if not texts:
            return []
        result = list(texts)
        prepared = [prepare_for_translation(text) for text in texts]
        indices = [index for index, value in enumerate(prepared) if value.strip()]
        if not indices:
            return result
        values = [prepared[index] for index in indices]
        if self._backend in {"chatgpt", "codex-cli"}:
            self._last_originals = [texts[index] for index in indices]
        try:
            translated = validate_translations(values, self._translate_batch(values), self.backend_name())
            check_cancelled()
            for index, value in zip(indices, translated):
                result[index] = recover_after_translation(texts[index], value)
            return result
        finally:
            self._last_originals = None

    # Backends -------------------------------------------------------------

    def _deep_translate_batch(self, texts: Sequence[str]) -> List[str]:
        check_cancelled()
        try:
            batch = getattr(self._translator, "translate_batch", None)
            values = list(batch(list(texts))) if callable(batch) else [self._translator.translate(t) for t in texts]
            return validate_translations(texts, values, "Google")
        except Exception:
            check_cancelled()
            return self._google_free_translate_batch(texts)

    def _deepl_translate_batch(self, texts: Sequence[str]) -> List[str]:
        return DeepLProvider(self._translator, self.source_lang, self.target_lang).translate(texts)

    def _codex_translate_batch(self, texts: Sequence[str]) -> List[str]:
        self._ensure_codex_document_analysis()
        provider: BatchProvider = CodexProvider(self._translator, self._provider_context(), self._codex_system_content())
        return provider.translate(texts)

    def _codex_system_content(self) -> str:
        source_label = language_prompt_label(
            self.source_lang,
            "auto",
            allow_auto=True,
        )
        target_label = language_prompt_label(self.target_lang, "ru")

        if self.system_prompt_template:
            content = self.system_prompt_template.replace(
                "{source_lang}", source_label
            ).replace("{target_lang}", target_label)
        else:
            content = (
                "You are a professional technical translator of engineering documents. "
                f"Translate every provided value from {source_label} to {target_label}. "
                "All values belong to the same document; keep terminology consistent. "
                "Preserve numbers, identifiers, placeholders such as '__DXF_DIM__', "
                "format markers, tabs, line breaks, and DXF control sequences."
            )
        if self._document_analysis:
            content += (
                f"\n\n[{self._context_label} PRE-TRANSLATION ANALYSIS]:\n"
                + self._document_analysis
            )
        elif self._drawing_context:
            content += (
                f"\n\n[{self._context_label} CONTEXT - all texts for reference]:\n"
                + self._drawing_context
            )
        if self._glossary:
            glossary_lines = [
                f'"{source}" -> "{target}"'
                for source, target in list(self._glossary.items())[:100]
            ]
            content += (
                "\n\n[GLOSSARY - use these translations consistently]:\n"
                + "\n".join(glossary_lines)
            )
        return content

    def _chatgpt_translate_batch(self, texts: Sequence[str]) -> List[str]:
        return self._openai_provider().translate(texts)

    def _openai_generation_kwargs(self, model: Optional[str], *, for_responses: bool = False) -> Dict[str, Any]:
        return self._openai_provider()._openai_generation_kwargs(model, for_responses=for_responses)

    def _should_retry_openai(self, exc: Exception) -> bool:
        return self._openai_provider()._should_retry_openai(exc)

    def _split_openai_payload(self, text: str, limit: int = OPENAI_SAFE_TEXT) -> List[str]:
        return split_payload(text, limit)

    def _as_responses_input(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._openai_provider()._as_responses_input(messages)

    def _openai_responses_text_config(self, model: Optional[str]) -> Dict[str, Any]:
        return self._openai_provider()._openai_responses_text_config(model)

    def _openai_responses_reasoning(self, model: Optional[str]) -> Optional[Dict[str, str]]:
        return self._openai_provider()._openai_responses_reasoning(model)

    def _openai_metadata(self) -> Dict[str, str]:
        return self._openai_provider()._openai_metadata()

    def _log_openai_response(self, response_obj: object) -> None:
        return self._openai_provider()._log_openai_response(response_obj)

    def _reasoning_effort_value(self, model: Optional[str] = None) -> Optional[str]:
        return self._openai_provider()._reasoning_effort_value(model)

    def _response_text(self, response_obj: object) -> str:
        return self._openai_provider()._response_text(response_obj)

    def _openai_reasoning_note(self, model: Optional[str]) -> str:
        return self._openai_provider()._openai_reasoning_note(model)

    def _strict_descriptor(self) -> str:
        value = self.openai_strict_value
        if value <= 0.34:
            return "low"
        if value <= 0.67:
            return "medium"
        return "high"

    def _google_free_translate_batch(self, texts: Sequence[str]) -> List[str]:
        return GoogleHttpProvider(self.source_lang, self.target_lang).translate(texts)

    def _identity_translate_batch(self, texts: Sequence[str]) -> List[str]:
        return list(texts)


def auto_translate(
    texts: Sequence[str],
    provider: str = "google",
    source_lang: str = "auto",
    target_lang: str = "ru",
    deepl_auth_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_project: Optional[str] = None,
    openai_temperature: float = 0.2,
    openai_reasoning_effort: Optional[str] = None,
    openai_verbosity: Optional[str] = None,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
    codex_cli_path: Optional[str] = None,
    codex_model: Optional[str] = None,
    codex_reasoning_effort: Optional[str] = None,
    codex_analysis_model: Optional[str] = None,
    codex_analysis_reasoning_effort: Optional[str] = None,
    codex_timeout_seconds: int = CODEX_DEFAULT_TIMEOUT_SECONDS,
) -> List[str]:
    engine = TranslationEngine(
        provider=provider,
        source_lang=source_lang,
        target_lang=target_lang,
        deepl_auth_key=deepl_auth_key,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_project=openai_project,
        openai_temperature=openai_temperature,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_verbosity=openai_verbosity,
        openai_strict_mode=openai_strict_mode,
        openai_strict_value=openai_strict_value,
        codex_cli_path=codex_cli_path,
        codex_model=codex_model,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_analysis_model=codex_analysis_model,
        codex_analysis_reasoning_effort=codex_analysis_reasoning_effort,
        codex_timeout_seconds=codex_timeout_seconds,
    )
    return engine.translate_many(texts)
