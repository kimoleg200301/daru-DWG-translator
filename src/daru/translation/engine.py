"""Translation engine supporting Google, DeepL, ChatGPT and identity backends."""

import json
import os
import re
import time
from http.client import RemoteDisconnected
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import (
    OPENAI_DEFAULT_MODEL,
    get_openai_model_profile,
    normalize_openai_base_url,
    normalize_translator_name,
)
from .codex_cli import (
    CODEX_DEFAULT_ANALYSIS_MODEL,
    CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
    CODEX_DEFAULT_MODEL,
    CODEX_DEFAULT_REASONING_EFFORT,
    CODEX_DEFAULT_TIMEOUT_SECONDS,
    CODEX_REASONING_EFFORTS,
    CodexCliError,
    CodexCliTranslator,
)
from .analysis import CodexAnalysisReview, CodexAnalysisSession

try:
    from requests.exceptions import ReadTimeout
except Exception:
    class ReadTimeout(Exception):  # type: ignore[no-redef]
        pass

DIM_PLACEHOLDER = "__DXF_DIM__"
OPENAI_SAFE_TEXT = 100000
OPENAI_RETRY_ATTEMPTS = 5
OPENAI_RETRY_DELAY = 3.0
CODEX_MAX_ITEMS = 50
CODEX_MAX_CHARS = 40000
CODEX_ANALYSIS_MAX_ITEMS = 120
CODEX_ANALYSIS_MAX_CHARS = 12000


def _extract_google_free_translation(payload: Any) -> Optional[str]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
        return None
    parts: List[str] = []
    for segment in payload[0]:
        if isinstance(segment, list) and segment and isinstance(segment[0], str):
            parts.append(segment[0])
    return "".join(parts) or None


def chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _group_split_jobs_into_batches(
    split_jobs: List[Tuple[str, int, int, str]],
    part_counts: Dict[int, int],
    batch_size: int,
) -> List[List[Tuple[str, int, int, str]]]:
    """Group split_jobs into batches ensuring all parts of one text stay together."""
    by_orig: Dict[int, List[Tuple[str, int, int, str]]] = {}
    seen_order: List[int] = []
    for job in split_jobs:
        orig_idx = job[1]
        if orig_idx not in by_orig:
            seen_order.append(orig_idx)
        by_orig.setdefault(orig_idx, []).append(job)

    batches: List[List[Tuple[str, int, int, str]]] = []
    current: List[Tuple[str, int, int, str]] = []
    for orig_idx in seen_order:
        group = by_orig[orig_idx]
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def restore_edge_whitespace(original: str, translated: str) -> str:
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
        self.source_lang = source_lang or "auto"
        self.target_lang = target_lang or "ru"
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
            analysis_effort
            if analysis_effort in CODEX_REASONING_EFFORTS
            else CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
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

            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            if self.openai_base_url:
                os.environ["OPENAI_BASE_URL"] = self.openai_base_url
            if self.openai_project:
                os.environ["OPENAI_PROJECT"] = self.openai_project
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
        source_label = (self.source_lang or "auto").strip() or "auto"
        target_label = (self.target_lang or "ru").strip() or "ru"
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
        if not texts:
            return []
        prepared = [prepare_for_translation(t) for t in texts]
        reset_required = False
        if self._backend in {"chatgpt", "codex-cli"}:
            self._last_originals = list(texts)
            reset_required = True
        try:
            translated = self._translate_batch(prepared)
        finally:
            if reset_required:
                self._last_originals = None
        result: List[str] = []
        for original, prepared_text, translated_text in zip(texts, prepared, translated):
            if not translated_text:
                translated_text = original
            result.append(recover_after_translation(original, translated_text))
        return result

    # Backends -------------------------------------------------------------

    def _deep_translate_batch(self, texts: Sequence[str]) -> List[str]:
        try:
            return list(self._translator.translate_batch(list(texts)))  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return [self._translator.translate(t) for t in texts]  # type: ignore[attr-defined]
            except Exception:
                return self._google_free_translate_batch(texts)
        except Exception:
            return self._google_free_translate_batch(texts)

    def _deepl_translate_batch(self, texts: Sequence[str]) -> List[str]:
        translator = self._translator
        results: List[str] = []
        source_lang = (self.source_lang or "").strip()
        source = None if not source_lang or source_lang.lower() == "auto" else source_lang.upper()
        target_lang = (self.target_lang or "ru").strip() or "RU"
        target = target_lang.upper()

        for chunk in chunked(texts, 40):
            try:
                translated = translator.translate_text(list(chunk), target_lang=target, source_lang=source)
            except Exception:
                chunk_results: List[str] = []
                for piece in chunk:
                    try:
                        single = translator.translate_text(piece, target_lang=target, source_lang=source)
                        text = getattr(single, "text", None)
                        chunk_results.append(text if text is not None else piece)
                    except Exception:
                        chunk_results.append(piece)
                results.extend(chunk_results)
                continue

            if not isinstance(translated, list):
                translated = [translated]

            chunk_results = []
            for original, item in zip_longest(chunk, translated):
                if item is None:
                    chunk_results.append(original)
                    continue
                text = getattr(item, "text", None)
                chunk_results.append(text if text is not None else original)
            results.extend(chunk_results)

        return results

    def _codex_translate_batch(self, texts: Sequence[str]) -> List[str]:
        originals = list(self._last_originals or texts)
        results = list(texts)
        split_jobs: List[Tuple[str, int, int, str]] = []
        part_counts: Dict[int, int] = {}
        part_sources: Dict[Tuple[int, int], str] = {}

        for original_index, text in enumerate(texts):
            parts = self._split_openai_payload(text, CODEX_MAX_CHARS)
            part_counts[original_index] = len(parts)
            for part_index, part in enumerate(parts):
                job_id = f"{original_index}:{part_index}"
                split_jobs.append((job_id, original_index, part_index, part))
                part_sources[(original_index, part_index)] = part

        batches: List[List[Tuple[str, int, int, str]]] = []
        current: List[Tuple[str, int, int, str]] = []
        current_chars = 0
        for job in split_jobs:
            job_chars = len(job[3])
            if current and (
                len(current) >= CODEX_MAX_ITEMS
                or current_chars + job_chars > CODEX_MAX_CHARS
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(job)
            current_chars += job_chars
        if current:
            batches.append(current)

        assembled_parts: Dict[int, Dict[int, str]] = {}
        self._ensure_codex_document_analysis()
        instructions = self._codex_system_content()
        translator: CodexCliTranslator = self._translator
        for batch in batches:
            items: List[Dict[str, str]] = []
            for job_id, original_index, _part_index, text in batch:
                item = {"id": job_id, "text": text}
                if self._entity_types:
                    original = (
                        originals[original_index]
                        if original_index < len(originals)
                        else text
                    )
                    entity_type = self._entity_types.get(original)
                    if entity_type:
                        item["type"] = entity_type
                items.append(item)

            translated_values = translator.translate(items, instructions=instructions)
            for job, translated in zip(batch, translated_values):
                _job_id, original_index, part_index, source_part = job
                assembled_parts.setdefault(original_index, {})[part_index] = translated
                if translated and translated != source_part:
                    original = (
                        originals[original_index]
                        if original_index < len(originals)
                        else source_part
                    )
                    self._glossary[original.strip()] = translated.strip()

        for original_index, source_text in enumerate(texts):
            total_parts = part_counts.get(original_index, 0)
            translated_parts = assembled_parts.get(original_index, {})
            combined = [
                translated_parts.get(
                    part_index,
                    part_sources.get((original_index, part_index), ""),
                )
                for part_index in range(total_parts)
            ]
            translated_text = "".join(combined)
            results[original_index] = translated_text if translated_text else source_text
        return results

    def _codex_system_content(self) -> str:
        source_label = (self.source_lang or "auto").strip()
        target_label = (self.target_lang or "ru").strip() or "ru"
        if source_label.lower() == "auto":
            source_label = "auto-detected"

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
        originals = list(self._last_originals or [])
        translator = self._translator or {}
        mode = translator.get("mode", "new")
        client = translator.get("client")
        module = translator.get("module")
        model = translator.get("model")
        batch_size = translator.get("batch_size", 12)
        ADAPTIVE_SINGLE_REQUEST_CHARS = 50000
        total_chars = sum(len(t) for t in texts)
        if total_chars <= ADAPTIVE_SINGLE_REQUEST_CHARS:
            batch_size = max(batch_size, len(texts))

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = self.openai_base_url

        chat_completion_create: Optional[Callable[..., object]] = None
        if module is not None:
            chat_namespace = getattr(module, "chat", None)
            completions_namespace = getattr(chat_namespace, "completions", None) if chat_namespace else None
            create_attr = getattr(completions_namespace, "create", None) if completions_namespace else None
            if callable(create_attr):
                chat_completion_create = create_attr

        openai_major_version: Optional[int] = None
        if module:
            version_value = getattr(module, "__version__", None)
            if isinstance(version_value, str):
                major_part = version_value.split(".", 1)[0]
                if major_part.isdigit():
                    openai_major_version = int(major_part)

        fallback_client = None
        if mode == "legacy":
            fallback_client = client
            if fallback_client is None and module and hasattr(module, "OpenAI"):
                client_kwargs = {"api_key": self.openai_api_key}
                if self.openai_base_url:
                    client_kwargs["base_url"] = self.openai_base_url
                if self.openai_project:
                    client_kwargs["project"] = self.openai_project
                try:
                    fallback_client = module.OpenAI(**client_kwargs)
                except TypeError:
                    fallback_client = None
                    try:
                        client_kwargs.pop("base_url", None)
                        client_kwargs.pop("project", None)
                        fallback_client = module.OpenAI(**client_kwargs)
                        if self.openai_base_url and fallback_client is not None:
                            with_options = getattr(fallback_client, "with_options", None)
                            if callable(with_options):
                                fallback_client = with_options(base_url=self.openai_base_url)
                            else:
                                try:
                                    setattr(fallback_client, "base_url", self.openai_base_url)
                                except Exception:
                                    pass
                    except Exception:
                        fallback_client = None
                except Exception:
                    fallback_client = None

            if fallback_client is not None:
                client = fallback_client
                mode = "new"
            elif not chat_completion_create and openai_major_version and openai_major_version >= 1:
                raise RuntimeError(
                    "Установлен openai>=1.0.0, но не удалось инициализировать новый клиент. "
                    "Переустановите пакет 'openai' или закрепите версию <1.0."
                )

        results: List[str] = list(texts)
        if mode == "legacy":
            if (not chat_completion_create) and (not module or not model):
                return results
        else:
            if not client or not model:
                return results

        def should_use_ai(src: str) -> bool:
            stripped = (src or "").strip()
            if len(stripped) < 3:
                return False
            if self.source_lang == 'en':
                if not re.search(r"[A-Za-z]", stripped):
                    return False
            if re.fullmatch(r"[A-Za-z]\.?", stripped):
                return False
            if re.fullmatch(r"[A-Za-z]\d+", stripped):
                return False
            if stripped.isupper() and len(stripped) <= 3 and " " not in stripped:
                return False
            if re.fullmatch(r"[-+]?\d+[\d\s./-]*", stripped):
                return False
            return True

        def extract_message(completion_obj) -> str:
            if completion_obj is None:
                return ""

            choices_obj = getattr(completion_obj, "choices", None)
            if choices_obj:
                first_choice = choices_obj[0]
                message_obj = getattr(first_choice, "message", None)
                if message_obj is not None:
                    content_attr = getattr(message_obj, "content", None)
                    if isinstance(content_attr, str) and content_attr:
                        return content_attr
                    if isinstance(message_obj, dict):
                        content = message_obj.get("content")
                        if isinstance(content, str) and content:
                            return content
                text_attr = getattr(first_choice, "text", None)
                if isinstance(text_attr, str) and text_attr:
                    return text_attr
                if isinstance(first_choice, dict):
                    message_dict = first_choice.get("message", {})
                    if isinstance(message_dict, dict):
                        content = message_dict.get("content")
                        if isinstance(content, str) and content:
                            return content
                    text_val = first_choice.get("text")
                    if isinstance(text_val, str) and text_val:
                        return text_val

            if isinstance(completion_obj, dict):
                choices_dict = completion_obj.get("choices", [])
                if isinstance(choices_dict, list) and choices_dict:
                    first_choice = choices_dict[0]
                    if isinstance(first_choice, dict):
                        message_dict = first_choice.get("message", {})
                        if isinstance(message_dict, dict):
                            content = message_dict.get("content")
                            if isinstance(content, str) and content:
                                return content
                        text_val = first_choice.get("text")
                        if isinstance(text_val, str) and text_val:
                            return text_val

            model_dump = getattr(completion_obj, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump()
                except Exception:
                    dumped = None
                if isinstance(dumped, dict):
                    return extract_message(dumped)

            return ""

        candidate_indices = []
        for idx, prepared in enumerate(texts):
            original = originals[idx] if idx < len(originals) else prepared
            if should_use_ai(original):
                candidate_indices.append(idx)
            else:
                results[idx] = prepared

        if not candidate_indices:
            return results

        split_jobs: List[Tuple[str, int, int, str]] = []
        part_counts: Dict[int, int] = {}
        part_sources: Dict[Tuple[int, int], str] = {}
        for idx in candidate_indices:
            segments = self._split_openai_payload(texts[idx])
            part_counts[idx] = len(segments)
            for part_idx, segment in enumerate(segments):
                job_id = f"{idx}:{part_idx}"
                split_jobs.append((job_id, idx, part_idx, segment))
                part_sources[(idx, part_idx)] = segment

        if not split_jobs:
            return results

        reasoning_note = self._openai_reasoning_note(model)
        def _prompt_lang(value: Optional[str], default: str, *, allow_auto: bool) -> str:
            cleaned = (value or "").strip()
            if not cleaned:
                return default
            if cleaned.lower() == "auto":
                return "auto-detected" if allow_auto else default
            return cleaned

        source_label = _prompt_lang(self.source_lang, "auto-detected", allow_auto=True)
        target_label = _prompt_lang(self.target_lang, "ru", allow_auto=False)
        if self.system_prompt_template:
            system_content = self.system_prompt_template
            system_content = system_content.replace("{source_lang}", source_label).replace("{target_lang}", target_label)
        else:
            system_content = (
                "You are a professional technical translator of elevator drawing manuals. "
                f"Translate the provided values from {source_label} to {target_label}. "
                "All texts belong to the SAME technical drawing — maintain consistent "
                "terminology across all items. Preserve numbers, "
                "placeholders like '__DXF_DIM__', and DXF control sequences such as \"\\P\". "
                "Each item may include a \"type\" field indicating the DXF entity type "
                "(TEXT, MTEXT, TABLE, DIMENSION, etc.) — use it to infer context. "
                "Respond with strict JSON: "
                "{\"translations\": [{\"id\": \"<id>\", \"text\": \"<translated>\"}, ...]}"
            )
        if self._drawing_context:
            system_content += (
                f"\n\n[{self._context_label} CONTEXT — all texts for reference]:\n"
                + self._drawing_context
            )
        if self._glossary:
            glossary_lines = [f'"{k}" → "{v}"' for k, v in list(self._glossary.items())[:100]]
            system_content += (
                "\n\n[GLOSSARY — previously translated terms, use consistently]:\n"
                + "\n".join(glossary_lines)
            )
        if reasoning_note:
            system_content = f"{system_content}\n\n{reasoning_note}"

        assembled_parts: Dict[int, Dict[int, str]] = {}

        def _build_item(job_id: str, orig_idx: int, text: str) -> Dict[str, str]:
            item: Dict[str, str] = {"id": job_id, "text": text}
            if self._entity_types:
                original = originals[orig_idx] if orig_idx < len(originals) else text
                etype = self._entity_types.get(original)
                if etype:
                    item["type"] = etype
            return item

        if mode == "legacy":
            for job_id, orig_idx, part_idx, part_text in split_jobs:
                single_payload = json.dumps(
                    {"items": [_build_item(job_id, orig_idx, part_text)]},
                    ensure_ascii=False,
                )
                single_messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": single_payload},
                ]
                base_kwargs = {
                    "model": model,
                    "messages": single_messages,
                    "store": True,
                }
                base_kwargs.update(self._openai_generation_kwargs(model))

                completion = None
                for attempt in range(OPENAI_RETRY_ATTEMPTS):
                    try:
                        if chat_completion_create:
                            modern_kwargs = dict(base_kwargs)
                            modern_kwargs["response_format"] = {"type": "json_object"}
                            completion = chat_completion_create(**modern_kwargs)
                        elif module is not None:
                            completion = module.ChatCompletion.create(**base_kwargs)
                        else:
                            raise RuntimeError("OpenAI legacy клиент недоступен")
                        break
                    except Exception as exc:
                        if (attempt + 1 == OPENAI_RETRY_ATTEMPTS) or not self._should_retry_openai(exc):
                            raise RuntimeError(f"ChatGPT translation failed: {exc}") from exc
                        time.sleep(OPENAI_RETRY_DELAY)
                if completion is None:
                    continue
                self._log_openai_response(completion)

                message = extract_message(completion) or "{}"
                translated_text = part_text
                try:
                    data = json.loads(message or "{}")
                    for item in data.get("translations", []):
                        if isinstance(item, dict) and item.get("id") == job_id:
                            candidate = item.get("text")
                            if isinstance(candidate, str) and candidate:
                                translated_text = candidate
                                break
                except json.JSONDecodeError:
                    cleaned = message.strip()
                    if cleaned:
                        translated_text = cleaned
                assembled_parts.setdefault(orig_idx, {})[part_idx] = translated_text
                if translated_text != part_text:
                    original = originals[orig_idx] if orig_idx < len(originals) else part_text
                    self._glossary[original.strip()] = translated_text.strip()
        else:
            split_jobs_grouped = _group_split_jobs_into_batches(split_jobs, part_counts, batch_size)
            for chunk in split_jobs_grouped:
                payload = [
                    _build_item(job_id, orig_idx, text_part)
                    for job_id, orig_idx, _, text_part in chunk
                ]
                user_content = json.dumps({"items": payload}, ensure_ascii=False)
                messages_payload = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                responses_input = self._as_responses_input(messages_payload)
                text_config = self._openai_responses_text_config(model)
                reasoning_config = self._openai_responses_reasoning(model)

                completion = None
                for attempt in range(OPENAI_RETRY_ATTEMPTS):
                    try:
                        modern_kwargs: Dict[str, Any] = {
                            "model": model,
                            "input": responses_input,
                            "store": True,
                        }
                        if text_config:
                            modern_kwargs["text"] = text_config
                        if reasoning_config:
                            modern_kwargs["reasoning"] = reasoning_config
                        metadata = self._openai_metadata()
                        if metadata:
                            modern_kwargs["metadata"] = metadata
                        modern_kwargs.update(self._openai_generation_kwargs(model, for_responses=True))
                        completion = client.responses.create(**modern_kwargs)
                        break
                    except Exception as exc:
                        if (attempt + 1 == OPENAI_RETRY_ATTEMPTS) or not self._should_retry_openai(exc):
                            raise RuntimeError(f"ChatGPT translation failed: {exc}") from exc
                        time.sleep(OPENAI_RETRY_DELAY)
                if completion is None:
                    continue
                self._log_openai_response(completion)

                message = self._response_text(completion)
                if not message:
                    raise RuntimeError("ChatGPT не вернул данных ответа")
                try:
                    data = json.loads(message or "{}")
                except json.JSONDecodeError as exc:
                    raise RuntimeError("ChatGPT не вернул корректный JSON") from exc

                mapping = {
                    item.get("id"): item.get("text")
                    for item in data.get("translations", [])
                    if isinstance(item, dict)
                }

                for job_id, orig_idx, part_idx, part_text in chunk:
                    translated = mapping.get(job_id)
                    assembled_parts.setdefault(orig_idx, {})[part_idx] = translated if translated else part_text
                    if translated and translated != part_text:
                        original = originals[orig_idx] if orig_idx < len(originals) else part_text
                        self._glossary[original.strip()] = translated.strip()

        for idx in candidate_indices:
            total_parts = part_counts.get(idx, 0)
            part_dict = assembled_parts.get(idx, {})
            if total_parts <= 1:
                translated_value = part_dict.get(0)
                if translated_value is None:
                    translated_value = part_sources.get((idx, 0), texts[idx])
                results[idx] = translated_value if translated_value else texts[idx]
                continue

            combined: List[str] = []
            for part_idx in range(total_parts):
                piece = part_dict.get(part_idx)
                if piece is None:
                    piece = part_sources.get((idx, part_idx), "")
                combined.append(piece)
            merged = "".join(combined)
            results[idx] = merged if merged else texts[idx]

        return results

    def _openai_generation_kwargs(self, model: Optional[str], *, for_responses: bool = False) -> Dict[str, Any]:
        profile = get_openai_model_profile(model or "")
        if profile["reasoning_efforts"]:
            effort = self._reasoning_effort_value(model)
            if effort and not for_responses:
                return {"reasoning_effort": effort}
            return {}
        if profile["supports_temperature"]:
            return {"temperature": self.openai_temperature}
        return {}

    def _should_retry_openai(self, exc: Exception) -> bool:
        if isinstance(exc, (ReadTimeout, RemoteDisconnected)):
            return True
        message = str(exc).lower()
        retry_markers = ("timed out", "timeout", "remote end closed connection", "connection aborted", "rate limit", "rate_limit", "429")
        return any(marker in message for marker in retry_markers)

    def _split_openai_payload(self, text: str, limit: int = OPENAI_SAFE_TEXT) -> List[str]:
        if len(text) <= limit or limit <= 0:
            return [text]
        segments: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + limit, length)
            split_pos = end
            if end < length:
                candidates: List[int] = []
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    candidates.append(newline_pos + 1)
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    candidates.append(space_pos + 1)
                for char in ".!?;:,":
                    pos = text.rfind(char, start, end)
                    if pos > start:
                        candidates.append(pos + 1)
                if candidates:
                    split_pos = max(candidates)
            piece = text[start:split_pos]
            if not piece:
                piece = text[start:end]
                split_pos = end
                if not piece:
                    break
            segments.append(piece)
            start = split_pos
        return segments if segments else [text]

    def _as_responses_input(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "system":
                role = "developer"
            content = message.get("content", "")
            if isinstance(content, list):
                formatted.append({"role": role, "content": content})
                continue
            formatted.append({"role": role, "content": str(content)})
        return formatted

    def _openai_responses_text_config(self, model: Optional[str]) -> Dict[str, Any]:
        config: Dict[str, Any] = {"format": {"type": "json_object"}}
        profile = get_openai_model_profile(model or "")
        if profile["supports_verbosity"]:
            config["verbosity"] = self.openai_verbosity
        return config

    def _openai_responses_reasoning(self, model: Optional[str]) -> Optional[Dict[str, str]]:
        effort = self._reasoning_effort_value(model)
        return {"effort": effort} if effort else None

    def _openai_metadata(self) -> Dict[str, str]:
        meta: Dict[str, str] = {
            "app": "daru-translator",
            "source_lang": (self.source_lang or "auto")[:32],
            "target_lang": (self.target_lang or "ru")[:32],
        }
        provider = (self.provider or "").strip()
        if provider:
            meta["provider"] = provider[:32]
        if self.openai_project:
            meta["project"] = self.openai_project[:64]
        return meta

    def _log_openai_response(self, response_obj: object) -> None:
        response_id = getattr(response_obj, "id", None)
        if not response_id and isinstance(response_obj, dict):
            response_id = response_obj.get("id")
        if response_id:
            print(f"[OpenAI] response stored: {response_id}")

    def _reasoning_effort_value(self, model: Optional[str] = None) -> Optional[str]:
        profile = get_openai_model_profile(model or self.openai_model)
        efforts = tuple(profile["reasoning_efforts"])
        if self.openai_reasoning_effort in efforts:
            return self.openai_reasoning_effort
        fallback = str(profile["default_reasoning_effort"])
        return fallback if fallback in efforts else None

    def _response_text(self, response_obj: object) -> str:
        text_value = getattr(response_obj, "output_text", None)
        if isinstance(text_value, str) and text_value.strip():
            return text_value
        fragments: List[str] = []
        output = getattr(response_obj, "output", None)
        if output:
            for item in output:
                contents = getattr(item, "content", None) or []
                for content in contents:
                    value = getattr(content, "text", None)
                    if isinstance(value, str):
                        fragments.append(value)
        return "".join(fragments)

    def _openai_reasoning_note(self, model: Optional[str]) -> str:
        effort = self._reasoning_effort_value(model)
        if not effort:
            return ""
        mapping = {
            "none": "Prioritise a direct, low-latency translation.",
            "minimal": "Use minimal reasoning while preserving translation accuracy.",
            "low": "Use efficient reasoning and prioritise terminology consistency.",
            "medium": "Balance reasoning depth, accuracy, and latency.",
            "high": "Apply careful reasoning to maximise translation accuracy and nuance.",
            "xhigh": "Apply the highest available reasoning effort to difficult translation ambiguities.",
        }
        return mapping.get(effort, "")

    def _strict_descriptor(self) -> str:
        value = self.openai_strict_value
        if value <= 0.34:
            return "low"
        if value <= 0.67:
            return "medium"
        return "high"

    def _google_free_translate_batch(self, texts: Sequence[str]) -> List[str]:
        import json as _json
        import urllib.parse
        import urllib.request

        results: List[str] = []
        source = (self.source_lang or "auto").lower()
        target = (self.target_lang or "ru").lower()

        base_params = {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
        }
        base_query = urllib.parse.urlencode(base_params)

        for text in texts:
            q = urllib.parse.urlencode({"q": text})
            url = f"https://translate.googleapis.com/translate_a/single?{base_query}&{q}"

            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    payload = resp.read().decode("utf-8")
                data = _json.loads(payload)
                translation = _extract_google_free_translation(data)
            except Exception:
                translation = None

            results.append(translation if translation is not None else text)

        return results

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
