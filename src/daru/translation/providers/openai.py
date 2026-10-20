"""OpenAI request construction, retries and response decoding."""

import json
import re
from http.client import RemoteDisconnected
from typing import Any, Callable, Dict, List, Optional, Sequence

from requests.exceptions import ReadTimeout

from ...config import get_openai_model_profile, language_prompt_label
from ..batching import _group_split_jobs_into_batches, split_payload
from ..contracts import TranslationBatchError, TranslationContext, translation_mapping
from ..job import check_cancelled, retry_wait
from ..options import TranslationOptions

OPENAI_RETRY_ATTEMPTS = 5
OPENAI_RETRY_DELAY = 3.0


def _chat_message_text(completion_obj) -> str:
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
            return _chat_message_text(dumped)

    return ""


class OpenAIProvider:
    def __init__(
        self, options: TranslationOptions, connection: dict, context: TranslationContext
    ) -> None:
        self.options = options
        self.connection = connection
        self.context = context

    def translate(self, texts: Sequence[str]) -> List[str]:
        originals = list(self.context.originals or texts)
        mode, client, module, legacy_create = self._resolve_transport()
        model = self.connection.get("model")
        if not model or (mode != "legacy" and client is None):
            raise TranslationBatchError("OpenAI: клиент или модель не настроены")
        batch_size = self.connection.get("batch_size", 12)
        if sum(map(len, texts)) <= 50000:
            batch_size = max(batch_size, len(texts))
        results = list(texts)
        jobs = []
        part_counts = {}
        for index, text in enumerate(texts):
            original = originals[index] if index < len(originals) else text
            if not self._should_translate(original):
                continue
            parts = split_payload(text)
            part_counts[index] = len(parts)
            jobs.extend(
                (f"{index}:{part}", index, part, value)
                for part, value in enumerate(parts)
            )
        if not jobs:
            return results
        instructions = self._instructions(model)
        batches = (
            [[job] for job in jobs]
            if mode == "legacy"
            else _group_split_jobs_into_batches(jobs, part_counts, batch_size)
        )
        assembled: Dict[int, Dict[int, str]] = {}
        for batch in batches:
            check_cancelled()
            items = []
            for job_id, index, _part, text in batch:
                item = {"id": job_id, "text": text}
                if self.context.entity_types:
                    entity = self.context.entity_types.get(originals[index])
                    if entity:
                        item["type"] = entity
                items.append(item)
            messages = [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": json.dumps({"items": items}, ensure_ascii=False),
                },
            ]
            if mode == "legacy":
                create = legacy_create or (
                    module.ChatCompletion.create if module is not None else None
                )
                if create is None:
                    raise TranslationBatchError("OpenAI legacy клиент недоступен")
                kwargs = {"model": model, "messages": messages, "store": True}
                if legacy_create:
                    kwargs["response_format"] = {"type": "json_object"}
                kwargs.update(self._openai_generation_kwargs(model))
                completion = self._request(create, kwargs)
                message = _chat_message_text(completion)
            else:
                completion = self._request(
                    client.responses.create, self._response_kwargs(model, messages)
                )
                message = self._response_text(completion)
            try:
                data = json.loads(message)
            except (TypeError, json.JSONDecodeError) as exc:
                raise TranslationBatchError(
                    "ChatGPT не вернул корректный JSON"
                ) from exc
            mapping = translation_mapping(
                data, {job[0]: job[3] for job in batch}, "OpenAI"
            )
            for job_id, index, part, _source in batch:
                assembled.setdefault(index, {})[part] = mapping[job_id]
        for index, count in part_counts.items():
            results[index] = "".join(assembled[index][part] for part in range(count))
            if results[index] != texts[index]:
                self.context.glossary[originals[index].strip()] = results[index].strip()
        return results

    def _should_translate(self, source: str) -> bool:
        stripped = source.strip()
        if len(stripped) < 3:
            return False
        if self.options.source_lang == "en" and not re.search(r"[A-Za-z]", stripped):
            return False
        if re.fullmatch(r"[A-Za-z]\.?|[A-Za-z]\d+|[-+]?\d+[\d\s./-]*", stripped):
            return False
        return not (stripped.isupper() and len(stripped) <= 3 and " " not in stripped)

    def _request(self, create: Callable[..., object], kwargs: Dict[str, Any]) -> object:
        for attempt in range(OPENAI_RETRY_ATTEMPTS):
            check_cancelled()
            try:
                completion = create(**kwargs)
            except Exception as exc:
                check_cancelled()
                if (
                    attempt + 1 == OPENAI_RETRY_ATTEMPTS
                    or not self._should_retry_openai(exc)
                ):
                    raise TranslationBatchError(
                        f"Ошибка перевода OpenAI: {exc}"
                    ) from exc
                retry_wait(OPENAI_RETRY_DELAY)
                continue
            check_cancelled()
            if completion is None:
                raise TranslationBatchError("OpenAI: пустой ответ")
            self._log_openai_response(completion)
            return completion
        raise TranslationBatchError("OpenAI: исчерпаны попытки перевода")

    def _response_kwargs(self, model, messages) -> Dict[str, Any]:
        kwargs = {
            "model": model,
            "input": self._as_responses_input(messages),
            "store": True,
        }
        text_config = self._openai_responses_text_config(model)
        reasoning = self._openai_responses_reasoning(model)
        if text_config:
            kwargs["text"] = text_config
        if reasoning:
            kwargs["reasoning"] = reasoning
        metadata = self._openai_metadata()
        if metadata:
            kwargs["metadata"] = metadata
        kwargs.update(self._openai_generation_kwargs(model, for_responses=True))
        return kwargs

    def _instructions(self, model) -> str:
        reasoning_note = self._openai_reasoning_note(model)
        source_label = language_prompt_label(
            self.options.source_lang,
            "auto",
            allow_auto=True,
        )
        target_label = language_prompt_label(self.options.target_lang, "ru")
        if self.options.system_prompt_template:
            system_content = self.options.system_prompt_template
            system_content = system_content.replace(
                "{source_lang}", source_label
            ).replace("{target_lang}", target_label)
        else:
            system_content = (
                "You are a professional technical translator of elevator drawing manuals. "
                f"Translate the provided values from {source_label} to {target_label}. "
                "All texts belong to the SAME technical drawing — maintain consistent "
                "terminology across all items. Preserve numbers, "
                "placeholders like '__DXF_DIM__', and DXF control sequences such as \"\\P\". "
                'Each item may include a "type" field indicating the DXF entity type '
                "(TEXT, MTEXT, TABLE, DIMENSION, etc.) — use it to infer context. "
                "Respond with strict JSON: "
                '{"translations": [{"id": "<id>", "text": "<translated>"}, ...]}'
            )
        if self.context.document_text:
            system_content += (
                f"\n\n[{self.context.label} CONTEXT — all texts for reference]:\n"
                + self.context.document_text
            )
        if self.context.glossary:
            glossary_lines = [
                f'"{k}" → "{v}"' for k, v in list(self.context.glossary.items())[:100]
            ]
            system_content += (
                "\n\n[GLOSSARY — previously translated terms, use consistently]:\n"
                + "\n".join(glossary_lines)
            )
        if reasoning_note:
            system_content = f"{system_content}\n\n{reasoning_note}"

        return system_content

    def _resolve_transport(self):
        mode = self.connection.get("mode", "new")
        client = self.connection.get("client")
        module = self.connection.get("module")
        chat_completion_create: Optional[Callable[..., object]] = None
        if module is not None:
            chat_namespace = getattr(module, "chat", None)
            completions_namespace = (
                getattr(chat_namespace, "completions", None) if chat_namespace else None
            )
            create_attr = (
                getattr(completions_namespace, "create", None)
                if completions_namespace
                else None
            )
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
                client_kwargs = {"api_key": self.options.openai_api_key}
                if self.options.openai_base_url:
                    client_kwargs["base_url"] = self.options.openai_base_url
                if self.options.openai_project:
                    client_kwargs["project"] = self.options.openai_project
                try:
                    fallback_client = module.OpenAI(**client_kwargs)
                except TypeError:
                    fallback_client = None
                    try:
                        client_kwargs.pop("base_url", None)
                        client_kwargs.pop("project", None)
                        fallback_client = module.OpenAI(**client_kwargs)
                        if self.options.openai_base_url and fallback_client is not None:
                            with_options = getattr(
                                fallback_client, "with_options", None
                            )
                            if callable(with_options):
                                fallback_client = with_options(
                                    base_url=self.options.openai_base_url
                                )
                            else:
                                try:
                                    setattr(
                                        fallback_client,
                                        "base_url",
                                        self.options.openai_base_url,
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        fallback_client = None
                except Exception:
                    fallback_client = None

            if fallback_client is not None:
                client = fallback_client
                mode = "new"
            elif (
                not chat_completion_create
                and openai_major_version
                and openai_major_version >= 1
            ):
                raise RuntimeError(
                    "Установлен openai>=1.0.0, но не удалось инициализировать новый клиент. "
                    "Переустановите пакет 'openai' или закрепите версию <1.0."
                )

        return mode, client, module, chat_completion_create

    def _openai_generation_kwargs(
        self, model: Optional[str], *, for_responses: bool = False
    ) -> Dict[str, Any]:
        profile = get_openai_model_profile(model or "")
        if profile["reasoning_efforts"]:
            effort = self._reasoning_effort_value(model)
            if effort and not for_responses:
                return {"reasoning_effort": effort}
            return {}
        if profile["supports_temperature"]:
            return {"temperature": self.options.openai_temperature}
        return {}

    def _should_retry_openai(self, exc: Exception) -> bool:
        if isinstance(exc, (ReadTimeout, RemoteDisconnected)):
            return True
        message = str(exc).lower()
        retry_markers = (
            "timed out",
            "timeout",
            "remote end closed connection",
            "connection aborted",
            "rate limit",
            "rate_limit",
            "429",
        )
        return any(marker in message for marker in retry_markers)

    def _as_responses_input(
        self, messages: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
            config["verbosity"] = self.options.openai_verbosity
        return config

    def _openai_responses_reasoning(
        self, model: Optional[str]
    ) -> Optional[Dict[str, str]]:
        effort = self._reasoning_effort_value(model)
        return {"effort": effort} if effort else None

    def _openai_metadata(self) -> Dict[str, str]:
        meta: Dict[str, str] = {
            "app": "daru-translator",
            "source_lang": (self.options.source_lang or "auto")[:32],
            "target_lang": (self.options.target_lang or "ru")[:32],
        }
        provider = (self.options.provider or "").strip()
        if provider:
            meta["provider"] = provider[:32]
        if self.options.openai_project:
            meta["project"] = self.options.openai_project[:64]
        return meta

    def _log_openai_response(self, response_obj: object) -> None:
        response_id = getattr(response_obj, "id", None)
        if not response_id and isinstance(response_obj, dict):
            response_id = response_obj.get("id")
        if response_id:
            print(f"[OpenAI] response stored: {response_id}")

    def _reasoning_effort_value(self, model: Optional[str] = None) -> Optional[str]:
        profile = get_openai_model_profile(model or self.options.openai_model)
        efforts = tuple(profile["reasoning_efforts"])
        if self.options.openai_reasoning_effort in efforts:
            return self.options.openai_reasoning_effort
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
