"""Codex CLI batching with lossless split assembly."""

from typing import Dict, List, Sequence, Tuple
from ..batching import split_payload
from ..codex_cli import CodexCliTranslator
from ..contracts import TranslationContext, validate_translations
from ..job import check_cancelled

CODEX_MAX_ITEMS = 50
CODEX_MAX_CHARS = 40000


class CodexProvider:
    def __init__(
        self, client: CodexCliTranslator, context: TranslationContext, instructions: str
    ) -> None:
        self.client = client
        self.context = context
        self.instructions = instructions

    def translate(self, texts: Sequence[str]) -> List[str]:
        originals = list(self.context.originals or texts)
        results = list(texts)
        split_jobs: List[Tuple[str, int, int, str]] = []
        part_counts: Dict[int, int] = {}
        part_sources: Dict[Tuple[int, int], str] = {}

        for original_index, text in enumerate(texts):
            parts = split_payload(text, CODEX_MAX_CHARS)
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
        instructions = self.instructions
        translator: CodexCliTranslator = self.client
        for batch in batches:
            check_cancelled()
            items: List[Dict[str, str]] = []
            for job_id, original_index, _part_index, text in batch:
                item = {"id": job_id, "text": text}
                if self.context.entity_types:
                    original = (
                        originals[original_index]
                        if original_index < len(originals)
                        else text
                    )
                    entity_type = self.context.entity_types.get(original)
                    if entity_type:
                        item["type"] = entity_type
                items.append(item)

            translated_values = validate_translations(
                [job[3] for job in batch],
                translator.translate(items, instructions=instructions),
                "Codex",
            )
            for job, translated in zip(batch, translated_values):
                _job_id, original_index, part_index, source_part = job
                assembled_parts.setdefault(original_index, {})[part_index] = translated
                if translated and translated != source_part:
                    original = (
                        originals[original_index]
                        if original_index < len(originals)
                        else source_part
                    )
                    self.context.glossary[original.strip()] = translated.strip()

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
            results[original_index] = (
                translated_text if translated_text else source_text
            )
        return results
