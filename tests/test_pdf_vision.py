from __future__ import annotations

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from daru.pdf import _core
from daru.pdf.vision import (
    CodexCliVisionLayoutRefiner,
    CodexCliVisionQaReviewer,
    OpenAIVisionLayoutRefiner,
    VisionBlockUpdate,
    VisionQaIssue,
)


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _FakeCodexStructuredClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class _EchoLayoutBatchClient:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["prompt"].rsplit("\n\n", 1)[-1])
        return {
            "blocks": [
                {
                    "stable_id": block["stable_id"],
                    "corrected_text": block["text"],
                    "semantic_type": "body",
                    "translate": True,
                    "fit_strategy": "default",
                    "font_weight": "normal",
                    "alignment": "left",
                    "confidence": 0.9,
                    "qa_flags": [],
                }
                for block in payload["blocks"]
            ]
        }


class _EmptyQaBatchClient:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"issues": []}


def _image_from_data_url(data_url):
    encoded = data_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(encoded)))


def test_vision_refiner_uses_structured_outputs_and_stable_id_guard():
    responses = _FakeResponses(
        {
            "blocks": [
                {
                    "stable_id": "ok",
                    "corrected_text": "Hyundai Elevator",
                    "semantic_type": "brand",
                    "translate": False,
                    "fit_strategy": "preserve_small",
                    "font_weight": "bold",
                    "alignment": "center",
                    "confidence": 0.91,
                    "qa_flags": ["logo"],
                },
                {
                    "stable_id": "invented",
                    "corrected_text": "Ignored",
                    "semantic_type": "body",
                    "translate": True,
                    "fit_strategy": "default",
                    "font_weight": "normal",
                    "alignment": "left",
                    "confidence": 1.0,
                    "qa_flags": [],
                },
            ]
        }
    )
    client = SimpleNamespace(responses=responses)
    block = SimpleNamespace(
        stable_id="ok",
        block_type="TEXT",
        source_text="HYUNDAI ELEVAT0R",
        bbox=(10, 20, 200, 50),
        is_table=False,
        alignment="left",
        confidence=98.0,
        parent_block_type="FIGURE",
    )
    refiner = OpenAIVisionLayoutRefiner(client=client, model="gpt-5.5")

    updates = refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (320, 200), "white"),
        blocks=[block],
    )

    assert set(updates) == {"ok"}
    assert updates["ok"].corrected_text == "Hyundai Elevator"
    assert updates["ok"].semantic_type == "brand"
    assert updates["ok"].font_weight == "bold"
    assert updates["ok"].alignment == "center"
    call = responses.calls[0]
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    user_content = call["input"][1]["content"]
    assert any(part["type"] == "input_image" for part in user_content)
    block_schema = call["text"]["format"]["schema"]["properties"]["blocks"]["items"]
    assert "font_weight" in block_schema["required"]
    assert "alignment" in block_schema["required"]


def test_openai_vision_refiner_stable_quality_downscales_image():
    responses = _FakeResponses(
        {
            "blocks": [
                {
                    "stable_id": "ok",
                    "corrected_text": "Text",
                    "semantic_type": "body",
                    "translate": True,
                    "fit_strategy": "default",
                    "font_weight": "normal",
                    "alignment": "left",
                    "confidence": 0.9,
                    "qa_flags": [],
                }
            ]
        }
    )
    client = SimpleNamespace(responses=responses)
    block = SimpleNamespace(
        stable_id="ok",
        block_type="TEXT",
        source_text="Text",
        bbox=(10, 20, 200, 50),
        is_table=False,
        alignment="left",
        confidence=98.0,
        parent_block_type="",
    )
    refiner = OpenAIVisionLayoutRefiner(
        client=client,
        model="gpt-5.5",
        image_quality="stable",
    )

    refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (5000, 3000), "white"),
        blocks=[block],
    )

    user_content = responses.calls[0]["input"][1]["content"]
    image_part = next(part for part in user_content if part["type"] == "input_image")
    image = _image_from_data_url(image_part["image_url"])
    assert image.width * image.height <= 4_000_000


def test_codex_cli_vision_refiner_attaches_page_and_guards_stable_ids():
    client = _FakeCodexStructuredClient(
        {
            "blocks": [
                {
                    "stable_id": "ok",
                    "corrected_text": "Corrected",
                    "semantic_type": "heading",
                    "translate": True,
                    "fit_strategy": "shrink",
                    "font_weight": "bold",
                    "alignment": "center",
                    "confidence": 0.88,
                    "qa_flags": [],
                },
                {
                    "stable_id": "invented",
                    "corrected_text": "Ignore",
                    "semantic_type": "body",
                    "translate": True,
                    "fit_strategy": "default",
                    "font_weight": "normal",
                    "alignment": "left",
                    "confidence": 1.0,
                    "qa_flags": [],
                },
            ]
        }
    )
    block = SimpleNamespace(
        stable_id="ok",
        block_type="TEXT",
        source_text="OCR text",
        bbox=(10, 20, 200, 50),
        is_table=False,
        alignment="left",
        confidence=98.0,
        parent_block_type="",
    )
    refiner = CodexCliVisionLayoutRefiner(client=client)

    updates = refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (320, 200), "white"),
        blocks=[block],
    )

    assert set(updates) == {"ok"}
    assert updates["ok"].corrected_text == "Corrected"
    assert updates["ok"].font_weight == "bold"
    assert updates["ok"].alignment == "center"
    call = client.calls[0]
    assert call["schema"]["required"] == ["blocks"]
    assert len(call["images"]) == 1
    assert call["images"][0][1].startswith(b"\x89PNG")
    assert '"stable_id": "ok"' in call["prompt"]


def test_codex_cli_vision_refiner_batches_large_pages_and_downscales_image():
    client = _EchoLayoutBatchClient()
    blocks = [
        SimpleNamespace(
            stable_id=f"block-{index}",
            block_type="TEXT",
            source_text=f"Text {index}",
            bbox=(10, 20 + index, 200, 50 + index),
            is_table=False,
            alignment="left",
            font_weight="auto",
            confidence=98.0,
            parent_block_type="",
        )
        for index in range(45)
    ]
    refiner = CodexCliVisionLayoutRefiner(client=client)

    updates = refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (3760, 2659), "white"),
        blocks=blocks,
    )

    assert len(client.calls) == 3
    assert set(updates) == {f"block-{index}" for index in range(45)}
    for call in client.calls:
        image = Image.open(BytesIO(call["images"][0][1]))
        assert image.width * image.height <= 4_000_000
        payload = json.loads(call["prompt"].rsplit("\n\n", 1)[-1])
        assert len(payload["blocks"]) <= 16


def test_codex_cli_vision_refiner_original_quality_keeps_source_dimensions():
    client = _EchoLayoutBatchClient()
    block = SimpleNamespace(
        stable_id="block",
        block_type="TEXT",
        source_text="Text",
        bbox=(10, 20, 200, 50),
        is_table=False,
        alignment="left",
        font_weight="auto",
        confidence=98.0,
        parent_block_type="",
    )
    refiner = CodexCliVisionLayoutRefiner(
        client=client,
        image_quality="original",
    )

    updates = refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (5000, 3000), "white"),
        blocks=[block],
    )

    assert set(updates) == {"block"}
    image = Image.open(BytesIO(client.calls[0]["images"][0][1]))
    assert (image.width, image.height) == (5000, 3000)


def test_codex_cli_vision_refiner_can_send_single_page_request():
    client = _EchoLayoutBatchClient()
    blocks = [
        SimpleNamespace(
            stable_id=f"block-{index}",
            block_type="TEXT",
            source_text=f"Text {index}",
            bbox=(10, 20 + index, 200, 50 + index),
            is_table=False,
            alignment="left",
            font_weight="auto",
            confidence=98.0,
            parent_block_type="",
        )
        for index in range(45)
    ]
    refiner = CodexCliVisionLayoutRefiner(
        client=client,
        request_mode="single_page",
    )

    updates = refiner.refine_page(
        page_index=0,
        image=Image.new("RGB", (3760, 2659), "white"),
        blocks=blocks,
    )

    assert len(client.calls) == 1
    assert set(updates) == {f"block-{index}" for index in range(45)}
    call = client.calls[0]
    assert len(call["images"]) == 1
    image = Image.open(BytesIO(call["images"][0][1]))
    assert image.width * image.height <= 4_000_000
    payload = json.loads(call["prompt"].rsplit("\n\n", 1)[-1])
    assert payload["request_mode"] == "single_page"
    assert len(payload["blocks"]) == 45


def test_codex_cli_vision_qa_attaches_source_and_rendered_pages():
    client = _FakeCodexStructuredClient(
        {
            "issues": [
                {
                    "stable_id": "stable",
                    "issue_type": "overflow",
                    "severity": "high",
                    "action": "shrink_text",
                    "confidence": 0.9,
                    "message": "Clipped",
                }
            ]
        }
    )
    reviewer = CodexCliVisionQaReviewer(client=client)
    region = SimpleNamespace(
        stable_id="stable",
        region_id=1,
        source_text="Source",
        translated_text="Translated",
        bbox=(0, 0, 100, 40),
        semantic_type="body",
        fit_strategy="default",
    )

    issues = reviewer.review_page(
        page_index=0,
        source_image=Image.new("RGB", (100, 100), "white"),
        rendered_image=Image.new("RGB", (100, 100), "gray"),
        regions=[region],
    )

    assert len(issues) == 1
    assert issues[0].action == "shrink_text"
    call = client.calls[0]
    assert [name for name, _content in call["images"]] == [
        "source.png",
        "rendered.png",
    ]


def test_codex_cli_vision_qa_batches_regions_and_downscales_both_images():
    client = _EmptyQaBatchClient()
    reviewer = CodexCliVisionQaReviewer(client=client)
    regions = [
        SimpleNamespace(
            stable_id=f"stable-{index}",
            region_id=index,
            source_text=f"Source {index}",
            translated_text=f"Translated {index}",
            bbox=(0, index, 100, 40),
            semantic_type="body",
            fit_strategy="default",
            font_weight="normal",
            is_bold=False,
            alignment="left",
            source_font_size_estimate=20,
            rendered_font_size=20,
        )
        for index in range(41)
    ]

    issues = reviewer.review_page(
        page_index=0,
        source_image=Image.new("RGB", (3760, 2659), "white"),
        rendered_image=Image.new("RGB", (3760, 2659), "gray"),
        regions=regions,
    )

    assert issues == []
    assert len(client.calls) == 3
    for call in client.calls:
        assert len(call["images"]) == 2
        for _name, content in call["images"]:
            image = Image.open(BytesIO(content))
            assert image.width * image.height <= 4_000_000
        payload = json.loads(call["prompt"].rsplit("\n\n", 1)[-1])
        assert len(payload["regions"]) <= 16


def test_codex_cli_vision_qa_can_send_single_page_request():
    client = _EmptyQaBatchClient()
    reviewer = CodexCliVisionQaReviewer(
        client=client,
        request_mode="single_page",
    )
    regions = [
        SimpleNamespace(
            stable_id=f"stable-{index}",
            region_id=index,
            source_text=f"Source {index}",
            translated_text=f"Translated {index}",
            bbox=(0, index, 100, 40 + index),
            semantic_type="body",
            fit_strategy="default",
            font_weight="normal",
            is_bold=False,
            alignment="left",
            source_font_size_estimate=20,
            rendered_font_size=20,
        )
        for index in range(41)
    ]

    issues = reviewer.review_page(
        page_index=0,
        source_image=Image.new("RGB", (3760, 2659), "white"),
        rendered_image=Image.new("RGB", (3760, 2659), "gray"),
        regions=regions,
    )

    assert issues == []
    assert len(client.calls) == 1
    call = client.calls[0]
    assert len(call["images"]) == 2
    for _name, content in call["images"]:
        image = Image.open(BytesIO(content))
        assert image.width * image.height <= 4_000_000
    payload = json.loads(call["prompt"].rsplit("\n\n", 1)[-1])
    assert payload["request_mode"] == "single_page"
    assert len(payload["regions"]) == 41


def test_pdf_vision_reviewer_uses_independent_codex_settings(monkeypatch):
    captured = {}

    class FakeReviewer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(_core, "CodexCliVisionQaReviewer", FakeReviewer)
    config = _core.PdfProcessingConfig(
        vision_backend="codex_cli",
        vision_model="gpt-5.5",
        vision_reasoning_effort="high",
        vision_codex_cli_path="C:/vision/codex.exe",
        vision_codex_timeout_seconds=420,
        vision_request_mode="single_page",
        vision_image_quality="original",
    ).normalized()

    reviewer = _core._make_vision_qa_reviewer(
        config=config,
        openai_key="translation-api-key",
        openai_base_url="https://translation.example/v1",
        openai_project="translation-project",
        codex_cli_path="C:/translation/codex.exe",
        codex_timeout_seconds=60,
        log=lambda _message: None,
    )

    assert isinstance(reviewer, FakeReviewer)
    assert captured["cli_path"] == "C:/vision/codex.exe"
    assert captured["model"] == "gpt-5.5"
    assert captured["reasoning_effort"] == "high"
    assert captured["timeout_seconds"] == 420
    assert captured["request_mode"] == "single_page"
    assert captured["image_quality"] == "original"


def test_apply_vision_block_updates_never_uses_unknown_ids():
    block = _core.BlockRegion(
        page_index=0,
        block_id="b1",
        block_type="TEXT",
        bbox=(10, 10, 120, 40),
        source_text="HYUNDAI ELEVAT0R",
        stable_id="stable",
    )
    updates = {
        "stable": VisionBlockUpdate(
            stable_id="stable",
            corrected_text="HYUNDAI ELEVATOR",
            semantic_type="brand",
            translate=False,
            fit_strategy="preserve_small",
            font_weight="bold",
            alignment="center",
            confidence=0.9,
            qa_flags=["logo"],
        ),
        "other": VisionBlockUpdate(
            stable_id="other",
            corrected_text="wrong",
            semantic_type="body",
            translate=True,
        ),
    }

    applied = _core._apply_vision_block_updates([block], updates)

    assert applied == 1
    assert block.source_text == "HYUNDAI ELEVATOR"
    assert block.corrected_text == "HYUNDAI ELEVATOR"
    assert block.semantic_type == "brand"
    assert block.font_weight == "bold"
    assert block.alignment == "center"
    assert block.should_translate is False
    assert block.render_enabled is False
    assert block.qa_flags == ["logo"]


def test_vision_qa_shrink_issue_updates_region_for_rerender():
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(0, 0, 100, 40),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Source",
        translated_text="Translated text",
        font_size_estimate=20,
        stable_id="stable",
    )
    issue = VisionQaIssue(
        stable_id="stable",
        issue_type="overflow",
        severity="high",
        action="shrink_text",
        confidence=0.95,
        message="Text is clipped",
    )

    changed = _core._apply_vision_qa_issues([region], [issue])

    assert changed is True
    assert region.fit_strategy == "shrink"
    assert region.font_size_estimate == 17
    assert "overflow" in region.qa_flags


def test_vision_formatting_reaches_region_without_preemptive_font_reduction():
    block = _core.BlockRegion(
        page_index=0,
        block_id="caption",
        block_type="TEXT",
        bbox=(20, 20, 220, 60),
        lines=[
            _core.LineRegion(
                page_index=0,
                line_id="line",
                bbox=(20, 20, 220, 50),
                source_text="Caption",
                word_boxes=[(20, 20, 100, 30)],
            )
        ],
        source_text="Caption",
        translated_text="Подпись",
        render_id=1,
        stable_id="stable",
        semantic_type="caption",
        fit_strategy="preserve_small",
        font_weight="bold",
        alignment="right",
    )
    page = np.full((100, 260, 3), 255, dtype=np.uint8)

    region = _core._block_to_region_info(
        block,
        page,
        _core.PdfProcessingConfig().normalized(),
        page_width=260,
        page_height=100,
        candidates=[block],
    )

    assert region is not None
    assert region.font_size_estimate == 30
    assert region.source_font_size_estimate == 30
    assert region.font_weight == "bold"
    assert region.is_bold is True
    assert region.alignment == "right"


def test_vision_qa_formatting_actions_do_not_reduce_font_size():
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(0, 0, 240, 80),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Source",
        translated_text="Translated",
        font_size_estimate=20,
        source_font_size_estimate=20,
        stable_id="stable",
    )
    issues = [
        VisionQaIssue(
            stable_id="stable",
            issue_type="font_weight",
            severity="medium",
            action="set_bold",
            confidence=0.9,
        ),
        VisionQaIssue(
            stable_id="stable",
            issue_type="alignment",
            severity="medium",
            action="align_center",
            confidence=0.9,
        ),
        VisionQaIssue(
            stable_id="stable",
            issue_type="padding",
            severity="medium",
            action="reduce_padding",
            confidence=0.9,
        ),
    ]

    changed = _core._apply_vision_qa_issues([region], issues)

    assert changed is True
    assert region.is_bold is True
    assert region.font_weight == "bold"
    assert region.alignment == "center"
    assert region.fit_strategy == "tight"
    assert region.font_size_estimate == 20


def test_vision_qa_can_restore_source_font_size():
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(0, 0, 240, 80),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Source",
        translated_text="Translated",
        font_size_estimate=14,
        source_font_size_estimate=22,
        stable_id="stable",
    )

    changed = _core._apply_vision_qa_issues(
        [region],
        [
            VisionQaIssue(
                stable_id="stable",
                issue_type="unnecessary_font_reduction",
                severity="medium",
                action="preserve_font_size",
                confidence=0.9,
            )
        ],
    )

    assert changed is True
    assert region.font_size_estimate == 22
    assert region.fit_strategy == "tight"


def test_renderer_preserves_source_font_size_when_text_fits():
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (420, 120), "white"),
        width=420,
        height=120,
        dpi=300,
    )
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(10, 10, 400, 100),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Short text",
        translated_text="Короткий текст",
        font_size_estimate=24,
        source_font_size_estimate=24,
        stable_id="stable",
    )

    _core.Renderer(_core.PdfProcessingConfig()).render_page(page, [region])

    assert region.rendered_font_size == 24


def test_wrap_text_splits_long_tokens_to_fit_width():
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    font = _core._FontLoader(None).get(12)
    lines = _core._wrap_text(
        "Supercalifragilisticexpialidocious",
        draw,
        font,
        45,
    )

    assert len(lines) > 1
    assert all(_core._measure_text(line, draw, font)[0] <= 45 for line in lines)
