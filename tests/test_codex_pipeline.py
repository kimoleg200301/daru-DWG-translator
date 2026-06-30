"""Propagation tests for Codex settings through public pipelines."""

from __future__ import annotations

import sys

import ezdxf

from daru.dxf import pipeline


def test_dxf_pipeline_passes_codex_settings_to_engine(tmp_path, monkeypatch):
    source = tmp_path / "source.dxf"
    output = tmp_path / "translated.dxf"
    document = ezdxf.new()
    document.modelspace().add_text("Unique Codex pipeline text")
    document.saveas(source)
    captured = {}

    class CapturingEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def backend_name(self):
            return "codex-cli"

        def set_drawing_context(self, *_args, **_kwargs):
            pass

        def translate_many(self, texts):
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(pipeline, "TranslationEngine", CapturingEngine)
    monkeypatch.setattr(pipeline, "DEFAULT_MAP_CACHE", tmp_path / "map-cache.csv")

    result = pipeline.translate_dxf(
        input_path=source,
        output_path=output,
        output_format="dxf",
        translator_name="codex",
        source_lang="en",
        target_lang="ru",
        codex_cli_path="C:/tools/codex.cmd",
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="low",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="high",
        codex_timeout_seconds=180,
        save_map=False,
        save_txt=False,
    )

    assert result["backend"] == "codex-cli"
    assert captured["provider"] == "codex"
    assert captured["codex_cli_path"] == "C:/tools/codex.cmd"
    assert captured["codex_model"] == "gpt-5.4-mini"
    assert captured["codex_reasoning_effort"] == "low"
    assert captured["codex_analysis_model"] == "gpt-5.5"
    assert captured["codex_analysis_reasoning_effort"] == "high"
    assert captured["codex_timeout_seconds"] == 180


def test_dxf_cli_accepts_separate_codex_analysis_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daru-translate",
            "source.dxf",
            "translated.dxf",
            "--codex-analysis-model",
            "gpt-5.5",
            "--codex-analysis-reasoning-effort",
            "xhigh",
            "--restart-translation",
        ],
    )

    args = pipeline.parse_args()

    assert args.codex_analysis_model == "gpt-5.5"
    assert args.codex_analysis_reasoning_effort == "xhigh"
    assert args.restart_translation is True
