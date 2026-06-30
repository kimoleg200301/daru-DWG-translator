import pytest
import ezdxf

from daru.dxf import pipeline


def test_dxf_resume_uses_checkpoint_after_batch_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.dxf"
    output = tmp_path / "translated.dxf"
    checkpoint = tmp_path / "checkpoint.json"
    document = ezdxf.new()
    modelspace = document.modelspace()
    modelspace.add_text("Alpha text")
    modelspace.add_text("Beta text")
    modelspace.add_text("Gamma text")
    document.saveas(source)

    class FailingAfterFirstBatch:
        calls = 0

        def __init__(self, **_kwargs):
            pass

        def backend_name(self):
            return "fake"

        def set_drawing_context(self, *_args, **_kwargs):
            pass

        def translate_many(self, texts):
            type(self).calls += 1
            if type(self).calls > 1:
                raise RuntimeError("boom")
            return [f"ru-{index}" for index, _text in enumerate(texts)]

    monkeypatch.setattr(pipeline, "TranslationEngine", FailingAfterFirstBatch)
    monkeypatch.setattr(pipeline, "DEFAULT_MAP_CACHE", tmp_path / "missing-map.csv")

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.translate_dxf(
            input_path=source,
            output_path=output,
            output_format="dxf",
            translator_name="fake",
            source_lang="en",
            target_lang="ru",
            save_map=False,
            save_txt=False,
            checkpoint_path=checkpoint,
        )

    assert checkpoint.exists()

    class ResumeEngine(FailingAfterFirstBatch):
        received = []
        translated_count = 0

        def translate_many(self, texts):
            type(self).received.extend(texts)
            result = []
            for _text in texts:
                result.append(f"ru-resumed-{type(self).translated_count}")
                type(self).translated_count += 1
            return result

    monkeypatch.setattr(pipeline, "TranslationEngine", ResumeEngine)

    pipeline.translate_dxf(
        input_path=source,
        output_path=output,
        output_format="dxf",
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
        save_map=False,
        save_txt=False,
        checkpoint_path=checkpoint,
    )

    assert len(ResumeEngine.received) == 2
    translated_doc = ezdxf.readfile(output)
    translated_values = {
        entity.dxf.text for entity in translated_doc.modelspace().query("TEXT")
    }
    assert translated_values == {"ru-0", "ru-resumed-0", "ru-resumed-1"}
