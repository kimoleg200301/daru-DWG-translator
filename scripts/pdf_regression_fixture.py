"""Render deterministic PDF layout fixtures without OCR/API calls.

Run against two package paths, then compare the rendered PNG pixels and the
extracted editable text. Temporary artifacts belong under tmp/pdfs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-path", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_path))
    import fitz
    from PIL import Image, ImageDraw

    from daru.pdf import _core
    from daru.pdf.editable import EditableTextLayer, supports_editable_rotation

    args.output.mkdir(parents=True, exist_ok=True)
    config = _core.PdfProcessingConfig(dpi=150, render_dpi=150).normalized()
    renderer = _core.Renderer(config)
    font = renderer.font_loader.get(22)
    pages, page_regions, backgrounds, raster_pages = [], [], [], []
    for index, (width, height) in enumerate([(1200, 850), (850, 1200), (1100, 780)]):
        image = Image.new("RGB", (width, height), (245, 248, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 25, width - 30, 115), fill=(210, 229, 246), outline=(35, 80, 125), width=2)
        draw.text((50, 52), "Electrical installation - reference drawing", fill="black", font=font)
        blocks = [_core.BlockRegion(index, "heading", "TITLE", (45, 40, width - 45, 95),
                    source_text="Electrical installation - reference drawing",
                    translated_text="Электроустановка — справочный чертёж", font_weight="bold",
                    word_boxes=[(50, 52, 525, 28)])]
        if index == 0:
            for y in (150, 230, 310, 390):
                draw.line((45, y, 820, y), fill=(40, 50, 60), width=3)
            for x in (45, 490, 820):
                draw.line((x, 150, x, 390), fill=(40, 50, 60), width=3)
            for row, (source, translated, value) in enumerate([
                ("Rated current", "Номинальный ток", "16 A"),
                ("Supply voltage", "Напряжение питания", "400 V"),
                ("Protection class", "Класс защиты", "IP54"),
            ]):
                y = 174 + row * 80
                draw.text((65, y), source, fill="black", font=font)
                draw.text((520, y), value, fill="black", font=font)
                blocks.append(_core.BlockRegion(index, f"cell-{row}", "CELL", (55, y - 4, 480, y + 35),
                    source_text=source, translated_text=translated, is_table=True,
                    word_boxes=[(65, y + 3, 265, 26)], parent_bbox=(45, 150 + row * 80, 490, 230 + row * 80)))
            draw.rectangle((910, 160, 1100, 345), outline=(35, 80, 125), width=5)
            draw.ellipse((930, 185, 1080, 320), outline=(180, 60, 50), width=6)
        elif index == 1:
            draw.rectangle((200, 180, 710, 760), outline=(40, 90, 145), width=5)
            label = Image.new("RGBA", (500, 65), (0, 0, 0, 0))
            ImageDraw.Draw(label).text((12, 10), "Disconnect before maintenance", fill="black", font=font)
            rotated = label.rotate(90, expand=True)
            image.paste(rotated, (80, 200), rotated)
            left, top, right, bottom = rotated.getbbox()
            blocks.append(_core.BlockRegion(index, "vertical", "TEXT", (75, 180, 145, 735),
                source_text="Disconnect before maintenance", translated_text="Отключить перед обслуживанием",
                rotation_deg=90, semantic_type="vertical", fit_strategy="vertical",
                word_boxes=[(80 + left, 200 + top, right - left, bottom - top)]))
            label.close()
            rotated.close()
        else:
            draw.rectangle((60, 165, 640, 365), fill=(230, 238, 218), outline=(65, 95, 55), width=3)
            text = "Inspect the valve before operation.\nRecord the result in the maintenance log."
            for line, y in zip(text.splitlines(), (192, 230)):
                draw.text((80, y), line, font=font, fill=(35, 45, 35))
            blocks.append(_core.BlockRegion(index, "paragraph", "TEXT", (70, 178, 630, 350),
                source_text=text, translated_text="Проверьте клапан перед запуском.\nЗапишите результат в журнал обслуживания.",
                alignment="left", word_boxes=[(80, 192, 515, 28), (80, 230, 515, 28)]))
            for x in range(745, 1000, 30):
                draw.line((x, 170, x - 80, 390), fill=(165, 180, 195), width=2)
        page = _core.PageImage(index, image, width, height, 150, (width * 72 / 150, height * 72 / 150))
        _core._assign_render_ids_for_page(blocks, page)
        regions = _core._build_regions_from_blocks(page, blocks, config)
        cleaned = renderer.clean_page(page, regions)
        cleaned.save(args.output / f"cleaned-{index + 1}.png")
        raster = renderer.render_text_layer(cleaned, regions)
        raster.save(args.output / f"raster-{index + 1}.png")
        pages.append(page)
        page_regions.append(regions)
        backgrounds.append(cleaned)
        raster_pages.append(raster)

    editable = EditableTextLayer(None, lambda message: None)
    def overlay(index, pdf_page):
        regions = [region for region in page_regions[index] if supports_editable_rotation(region)]
        editable.insert_page(pdf_page, pages[index], regions)
    _core.images_to_pdf(backgrounds, args.output / "editable.pdf", dpi=150, input_dpi=150,
                        log=lambda message: None, page_sizes_points=[page.physical_size_points for page in pages],
                        page_overlay=overlay)
    text_pages = []
    with fitz.open(args.output / "editable.pdf") as document:
        for index, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(str(args.output / f"editable-{index + 1}.png"))
            text_pages.append(page.get_text())
    (args.output / "text.json").write_text(json.dumps(text_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    for image in backgrounds + raster_pages + [page.image for page in pages]:
        image.close()
    print(json.dumps({"pages": len(pages), "warnings": editable.warnings, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
