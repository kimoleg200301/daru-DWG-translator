import numpy as np
import pytest
from PIL import Image

from daru.pdf.config import PdfProcessingConfig
from daru.pdf.imaging import _estimate_text_color
from daru.pdf.models import RegionInfo
from daru.pdf.rendering import Renderer


@pytest.mark.parametrize("background,ink", [
    ((245, 245, 245), (0, 0, 0)),
    ((30, 30, 30), (240, 245, 250)),
    ((220, 235, 245), (35, 45, 170)),
])
def test_sparse_glyphs_keep_ink_color_when_word_box_is_mostly_background(background, ink):
    roi = np.full((40, 200, 3), background, dtype=np.uint8)
    roi[10:30, 45:48] = ink
    roi[10:30, 65:68] = ink
    word_box_mask = np.zeros((40, 200), dtype=np.uint8)
    word_box_mask[5:35, 35:170] = 255
    assert _estimate_text_color(roi, word_box_mask) == ink


def test_raster_heading_is_anchored_to_source_text_inside_cleaning_margin():
    region = RegionInfo(
        page_index=0, region_id=0, bbox=(0, 10, 500, 100),
        content_bbox=(80, 20, 320, 28), text_orientation_deg=0,
        source_text="Drawing title", translated_text="Technical drawing",
        text_color=(0, 0, 0), background_color=(255, 255, 255),
        font_size_estimate=24, block_type="TITLE", is_bold=True,
    )
    with Image.new("RGB", (520, 150), "white") as background:
        rendered = Renderer(PdfProcessingConfig()).render_text_layer(background, [region])
    try:
        pixels = np.asarray(rendered)
        dark_y, dark_x = np.where(np.any(pixels < 245, axis=2))
        assert dark_x.size
        assert 80 <= dark_x.min() <= 90
        assert dark_y.min() >= 30
        assert dark_x.max() < 500
        assert region.rendered_font_size == 24
        assert region.bbox == (0, 10, 500, 100)  # Cleaning geometry remains intact.
    finally:
        rendered.close()
