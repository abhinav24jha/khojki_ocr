import numpy as np
import pytest
from PIL import Image

from khojki_ocr.segment import detect_lines, ink_mask, overlay


@pytest.mark.parametrize("mode", ["adaptive", "robust", "projection"])
def test_generated_lines_preserve_reading_order_and_bounds(printed_page, mode):
    lines, _region = detect_lines(printed_page, prefix="scan", mode=mode)
    assert len(lines) == 3
    assert [line.id for line in lines] == ["scan-01", "scan-02", "scan-03"]
    assert [line.box[1] for line in lines] == sorted(line.box[1] for line in lines)
    for line, baseline in zip(lines, (60, 140, 220)):
        left, top, right, bottom = line.box
        assert 0 <= left < right <= printed_page.width
        assert 0 <= top <= baseline < baseline + 19 < bottom <= printed_page.height


def test_assisted_region_excludes_outside_lines(printed_page):
    lines, region = detect_lines(printed_page, region=(30, 110, 500, 190))
    assert region == [30, 110, 500, 190]
    assert len(lines) == 1
    assert 110 <= lines[0].box[1] < lines[0].box[3] <= 190


def test_blank_scan_and_single_pixel_noise_have_no_lines():
    image = Image.new("L", (300, 200), 255)
    image.putpixel((100, 100), 0)
    assert not ink_mask(image).any()
    assert detect_lines(image)[0] == []


def test_overlay_does_not_mutate_source(printed_page):
    original = np.asarray(printed_page).copy()
    lines, _ = detect_lines(printed_page)
    rendered = overlay(printed_page, lines)
    assert np.array_equal(np.asarray(printed_page), original)
    assert not np.array_equal(np.asarray(rendered), original)


@pytest.mark.parametrize("size", [(1, 1), (5, 10), (20, 20)])
def test_tiny_blank_images_return_no_lines(size):
    assert detect_lines(Image.new("L", size, 255))[0] == []


def test_unknown_mode_is_rejected(printed_page):
    with pytest.raises(ValueError, match="mode"):
        detect_lines(printed_page, mode="typo")


@pytest.mark.parametrize("region", [(-1, 0, 20, 20), (20, 0, 10, 20), (0, 0, 700, 20)])
def test_invalid_regions_are_rejected(printed_page, region):
    with pytest.raises(ValueError):
        detect_lines(printed_page, region=region)
