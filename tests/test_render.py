"""Shaping regressions require the documented train extra, never model inference."""

import numpy as np
import pytest

from khojki_ocr.render import Renderer
from khojki_ocr.resources import review_font


@pytest.fixture
def renderer():
    return Renderer(review_font(), size=40)


def test_shaped_khojki_has_ink_and_white_padding(renderer):
    image = renderer.render("\U00011208\U0001122c\U0001123a", pad=4)
    pixels = np.asarray(image)
    assert image.mode == "L"
    assert np.any(pixels < 128)
    assert np.all(pixels[:4] == 255)
    assert np.all(pixels[-4:] == 255)
    assert np.all(pixels[:, :4] == 255)
    assert np.all(pixels[:, -4:] == 255)


def test_empty_text_is_a_valid_blank_image(renderer):
    image = renderer.render("")
    assert image.width > 0 and image.height > 0
    assert np.all(np.asarray(image) == 255)


def test_missing_glyph_is_rejected_instead_of_training_on_tofu(renderer):
    with pytest.raises(ValueError, match="lacks a shaped glyph"):
        renderer.render("😀")


@pytest.mark.parametrize("size", [0, -1, True, 40.5])
def test_invalid_font_size_is_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        Renderer(review_font(), size=size)


def test_negative_padding_is_rejected(renderer):
    with pytest.raises(ValueError, match="nonnegative integer"):
        renderer.render("\U00011208", pad=-1)
