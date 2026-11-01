import numpy as np
from PIL import Image, ImageDraw

from khojki_ocr.word_augmentation import (
    SEP,
    concatenate_words,
    separator_positions,
    training_words,
)


def test_separator_geometry_and_label_count_must_agree(word_strip):
    image, text = word_strip
    positions = separator_positions(image)
    assert [round(center) for center, _ in positions] == [60, 145]
    words = training_words(image, text)
    assert [label for _, label in words] == [
        word + SEP for word in text.split(SEP)[:-1]
    ]
    assert training_words(image, text + SEP) == []
    assert training_words(image, SEP + text.split(SEP, 1)[1]) == []
    assert training_words(Image.new("L", (180, 48), 255), text) == []


def test_rgb_padding_is_white_not_red(word_strip):
    image, text = word_strip
    words = training_words(image, text, preserve_height=True)
    assert len(words) == 2
    for crop, _ in words:
        assert crop.mode == "L"
        assert crop.height == image.height
        assert np.all(np.asarray(crop)[:, :3] == 255)
        assert np.all(np.asarray(crop)[:, -3:] == 255)
    legacy = training_words(image, text, preserve_height=True, white_padding=False)
    assert legacy[0][0].getpixel((0, 0)) == (255, 0, 0)


def test_concatenation_preserves_label_order_and_ink_height():
    image = Image.new("L", (60, 48), 255)
    ImageDraw.Draw(image).rectangle((10, 10, 50, 37), fill=0)
    combined, text = concatenate_words(
        [(image, "x"), (image, "y")], preserve_height=True
    )
    assert combined.size == (131, 48)
    assert text == "xy"
    y, _ = np.where(np.asarray(combined) < 128)
    assert (int(y.min()), int(y.max())) == (10, 37)


def test_empty_bank_and_negative_gap_are_rejected(word_strip):
    import pytest

    with pytest.raises(ValueError):
        concatenate_words([])
    with pytest.raises(ValueError):
        concatenate_words([(word_strip[0], word_strip[1])], gap=-1)
