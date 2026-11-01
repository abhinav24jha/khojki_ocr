"""Small checkpoint and recognizer integration cases, with no training downloads."""

import pytest
from PIL import Image, ImageDraw

from khojki_ocr.vocabulary import VOCAB

pytestmark = pytest.mark.integration


@pytest.fixture
def checkpoint_state():
    import torch

    from khojki_ocr.model import CRNN

    torch.set_num_threads(2)
    torch.manual_seed(11)
    return {
        "model": CRNN().state_dict(),
        "vocab": VOCAB,
        "height": 48,
        "layout": "robust",
    }


def test_compact_ensemble_matches_identical_single_member(tmp_path, checkpoint_state):
    import torch

    from khojki_ocr.recognize import Recognizer

    single = tmp_path / "single.pt"
    ensemble = tmp_path / "ensemble.pt"
    torch.save(checkpoint_state, single)
    torch.save(
        {
            "ensemble_states": [{"model": checkpoint_state["model"], "packed": False}]
            * 2,
            "vocab": VOCAB,
            "height": 48,
            "layout": "robust",
        },
        ensemble,
    )
    first, combined = Recognizer(single), Recognizer(ensemble)
    image = Image.new("L", (80, 48), 255)
    assert len(combined.models) == 2
    assert first(image) == combined(image)


def test_checkpoint_vocabulary_mismatch_is_rejected(tmp_path, checkpoint_state):
    import torch

    from khojki_ocr.recognize import Recognizer

    checkpoint_state["vocab"] = "wrong ordering"
    path = tmp_path / "bad.pt"
    torch.save(checkpoint_state, path)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        Recognizer(path)


def test_ensemble_with_mixed_image_heights_is_rejected(tmp_path, checkpoint_state):
    import torch

    from khojki_ocr.recognize import Recognizer

    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(checkpoint_state, first)
    torch.save({**checkpoint_state, "height": 32}, second)
    with pytest.raises(ValueError, match="heights must match"):
        Recognizer([first, second])


def test_generated_wide_gutter_splits_columns_but_keeps_line_height():
    from khojki_ocr.recognize import split_columns

    image = Image.new("L", (360, 48), 255)
    draw = ImageDraw.Draw(image)
    for x in (10, 35, 60, 250, 275, 300):
        draw.rectangle((x, 12, x + 14, 32), fill=0)
    chunks = split_columns(image, "robust")
    assert len(chunks) == 2
    assert all(chunk.height == 48 for chunk in chunks)
    assert all(chunk.width < 120 for chunk in chunks)


@pytest.mark.parametrize("mode", ["raw", "contrast", "binary"])
def test_preprocess_accepts_rgb_and_returns_grayscale(mode):
    from khojki_ocr.recognize import preprocess

    image = Image.new("RGB", (80, 48), "white")
    processed = preprocess(image, mode)
    assert processed.mode == "L"
    assert processed.size == image.size
