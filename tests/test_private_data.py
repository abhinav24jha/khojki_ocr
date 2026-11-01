"""Optional regressions requiring the exercise scans, never required by public CI."""

import json

import pytest
from PIL import Image

pytestmark = pytest.mark.private_data


def require_private(project_root, relative):
    path = project_root / relative
    if not path.is_file():
        pytest.skip(f"Private exercise asset not distributed: {relative}")
    return path


def test_robust_layout_keeps_visually_audited_faint_refrain(project_root):
    from khojki_ocr.segment import detect_lines

    path = require_private(project_root, "data/pages/p20.png")
    with Image.open(path) as page:
        lines, _ = detect_lines(page, mode="robust")
    assert len(lines) == 16


@pytest.mark.integration
def test_real_number_gutters_are_preserved(project_root):
    paths = [
        require_private(project_root, f"data/lines/{identifier}.png")
        for identifier in ("p19r-01", "p19r-02")
    ]
    from khojki_ocr.recognize import split_columns

    for path in paths:
        with Image.open(path) as image:
            assert len(split_columns(image, "robust")) == 2


def test_real_training_word_boundaries_match_label_count(project_root):
    from khojki_ocr.word_augmentation import SEP, training_words

    rows = json.loads((project_root / "data/annotations.json").read_text())
    row = next(row for row in rows if row["id"] == "p17r-01")
    path = require_private(project_root, "data/" + row["image"])
    with Image.open(path) as image:
        bank = training_words(image, row["text"])
        assert len(bank) == row["text"].count(SEP)
        assert training_words(image, row["text"] + SEP) == []
