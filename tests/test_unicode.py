import pytest

from khojki_ocr.unicode import inspect_text, plausible_prefix
from khojki_ocr.vocabulary import VOCAB


def test_validator_reports_without_repairing_raw_text():
    text = "\U00011200\U0001122c\U00011230"
    report = inspect_text(text)
    assert "non_atomic_vowel" in {issue["code"] for issue in report["issues"]}
    assert "".join(char["character"] for char in report["codepoints"]) == text


@pytest.mark.parametrize("text", ["क", "\U00011212", "😀"])
def test_unsupported_inventory_is_flagged(text):
    assert not inspect_text(text)["valid_inventory"]


def test_valid_khojki_and_gujarati_digits_need_no_repair():
    assert inspect_text("\U00011208\U00011232\U0001123a૧")["issues"] == []
    assert inspect_text("\U0001122c")["issues"][0]["code"] == "orphan_combining_sign"


def test_optional_prefix_filter_is_structural_not_a_dictionary():
    assert plausible_prefix("\U00011208\U0001122c\U00011234")
    assert not plausible_prefix("\U0001122c")
    assert not plausible_prefix("\U00011208\U0001122c\U00011230")
    assert plausible_prefix("\U00011219")
    assert plausible_prefix("\U00011224")


def test_vocabulary_has_unique_scalars_and_excludes_unassigned_slot():
    assert len(VOCAB) == len(set(VOCAB))
    assert "\U00011212" not in VOCAB
    assert set("૦૧૨૩૪૫૬૭૮૯").issubset(VOCAB)
    assert all(len(char) == 1 for char in VOCAB)
