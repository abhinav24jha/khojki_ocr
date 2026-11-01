import pytest

from khojki_ocr.metrics import alignment, score


@pytest.mark.parametrize(
    "reference,prediction,edits",
    [
        ("", "", (0, 0, 0)),
        ("abc", "abc", (0, 0, 0)),
        ("abc", "adc", (1, 0, 0)),
        ("abc", "ac", (0, 1, 0)),
        ("ac", "abc", (0, 0, 1)),
        ("abc", "", (0, 3, 0)),
        ("", "abc", (0, 0, 3)),
        ("kitten", "sitting", (2, 0, 1)),
    ],
)
def test_edit_counts_and_alignment_reconstruct_inputs(reference, prediction, edits):
    result = score(reference, prediction)
    assert tuple(result[key] for key in ("substitute", "delete", "insert")) == edits
    assert result["errors"] == sum(edits)
    steps = alignment(reference, prediction)
    assert "".join(step[1] for step in steps) == reference
    assert "".join(step[2] for step in steps) == prediction


def test_cer_counts_unicode_scalars_and_preserves_combining_marks():
    reference = "\U00011208\U0001122c\U0001123a૧"
    result = score(reference, reference[:-1] + "૨")
    assert result["reference_characters"] == 4
    assert result["errors"] == 1
    assert result["cer"] == 0.25
    assert score("é", "e\u0301")["errors"] == 2


def test_empty_reference_and_unbounded_cer_are_explicit():
    assert score("", "")["cer"] == 0
    assert score("", "x")["cer"] is None
    assert score("a", "abc")["cer"] == 2
