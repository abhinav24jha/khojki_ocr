import itertools

import numpy as np
import pytest

from khojki_ocr.decoding import add, beam_decode
from khojki_ocr.vocabulary import VOCAB


def full_lattice(probabilities):
    probabilities = np.asarray(probabilities)
    lattice = np.full((len(probabilities), len(VOCAB) + 1), -np.inf)
    with np.errstate(divide="ignore"):
        lattice[:, : probabilities.shape[1]] = np.log(probabilities)
    return lattice


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.3, 0.5, 0.2]],
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
        [[0.01, 0.49, 0.5], [0.01, 0.49, 0.5], [0.01, 0.49, 0.5]],
    ],
)
def test_beam_matches_exhaustive_ctc_path_probability_sums(probabilities):
    probabilities = np.asarray(probabilities)
    totals = {}
    for path in itertools.product(range(3), repeat=len(probabilities)):
        collapsed, previous = [], 0
        for token in path:
            if token and token != previous:
                collapsed.append(VOCAB[token - 1])
            previous = token
        text = "".join(collapsed)
        totals[text] = totals.get(text, 0) + np.prod(
            [probabilities[t, token] for t, token in enumerate(path)]
        )
    decoded = beam_decode(full_lattice(probabilities), width=100, topk=3)
    assert totals[decoded] == pytest.approx(max(totals.values()), abs=1e-12)


def test_blank_is_retained_even_when_outside_topk():
    lattice = full_lattice([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]])
    assert beam_decode(lattice, width=20, topk=1) == VOCAB[0]


def test_empty_timesteps_and_impossible_paths():
    assert beam_decode(np.empty((0, len(VOCAB) + 1))) == ""
    with pytest.raises(ValueError, match="No valid decoding path"):
        beam_decode(full_lattice([[1.0, 0.0]]), prefix_valid=lambda text: False)
    with pytest.raises(ValueError, match="No valid decoding path"):
        beam_decode(np.full((1, len(VOCAB) + 1), -np.inf))


@pytest.mark.parametrize(
    "keyword,value",
    [("width", 0), ("width", -1), ("width", True), ("topk", 0), ("topk", 1.5)],
)
def test_invalid_search_limits(keyword, value):
    with pytest.raises(ValueError, match="positive integer"):
        beam_decode(full_lattice([[1.0, 0.0]]), **{keyword: value})


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(3),
        np.zeros((2, 3)),
        np.full((1, len(VOCAB) + 1), np.nan),
        np.full((1, len(VOCAB) + 1), np.inf),
    ],
)
def test_invalid_lattice(invalid):
    with pytest.raises(ValueError):
        beam_decode(invalid)


def test_log_add_is_stable_at_extreme_magnitudes():
    assert add(-np.inf, -1000) == -1000
    assert add(-1000, -1000) == pytest.approx(-1000 + np.log(2))
