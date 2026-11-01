"""CTC prefix beam search, without a dictionary or language-model prior."""

import math
import numpy as np
from .vocabulary import VOCAB

NEG = -float("inf")


def add(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b) - m))


def beam_decode(log_probs, width=10, topk=8, prefix_valid=None):
    """Decode a (time, classes) log-probability matrix using CTC prefix search.

    Blank is class zero and is always considered, independently of topk.
    An impossible constraint or probability lattice raises ValueError rather
    than silently emitting a prefix with zero probability.
    """
    for name, value in (("width", width), ("topk", topk)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    log_probs = np.asarray(log_probs)
    if log_probs.ndim != 2 or log_probs.shape[1] != len(VOCAB) + 1:
        raise ValueError(f"Expected (time, {len(VOCAB) + 1}) log probabilities")
    if np.isnan(log_probs).any() or np.isposinf(log_probs).any():
        raise ValueError("Log probabilities cannot contain NaN or positive infinity")
    beams = {(): (0.0, NEG)}
    for row in log_probs:
        k = min(topk, len(row))
        labels = np.argpartition(row, -k)[-k:]
        following = {}

        def update(prefix, b=NEG, n=NEG):
            if prefix_valid is not None and not prefix_valid(
                "".join(VOCAB[i - 1] for i in prefix)
            ):
                return
            old = following.get(prefix, (NEG, NEG))
            following[prefix] = (add(old[0], b), add(old[1], n))

        for prefix, (blank, nonblank) in beams.items():
            total = add(blank, nonblank)
            update(prefix, b=total + float(row[0]))
            for c in labels:
                c = int(c)
                if c == 0:
                    continue
                prob = float(row[c])
                if prefix and prefix[-1] == c:
                    update(prefix, n=nonblank + prob)
                    update(prefix + (c,), n=blank + prob)
                else:
                    update(prefix + (c,), n=total + prob)
        beams = dict(
            sorted(following.items(), key=lambda x: add(*x[1]), reverse=True)[:width]
        )
        if not beams or all(add(*scores) == NEG for scores in beams.values()):
            raise ValueError(
                "No valid decoding path remains under the supplied constraints"
            )
    best = max(beams, key=lambda x: add(*beams[x]))
    return "".join(VOCAB[i - 1] for i in best)
