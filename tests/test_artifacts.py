"""Frozen references are audit records, not claims of expert ground truth."""

import hashlib
import json

import pytest


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_original_frozen_reference_is_unchanged_and_disjoint(project_root):
    rows = read_json(project_root / "data/annotations.json")
    frozen = read_json(project_root / "data/annotations-v2-frozen.json")
    assert [(r["id"], r["text"], r["split"]) for r in rows] == [
        (r["id"], r["text"], r["split"]) for r in frozen
    ]
    assert len({row["id"] for row in rows}) == len(rows)
    groups = {
        split: {row["id"] for row in rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    assert len(groups["test"]) == 10
    assert not groups["test"] & (groups["train"] | groups["dev"])
    assert not groups["train"] & groups["dev"]
    assert all(row["page"] != 19 for row in rows if row["split"] in ("train", "dev"))


@pytest.mark.parametrize(
    "round_name,training_name",
    [
        ("round2", "train-dev.json"),
        ("round3", "train-dev.json"),
        ("round4", "train-dev-corrected.json"),
    ],
)
def test_later_holdouts_match_frozen_hashes_and_have_no_training_overlap(
    project_root, round_name, training_name
):
    directory = project_root / "data" / round_name
    path = directory / "test.json"
    metadata = read_json(directory / "test-freeze.json")
    expected = metadata.get("sha256", metadata.get("annotation_sha256"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    holdout = read_json(path)
    training = read_json(directory / training_name)
    assert len(holdout) == 5
    assert all(row["split"] == "test" for row in holdout)
    assert not {row["id"] for row in holdout} & {row["id"] for row in training}
    assert sum(len(row["text"]) for row in holdout) == metadata["characters"]
    assert not {row["image"] for row in holdout} & {row["image"] for row in training}
    assert len({row["id"] for row in training}) == len(training)
    assert all(row["text"] for row in training)
