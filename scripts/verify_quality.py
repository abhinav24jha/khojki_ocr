"""Run bounded CPU release checks in Colab/CI and preserve their evidence.

This runner intentionally imports no ML runtime until after pytest exits. Use
--private-data only when the original exercise scans have been restored. No
historical evaluation or deployed checkpoint is overwritten.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_logged(command, log_path, timeout=900):
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log_path}")


def verify_fresh_predictions(output):
    annotations = ROOT / "data/round4/test.json"
    freeze = json.loads((ROOT / "data/round4/test-freeze.json").read_text())
    assert (
        hashlib.sha256(annotations.read_bytes()).hexdigest()
        == freeze["annotation_sha256"]
    )
    for relative, digest in freeze["image_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "data" / relative).read_bytes()).hexdigest()
            == digest
        )
    run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_pages.py"),
            "--checkpoint",
            str(ROOT / "models/printed-khojki.pt"),
            "--annotations",
            str(annotations),
            "--split",
            "test",
            "--device",
            "cpu",
            "--output",
            str(output / "fresh-evaluation"),
        ],
        output / "fresh-evaluation.log",
    )
    original = json.loads(
        (ROOT / "results/round4-final-r4/evaluation.json").read_text()
    )
    observed = json.loads((output / "fresh-evaluation/evaluation.json").read_text())
    keys = (
        "reference",
        "automatic_prediction",
        "checked_crop_prediction",
        "detected_box",
        "automatic",
        "checked_crop",
    )
    old = {row["id"]: row for row in original["lines"]}
    new = {row["id"]: row for row in observed["lines"]}
    assert old.keys() == new.keys(), "Evaluation line IDs changed"
    for identifier in old:
        for key in keys:
            assert old[identifier][key] == new[identifier][key], (
                f"Changed {identifier}: {key}"
            )
    assert original["summary"] == observed["summary"], (
        "Frozen evaluation summary changed"
    )
    return {
        "exact_per_line_match": True,
        "summary": observed["summary"],
        "reference_sha256": freeze["annotation_sha256"],
    }


def assert_equal_state(left, right, path="checkpoint"):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), f"Tensor mismatch at {path}"
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), f"Array mismatch at {path}"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"Key mismatch at {path}"
        for key in left:
            assert_equal_state(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), f"Length mismatch at {path}"
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal_state(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"Value mismatch at {path}"


def training_smoke(output, private_data):
    import torch

    from khojki_ocr.checkpoints import save_checkpoint
    from khojki_ocr.vocabulary import VOCAB

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    smoke = output / "training-smoke"
    smoke.mkdir()
    (smoke / "assets").symlink_to(ROOT / "assets", target_is_directory=True)
    synthetic = load_script("train_synthetic")
    original_synthetic = synthetic.Synthetic

    def small_synthetic(n, seed, augment, fonts, repertoire="legacy"):
        return original_synthetic(
            min(n, 8 if augment else 4), seed, augment, fonts, repertoire
        )

    # Exercise real shaping, augmentation, CTC, optimizer, and checkpoint writing,
    # while bounding the fixture bank instead of generating 3,120 examples.
    with (
        patch.object(synthetic, "ROOT", smoke),
        patch.object(synthetic, "Synthetic", small_synthetic),
    ):
        synthetic.main(
            ["--steps", "1", "--batch", "2", "--device", "cpu", "--output", "synthetic"]
        )
    trained = torch.load(
        smoke / "experiments/synthetic/last.pt", map_location="cpu", weights_only=False
    )
    assert trained["step"] == 1 and trained["warm_start_only"] is True
    assert all(torch.isfinite(value).all() for value in trained["model"].values())

    if private_data:
        (smoke / "data").symlink_to(ROOT / "data", target_is_directory=True)
        annotations = ROOT / "data/round4/train-dev-corrected.json"
    else:
        # Public CI uses generated glyph strips with separate train/dev records.
        from khojki_ocr.render import Renderer
        from khojki_ocr.resources import review_font

        data = smoke / "data"
        data.mkdir()
        renderer = Renderer(review_font(), 36)
        rows = []
        for index, (split, text) in enumerate(
            [
                ("train", "\U00011208\U0001122c"),
                ("train", "\U00011219\U0001122c"),
                ("dev", "\U00011224\U0001122c"),
            ]
        ):
            filename = f"line-{index}.png"
            renderer.render(text).save(data / filename)
            rows.append(
                {
                    "id": f"generated-{index}",
                    "split": split,
                    "image": filename,
                    "text": text,
                }
            )
        annotations = data / "annotations.json"
        annotations.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    deployed = torch.load(
        ROOT / "models/printed-khojki.pt", map_location="cpu", weights_only=False
    )
    member = deployed["ensemble_states"][0]
    initial = smoke / "initial.pt"
    save_checkpoint({"model": member["model"], "vocab": VOCAB}, initial)
    finetune = load_script("finetune")
    common = [
        "--device",
        "cpu",
        "--annotations",
        str(annotations),
        "--layout",
        "robust",
        "--seed",
        "31",
        "--synthetic-fraction",
        "0.25",
    ]
    if member.get("packed", False):
        common.append("--packed")
    with patch.object(finetune, "ROOT", smoke):
        finetune.main(
            common
            + [
                "--checkpoint",
                str(initial),
                "--steps",
                "2",
                "--output",
                "uninterrupted",
            ]
        )
        finetune.main(
            common
            + ["--checkpoint", str(initial), "--steps", "1", "--output", "resumed"]
        )
        resumed_path = smoke / "experiments/resumed/last.pt"
        finetune.main(
            common
            + [
                "--checkpoint",
                str(resumed_path),
                "--resume",
                "--steps",
                "2",
                "--output",
                "resumed",
            ]
        )
    uninterrupted = torch.load(
        smoke / "experiments/uninterrupted/last.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "step",
        "annotation_sha256",
    ):
        assert_equal_state(uninterrupted[key], resumed[key], key)
    # The interrupted run additionally evaluates step 1. Compare the common
    # step-2 record, whose elapsed time naturally differs between runs.
    old_record = {
        k: v for k, v in uninterrupted["history"][-1].items() if k != "seconds"
    }
    new_record = {k: v for k, v in resumed["history"][-1].items() if k != "seconds"}
    assert_equal_state(old_record, new_record, "step_2_history")
    config = json.loads((smoke / "experiments/resumed/config.json").read_text())
    return {
        "synthetic_steps": 1,
        "synthetic_training_examples": 8,
        "synthetic_validation_examples": 4,
        "finetune_steps": 2,
        "resume_model_optimizer_rng_exact_match": True,
        "common_step_history_match_ignoring_time": True,
        "train_line_ids": config["train_line_ids"],
        "validation_line_ids": config["validation_line_ids"],
        "private_data": private_data,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory; never overwritten",
    )
    parser.add_argument(
        "--private-data",
        action="store_true",
        help="Also validate original scans and frozen predictions",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Only for a job that has already run the full suite",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "2")
    started = time.perf_counter()
    summary = {
        "status": "running",
        "private_data": args.private_data,
        "python": sys.version,
    }
    summary_path = output / "verification.json"
    try:
        if not args.skip_pytest:
            command = [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "--cov=khojki_ocr",
                "--cov-report=term-missing",
                f"--cov-report=xml:{output / 'coverage.xml'}",
                f"--junitxml={output / 'test-results.xml'}",
            ]
            if not args.private_data:
                command += ["-m", "not private_data"]
            run_logged(command, output / "pytest.log")
            summary["pytest"] = "passed"
        else:
            summary["pytest"] = "skipped_by_explicit_flag"
        if args.private_data:
            summary["fresh_predictions"] = verify_fresh_predictions(output)
        summary["training"] = training_smoke(output, args.private_data)
        summary["status"] = "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Resource links are runtime conveniences, not duplicate evidence artifacts.
        for name in ("assets", "data"):
            link = output / "training-smoke" / name
            if link.is_symlink():
                link.unlink()
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
