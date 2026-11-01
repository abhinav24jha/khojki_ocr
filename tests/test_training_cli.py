"""Argument validation runs before importing a neural runtime or allocating data."""

import importlib.util

import pytest


@pytest.fixture
def training_scripts(project_root):
    modules = {}
    for name in ("finetune", "train_synthetic"):
        spec = importlib.util.spec_from_file_location(
            name, project_root / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.mark.parametrize(
    "arguments", [["--steps", "0"], ["--batch", "0"], ["--output", "../escape"]]
)
def test_synthetic_training_rejects_bad_arguments_before_ml_import(
    training_scripts, arguments
):
    with pytest.raises(SystemExit) as result:
        training_scripts["train_synthetic"].parse_args(arguments)
    assert result.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--lr", "nan"],
        ["--lr", "-1"],
        ["--train-limit", "0"],
        ["--synthetic-fraction", "0.8", "--word-fraction", "0.3"],
        ["--synthetic-fraction", "-0.1"],
        ["--seed", "-1"],
        ["--output", "../escape"],
    ],
)
def test_finetune_rejects_bad_arguments_before_loading_files(
    training_scripts, arguments
):
    with pytest.raises(SystemExit) as result:
        training_scripts["finetune"].parse_args(
            ["--checkpoint", "unused.pt", *arguments]
        )
    assert result.value.code == 2


def test_finetune_accepts_complete_reproducible_configuration(
    training_scripts, tmp_path
):
    checkpoint, annotations = tmp_path / "initial.pt", tmp_path / "rows.json"
    checkpoint.write_bytes(b"Not loaded during parsing")
    annotations.write_text("[]")
    args = training_scripts["finetune"].parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--annotations",
            str(annotations),
            "--steps",
            "2",
            "--device",
            "cpu",
            "--packed",
            "--seed",
            "31",
            "--synthetic-fraction",
            "0.25",
        ]
    )
    assert args.steps == 2 and args.packed and args.seed == 31
    assert args.synthetic_fraction == 0.25
