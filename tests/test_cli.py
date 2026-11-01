import hashlib
import json
import subprocess
import sys

import pytest
from PIL import Image


def run_cli(*arguments, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "khojki_ocr", *map(str, arguments)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        cwd=cwd,
    )


def test_help_is_available_without_model_loading():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--pdf-native" in result.stdout
    assert "--region" in result.stdout
    assert "--threads" in result.stdout


def test_missing_source_is_an_argparse_error(tmp_path):
    result = run_cli(tmp_path / "missing.png", "--output", tmp_path / "output")
    assert result.returncode == 2
    assert "Source does not exist" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--threads", "0"], "--threads must be positive"),
        (["--region", "-1", "0", "10", "10"], "Invalid crop coordinates"),
        (["--region", "5", "0", "5", "10"], "Invalid crop coordinates"),
    ],
)
def test_invalid_arguments_fail_before_reading_checkpoint(tmp_path, arguments, message):
    image = tmp_path / "page.png"
    Image.new("RGB", (40, 40), "white").save(image)
    checkpoint = tmp_path / "not-a-checkpoint.pt"
    checkpoint.write_bytes(b"Invalid; validation must happen before torch.load")
    result = run_cli(
        image, "--checkpoint", checkpoint, "--output", tmp_path / "output", *arguments
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.integration
def test_bundled_checkpoint_cli_writes_complete_run_without_training(
    tmp_path, project_root
):
    source = tmp_path / "blank.png"
    output = tmp_path / "ocr"
    Image.new("RGB", (300, 200), "white").save(source)
    checkpoint = project_root / "models/printed-khojki.pt"
    result = run_cli(
        source, "--checkpoint", checkpoint, "--output", output, "--threads", "2"
    )
    assert result.returncode == 0, result.stderr
    run = json.loads((output / "run.json").read_text())
    assert (
        run["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert run["device"] == "cpu"
    assert run["pages"][0]["line_count"] == 0
    assert (output / "page-001/transcription.txt").read_text() == ""
    assert (output / "page-001/review.html").is_file()
    assert (output / "page-001/NotoSansKhojki-Regular.ttf").is_file()
