"""Additional Colab notebook cell: verify an installed wheel outside the checkout."""

from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path("/content/khojki-quality")
output = ROOT / "results/quality-validation"
wheel_target = Path("/content/khojki-wheel-installed")
work = Path("/content/khojki-wheel-smoke")
work.mkdir(exist_ok=True)
with (output / "wheel.log").open("w") as stream:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
    wheel = next((ROOT / "dist").glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(wheel_target),
            str(wheel),
        ],
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
    env = dict(os.environ, PYTHONPATH=str(wheel_target))
    code = """from PIL import Image
from khojki_ocr.resources import default_checkpoint,review_font
assert '/content/khojki-wheel-installed/' in str(default_checkpoint())
assert '/content/khojki-wheel-installed/' in str(review_font())
Image.new('RGB',(300,200),'white').save('blank.png')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=work,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "khojki_ocr",
            "blank.png",
            "--output",
            "ocr",
            "--threads",
            "2",
        ],
        cwd=work,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
report = json.loads((work / "ocr/run.json").read_text())
assert report["pages"][0]["line_count"] == 0
for name in [
    "review.html",
    "OFL.txt",
    "NotoSansKhojki-Regular.ttf",
    "result.json",
    "overlay.png",
    "transcription.txt",
]:
    assert (work / "ocr/page-001" / name).is_file(), name
(output / "wheel-verification.json").write_text(
    json.dumps(
        {
            "installed_outside_checkout": True,
            "blank_image_inference": True,
            "all_outputs_and_font_license_present": True,
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
        indent=2,
    )
)
print("Installed-wheel inference and resources: PASS")
print((output / "pytest.log").read_text()[-2200:])
