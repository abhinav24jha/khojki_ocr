"""Prepare an immutable verification bundle and notebook before connecting Colab."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    files = {ROOT / name for name in tracked if name and (ROOT / name).is_file()}
    for directory in ("src", "scripts", "tests", ".github"):
        files.update(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and not any(
                part == "__pycache__" or part.endswith(".egg-info") for part in p.parts
            )
        )
    files.update(
        ROOT / name for name in ("setup.py", "MANIFEST.in", "technical-retro.md")
    )
    files.update((ROOT / "data/pages").glob("*.png"))
    files.update((ROOT / "data/lines").glob("*.png"))
    archive = ROOT / "cloud/khojki-quality-input.zip"
    # Existing execution notebooks are records, not inputs to this verification.
    files = {
        p
        for p in files
        if p.relative_to(ROOT).parts[0] not in {"cloud"} and p.suffix not in {".pyc"}
    }
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(files):
            bundle.write(file, str(file.relative_to(ROOT)))
        bundle.writestr("quality-input-manifest.json", json.dumps(manifest, indent=2))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    setup = f"""from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile
BUNDLE = Path('/content/{archive.name}')
assert BUNDLE.is_file(), 'Upload the prepared ZIP through Files before running this cell.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == {digest!r}
ROOT = Path('/content/khojki-quality')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    for name in z.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    z.extractall(ROOT)
manifest = json.loads((ROOT/'quality-input-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT)+'[test,train]'], check=True)
print('Verified input files:', len(manifest), 'Bundle SHA256:', {digest!r})
"""
    run = """output = ROOT/'results/quality-validation'
log = ROOT/'quality-run.log'
with log.open('w') as stream:
    completed = subprocess.run([sys.executable, str(ROOT/'scripts/verify_quality.py'), '--private-data', '--output', str(output)], stdout=stream, stderr=subprocess.STDOUT)
print(log.read_text()[-5000:])
print('Return code:', completed.returncode)
if (output/'verification.json').exists():
    print((output/'verification.json').read_text())
"""
    export = """import base64
from IPython.display import HTML, display
archive = Path('/content/khojki-quality-results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(log, 'quality-run.log')
    z.write(ROOT/'quality-input-manifest.json', 'quality-input-manifest.json')
    for file in output.rglob('*'):
        if file.is_file() and not file.is_symlink() and not any(parent.is_symlink() for parent in file.parents if parent != output):
            z.write(file, str(file.relative_to(ROOT)))
raw = archive.read_bytes()
print(json.dumps({'name': archive.name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}))
display(HTML('<a download="'+archive.name+'" href="data:application/zip;base64,'+base64.b64encode(raw).decode()+'">Save verified quality results</a>'))
"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Khojki release verification\nCPU-only. Upload `khojki-quality-input.zip` in Files, then run cells in order. Save and verify the result archive before disconnecting. This notebook changes no historical experiment or frozen model."
            ],
        }
    ]
    for source in (setup, run, export):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": source.splitlines(True),
                "outputs": [],
                "execution_count": None,
            }
        )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "colab": {"name": "Khojki_quality_verification.ipynb"},
        },
        "cells": cells,
    }
    (ROOT / "cloud/Khojki_quality_verification.ipynb").write_text(
        json.dumps(notebook, indent=2)
    )
    receipt = {
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "files": len(manifest),
    }
    (ROOT / "cloud/quality-input-receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
