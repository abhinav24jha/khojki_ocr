"""Restore exercise-only scan assets from the supplied PDF and frozen boxes.

Does not change references, split assignments, or detection parameters.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    a = p.parse_args()
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    if hashlib.sha256(a.pdf.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError(
            "PDF differs from the frozen assessment source; do not reuse its page coordinates."
        )
    from pypdf import PdfReader

    reader = PdfReader(a.pdf)
    for directory in ["pages", "lines", "reference"]:
        (ROOT / "data" / directory).mkdir(parents=True, exist_ok=True)
    rows_by_id = {}
    for name in [
        "data/annotations.json",
        "data/round2/expanded-train-dev.json",
        "data/round2/test.json",
        "data/round3/train-dev.json",
        "data/round3/test.json",
        "data/round4/test.json",
    ]:
        path = ROOT / name
        if path.exists():
            for row in json.loads(path.read_text()):
                rows_by_id[row["id"]] = row
    rows = list(rows_by_id.values())
    for page in sorted({4, 17, 18, 19, 20, 23} | {r["page"] for r in rows}):
        im = max(
            reader.pages[page - 1].images,
            key=lambda item: item.image.width * item.image.height,
        ).image.convert("RGB")
        im.save(ROOT / f"data/pages/p{page:02d}.png")
        for row in [r for r in rows if r["page"] == page]:
            im.crop(row["box"]).save(ROOT / "data" / row["image"])
    print(
        f"Restored source images and {len(rows)} frozen line crops; references unchanged."
    )


if __name__ == "__main__":
    main()
