"""Frozen-model page evaluation; match detected lines geometrically, never by text.

Only pre-transcribed regions can be scored. Unlabelled page output is saved but
must not be described as evaluated full-page accuracy.
"""

import argparse
import collections
import json
from pathlib import Path

from PIL import Image

from khojki_ocr.metrics import alignment, score
from khojki_ocr.pipeline import process_page
from khojki_ocr.unicode import inspect_text

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--annotations", type=Path, default=ROOT / "data/annotations.json")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    rows = [r for r in json.loads(a.annotations.read_text()) if r["split"] == a.split]
    if not rows or not sum(len(row["text"]) for row in rows):
        raise ValueError("The requested split must contain nonempty reference text")
    import torch

    from khojki_ocr.recognize import Recognizer

    torch.set_num_threads(2)
    rec = Recognizer(a.checkpoint, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    page_reports = []
    confusions = collections.Counter()
    for page in sorted({r["page"] for r in rows}):
        with Image.open(ROOT / f"data/pages/p{page:02d}.png") as image:
            report = process_page(image, rec, a.output / f"p{page:02d}", f"p{page:02d}")
        page_reports.append(
            {"page": page, **{k: v for k, v in report.items() if k != "lines"}}
        )
        used = set()
        for row in [r for r in rows if r["page"] == page]:
            y0, y1 = row["box"][1], row["box"][3]

            def overlap(line, y0=y0, y1=y1):
                b = line["box"]
                intersection = max(0, min(y1, b[3]) - max(y0, b[1]))
                return intersection / max(y1 - y0, b[3] - b[1])

            candidates = [
                line
                for line in report["lines"]
                if line["id"] not in used and overlap(line) > 0.5
            ]
            line = max(candidates, key=overlap) if candidates else None
            if line:
                used.add(line["id"])
            prediction = line["text"] if line else ""
            with Image.open(ROOT / "data" / row["image"]) as checked_image:
                checked = rec(checked_image)
            auto_score = score(row["text"], prediction)
            crop_score = score(row["text"], checked)
            for op, r, q, _, _ in alignment(row["text"], prediction):
                if op != "equal":
                    confusions[(op, r, q)] += 1
            results.append(
                {
                    "id": row["id"],
                    "page": page,
                    "reference": row["text"],
                    "automatic_prediction": prediction,
                    "checked_crop_prediction": checked,
                    "reference_box": row["box"],
                    "detected_box": line["box"] if line else None,
                    "detected_id": line["id"] if line else None,
                    "vertical_overlap": overlap(line) if line else 0,
                    "automatic": auto_score,
                    "checked_crop": crop_score,
                    "net_crop_error_difference": auto_score["errors"]
                    - crop_score["errors"],
                    "encoding": inspect_text(prediction),
                }
            )
    summary = {
        "lines": len(results),
        "characters": sum(r["automatic"]["reference_characters"] for r in results),
        "missed_lines": sum(r["detected_id"] is None for r in results),
        "automatic_errors": sum(r["automatic"]["errors"] for r in results),
        "checked_crop_errors": sum(r["checked_crop"]["errors"] for r in results),
        "crop_changes_prediction_lines": sum(
            r["automatic_prediction"] != r["checked_crop_prediction"] for r in results
        ),
        "invalid_inventory_characters": sum(
            sum(i["code"] == "unsupported_codepoint" for i in r["encoding"]["issues"])
            for r in results
        ),
        "utf8_failures": sum(not r["encoding"]["utf8_roundtrip"] for r in results),
    }
    summary["automatic_cer"] = summary["automatic_errors"] / summary["characters"]
    summary["checked_crop_cer"] = summary["checked_crop_errors"] / summary["characters"]
    summary["net_crop_error_difference"] = (
        summary["automatic_errors"] - summary["checked_crop_errors"]
    )
    out = {
        "summary": summary,
        "pages": page_reports,
        "lines": results,
        "confusions": [
            {"operation": op, "reference": r, "prediction": q, "count": n}
            for (op, r, q), n in confusions.most_common()
        ],
        "interpretation": "CER covers only selected transcribed lines. Extra detections elsewhere are unscored. Net crop error difference is a paired intervention, not a causal decomposition; recognition and segmentation interact.",
    }
    (a.output / "evaluation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
