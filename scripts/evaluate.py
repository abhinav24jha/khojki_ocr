"""Evaluate frozen line references without changing predictions or reference text."""

import argparse
import collections
import json
import time
from pathlib import Path

from PIL import Image

from khojki_ocr.metrics import alignment, score

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument(
        "--split", default="dev", choices=["train", "dev", "test", "all"]
    )
    parser.add_argument(
        "--preprocess", default="contrast", choices=["raw", "contrast", "binary"]
    )
    parser.add_argument("--no-columns", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", default=str(ROOT / "data/annotations.json"))
    parser.add_argument("--layout", choices=["legacy", "robust"])
    parser.add_argument("--constrained", action="store_true", default=None)
    args = parser.parse_args(argv)
    if args.beam is not None and args.beam < 1:
        parser.error("--beam must be positive")
    return args


def evaluate_rows(rows, recognizer):
    results = []
    confusions = collections.Counter()
    for row in rows:
        with Image.open(ROOT / "data" / row["image"]) as image:
            start = time.perf_counter()
            prediction = recognizer(image)
            elapsed = time.perf_counter() - start
        result = {
            "id": row["id"],
            "reference": row["text"],
            "prediction": prediction,
            "seconds": elapsed,
            **score(row["text"], prediction),
        }
        results.append(result)
        for operation, reference, predicted, _, _ in alignment(row["text"], prediction):
            if operation != "equal":
                confusions[(operation, reference, predicted)] += 1
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return results, confusions


def main(argv=None):
    args = parse_args(argv)
    rows = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    rows = [row for row in rows if args.split == "all" or row["split"] == args.split]
    if not rows or not sum(len(row["text"]) for row in rows):
        raise ValueError("The requested split must contain nonempty reference text")

    import torch

    from khojki_ocr.recognize import Recognizer

    torch.set_num_threads(4)
    start = time.perf_counter()
    recognizer = Recognizer(
        args.checkpoint,
        args.device,
        args.preprocess,
        not args.no_columns,
        args.beam,
        args.layout,
        args.constrained,
    )
    load_seconds = time.perf_counter() - start
    results, confusions = evaluate_rows(rows, recognizer)
    summary = {
        "lines": len(rows),
        "characters": sum(r["reference_characters"] for r in results),
        "errors": sum(r["errors"] for r in results),
        "load_seconds": load_seconds,
        "inference_seconds": sum(r["seconds"] for r in results),
    }
    summary["cer"] = summary["errors"] / summary["characters"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": vars(args),
                "summary": summary,
                "lines": results,
                "confusions": [
                    {"operation": op, "reference": r, "prediction": q, "count": n}
                    for (op, r, q), n in confusions.most_common()
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
