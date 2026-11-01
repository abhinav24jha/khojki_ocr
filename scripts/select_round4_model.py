"""Development-only fixed-decoder selection for the padding/normalization audit."""

from pathlib import Path
import argparse, json, hashlib, datetime, copy
import torch
from PIL import Image
from khojki_ocr.recognize import Recognizer, split_columns, preprocess
from khojki_ocr.model import image_tensor, collate
from khojki_ocr.metrics import score

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--phase", choices=["pilot", "final"], required=True)
p.add_argument("--device", default="cuda")
a = p.parse_args()
torch.set_num_threads(2)
rows = [
    r
    for r in json.loads((ROOT / "data/round2/train-dev.json").read_text())
    if r["split"] == "dev"
]
n = sum(len(r["text"]) for r in rows)
names = [
    "r4-control",
    "r4-word-red",
    "r4-word-white",
    "r4-freezebn",
    "r4-word-white-freezebn",
    "r4-freezernn",
]
cases = [
    ("incumbent", [ROOT / "models/round3-selected.pt"]),
    ("packed-start", [ROOT / "experiments/precision-packed/best.pt"]),
] + [(name, [ROOT / f"experiments/{name}/best.pt"]) for name in names]
records = []


def evaluate(name, paths):
    rec = Recognizer(paths, a.device, beam_width=10, constrained=True)
    lines = []
    for row in rows:
        pred = rec(Image.open(ROOT / "data" / row["image"]))
        lines.append(
            {
                "id": row["id"],
                "reference": row["text"],
                "prediction": pred,
                **score(row["text"], pred),
            }
        )
    record = {
        "name": name,
        "paths": [str(p.relative_to(ROOT)) for p in paths],
        "errors": sum(x["errors"] for x in lines),
        "characters": n,
        "lines": lines,
    }
    record["cer"] = record["errors"] / n
    records.append(record)
    print(json.dumps({k: v for k, v in record.items() if k != "lines"}), flush=True)


for name, paths in cases:
    evaluate(name, paths)
if a.phase == "final":
    top = sorted([r for r in records if r["name"] in names], key=lambda r: r["errors"])[
        :2
    ]
    evaluate("new-ensemble", [ROOT / r["paths"][0] for r in top])
    # Compare one incumbent member + strongest new member, not a decoder grid.
    evaluate(
        "mixed-ensemble",
        [ROOT / "experiments/long-control/best.pt", ROOT / top[0]["paths"][0]],
    )
best = min(
    records,
    key=lambda r: (
        r["errors"],
        2 if r["name"] in ["incumbent", "new-ensemble", "mixed-ensemble"] else 1,
        r["name"] != "incumbent",
    ),
)
report = {
    "phase": a.phase,
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "test_used": False,
    "annotation_policy": "Original round-two training/development labels unchanged. Previously viewed test sets excluded from candidate selection; later scores on them are diagnostic, not new blind tests.",
    "experiments": records,
    "selected": {k: v for k, v in best.items() if k != "lines"},
}
if a.phase == "final":
    saved = [
        torch.load(ROOT / x, map_location="cpu", weights_only=False)
        for x in best["paths"]
    ]
    if best["name"] == "incumbent":
        compact = saved[0]
    else:
        compact = {
            "vocab": saved[0]["vocab"],
            "height": 48,
            "layout": "robust",
            "beam_width": 10,
            "constrained": True,
        }
        if len(saved) == 1:
            compact.update(
                model=saved[0]["model"], packed=saved[0].get("packed", False)
            )
        else:
            compact["ensemble_states"] = [
                {"model": s["model"], "packed": s.get("packed", False)} for s in saved
            ]
    out = ROOT / "models/round4-selected.pt"
    torch.save(compact, out)
    rec = Recognizer(out, a.device)
    assert [rec(Image.open(ROOT / "data" / r["image"])) for r in rows] == [
        r["prediction"] for r in best["lines"]
    ]
    report["checkpoint_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    report["serialized_prediction_match"] = True
(ROOT / f"experiments/round4-{a.phase}-selection.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
