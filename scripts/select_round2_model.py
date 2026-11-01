"""Development-only decoder/model selection; fresh test data must be absent."""

from pathlib import Path
import argparse, json, hashlib, datetime, time, itertools
import torch
from PIL import Image
from khojki_ocr.recognize import Recognizer
from khojki_ocr.metrics import score

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
a = p.parse_args()
torch.set_num_threads(2)
rows = [
    r
    for r in json.loads((ROOT / "data/round2/train-dev.json").read_text())
    if r["split"] == "dev"
]
assert len(rows) == 16
pool = {}
for name in [
    "long-control",
    "long-bold",
    "precision-packed",
    "geometry-legacy",
    "geometry-preserve",
]:
    path = ROOT / f"experiments/{name}/best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    pool[name] = {
        "path": path,
        "errors": round(saved["validation_cer"] * 405),
        "height": saved.get("height", 48),
    }
    del saved
ranked = sorted(pool, key=lambda n: pool[n]["errors"])
# Greedy accuracy has already been measured while training; compare the top two.
chosen = ranked[:2]
cases = []
for name in chosen:
    for beam, constrained in [(1, False), (10, False), (10, True)]:
        cases.append(([name], beam, constrained))
if pool[chosen[0]]["height"] == pool[chosen[1]]["height"]:
    for beam, constrained in [(1, False), (10, True)]:
        cases.append((chosen, beam, constrained))
records = []
for names, beam, constrained in cases:
    rec = Recognizer(
        [pool[n]["path"] for n in names],
        a.device,
        beam_width=beam,
        constrained=constrained,
    )
    predictions = []
    start = time.perf_counter()
    for row in rows:
        prediction = rec(Image.open(ROOT / "data" / row["image"]))
        predictions.append(
            {
                "id": row["id"],
                "reference": row["text"],
                "prediction": prediction,
                **score(row["text"], prediction),
            }
        )
    errors = sum(r["errors"] for r in predictions)
    record = {
        "models": names,
        "beam_width": beam,
        "constrained": constrained,
        "errors": errors,
        "characters": 405,
        "cer": errors / 405,
        "seconds": time.perf_counter() - start,
        "lines": predictions,
    }
    records.append(record)
    print(json.dumps({k: v for k, v in record.items() if k != "lines"}), flush=True)
    del rec
best = min(
    records,
    key=lambda x: (x["errors"], len(x["models"]), x["beam_width"], x["constrained"]),
)
saved = [
    torch.load(pool[n]["path"], map_location="cpu", weights_only=False)
    for n in best["models"]
]
compact = {
    "vocab": saved[0]["vocab"],
    "height": saved[0].get("height", 48),
    "layout": "robust",
    "beam_width": best["beam_width"],
    "constrained": best["constrained"],
}
if len(saved) == 1:
    compact.update(model=saved[0]["model"], packed=saved[0].get("packed", False))
else:
    compact["ensemble_states"] = [
        {"model": s["model"], "packed": s.get("packed", False)} for s in saved
    ]
out = ROOT / "models/round2-selected.pt"
out.parent.mkdir(exist_ok=True)
torch.save(compact, out)
# Verify serialized deployment model gives precisely the measured predictions.
rec = Recognizer(out, a.device)
assert [rec(Image.open(ROOT / "data" / r["image"])) for r in rows] == [
    r["prediction"] for r in best["lines"]
]
report = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "test_used": False,
    "selection_rule": "Minimum development edits, then fewer models, then smaller beam, then unconstrained. Only top two training-selected models are compared for decoding.",
    "candidate_training_errors": {n: v["errors"] for n, v in pool.items()},
    "experiments": records,
    "selected": {k: v for k, v in best.items() if k != "lines"},
    "checkpoint_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "source_checkpoints": {
        n: hashlib.sha256(pool[n]["path"].read_bytes()).hexdigest()
        for n in best["models"]
    },
    "serialized_prediction_match": True,
}
(ROOT / "experiments/round2-selection.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(
    "SELECTED", json.dumps(report["selected"]), report["checkpoint_sha256"], flush=True
)
