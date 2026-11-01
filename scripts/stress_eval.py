"""Fixed robustness probes on development data; never used as test claims."""

from pathlib import Path
import argparse, json
import torch
from PIL import Image, ImageFilter
from khojki_ocr.recognize import Recognizer
from khojki_ocr.metrics import score

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--output", required=True)
p.add_argument("--annotations", type=Path, default=ROOT / "data/annotations.json")
a = p.parse_args()
torch.set_num_threads(2)
rec = Recognizer(a.checkpoint, a.device)
rows = [r for r in json.loads(a.annotations.read_text()) if r["split"] == "dev"]
transforms = {
    "unchanged": lambda im: im,
    "half_resolution": lambda im: im.resize(
        (im.width // 2, im.height // 2), Image.Resampling.BILINEAR
    ),
    "blur_0.8px": lambda im: im.filter(ImageFilter.GaussianBlur(0.8)),
    "skew_1_degree": lambda im: im.rotate(
        1, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255
    ),
}
out = {}
for name, transform in transforms.items():
    predictions = []
    for row in rows:
        pred = rec(transform(Image.open(ROOT / "data" / row["image"]).convert("L")))
        predictions.append(
            {"id": row["id"], "prediction": pred, **score(row["text"], pred)}
        )
    errors = sum(r["errors"] for r in predictions)
    n = sum(r["reference_characters"] for r in predictions)
    out[name] = {
        "errors": errors,
        "characters": n,
        "cer": errors / n,
        "lines": predictions,
    }
Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(
    json.dumps(
        {k: {a: b for a, b in v.items() if a != "lines"} for k, v in out.items()}
    )
)
