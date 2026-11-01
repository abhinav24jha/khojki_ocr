"""Image/label audit without model imports or predictions."""

from pathlib import Path
import json, collections, hashlib
import cv2, numpy as np
from PIL import Image, ImageDraw
from khojki_ocr.word_augmentation import training_words

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/round4"
OUT.mkdir(exist_ok=True)
rows = {}
for f in [
    "data/annotations.json",
    "data/round2/expanded-train-dev.json",
    "data/round2/test.json",
    "data/round3/test.json",
]:
    for r in json.loads((ROOT / f).read_text()):
        rows[r["id"]] = r
words = []
rejected = []
for r in rows.values():
    crops = training_words(
        Image.open(ROOT / "data" / r["image"]), r["text"], preserve_height=True
    )
    if not crops:
        rejected.append(r["id"])
    for i, (im, text) in enumerate(crops):
        a = np.array(im.convert("L"))
        bg = cv2.GaussianBlur(a, (0, 0), 9)
        ink = np.clip((bg.astype(float) - a) / 110, 0, 1)
        ys, xs = np.where(ink > 0.25)
        if len(xs) < 20:
            continue
        ink = ink[
            max(0, ys.min() - 2) : ys.max() + 3, max(0, xs.min() - 2) : xs.max() + 3
        ]
        ar = ink.shape[1] / ink.shape[0]
        norm = cv2.resize(ink, (96, 40))
        words.append(
            {
                "id": r["id"] + f"-w{i + 1}",
                "text": text,
                "image": im,
                "norm": norm,
                "ar": ar,
            }
        )
pairs = []
for i, a in enumerate(words):
    for j in range(i):
        b = words[j]
        if a["text"] == b["text"] or abs(np.log(a["ar"] / b["ar"])) > 0.12:
            continue
        mse = float(np.mean((a["norm"] - b["norm"]) ** 2))
        pairs.append((mse, i, j))
pairs.sort()
pairs = pairs[:24]
sheet = Image.new("RGB", (1450, len(pairs) * 125), "white")
d = ImageDraw.Draw(sheet)
report = []
for k, (distance, i, j) in enumerate(pairs):
    a, b = words[i], words[j]
    d.text((10, k * 125 + 3), f"{distance:.4f} {a['id']} / {b['id']}", fill="black")
    for x, w in [(10, a), (710, b)]:
        im = w["image"].copy()
        im.thumbnail((670, 85))
        im = im.resize((min(670, im.width * 2), min(85, im.height * 2)))
        sheet.paste(im, (x, k * 125 + 25))
    report.append(
        {
            "distance": distance,
            "first": {v: a[v] for v in ["id", "text"]},
            "second": {v: b[v] for v in ["id", "text"]},
        }
    )
sheet.save(OUT / "label-consistency-candidates.png")
train = [r for r in rows.values() if r["id"].startswith("p17r")]
counts = collections.Counter("".join(r["text"] for r in train))
coverage = []
for r in rows.values():
    unseen = collections.Counter(c for c in r["text"] if counts[c] == 0)
    rare = collections.Counter(c for c in r["text"] if counts[c] < 3)
    coverage.append(
        {
            "id": r["id"],
            "characters": len(r["text"]),
            "unseen": dict(unseen),
            "rare_less_than_three": dict(rare),
        }
    )
report = {
    "method": "Pixel-normalized visual similarity only; candidates for human visual audit, not automatic reference edits. No model predictions read.",
    "line_count": len(rows),
    "word_count": len(words),
    "separator_rejected_lines": rejected,
    "candidate_pairs": report,
    "training_counts": dict(counts),
    "coverage": coverage,
}
(OUT / "dataset-audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print("lines", len(rows), "words", len(words), "rejected", rejected)
print("unseen", [(r["id"], r["unseen"]) for r in coverage if r["unseen"]])
