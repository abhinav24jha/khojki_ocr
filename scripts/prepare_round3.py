"""Failure-driven data expansion; preserve earlier test outcomes explicitly."""

from pathlib import Path
import json, hashlib, datetime
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/round3"
OUT.mkdir(exist_ok=True)
rows = json.loads((ROOT / "data/round2/train-dev.json").read_text())
oldtest = json.loads((ROOT / "data/round2/test.json").read_text())
for row in oldtest:
    row["former_split"] = "round2_test"
    row["split"] = "train" if row["id"] in ["p19r-06", "p19r-08", "p19r-09"] else "dev"
    rows.append(row)
extra = json.loads((ROOT / "data/round2/expanded-train-dev.json").read_text())
rows.extend(r for r in extra if r["id"] in ["p18r-05", "p18r-09"])
p = OUT / "train-dev.json"
p.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
ids = [f"p19r-{i:02d}" for i in range(11, 16)]
(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": "Round2 fresh evaluation exposed BHA/BA and related novel-form errors. Three former test lines become training; two become development. Two previously annotated p18 lines add BBA+I and JA+U forms.",
            "train_ids": [r["id"] for r in rows if r["split"] == "train"],
            "dev_ids": [r["id"] for r in rows if r["split"] == "dev"],
            "new_test_ids": ids,
            "new_test_status": "Selected before round3 training. References must be frozen before reading model predictions. These source images were included in previous whole-page inference but their unscored predictions have not been inspected. Same book/page, not book-disjoint.",
            "training_annotation_caveat": "All labels are non-expert. Earlier ambiguities remain recorded; no reference edits to match predictions.",
            "train_dev_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
sheet = Image.new("RGB", (1500, 5 * 170), "white")
d = ImageDraw.Draw(sheet)
for i, id in enumerate(ids):
    im = Image.open(ROOT / f"data/lines/{id}.png").convert("RGB")
    im = im.resize((im.width * 2, im.height * 2))
    sheet.paste(im, (10, i * 170 + 25))
    d.text((10, i * 170 + 3), id, fill="black")
sheet.save(OUT / "new-test-source.png")
print(
    {
        split: (
            sum(r["split"] == split for r in rows),
            sum(len(r["text"]) for r in rows if r["split"] == split),
        )
        for split in ["train", "dev"]
    }
)
