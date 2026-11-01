"""Small visually checked training expansion. Test references are not consulted."""

from pathlib import Path
import json, hashlib, datetime
from PIL import Image, ImageDraw
from write_annotations import encode
from khojki_ocr.render import Renderer

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "p18r-05": "aa pa : bba.i ra.an ja na : u pa na.ii :",
    "p18r-07": "aa pa sa : ma.ii ha.e.an pa.ii : u pa na.ii :",
    "p18r-09": "ka.e ta.e ka : ja.u ga : ta.o : sa ta.aa ra.aa : ra ha.e aa : SPACE 1",
    "p18r-10": "u pa ka.aa : ma.e da : ma.i va.e : na : la ha.e aa : SPACE 2",
    "p18r-11": "aa pa.e : aa pa : sa.i ra ja nna ha.aa ra.aa : SPACE 3",
}
protocol = json.loads((ROOT / "data/protocol.json").read_text())
lookup = {r["id"]: r for r in protocol["lines"]}
rows = json.loads((ROOT / "data/round2/train-dev.json").read_text())
render = Renderer(ROOT / "assets/NotoSerifKhojki-Bold.ttf", 48)
for id, spec in SPECS.items():
    row = lookup[id].copy()
    row.update(
        split="train",
        text=encode(spec),
        annotation_tokens=spec,
        annotation_method="Non-expert AI visual chart transcription, with earlier OCR predictions used as an annotation aid and then checked visually. Training only.",
        uncertain_positions=[],
    )
    rows.append(row)
    im = Image.open(ROOT / "data" / row["image"]).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2))
    rend = render.render(row["text"])
    sheet = Image.new(
        "RGB", (max(im.width, rend.width) + 20, im.height + rend.height + 60), "white"
    )
    sheet.paste(im, (10, 20))
    sheet.paste(rend, (10, im.height + 50))
    ImageDraw.Draw(sheet).text(
        (10, 0), id + " candidate training source / encoded reference", fill="black"
    )
    sheet.save(ROOT / f"data/round2/new-train-{id}.png")
out = ROOT / "data/round2/expanded-train-dev.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(
    len(rows),
    "rows; five new training lines,",
    sum(len(r["text"]) for r in rows if r["split"] == "train"),
    "training codepoints",
)
