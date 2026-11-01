"""Source/chart transcription before any page-21 OCR output is inspected.

Alternatives are declared from glyph ambiguity, never from model output.
These are non-expert references, NOT expert-certified ground truth.
"""

from pathlib import Path
import json, hashlib, datetime
from PIL import Image, ImageDraw
from write_annotations import encode
from khojki_ocr.render import Renderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/round4"
SPECS = [
    "aa ddda : pa.u ra sa ka.aa : e ha ta.ii ja : pa.e la.aa : SPACE 1",
    "ta.ii na : ddda.e va : e ka : ma.u ra ta : ba.o la.aa : SPACE 2",
    "sa ta ga.u ra : ma.ii ha.e : ra e aa : a ka.e la.aa : SPACE 3",
    "bba.i ta : u ttha : ka ra.e : ha ra kha.u.an : ja.e la.aa : SPACE 4",
    "ja.o tha.ii : ma e aa : ma.ii ha.e : bba.ii cha.ii : a na.e kha.ii : SPACE 5",
]
ALTS = {
    3: [SPECS[3].replace("ja.e la.aa", "ddda.e la.aa")],
    4: [SPECS[4].replace("cha.ii", "ha.ii")],
}
NOTES = {
    3: "Last word initial base JA versus DDDA is uncertain in the scan; primary JA selected from lower hooked outline and right stem.",
    4: "Fourth word second base CHA versus HA is uncertain; primary CHA selected from its upper loop. Both alternatives frozen before recognition.",
}


def main():
    if (OUT / "test-freeze.json").exists():
        raise RuntimeError(
            "Already frozen; do not overwrite after viewing predictions."
        )
    rows = json.loads((OUT / "fresh-selection.json").read_text())["rows"]
    ren = Renderer(ROOT / "assets/NotoSansKhojki-Regular.ttf", 90)
    for i, (row, spec) in enumerate(zip(rows, SPECS)):
        row.update(
            text=encode(spec),
            annotation_tokens=spec,
            annotation_method="AI assistant visual source/chart transcription before any OCR outputs for this page; non-expert",
            uncertainty=NOTES.get(i, ""),
            alternate_references=[encode(s) for s in ALTS.get(i, [])],
        )
        im = Image.open(ROOT / "data" / row["image"])
        im = im.resize((im.width * 3, im.height * 3))
        render = ren.render(row["text"])
        sheet = Image.new(
            "RGB",
            (max(im.width, render.width) + 20, im.height + render.height + 65),
            "white",
        )
        sheet.paste(im, (10, 10))
        sheet.paste(render, (10, im.height + 25))
        ImageDraw.Draw(sheet).text(
            (10, sheet.height - 20), row["id"] + " " + spec, fill="black"
        )
        sheet.save(OUT / f"frozen-{row['id']}.png")
    dest = OUT / "test.json"
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    frozen = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "annotation_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "prediction_seen": False,
        "selection_policy": "First five detected body lines selected before round4 training outputs; all retained, including uncertainty.",
        "reference_status": "Non-expert, source/chart based; expert validation required. Do not equate CER with certified accuracy.",
        "characters": sum(len(r["text"]) for r in rows),
        "uncertain_lines": [r["id"] for r in rows if r["uncertainty"]],
        "image_sha256": {
            r["image"]: hashlib.sha256(
                (ROOT / "data" / r["image"]).read_bytes()
            ).hexdigest()
            for r in rows
        },
    }
    (OUT / "test-freeze.json").write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
