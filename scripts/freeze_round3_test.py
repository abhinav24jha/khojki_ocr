"""Freeze additional numbered body-line references before prediction inspection."""

from pathlib import Path
import json, hashlib, datetime
from PIL import Image, ImageDraw
from write_annotations import encode
from khojki_ocr.render import Renderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/round3"
specs = {
    "p20r-09": "sa ta ga.u ra : ka pa.e aa : ta ba : a ma ra : ga.u aa ba : SPACE 10",
    "p20r-10": "aa pa nna.ii : ma.aa pa.i aa : aa pa nna.e : ha.aa ya : bha ba : SPACE 11",
    "p20r-11": "ja.u ga ta : ba.aa ta.ii : sa ta ga.u ra ka.e : cha ba : SPACE 12",
    "p18r-14": "u pa.e : sa ta.aa ra.aa : a ga.o ca ra : pa.e nya i : SPACE 6",
    "p18r-15": "ta ba ta.o : pha.i ra sa ta.aa : ka ra ta.aa : ka.aa ma.aa : SPACE 7",
}
alts = {
    "p20r-09": [("a ma ra", "a ta ra")],
    "p20r-10": [
        ("ma.aa pa.i", "ta.aa pa.i"),
        ("ha.aa ya", "ha.aa ya.u"),
        ("bha ba", "ma ba"),
        ("bha ba", "bba ba"),
    ],
    "p20r-11": [("ra ka.e", "ra cha.e"), ("ra ka.e", "ra ca.e"), ("cha ba", "bha ba")],
    "p18r-14": [("u pa.e", "u tha.e"), ("pa.e nya i", "tha.e nya i")],
    "p18r-15": [("ka.aa ma.aa", "ka.aa ta.aa")],
}
lookup = {
    r["id"]: r for r in json.loads((ROOT / "data/protocol.json").read_text())["lines"]
}
rows = []
render = Renderer(ROOT / "assets/NotoSerifKhojki-Bold.ttf", 46)
for id, spec in specs.items():
    row = lookup[id].copy()
    row.update(
        split="test",
        text=encode(spec),
        annotation_tokens=spec,
        annotation_method="Non-expert visual chart transcription, not copied from model output; historical forms need expert adjudication",
        uncertain_readings=[
            {
                "primary_tokens": a,
                "alternative_tokens": b,
                "alternative_text": encode(spec.replace(a, b)),
                "reason": "Visually ambiguous historical printed form relative to modern chart",
            }
            for a, b in alts.get(id, [])
        ],
    )
    rows.append(row)
    im = Image.open(ROOT / "data" / row["image"]).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2))
    ref = render.render(row["text"])
    sheet = Image.new(
        "RGB", (max(im.width, ref.width) + 20, im.height + ref.height + 55), "white"
    )
    sheet.paste(im, (10, 20))
    sheet.paste(ref, (10, im.height + 45))
    ImageDraw.Draw(sheet).text(
        (10, 0), id + " source / primary reference", fill="black"
    )
    sheet.save(OUT / f"reference-{id}.png")
p = OUT / "test.json"
if p.exists():
    raise RuntimeError("Do not overwrite frozen references")
p.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
(OUT / "test-freeze.json").write_text(
    json.dumps(
        {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "lines": len(rows),
            "characters": sum(len(r["text"]) for r in rows),
            "status": "Frozen before round3 training and before inspecting recognizer predictions for these lines. Same-book, same-page relationships exist; not independent-book accuracy.",
            "limitation": "Non-expert labels with explicit ambiguous readings. This set is selected for numbered body-line structure; footer/refrain abbreviations were excluded before prediction inspection because their mapping was unresolved.",
        },
        indent=2,
    )
)
p = OUT / "protocol.json"
protocol = json.loads(p.read_text())
protocol["initial_candidate_ids"] = protocol["new_test_ids"]
protocol["new_test_ids"] = list(specs)
protocol["selection_revision_reason"] = (
    "Before training or prediction inspection: p19 footer/refrain forms had unresolved chart identity. Use three numbered p20 body lines and two numbered p18 body lines. This selection limits generalization claims."
)
p.write_text(json.dumps(protocol, indent=2))
print(len(rows), sum(len(r["text"]) for r in rows))
