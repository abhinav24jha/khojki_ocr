"""Readable annotation source. These are visual, non-expert transcriptions.

Do not silently alter these in response to a recognizer's test predictions.
Any subsequent reference correction must have image-grounded rationale and
be recorded in data/annotation-revisions.json.
"""

from pathlib import Path
import json, hashlib, datetime
from khojki_ocr.render import Renderer
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
base_names = [
    "a",
    "aa",
    "i",
    "u",
    "e",
    "ai",
    "o",
    "au",
    "ka",
    "kha",
    "ga",
    "gga",
    "gha",
    "nga",
    "ca",
    "cha",
    "ja",
    "jja",
    None,
    "nya",
    "tta",
    "ttha",
    "dda",
    "ddha",
    "nna",
    "ta",
    "tha",
    "da",
    "ddda",
    "dha",
    "na",
    "pa",
    "pha",
    "ba",
    "bba",
    "bha",
    "ma",
    "ya",
    "ra",
    "la",
    "va",
    "sa",
    "ha",
    "lla",
]
BASE = {n: chr(0x11200 + i) for i, n in enumerate(base_names) if n}
SIGNS = {
    n: chr(0x1122C + i)
    for i, n in enumerate(
        ["aa", "i", "ii", "u", "e", "ai", "o", "au", "an", "vir", "nu", "sha"]
    )
}


def encode(spec):
    out = ""
    for token in spec.split():
        if token == ":":
            out += "\U0001123a"
        elif token == "SPACE":
            out += " "
        elif token.isdigit():
            out += "".join(chr(0xAE6 + int(c)) for c in token)
        else:
            parts = token.split(".")
            out += BASE[parts[0]] + "".join(SIGNS[x] for x in parts[1:])
    return out


# Created before reading learned-recognizer predictions for these lines.
# The spellings below describe glyph identity, not phonetic transliteration.
SPECS = {
    "p17r-01": "pa.e ha.e la.aa : sa.u pa.ii ja na : na.i ra.an ja na : ha.o ma.aa : SPACE 1",
    "p17r-02": "aa ddda : a.an ta : ma.ii ha.e : ta.e ja : ja : ha.o ma.aa : SPACE 2",
    "p17r-03": "ta ba : ja.aa : ta.e ja : bba.i.an bba : u pa.aa pa.i aa : SPACE 3",
    "p17r-04": "sa.aa ddda : sa bba ddda : sa ba ma.ii ha.e : sa ma.aa pa.i aa : SPACE 4",
    "p17r-05": "aa ddda : a.an ta : ha.o ma.aa : a ka.e la.aa : SPACE 5",
    "p17r-06": "ta ba : ta.o : sa ta ga.u ra : sa.u pa.ii ja na : ka.e : ba.e la.aa : SPACE 6",
    "p17r-07": "pa.i sa : pa.e la : ja.e : na ra : ka.e ra.aa : SPACE 7",
    "p17r-08": "ta ba : ta.o : sa ta ga.u ra : na.e : sa ta.aa ra.aa : SPACE 8",
    "p17r-09": "na.an ha.ii.an : ta ba : pa.ii ta : na.an ha.ii.an : ta ba : ra ma.aa : SPACE 9",
    "p17r-10": "na.an ha.ii.an : ta ba : na.aa ma : na.an ha.ii.an : ta ba.i : ja.aa ma.aa : SPACE 10",
    "p17r-11": "pa.e ha.e la.aa : a pa na.ii : kha.i.an ddha : u pa.aa pa.i aa : SPACE 11",
    "p17r-12": "sa.o : ta na : ma.ii ha.e.an pa.ii : na.u ra : na.i pa.aa pa.i aa : SPACE 12",
    "p17r-13": "sa.o : i : na.u ra ka.aa : sa ta ga.u ra : ha.u aa : SPACE 13",
    "p17r-14": "ta ba : pa.ii cha.e : pha.i ra : ra ca na.aa : ka.ii aa : SPACE 14",
    "p17r-15": "sa.aa pa.ii : ha.i ka ma ta : sa ta ga.u ra na.e : ka.ii ta.ii : SPACE 15",
    "p18r-01": "u pa na.ii : pha.i ra sa ta.aa : ta ba : ba.aa ta.ii : ba.ii ta.ii : SPACE 16",
    "p18r-02": "aa ga.o : aa da ma : pha.i ra sa ta.aa : sa.aa pa.ii : SPACE 17",
    "p18r-03": "sa.o : sa ba : ha.u pa.i : a pa nna.ii : ja.aa ta.ii : SPACE 18",
    "p19r-01": "ta ba : ta.o : sa ta ga.u ra : a ga.o ca ra : ha.e ta.e : SPACE 9",
    "p19r-02": "u sa ha.ii : ta.e ja : ma.ii ha.e : pha.i ra sa ta.e : na : pa.o ta.e : SPACE 10",
    "p19r-03": "pa.ii cha.e : ra ca na.aa : ra ca.ii : a pa.aa ra : SPACE 11",
    "p19r-04": "kha.i.an ddha ma.ii ha.e : u pa na.aa : sa.u pa.ii ja na ha.aa ra : SPACE 12",
    "p19r-05": "ta ba ha.ii : ra ca.e aa : dha.an dha.u ka.aa ra : SPACE 13",
    "p20r-01": "pa.i sa.aa : sa ta ga.u ra : ta ma.e : pa.aa pa.i ya.aa : na.i sa.e sa.o : SPACE 2",
    "p20r-02": "dha dha.u.an ka.aa ra : ma.ii ha.e : dha ra.e aa : dha.ii aa na : SPACE 3",
    "p20r-03": "sa ta ga.u ra : na.e : ta ba : ka pa.e aa : ga.ii aa na : SPACE 4",
    "p20r-04": "na.i ra.an ja na : ra ca na.aa : sa ba ha.ii : bba.ii dha.ii : SPACE 5",
    "p20r-05": "sa ta ka.ii : ddda.o pa.ii : sa ta ga.u ra na.e : sa.ii dha.ii : SPACE 6",
    "p20r-06": "aa pa.e.an : pa.e la : ra ca na.aa ka.aa : ka.ii ma.aa : SPACE 7",
    "p20r-07": "sa ta pa.an tha : ma.aa ra ga : sa ta ga.u ra ka.u.an : da.ii ma.aa : SPACE 8",
    "p20r-08": "kha.i.an ddha : ba.i ra ma.an ddha : ra ca.e aa : aa sa ma.aa na : SPACE 9",
}

if __name__ == "__main__":
    protocol = json.loads((ROOT / "data/protocol.json").read_text())
    records = []
    render = Renderer(ROOT / "assets/NotoSansKhojki-Regular.ttf", 44)
    for row in protocol["lines"]:
        if row["id"] not in SPECS:
            continue
        text = encode(SPECS[row["id"]])
        records.append(
            row
            | {
                "text": text,
                "annotation_tokens": SPECS[row["id"]],
                "annotation_method": "AI assistant visual transcription against Unicode reference; non-expert; not model output",
                "uncertain_positions": [],
            }
        )
        original = Image.open(ROOT / "data" / row["image"])
        original = original.resize((original.width * 2, original.height * 2))
        rendered = render.render(text)
        sheet = Image.new(
            "RGB",
            (
                max(original.width, rendered.width) + 20,
                original.height + rendered.height + 55,
            ),
            "white",
        )
        d = ImageDraw.Draw(sheet)
        sheet.paste(original, (10, 20))
        sheet.paste(rendered, (10, original.height + 45))
        d.text((10, 0), row["id"] + " source / reference rendering", fill="black")
        sheet.save(ROOT / "data/reference" / f"review-{row['id']}.png")
    dest = ROOT / "data/annotations.json"
    dest.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(len(records), "lines", sum(len(r["text"]) for r in records), "code points")
