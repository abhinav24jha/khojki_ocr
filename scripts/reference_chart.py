from pathlib import Path
import unicodedata
from PIL import Image, ImageDraw, ImageFont
from khojki_ocr.render import Renderer

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data/reference"
out.mkdir(parents=True, exist_ok=True)
latin = ImageFont.load_default(size=13)
for family in ["Sans", "Serif"]:
    r = Renderer(ROOT / f"assets/Noto{family}Khojki-Regular.ttf", 48)
    cps = [
        cp for cp in range(0x11200, 0x11242) if unicodedata.name(chr(cp), "")
    ] + list(range(0xAE6, 0xAF0))
    sheet = Image.new("RGB", (1000, ((len(cps) + 7) // 8) * 100), "white")
    d = ImageDraw.Draw(sheet)
    for i, cp in enumerate(cps):
        x, y = (i % 8) * 125, (i // 8) * 100
        try:
            glyph = r.render(chr(cp))
            glyph.thumbnail((110, 64))
            sheet.paste(glyph, (x + 5, y + 4))
        except ValueError:
            d.text((x + 5, y + 10), "UNSUPPORTED", font=latin, fill="red")
        d.text((x + 5, y + 69), f"{cp:04X}", font=latin, fill="black")
        name = (
            unicodedata.name(chr(cp))
            .replace("KHOJKI ", "")
            .replace("LETTER ", "")
            .replace("VOWEL SIGN ", "VS ")
            .replace("GUJARATI DIGIT ", "DIGIT ")
        )
        d.text((x + 5, y + 84), name, font=latin, fill="black")
    sheet.save(out / f"{family.lower()}-chart.png")
