"""Page -> detected lines -> raw Unicode -> auditable offline review artifacts."""

from pathlib import Path
import html
import math
import json
import shutil
import time
from PIL import Image, ImageOps
from .segment import detect_lines, overlay
from .unicode import inspect_text
from .resources import review_font, resource_path

MAX_PAGE_PIXELS = 30_000_000


def page_images(source, pages=None, native=False):
    """Yield one detached RGB page at a time, releasing PDF native resources.

    PDF pages use one-based indexing. Native mode requires a scan consisting
    of exactly one embedded image; otherwise the page is rendered at 150 DPI.
    """
    source = Path(source)
    if pages is not None:
        pages = list(pages)
        if not pages or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in pages
        ):
            raise ValueError("Pages must be nonempty positive one-based page numbers")
    if source.suffix.lower() != ".pdf":
        if pages is not None and any(n != 1 for n in pages):
            raise ValueError("A page image contains only page 1")
        with Image.open(source) as im:
            if im.width * im.height > MAX_PAGE_PIXELS:
                raise ValueError(
                    "Page exceeds 30 megapixels; downsample before processing."
                )
            yield 1, ImageOps.exif_transpose(im).convert("RGB")
        return
    if native:
        from pypdf import PdfReader

        with source.open("rb") as stream:
            pdf = PdfReader(stream)
            for n in pages or range(1, len(pdf.pages) + 1):
                if not 1 <= n <= len(pdf.pages):
                    raise ValueError(f"Page {n} outside PDF")
                images = list(pdf.pages[n - 1].images)
                if len(images) != 1:
                    raise ValueError(
                        "Native mode requires exactly one embedded image per page; omit --pdf-native to render."
                    )
                image = images[0].image
                if image.width * image.height > MAX_PAGE_PIXELS:
                    raise ValueError(
                        "Page exceeds 30 megapixels; downsample before processing."
                    )
                yield n, image.convert("RGB")
    else:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(source)
        try:
            for n in pages or range(1, len(pdf) + 1):
                if not 1 <= n <= len(pdf):
                    raise ValueError(f"Page {n} outside PDF")
                page = pdf[n - 1]
                try:
                    width, height = page.get_size()
                    if (
                        math.ceil(width * 150 / 72) * math.ceil(height * 150 / 72)
                        > MAX_PAGE_PIXELS
                    ):
                        raise ValueError(
                            "Page exceeds 30 megapixels at 150 DPI; downsample before processing."
                        )
                    bitmap = page.render(scale=150 / 72)
                    try:
                        yield n, bitmap.to_pil().convert("RGB")
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            pdf.close()


def process_page(image, recognizer, output, prefix="page", region=None):
    if not prefix or prefix in {".", ".."} or "/" in prefix or "\\" in prefix:
        raise ValueError("Prefix must be a simple filename component")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if image.width * image.height > MAX_PAGE_PIXELS:
        raise ValueError("Page exceeds 30 megapixels; downsample before processing.")
    start = time.perf_counter()
    layout = getattr(recognizer, "layout", "legacy")
    lines, roi = detect_lines(
        image,
        prefix,
        region=region,
        mode="robust" if layout == "robust" else "adaptive",
    )
    detection_seconds = time.perf_counter() - start
    overlay(image, lines).save(output / "overlay.png")
    result = []
    for line in lines:
        crop = image.crop(line.box)
        crop.save(output / f"{line.id}.png")
        t = time.perf_counter()
        text = recognizer(crop)
        elapsed = time.perf_counter() - t
        result.append(
            {
                "id": line.id,
                "box": line.box,
                "crop": f"{line.id}.png",
                "text": text,
                "validation": inspect_text(text),
                "recognition_seconds": elapsed,
                "review_required": True,
            }
        )
    text = "\n".join(r["text"] for r in result) + ("\n" if result else "")
    (output / "transcription.txt").write_text(text, encoding="utf-8")
    report = {
        "image_size": list(image.size),
        "region": roi,
        "manual_region": region is not None,
        "layout": layout,
        "line_count": len(lines),
        "detection_seconds": detection_seconds,
        "recognition_seconds": sum(r["recognition_seconds"] for r in result),
        "processing_seconds": time.perf_counter() - start,
        "scope": "Prototype for printed Khojki. All lines require review; handwriting is unvalidated.",
        "lines": result,
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_review(report, output)
    return report


def write_review(report, output):
    """Self-contained local UI; corrections export separately and never change raw OCR."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = review_font()
    shutil.copyfile(font, output / "NotoSansKhojki-Regular.ttf")
    shutil.copyfile(resource_path("assets", "OFL.txt"), output / "OFL.txt")
    rows = []
    for row in report["lines"]:
        issues = row["validation"]["issues"]
        names = " | ".join(
            c["codepoint"] + " " + c["name"] for c in row["validation"]["codepoints"]
        )
        rows.append(
            f"""<article><h2>{html.escape(row['id'])}</h2>
<img src="{html.escape(row['crop'])}" alt="Source scan of {html.escape(row['id'])}">
<label>Raw OCR</label><p class="khojki">{html.escape(row['text'])}</p>
<label>Reviewed transcription <textarea data-id="{html.escape(row['id'])}" class="khojki" spellcheck="false">{html.escape(row['text'])}</textarea></label>
<p>{html.escape(json.dumps(issues,ensure_ascii=False)) if issues else 'No encoding warning. This does not establish visual correctness.'}</p>
<details><summary>Unicode identities</summary><p>{html.escape(names)}</p></details></article>"""
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Khojki OCR review</title>
<style>@font-face{font-family:Khojki;src:url(NotoSansKhojki-Regular.ttf)}body{max-width:1000px;margin:36px auto;padding:0 20px;font:16px/1.5 system-ui;color:#202b37;background:#f5f4ef}article{background:white;border:1px solid #ddd;padding:20px;margin:20px 0}img{max-width:100%;display:block}.khojki{font-family:Khojki,serif;font-size:28px;line-height:1.8}textarea{display:block;width:98%;min-height:75px}button{padding:12px;background:#184e50;color:white;border:0;border-radius:6px}summary{cursor:pointer}label{font-weight:600}details p{font-size:12px;overflow-wrap:anywhere}</style>
<h1>Khojki OCR review</h1><p>Compare every line with the scan. Encoding checks cannot detect a plausible but wrong letter. Changes remain in this browser until exported; raw OCR files stay unchanged.</p>
<p><a href="overlay.png">View detected lines on the page</a> · <a href="transcription.txt">Raw UTF-8 text</a> · <a href="result.json">Detection and encoding details</a></p>
<button id="export">Export reviewed transcriptions</button>"""
        + "".join(rows)
        + """
<script>document.getElementById('export').onclick=()=>{const rows=Array.from(document.querySelectorAll('textarea')).map(e=>({id:e.dataset.id,text:e.value}));const u=URL.createObjectURL(new Blob([JSON.stringify({type:'human_review',lines:rows},null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='reviewed-transcriptions.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>"""
    )
    (output / "review.html").write_text(document, encoding="utf-8")
