"""Generate shareable tables and a reference comparison from saved results only."""

from pathlib import Path
import csv, html, json, hashlib, unicodedata, datetime, collections

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ev = json.loads((R / "test-pages/evaluation.json").read_text())
rows = ev["lines"]
with (R / "test-transcriptions.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "id",
            "page",
            "reference",
            "prediction",
            "substitutions",
            "deletions",
            "insertions",
            "reference_codepoints",
            "CER",
        ]
    )
    for row in rows:
        s = row["automatic"]
        writer.writerow(
            [
                row["id"],
                row["page"],
                row["reference"],
                row["automatic_prediction"],
                s["substitute"],
                s["delete"],
                s["insert"],
                s["reference_characters"],
                s["cer"],
            ]
        )
cards = []
for row in rows:
    s = row["automatic"]
    crop = f"test-pages/p{row['page']:02d}/{row['detected_id']}.png"
    cards.append(f'''<article><h2>{row["id"]} <small>{s["errors"]}/{s["reference_characters"]} errors · {s["cer"]:.1%} CER</small></h2>
<img src="{crop}" alt="Scanned line {row["id"]}"><p class="label">Frozen visual reference</p><p class="khojki">{html.escape(row["reference"])}</p>
<p class="label">Raw automatic prediction</p><p class="khojki">{html.escape(row["automatic_prediction"])}</p>
<details><summary>Reference code points</summary><p>{" ".join(f"U+{ord(c):04X}" for c in row["reference"])}</p></details></article>''')
(R / "test-comparison.html").write_text(
    """<!doctype html><html lang="en"><meta charset="utf-8"><title>Khojki held-out evaluation</title>
<style>@font-face{font-family:Khojki;src:url(../assets/NotoSansKhojki-Regular.ttf)}body{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;background:#f4f4ed;color:#20333c}h1{font-size:38px}h2{font-size:20px}small{float:right;font-weight:400}article{background:white;border:1px solid #ddd;border-radius:8px;padding:22px;margin:24px 0}img{width:100%;image-rendering:auto}.khojki{font:30px/1.8 Khojki,serif;margin:4px 0}.label{color:#506775;font-size:13px;margin-bottom:0}details{font-size:13px}.metric{font-size:25px;font-weight:700}</style>
<h1>Khojki OCR: held-out evaluation</h1><p class="metric">8.4% CER · 21 edits / 250 code points · 10 lines</p>
<p>Model and preprocessing selected on development data before this evaluation. References were visually transcribed with a Unicode chart by an AI assistant; they are not expert-certified. These are selected lines from two related printed pages, not a random sample of the collection. One of ten lines is exact.</p>
<p><a href="test-pages/p19/review.html">Review page 19</a> · <a href="test-pages/p20/review.html">Review page 20</a> · <a href="test-transcriptions.tsv">Download all reference/prediction pairs</a></p>"""
    + "".join(cards)
    + "</html>",
    encoding="utf-8",
)
stage = {
    "automatic_test_errors": 21,
    "line_detection": {
        "missed_selected_lines": 0,
        "net_character_error_change_vs_checked_crops": 0,
        "predictions_changed_by_crop": 2,
    },
    "column_layout": {
        "missing_column_spaces": 2,
        "evidence": "column-audit.json: both affected lines produced one chunk despite two reference columns",
    },
    "recognition": {"remaining_nonspace_edits": 19, "checked_crop_total_edits": 21},
    "unicode_mapping": {
        "out_of_inventory_characters": 0,
        "utf8_roundtrip_failures": 0,
        "multiple_vowel_warning_lines": 1,
    },
    "caveat": "Operational attribution: two space deletions assigned to column layout, nineteen remaining edits to recognition; mapping counts mean mechanical code-point/storage errors only. Equal aggregate checked/automatic CER does not prove segmentation has no effect.",
}
(R / "stage-audit.json").write_text(json.dumps(stage, indent=2))
model = ROOT / "models/printed-khojki.pt"
card = {
    "parameters": 1060946,
    "height": 48,
    "checkpoint_bytes": model.stat().st_size,
    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    "training": "3000 shaped synthetic examples, 1000 steps; fine-tuned on first10 of15 train lines, selected at step300 using development CER.",
    "development": {"lines": 6, "characters": 155, "errors": 10, "cer": 10 / 155},
    "test": ev["summary"],
    "scope": "Related printed pages only. Not validated for handwriting, all historical fonts, rare conjuncts, all punctuation, or the whole Khojki Unicode inventory.",
    "reference_quality": "AI visual transcription with font-chart review; no independent Khojki expert certification.",
    "training_inventory": "Legacy generator: U+11200..1122B excluding11212, U+1122C..11234, U+1123A, Gujarati digits and ASCII space. Other output classes exist but lack positive training examples.",
    "decision": "Retain original curve10. Cloud pilot, prefix beam, and ensemble did not improve development CER.",
}
(ROOT / "models/model-card.json").write_text(json.dumps(card, indent=2))
print("Generated comparison HTML, TSV, model card and stage audit.")
