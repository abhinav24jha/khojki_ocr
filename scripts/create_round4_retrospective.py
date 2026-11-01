"""One-page round-four retrospective, based solely on saved measured artifacts."""

from pathlib import Path
import json
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    FrameBreak,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def load(n):
    return json.loads((ROOT / n).read_text())


r = load("results/round4-report.json")
v = load("results/round4-verification.json")
shutdown = load("cloud/round4-shutdown.json")
f = r["fresh"]
seconds = v["cpu_page"]["processing_seconds"]
style = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=9.3,
    leading=11.8,
    textColor=HexColor("#253442"),
    spaceAfter=7,
)
heading = ParagraphStyle(
    "heading",
    parent=style,
    fontName="Helvetica-Bold",
    fontSize=10.2,
    leading=12.6,
    spaceBefore=3,
    spaceAfter=4,
    textColor=HexColor("#145c63"),
)
left = [
    (
        "Pipeline and honest outcome",
        "Page/PDF -> background-relative ink and frame removal -> lines and number-column chunks -> two 1.06M CNN-BiLSTM-CTC recognizers -> structural beam decoding -> UTF-8 and offline correction UI. HarfBuzz/FreeType shape Noto synthetic data; selected weights adapt on 15 real printed lines. Inference needs no API or retraining.",
    ),
    (
        "",
        f"<b>New frozen test: {f['automatic_errors']}/120 edits = {f['automatic_cer']:.2%} CER on five real page-21 lines.</b> The previous model has the same result: nine additional training runs did not beat it on development, so it was retained. Development remains 14/405 (3.46%), an optimistic selection score. All five source/chart references were frozen before predictions; two uncertain readings were declared in advance (both alternatives leave 17.50%). Related print and non-expert references limit generalization claims.",
    ),
    (
        "Errors by stage",
        f"<b>{f['missed_lines']} missed selected lines; {r['space_involving_edits']} space-involving edits; {r['other_edits']} other recognition edits; {f['invalid_inventory_characters']} invalid code points / {f['utf8_failures']} UTF-8 failures.</b> Automatic and checked crops yield identical predictions. Valid-but-wrong letters evade encoding validation. These are operational categories, not a uniquely identified causal allocation; unlabelled regions are unscored. Three alignments pair inserted PA with E-to-I substitution; E has no real training example.",
    ),
    (
        "What failed, with numbers",
        "Initial synthetic CER was 1.31%, but real-line CER was 48%; contrast/chunking reduced it to 29.33%. This round, six matched 600-step pilots gave 16-17/405 edits versus incumbent 14. A confirmed RGB-padding bug created red/dark borders: both padding variants gave 16 edits at 600 steps; at 3000, old artifact 14 versus fixed white 16. Correct image handling did not imply an accuracy gain. Slower 1000-step original/corrected-label/10%-word runs gave 16/18/18; none qualified for 5000. Five contrast/blur variants gave 15-18. Full histories and failed runs are retained.",
    ),
]
right = [
    (
        "Encoding decisions",
        "Dependent vowels: U+1122C-U+11233; anusvara U+11234; independent AA U+11201. O/AU signs are atomic U+11232/U+11233. Word separator: U+1123A; printed digits: Gujarati U+0AE6-U+0AEF. ASCII space marks the distant number column. CER counts Unicode scalars, including signs and digits. Inventory, orphan/adjacent signs, atomic vowels, NFC changes and UTF-8 round trips are checked; raw text is not silently repaired.",
    ),
    (
        "AI error caught in our own labels",
        "Image-only repeated-word matching plus enlarged scan/chart comparisons exposed likely errors in earlier AI-authored references (for example, U versus II and BA versus NA). Four proposed word revisions change the same old output from <b>26/123 to 15/128 edits</b>. This is a post-hoc reference effect, <b>not better OCR</b>; originals remain frozen and expert confirmation is needed. One training-only sign correction was tested separately and did not improve development. Earlier VLM probes also returned Kannada/Devanagari (2/2 wrong-script failures).",
    ),
    (
        "Single biggest bottleneck",
        "Obtain <b>200-500 independently adjudicated printed lines</b> spanning books, fonts, damage and rare signs; double-annotate a subset and measure a 50/100/200/500-line curve with book-disjoint tests. This is a planning estimate. Thirty-four supported Khojki characters have no examples in the 15 real training lines. Modern-font rendering cannot adjudicate every historical form. Initial 31-line authoring/freezing took about 14 minutes plus chart study and review; this is AI-assisted elapsed time, not expert productivity.",
    ),
    (
        "Scale, resources and handwriting",
        f"<b>{seconds:.2f} s per 16-line page</b> on a two-thread Colab CPU, plus {v['cpu_run']['model_load_seconds']:.3f} s loading; CPU/CUDA text matched. Roughly {47 * seconds:.0f} s for 47 comparable printed pages, excluding startup/PDF rendering and review. No equivalent quality is claimed for the mixed collection. T4 was about 1.15 units/hour; setup/idle time dominated. All archives verified; {shutdown['remaining_units']:.2f} units remain, zero sessions. Handwriting needs baseline/polygon detection, writer-disjoint evaluation and perhaps 1,000-3,000 diverse labelled lines initially; current handwriting output is unvalidated.",
    ),
]


def decorate(c, doc):
    c.setFillColor(HexColor("#173746"))
    c.setFont("Helvetica-Bold", 21)
    c.drawString(36, 756, "Khojki OCR | Technical retrospective")
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#586975"))
    c.drawString(
        36,
        738,
        "Round four - source audit, controlled experiments and frozen evaluation - 31 October 2026",
    )
    c.setFillColor(HexColor("#e9f1ef"))
    c.roundRect(36, 681, 540, 43, 6, fill=1, stroke=0)
    for x, label, value in [
        (48, "NEW FROZEN TEST CER", f"{f['automatic_cer']:.2%}"),
        (240, "ADDITIONAL TRAINING RUNS", "9"),
        (430, "CPU PAGE TIME", f"{seconds:.2f} s"),
    ]:
        c.setFillColor(HexColor("#4b6870"))
        c.setFont("Helvetica", 7.3)
        c.drawString(x, 710, label)
        c.setFillColor(HexColor("#163f46"))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, 691, value)
    c.setStrokeColor(HexColor("#d4dddf"))
    c.line(36, 35, 576, 35)
    c.setFont("Helvetica", 7.6)
    c.setFillColor(HexColor("#566772"))
    c.drawString(
        36,
        23,
        "Evidence: RESULTS.md, frozen references, source-label audit and complete experiment histories.",
    )
    c.drawRightString(576, 23, "1 / 1")


def main():
    output = ROOT / "technical-retro.pdf"
    doc = BaseDocTemplate(
        str(output),
        pagesize=(612, 792),
        title="Khojki OCR - Technical retrospective",
        author="Khojki OCR take-home",
    )
    doc.addPageTemplates(
        PageTemplate(
            id="two",
            frames=[
                Frame(
                    36,
                    45,
                    260,
                    620,
                    leftPadding=0,
                    rightPadding=0,
                    topPadding=0,
                    bottomPadding=0,
                ),
                Frame(
                    316,
                    45,
                    260,
                    620,
                    leftPadding=0,
                    rightPadding=0,
                    topPadding=0,
                    bottomPadding=0,
                ),
            ],
            onPage=decorate,
        )
    )
    story = []
    for column in [left, right]:
        if story:
            story.append(FrameBreak())
        for title, text in column:
            if title:
                story.append(Paragraph(title, heading))
            story.append(Paragraph(text, style))
    doc.build(story)
    assert len(PdfReader(output).pages) == 1
    print(output)


if __name__ == "__main__":
    main()
