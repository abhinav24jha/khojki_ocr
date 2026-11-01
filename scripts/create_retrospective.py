"""Render the required one-page retrospective from saved evaluation artifacts."""

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
if (ROOT / "results/round4-report.json").exists():
    from create_round4_retrospective import main

    main()
    raise SystemExit(0)
r = json.loads((ROOT / "results/round3-report.json").read_text())
e = json.loads((ROOT / "results/round3-evaluation-complete.json").read_text())
seconds = e["cpu_page"]["pages"][0]["processing_seconds"]
style = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=9.5,
    leading=12.2,
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
        "Pipeline and measured outcome",
        "Page/PDF -> background-relative ink and frame removal -> line peaks -> text/number chunks -> two 1.06M-parameter CNN-BiLSTM-CTC recognizers -> beam search -> UTF-8 and offline review. HarfBuzz/FreeType shape Noto synthetic examples; 15 real printed lines adapt the selected models. Inference needs no API or retraining.",
    ),
    (
        "",
        "<b>Latest frozen test: 26/123 edits, 21.14% CER on five real lines</b>; original baseline: 28/123 (22.76%). The preceding five-line test was 17/118 (14.41%), versus baseline 25/118 (21.19%). Development reached 14/405 (3.46%); the gap is real. Labels are non-expert, chart-transcribed and frozen before prediction inspection. Same-book/page relationships prevent collection-wide claims. Predeclared alternatives on the latest test give 20.16-25.20% CER, not a confidence interval.",
    ),
    (
        "Errors by stage",
        "Latest test: <b>0 missed selected lines; 2 space-involving edits; 24 other recognition edits; 0 invalid code points / UTF-8 failures</b>. Automatic and checked crops give identical predictions. Space edits are layout/recognition symptoms, not a proven causal allocation. Repeated errors include inserted SA and vowel U (two each). The revised detector also separates a short refrain previously merged on page 20. Unlabelled detections are not scored.",
    ),
    (
        "Experiments that failed",
        "Synthetic validation: 1.31% CER, but initial real-line CER: 48%; contrast/column splitting: 29.33%. On 405 development characters: 300-step control 6.91%, extended control 4.44%, 40% word recombination 8.64%, both 64-pixel pilots 8.64%, five extra labels 5.68%. Longer packed recurrence plus ensemble/filter reached 3.46%. A final targeted-data round gave 24/453 and 26/453 edits versus incumbent 20/453; both pilots were rejected. Histories include failed checkpoints, not just minima.",
    ),
]
right = [
    (
        "Encoding choices and validation",
        "Vowel signs use U+1122C-U+11233; anusvara U+11234; independent AA U+11201. O/AU are atomic U+11232/U+11233. Word separators use U+1123A; printed digits use Gujarati U+0AE6-U+0AEF. ASCII space marks a distant number column. CER counts Unicode scalar values, including signs/digits. Inventory, orphan/adjacent signs, atomic vowels, NFC changes and UTF-8 round trips are checked. Structural filtering cannot certify a plausible wrong letter.",
    ),
    (
        "Highest-value improvement",
        "Obtain <b>200-500 independently reviewed printed lines</b> spanning books, typefaces, damage and rare signs; double-annotate a subset, reserve entire books and measure a 50/100/200/500-line curve. This is a planning estimate. Further tuning against a tiny, uncertain development set is weak evidence. Historical TA/MA and NA/VA need adjudication. Some output classes have no positive examples. Initial 31-line authoring/freezing spanned about 14 minutes plus prior chart study and later review; this is not human-expert productivity.",
    ),
    (
        "Scale and handwriting",
        f"The selected model processes a 16-line page in <b>{seconds:.2f} s on a two-thread Colab CPU</b>, plus 0.044 s loading; startup/PDF rendering/review are excluded. Approximately {47 * seconds:.0f} s for 47 comparable printed pages, or {seconds:.2f}N s for N pages. The mixed collection will not share this accuracy. T4 was about 1.15 units/hour; setup/idle time dominated. Exports were hash-verified; final balance 167.31 units, zero active sessions. Handwriting needs baseline/polygon detection, writer-disjoint evaluation and roughly 1,000-3,000 diverse labelled lines as an initial planning range; current handwriting output is unvalidated.",
    ),
    (
        "One AI mistake caught",
        "Gemma3:4b and Gemma4:e2b returned repetitive Kannada/Devanagari instead of Khojki (37 s / 64 s; 2/2 wrong-script/format failures, no valid CER). Code-point inspection and the chart caught this. A render/Unicode verifier can reject wrong-script output but cannot identify every historical glyph; it was not used as an oracle for reinforcement learning.",
    ),
]
output = ROOT / "technical-retro.pdf"


def decorate(c, doc):
    c.setFillColor(HexColor("#173746"))
    c.setFont("Helvetica-Bold", 21)
    c.drawString(36, 756, "Khojki OCR | Technical retrospective")
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#586975"))
    c.drawString(
        36, 738, "Printed-page prototype - measured limitations - 31 October 2026"
    )
    c.setFillColor(HexColor("#e9f1ef"))
    c.roundRect(36, 681, 540, 43, 6, fill=1, stroke=0)
    for x, label, value in [
        (48, "LATEST FROZEN TEST CER", "21.14%"),
        (240, "PRIOR FROZEN TEST CER", "14.41%"),
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
        "Evidence: RESULTS.md, frozen references, visual comparisons and complete experiment histories.",
    )
    c.drawRightString(576, 23, "1 / 1")


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
assert len(PdfReader(output).pages) == 1, (
    "Retrospective exceeds one page; shorten prose."
)
print(output)
