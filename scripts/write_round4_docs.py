"""Regenerate the numerical results ledger; curated documentation is hand maintained."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads((ROOT / p).read_text())


def main():
    r = load("results/round4-report.json")
    v = load("results/round4-verification.json")
    f = r["fresh"]
    old = r["incumbent_fresh"]
    sens = r["predeclared_reference_sensitivity"]
    sec = v["cpu_page"]["processing_seconds"]
    sha = r["checkpoint_sha256"]
    selected = r["selected"]
    shutdown = load("cloud/round4-shutdown.json")
    balance = shutdown["remaining_units"]
    historical = ROOT / "RESULTS_ROUND3.md"
    if not historical.exists():
        shutil.copyfile(ROOT / "RESULTS.md", historical)
    rows = []
    for label, path in [
        ("600-step pilots", "experiments/round4-pilot-selection.json"),
        (
            "After paired 3000-step extensions",
            "experiments/round4-final-selection.json",
        ),
        ("1000-step low-rate pilots", "experiments/round4-followup-pilot.json"),
        (
            "Final development-only selection",
            "experiments/round4-followup-selection.json",
        ),
    ]:
        rows.append(
            "\n### " + label + "\n\n| Candidate | Edits / 405 | CER |\n|---|---:|---:|"
        )
        for e in load(path)["experiments"]:
            rows.append(f"| {e['name']} | {e['errors']} | {e['errors'] / 405:.2%} |")
    audit = "\n".join(
        f"| {a['model']} | {a['original_summary']['automatic_errors']}/{a['original_summary']['characters']} ({a['original_summary']['automatic_cer']:.2%}) | {a['proposed_errors']}/{a['proposed_characters']} ({a['proposed_cer']:.2%}) |"
        for a in r["reference_audit"]
    )
    steps = []
    for name in [
        "r4-control",
        "r4-word-red",
        "r4-word-white",
        "r4-freezebn",
        "r4-word-white-freezebn",
        "r4-freezernn",
        "r4-low-original",
        "r4-low-corrected",
        "r4-low-white",
    ]:
        h = load(f"experiments/{name}/history.json")
        best = min(h, key=lambda e: e["errors"])
        steps.append(
            f"| {name} | {h[-1]['step']} | {best['step']} | {best['errors']} |"
        )
    text = f"""# Round-four results and audit

The selected configuration is **{selected["name"]}**, chosen on the unchanged 16-line / 405-character development set: **{selected["errors"]}/405 ({selected["errors"] / 405:.2%} CER)**. On five newly frozen page-21 lines it produces **{f["automatic_errors"]}/120 ({f["automatic_cer"]:.2%} CER)**, versus **{old["automatic_errors"]}/120 ({old["automatic_cer"]:.2%})** for the previous model. These are non-expert chart transcriptions, not certified ground truth or collection-wide accuracy.

[Visual comparison](results/round4-comparison.html) · [Source-label audit](data/round4/reference-audit.html) · [Full prior experiment ledger](RESULTS_ROUND3.md)

## What the first-principles audit found

**A real augmentation defect:** Pillow scalar `fill=255` on RGB images creates red padding, not white. Grayscale conversion made those borders dark artificial strokes. `training_words` now converts to grayscale before padding; a regression test checks RGB input and white output margins. A legacy flag reproduces the old behavior for a matched comparison. Both variants reached 16 development edits after 600 steps; through 3000 steps the old artifact reached 14 and the fixed version 16. The fix is correct image handling, but this experiment does **not** establish an accuracy benefit. The default data-generation path is fixed; old trained candidates and their provenance are retained.

**Our AI-authored references contained likely mistakes:** image-only similarity screening found inconsistent labels on near-identical words. Enlarged scan/font comparisons support four historical-test word revisions and one training-only revision (p17r-10, removing an unsupported I sign). Originals remain frozen. Proposed corrections have explicit source evidence and are not expert-certified. They were not silently substituted into the headline test results.

| Same model, different reference version | Original historical test | Proposed source-audited reference |
|---|---:|---:|
{audit}

This table separates label effects from model effects. The incumbent's proposed-score reduction from 26/123 to 15/128 is **not an OCR improvement**. The changes were post-hoc, after earlier model outputs had been seen, so the revised score is diagnostic rather than a fresh unbiased estimate. Unresolved p18r-14 and p19r-06 readings remain unresolved. Re-rendering a modern font supports review but is not an oracle for historical glyph identity.

**Coverage remains narrow:** 34 supported Khojki characters have no positive examples in the 15 real training lines; some occur in synthetic replay only. Class coverage is recorded in `data/round4/class-coverage.json`. Output-vocabulary support is not recognition competence. A pixel-similarity audit examined 46 known lines / 190 separator-delimited word crops; three lines failed separator-count checks and were excluded from word crops. Similarity candidates are review prompts, not automatic relabelling.

## Controlled experiments

Nine new named training runs were completed. The initial six share the packed warm start, original labels, seed 503, learning rate 5e-5, 25% synthetic replay, robust layout and bold replay. They isolate no words, legacy red padding, fixed white padding, frozen BatchNorm statistics, fixed words plus frozen BatchNorm, and frozen recurrent weights. The frozen-recurrent pilot initially hit a cuDNN backward constraint; frozen weights now retain training mode with dropout disabled. It was rerun successfully; this failure is documented rather than counted as a completed accuracy result.

At most two 600-step pilots with <=16 edits were extended to 3000; ties chose red and white padding, preserving the paired comparison. Candidate selection also tested a two-new-member ensemble and a strongest-new/old-member ensemble. The single red-padding candidate tied the 14-edit incumbent; its historical regression totals were 17/118 and 26/123.

The follow-up uses seed 607 and a five-times-lower learning rate (1e-5): original labels, one corrected training label, and corrected labels plus 10% fixed word recombination. All first run 1000 steps. Up to two candidates with <=15 development edits extend to 5000, retaining the best checkpoint. The final selection compares these with incumbents and one mixed ensemble, then five fixed optical variants: weaker ink, stronger ink, mild blur, three-strength averaging and original/blur averaging. No new test predictions enter this selection. The small reused development set makes every selected minimum optimistic.

{chr(10).join(rows)}

### Training lengths and saved checkpoints

| Run | Total completed steps | Best greedy checkpoint step | Best greedy dev edits / 405 |
|---|---:|---:|---:|
{chr(10).join(steps)}

Training checkpoint choice uses greedy CER every 50 steps; the tables above re-evaluate best checkpoints with the fixed beam-10 structural decoder. The two criteria are deliberately named, not conflated. Full histories, including later regressions, are retained. Best/last checkpoints include optimizer and RNG state. Longer training was used when pilots met the predeclared rule, rather than assumed to improve quality.

## Fresh evaluation and stage attribution

The first five detected body lines on previously untranscribed page 21 were selected before round-four outputs. All five were retained. Chart-based references and two explicit alternatives were frozen before any recognition output for that page was read (`data/round4/test-freeze.json`). Two lines have unresolved base-letter ambiguity. Predeclared alternatives give **{sens["minimum"]["cer"]:.2%}–{sens["maximum"]["cer"]:.2%} CER**; this is a sensitivity range, not a confidence interval. The fixed primary denominator is 120 scalar values. Do not merge these sequential tests into an independent benchmark.

- Missed selected lines: **{f["missed_lines"]}**.
- Automatic versus checked crops: **{f["automatic_errors"]} versus {f["checked_crop_errors"]} edits**; {f["crop_changes_prediction_lines"]} changed predictions.
- Space-involving edits: **{r["space_involving_edits"]}**; other character edits: **{r["other_edits"]}**.
- Invalid-inventory characters: **{f["invalid_inventory_characters"]}**; UTF-8 failures: **{f["utf8_failures"]}**.
- Exact primary-reference lines: **{r["exact_lines"]}/5**.

This is an operational breakdown. Valid-but-wrong code points are recognition errors that an inventory check cannot detect; crop interventions do not uniquely prove causation. Unlabelled page regions are not scored. Three repeated alignments pair inserted PA with independent E-to-vowel-I substitution; a fourth E location is also confused. E has no positive real training example. E and PA+I have similar outlines but remain distinct encodings; this post-evaluation diagnostic does not change references. Detailed alignments and code points are in `results/round4-report.json`.

## Reproducibility, throughput and limits

The final serialized checkpoint reproduced every selected development prediction. All 16 regression tests passed in the final cloud evaluation. A subsequent diagnostic-only Unicode audit added missing atomic I/U/AI/AU checks from Unicode 17 Table 15-4; 14 focused tests passed locally without importing ML libraries. This adds warnings, not text rewriting or decoder changes. CPU and CUDA produced identical full-page-21 text. The CPU CLI processed {v["cpu_page"]["line_count"]} lines in **{sec:.3f} seconds**, plus **{v["cpu_run"]["model_load_seconds"]:.3f} seconds** loading, using two threads. This excludes Python startup/PDF rasterization and manual review. Rough extrapolation: **{47 * sec:.0f} seconds for 47 comparable printed pages**, or {sec:.2f}N seconds for N pages. The mixed handwritten collection cannot inherit this quality estimate.

Deployment contains {v["members"]} model member(s), {v["parameters"]:,} parameters in total; optical views are `{v["views"]}`. No API or retraining is required. SHA-256: `{sha}`. Model files, all nine resumable experiment archives, manifests, complete logs and outputs were saved and verified before runtime deletion. Final observed balance: **{balance:.2f} units**, **0 active sessions**, **0 ongoing usage**, above the 75-unit reserve.

The next high-value work is 200–500 independently adjudicated printed lines spanning books, rare forms and degradation, with book-disjoint tests and a measured learning curve. That is a planning estimate, not a promise. Handwriting needs writer-disjoint labels and a different geometry regime; no handwriting accuracy is claimed. Further tiny-development-set tuning cannot substitute for reference quality or coverage.
"""
    (ROOT / "RESULTS.md").write_text(text)
    print(
        "Updated RESULTS.md from verified artifacts; curated documentation is unchanged."
    )


if __name__ == "__main__":
    main()
