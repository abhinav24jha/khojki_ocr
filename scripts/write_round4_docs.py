"""Write current delivery docs from verified round-four artifacts, retaining history."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def main():
 r=load('results/round4-report.json');s=load('experiments/round4-followup-selection.json');v=load('results/round4-verification.json');f=r['fresh'];old=r['incumbent_fresh'];sens=r['predeclared_reference_sensitivity'];sec=v['cpu_page']['processing_seconds'];sha=r['checkpoint_sha256'];selected=r['selected'];shutdown=load('cloud/round4-shutdown.json');balance=shutdown['remaining_units']
 historical=ROOT/'RESULTS_ROUND3.md'
 if not historical.exists():shutil.copyfile(ROOT/'RESULTS.md',historical)
 rows=[]
 for label,path in [('600-step pilots','experiments/round4-pilot-selection.json'),('After paired 3000-step extensions','experiments/round4-final-selection.json'),('1000-step low-rate pilots','experiments/round4-followup-pilot.json'),('Final development-only selection','experiments/round4-followup-selection.json')]:
  rows.append('\n### '+label+'\n\n| Candidate | Edits / 405 | CER |\n|---|---:|---:|')
  for e in load(path)['experiments']:rows.append(f"| {e['name']} | {e['errors']} | {e['errors']/405:.2%} |")
 audit='\n'.join(f"| {a['model']} | {a['original_summary']['automatic_errors']}/{a['original_summary']['characters']} ({a['original_summary']['automatic_cer']:.2%}) | {a['proposed_errors']}/{a['proposed_characters']} ({a['proposed_cer']:.2%}) |" for a in r['reference_audit'])
 steps=[]
 for name in ['r4-control','r4-word-red','r4-word-white','r4-freezebn','r4-word-white-freezebn','r4-freezernn','r4-low-original','r4-low-corrected','r4-low-white']:
  h=load(f'experiments/{name}/history.json');best=min(h,key=lambda e:e['errors']);steps.append(f"| {name} | {h[-1]['step']} | {best['step']} | {best['errors']} |")
 text=f'''# Round-four results and audit

The selected configuration is **{selected['name']}**, chosen on the unchanged 16-line / 405-character development set: **{selected['errors']}/405 ({selected['errors']/405:.2%} CER)**. On five newly frozen page-21 lines it produces **{f['automatic_errors']}/120 ({f['automatic_cer']:.2%} CER)**, versus **{old['automatic_errors']}/120 ({old['automatic_cer']:.2%})** for the previous model. These are non-expert chart transcriptions, not certified ground truth or collection-wide accuracy.

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

The first five detected body lines on previously untranscribed page 21 were selected before round-four outputs. All five were retained. Chart-based references and two explicit alternatives were frozen before any recognition output for that page was read (`data/round4/test-freeze.json`). Two lines have unresolved base-letter ambiguity. Predeclared alternatives give **{sens['minimum']['cer']:.2%}–{sens['maximum']['cer']:.2%} CER**; this is a sensitivity range, not a confidence interval. The fixed primary denominator is 120 scalar values. Do not merge these sequential tests into an independent benchmark.

- Missed selected lines: **{f['missed_lines']}**.
- Automatic versus checked crops: **{f['automatic_errors']} versus {f['checked_crop_errors']} edits**; {f['crop_changes_prediction_lines']} changed predictions.
- Space-involving edits: **{r['space_involving_edits']}**; other character edits: **{r['other_edits']}**.
- Invalid-inventory characters: **{f['invalid_inventory_characters']}**; UTF-8 failures: **{f['utf8_failures']}**.
- Exact primary-reference lines: **{r['exact_lines']}/5**.

This is an operational breakdown. Valid-but-wrong code points are recognition errors that an inventory check cannot detect; crop interventions do not uniquely prove causation. Unlabelled page regions are not scored. Three repeated alignments pair inserted PA with independent E-to-vowel-I substitution; a fourth E location is also confused. E has no positive real training example. E and PA+I have similar outlines but remain distinct encodings; this post-evaluation diagnostic does not change references. Detailed alignments and code points are in `results/round4-report.json`.

## Reproducibility, throughput and limits

The final serialized checkpoint reproduced every selected development prediction. All 16 regression tests passed in the final cloud evaluation. A subsequent diagnostic-only Unicode audit added missing atomic I/U/AI/AU checks from Unicode 17 Table 15-4; 14 focused tests passed locally without importing ML libraries. This adds warnings, not text rewriting or decoder changes. CPU and CUDA produced identical full-page-21 text. The CPU CLI processed {v['cpu_page']['line_count']} lines in **{sec:.3f} seconds**, plus **{v['cpu_run']['model_load_seconds']:.3f} seconds** loading, using two threads. This excludes Python startup/PDF rasterization and manual review. Rough extrapolation: **{47*sec:.0f} seconds for 47 comparable printed pages**, or {sec:.2f}N seconds for N pages. The mixed handwritten collection cannot inherit this quality estimate.

Deployment contains {v['members']} model member(s), {v['parameters']:,} parameters in total; optical views are `{v['views']}`. No API or retraining is required. SHA-256: `{sha}`. Model files, all nine resumable experiment archives, manifests, complete logs and outputs were saved and verified before runtime deletion. Final observed balance: **{balance:.2f} units**, **0 active sessions**, **0 ongoing usage**, above the 75-unit reserve.

The next high-value work is 200–500 independently adjudicated printed lines spanning books, rare forms and degradation, with book-disjoint tests and a measured learning curve. That is a planning estimate, not a promise. Handwriting needs writer-disjoint labels and a different geometry regime; no handwriting accuracy is claimed. Further tiny-development-set tuning cannot substitute for reference quality or coverage.
'''
 (ROOT/'RESULTS.md').write_text(text)
 readme=f'''# Khojki OCR take-home

Page image/PDF → automatic lines → visual sequence recognition → Khojki UTF-8, with a saved model and offline correction interface.

**Latest frozen five-line test: {f['automatic_errors']}/120 = {f['automatic_cer']:.2%} CER**, versus the previous model's {old['automatic_errors']}/120 = {old['automatic_cer']:.2%}. Development: {selected['errors']}/405 = {selected['errors']/405:.2%}. References are non-expert chart transcriptions; two fresh lines have declared ambiguity. This is a measured printed-page prototype, not universal or near-perfect OCR.

## Inspect

- [One-page technical retrospective](technical-retro.pdf)
- [Latest source/reference/OCR comparison and experiment table](results/round4-comparison.html)
- [Dataset label audit with enlarged evidence](data/round4/reference-audit.html)
- [Current results](RESULTS.md) and [prior full ledger](RESULTS_ROUND3.md)
- [Full page-21 review interface](results/round4-cpu-demo/page-001/review.html)
- [Architecture](ARCHITECTURE.md) and [model provenance](MODEL_CARD.md)

The private ZIP includes fonts and scan crops. Open HTML locally. Corrections export to a separate JSON and never change raw predictions or frozen references; edits are not autosaved.

## Run without retraining

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m khojki_ocr data/pages/p21.png --output demo --device cpu
python -m khojki_ocr /path/to/scans.pdf --pages 19 20 --pdf-native --output pdf-demo --device cpu
```

Python 3.10+. The default `models/printed-khojki.pt` is byte-identical to `models/round4-final.pt`. It contains {v['members']} member(s) and requires no API key, Ollama or training. The CLI defaults to two CPU threads and sequential pages. CPU/CUDA parity was checked; MPS for the selected model was not verified. Further training belongs in Colab, following the earlier local-memory failure.

`--pdf-native` extracts a page's sole embedded scan; omit it to rasterize at 150 DPI. `--region LEFT TOP RIGHT BOTTOM` is an explicit assisted crop recorded in output JSON. Each page produces `transcription.txt`, `result.json`, `overlay.png`, line crops and `review.html`; `run.json` stores model hash and timings. All text requires review, especially handwriting and unfamiliar forms.

## Reproduce evaluation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/evaluate_pages.py --checkpoint models/printed-khojki.pt --annotations data/round4/test.json --split test --device cpu --output reproduced-test
python scripts/evaluate_pages.py --checkpoint models/printed-khojki.pt --annotations data/round2/train-dev.json --split dev --device cpu --output reproduced-dev
python scripts/audit_artifacts.py
```

Authoritative new evaluation: `results/round4-final-r4/evaluation.json`. CER counts Unicode scalars, including combining signs, Gujarati digits, U+1123A separators and number-column spaces. It is not a byte, UTF-16-unit or grapheme metric. Automatic lines are matched to reference boxes geometrically, never by OCR text. Unlabelled regions are unscored. Sixteen integration tests passed in Colab, followed by 14 focused atomic-vowel tests locally; CPU/CUDA page parity passed.

## What changed in round four

A confirmed RGB-padding bug now has a fix and regression test. Nine controlled training runs tested padding, normalization, frozen recurrence, a slower learning rate and one isolated training-label correction; promising candidates ran 3000 or 5000 total steps. Optical variants were selected only on development. The final configuration is `{selected['name']}` with views `{v['views']}`. Complete outcomes, including unsuccessful fixes, are in RESULTS.md.

Our earlier AI transcriptions had likely errors. Source comparisons propose four historical-test word corrections; original labels and original scores remain unchanged. Their lower revised-reference scores are not model gains. The fresh page-21 references were separately frozen before predictions and include declared alternatives.

## Data, provenance and resuming

The retained weights use the original 15 page-17 training lines (384 code points). The corrected-label trials (383 code points) were rejected; their isolated revision is documented in `data/round4/train-dev-corrected.json` and experiment configurations. Original development is 16 lines / 405 characters. The earlier fresh sets (118 and 123 characters) are now historical regression diagnostics; the new five-line set is 120 characters. All material is related print, not book-disjoint. No sequential tests should be pooled into a nominally independent benchmark.

The task requests exercise-only use. Private delivery includes images; Git ignores scan assets by default. Restore source images using:

```bash
python scripts/restore_data.py /path/to/sample_khojki_script_vatiations.pdf
```

Restoration verifies the source PDF hash and frozen coordinates without relabelling. Annotation-authoring scripts are historical records, not ordinary inference steps. Do not overwrite frozen labels after viewing output.

`cloud/Khojki_round4_complete.ipynb` contains the offline-prepared workflow. Upload its paired initial and follow-up ZIPs through Colab Files, then run; hashes are verified. The first data bundle is `khojki-round4.zip`, the second `khojki-round4-followup.zip`. Full archives remain in the working project; the compact submission retains checkpoints for deployment, code, manifests, configs, histories, logs and notebooks. `cloud/archive-index.json` identifies every preserved archive.

`finetune.py` saves best/last weights, optimizer, Python/NumPy/PyTorch/device RNG state, settings, step and annotation hash. Resume with `--resume --checkpoint .../last.pt`, identical settings/output name and a larger total `--steps`. Changed labels require a new named warm start. The initial synthetic trainer's resume is weights-only. No training is required to reproduce shipped predictions.

All cloud output was size/hash/ZIP-checked before shutdown. Final balance: **{balance:.2f} units; zero active sessions and ongoing use**. No repository was published and no email was sent. Font licence: `assets/OFL.txt`.
'''
 (ROOT/'README.md').write_text(readme)
 model=f'''# Deployment model and provenance

`models/printed-khojki.pt` is byte-identical to `models/round4-final.pt`: {(ROOT/'models/round4-final.pt').stat().st_size:,} bytes, SHA-256 `{sha}`.

Development-selected configuration: **{selected['name']}**. Source checkpoints: `{selected['paths']}`. Optical views are (ink-strength multiplier, Gaussian-blur radius): `{v['views']}`. {v['members']} CRNN member(s), {v['parameters']:,} total parameters; height 48, robust layout, beam 10, conservative structural filtering. Metadata preserves all inference choices. There is no dictionary, reference lookup, or language-model text replacement.

All selected members derive from the original 15 page-17 real lines plus shaped Noto synthetic replay. Follow-up corrected-label experiments change only the disputed p17r-10 I sign (384 → 383 real code points). Exact configuration, data hashes and source checkpoints are in `experiments/round4-followup-selection.json` and each named experiment directory. Original 10-line baseline and previous compact deployments remain available. A legacy RGB-padding ablation deliberately preserves an old artificial-border defect; this is recorded in its training config, not hidden.

Latest five-line test: {f['automatic_errors']}/120 = {f['automatic_cer']:.2%} CER, previous model {old['automatic_errors']}/120 = {old['automatic_cer']:.2%}. Predeclared alternatives: {sens['minimum']['cer']:.2%}–{sens['maximum']['cer']:.2%}. Development {selected['errors']}/405 is a repeatedly selected minimum, not unseen accuracy. References are non-expert and related print; no full-script, book-disjoint, handwriting or near-perfect claim is supported. Thirty-four Khojki classes lack real training examples.

The final serialized model reproduces selected development predictions; CPU and CUDA full-page text match. Two-thread CPU page time: {sec:.3f}s for {v['cpu_page']['line_count']} lines. MPS was not verified. Use an editable install so bundled fonts/assets remain available; load only trusted checkpoint files.

Frozen new references, alternatives and image hashes: `data/round4/test.json` and `test-freeze.json`. Proposed historical-reference corrections remain separate in `data/round4/proposed-reference-revisions.json`; lower scores from those corrections are not model improvements. Source PDF provenance is in `data/provenance.json`. The scans are for this private exercise; font licensing is bundled.
'''
 (ROOT/'MODEL_CARD.md').write_text(model)
 model_json=load('models/model-card.json');model_json.update(sha256=sha,bytes=(ROOT/'models/round4-final.pt').stat().st_size,primary_test=f,views=v['views'],round4_candidates_rejected=selected['name']=='incumbent');(ROOT/'models/model-card.json').write_text(json.dumps(model_json,indent=2))
 arch=ROOT/'ARCHITECTURE.md';a=arch.read_text();marker='\n## Round-four audit and resulting design\n';a=a.split(marker)[0];a=a.replace('## Final experiment outcomes','## Round-three outcomes before the later reference audit')
 a+=marker+f'''
The new audit separated three mechanisms: image generation defects, uncertain target sequences, and adaptation/inference sensitivity. RGB scalar padding introduced red borders that became dark strokes; normalization to grayscale fixes that contract. The matched 3000-step study did not show a recognition gain from the fix, so correctness of the data operation and measured utility of the augmentation remain separate claims. Freezing BatchNorm running statistics and recurrent weights also failed to beat the incumbent in 600-step pilots.

Nine training runs, conditional extensions and five fixed optical variants led to `{selected['name']}` ({selected['errors']}/405 development edits). Any saved optical views are averaged in probability space before fixed CTC beam decoding; they preserve image dimensions and temporal alignment. Selected metadata is replayed and checked against development predictions before test evaluation. The fresh page-21 result is {f['automatic_errors']}/120; previous-model result is {old['automatic_errors']}/120. This final test did not choose the model.

An image-only repeated-word audit also exposed likely mistakes in our earlier AI-authored labels. On the same incumbent output, proposed historical-reference revisions change 26/123 to 15/128 edits. That is a reference effect, not an OCR gain, and remains post-hoc/non-expert. Original frozen files remain intact. One training-only correction was isolated in a matched lower-rate study; new test labels retain explicit alternatives and were frozen before page-21 output.

Current performance and reference uncertainty both limit inference. Thirty-four supported Khojki characters have no real training example, and a modern-font shape cannot settle every historical variant. Independent adjudication and book-disjoint coverage are therefore stronger next investments than continuing to optimize this small reused development set. See RESULTS.md for all failed candidates and the original historical ledger for prior rounds.
'''
 arch.write_text(a)
 print('Updated README, RESULTS, MODEL_CARD and ARCHITECTURE from verified artifacts')
if __name__=='__main__':main()
