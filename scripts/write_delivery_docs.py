"""Assemble delivery documentation from measured artifacts; no model execution."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
selected=read('experiments/round3-final-selection.json');r=read('results/round3-report.json');e=read('results/round3-evaluation-complete.json');s=e['test_summary'];seconds=e['cpu_page']['pages'][0]['processing_seconds']
model=ROOT/'models/printed-khojki.pt';h=hashlib.sha256(model.read_bytes()).hexdigest()
readme='''# Khojki OCR take-home

A runnable page-image/PDF → line detection → visual sequence recognition → Khojki UTF-8 pipeline, with saved models, measured failures and an offline correction interface.

**Latest frozen test: 21.14% code-point CER (26 edits / 123 characters, five real printed lines).** The preceding five-line test was **14.41% (17/118)**. Development reached 3.46% (14/405), but that is not unseen accuracy. This is a useful printed-page prototype, not near-perfect OCR, a handwriting solution, or a collection-wide benchmark. All references are non-expert chart transcriptions with explicit limitations.

## Inspect the submission

- [One-page technical retrospective](technical-retro.pdf)
- [Latest five source/reference/OCR comparisons](results/round3-comparison.html)
- [Prior five-line comparison](results/round2-comparison.html)
- [Latest UTF-8 reference/prediction pairs](results/round3-transcriptions.tsv)
- [Results and complete experiment ledger](RESULTS.md)
- [Architecture and reasoning](ARCHITECTURE.md)
- [Page 20 correction interface](results/round3-cpu-demo/page-001/review.html)
- [Saved model and data provenance](MODEL_CARD.md)

Open the HTML files locally. The private assessment archive includes their fonts and source crops. **Export reviewed transcriptions** saves a separate JSON and never changes raw predictions or evaluation references. Edits are not autosaved. Rendering, editing and export were checked in the browser.

## Run without retraining

Use Python 3.10+ from this repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m khojki_ocr data/pages/p20.png --output demo --device cpu
```

The default checkpoint is `models/printed-khojki.pt`: two 1.06M-parameter recognizers, approximately 8.5 MB total. No API key, Ollama service or training is needed. The CLI defaults to two CPU threads and processes pages sequentially. CPU and CUDA were verified; the packed-model MPS path was not verified. Use CPU for ordinary local inference; all further training should run in Colab because earlier concurrent local VLM/training exhausted laptop memory.

Outputs include `transcription.txt`, `result.json`, `overlay.png`, line PNGs and `review.html` per page, plus `run.json` with checkpoint hash and timings. The selected model averages packed/unpacked recurrent recognizers, uses beam width 10 and a conservative structural prefix filter. There is no dictionary, reference lookup or language-model text replacement.

```bash
python -m khojki_ocr /path/to/scans.pdf --pages 19 20 --pdf-native --output pdf-demo --device cpu
```

`--pdf-native` preserves the embedded scan when each page has exactly one image; omit it to render at 150 DPI. `--region LEFT TOP RIGHT BOTTOM` provides an explicit assisted crop if automatic framing fails. JSON records that assistance. Every output line requires review, especially handwriting and unfamiliar typefaces.

## Reproduce the reported evaluation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/evaluate_pages.py --checkpoint models/printed-khojki.pt --annotations data/round3/test.json --split test --device cpu --output reproduced-test
python scripts/evaluate_pages.py --checkpoint models/printed-khojki.pt --annotations data/round3/train-dev.json --split dev --device cpu --output reproduced-dev
python scripts/audit_artifacts.py
```

The saved authoritative test is `results/round3-test/evaluation.json`. It includes reference boxes, automatic box matches, automatic/checked-crop predictions and edit counts. Only pre-transcribed lines are scored; unlabelled page regions are not counted as successful detections. Fifteen regression tests passed in Colab. CPU and CUDA page-20 text matched exactly. Editable installation and the console entry point were also verified.

CER = summed substitutions + deletions + insertions divided by summed reference Unicode scalar values. Combining marks, Gujarati digits, separators and number-column spaces count. Line-ending newlines, UTF-8 bytes and UTF-16 code units do not. No reference was changed to match an OCR prediction.

## Data and selection history

| Phase | Training candidates | Development | Fresh test at that phase |
|---|---|---|---|
| Initial baseline | 15 available / 10 selected, page 17 | 6 lines / 155 code points | 10 lines / 250 |
| Round 2 | 15 lines / 384 | 16 / 405, including original test | 5 / 118, page 19 lines 6–10 |
| Round 3 | 20 / 495, including 3 previous test lines | 18 / 453, including 2 previous test lines | 5 / 123, page 18 lines 14–15 and page 20 lines 9–11 |

Round-three candidates were rejected on development results. The **final weights remain the round-two ensemble trained on 15 page-17 lines**; they were reserialized without retraining. The round-three training-pool score is therefore not a training-fit score for the selected model. Original and round-two tests are historical evidence once used to guide later development; do not pool the sequential test sets into an independent benchmark.

Frozen labels, SHA-256 records, split promotion and uncertainty are in `data/round2/` and `data/round3/`. Round-three test candidates changed before training/prediction inspection because footer/refrain mappings were unresolved; numbered body lines were chosen instead. That selection limits claims. The source pages had appeared in earlier unscored whole-page inference, but these line predictions were not inspected to choose the final references. All sets share related books/pages; no book-disjoint claim is made.

## Restore exercise-only images

The supplied material requests exercise-only use. The private delivery archive contains inspection assets; they are ignored in Git by default. Restore them from the original supplied PDF after cloning:

```bash
python scripts/restore_data.py /path/to/sample_khojki_script_vatiations.pdf
```

This verifies the PDF hash and uses frozen coordinates; it does not regenerate labels. Annotation-authoring scripts are historical workflow records, not ordinary inference steps. Do not rerun them to alter frozen references after viewing predictions.

## Training, archives and resuming

Install `python -m pip install -e '.[train]'` in Colab. Prepared notebooks and exact paired input bundles are in `cloud/`; the notebook index links to saved Drive notebooks. Upload the notebook before connecting, then its paired ZIP through Colab's Files panel. Each verifies input hashes. Preserve at least 75 units and export/verify all output before disconnecting.

The final members came from `experiments/precision-packed/best.pt` (step 2450) and `experiments/long-control/best.pt` (step 2800). `experiments/round2-selection.json` records their development-only decoder selection; `round3-final-selection.json` records rejection of the new candidates. `cloud/build_round3.py` defines the latest matched pilots and conditional extension rule. Earlier promising pilots were extended to 3,000 steps; the latest weaker pilots were not.

`finetune.py` saves best/last weights, optimizer, Python/NumPy/PyTorch/device RNG states, settings, step, annotation hash and complete histories. Resume with the same data/settings, `--resume`, `--checkpoint .../last.pt`, the same `--output`, and a larger total `--steps`. Changing annotations requires a new named warm start. The synthetic script's `--resume PATH` is weights-only, not exact optimizer resume. Training is stochastic; reproduce the reported inference with the shipped compact checkpoint.

All full resumable checkpoints and ZIPs are retained in the working project. The compact submission archive includes deployment checkpoints, code, configuration/history/logs, references, results and notebooks; it excludes large training checkpoints and duplicate input/result ZIPs. `cloud/archive-index.json` identifies the preserved archive hashes. Exact package snapshots are recorded in results; install project dependencies rather than every unrelated Colab package.

## Limits and delivery

Unfamiliar historical forms, rare signs, degraded scans and handwritten pages remain hard. Some output classes have no positive training examples. Unicode validation catches malformed sequences, not a visually plausible wrong letter. The next defensible step is independently reviewed, diverse labels and book-disjoint evaluation, not reporting development CER as OCR accuracy.

The one-page retrospective covers every requested topic. The local Git repository and private ZIP are ready for review. No repository was published and no email was sent. Font licensing is in `assets/OFL.txt`. The final observed Colab balance was **167.31 units**, **zero active sessions** and **zero ongoing usage**.
'''
(ROOT/'README.md').write_text(readme)
old=ROOT/'RESULTS.md';archive=ROOT/'results/historical-baseline-notes.md'
if not archive.exists():archive.write_text('# Archived initial-baseline report\n\nThis is the report before subsequent rounds. “Final” below refers only to that historical phase. Current results are in ../RESULTS.md.\n\n'+old.read_text())
text='''# Results and experiment ledger

The selected model runs end to end, but its unseen-line accuracy is substantially worse than its development score. The latest five-line test is **26/123 = 21.14% CER**, compared with **28/123 = 22.76%** for the original baseline. The previous fresh test was **17/118 = 14.41%**, compared with **25/118 = 21.19%** for the original baseline. No near-perfect or handwriting accuracy claim is supported.

## Frozen evaluation sequence

| Phase / model | Reference set | Edits / code points | CER | Interpretation |
|---|---|---:|---:|---|
| Original baseline | Initial 10-line test | 21 / 250 | 8.40% | Historical; these lines later became development |
| Round-two ensemble | Expanded development | 14 / 405 | 3.46% | Repeated model/decoder selection, optimistic |
| Round-two ensemble | First fresh 5-line test | 17 / 118 | 14.41% | Stronger generalization check, later used for round-three design |
| Original baseline | Same first fresh test | 25 / 118 | 21.19% | Paired comparison |
| Final retained ensemble | Expanded round-three development | 20 / 453 | 4.42% | Used to reject two new training candidates |
| Final retained ensemble | Latest frozen 5-line test | 26 / 123 | 21.14% | Final primary result |
| Original baseline | Same latest test | 28 / 123 | 22.76% | Paired comparison |

The final weights are the round-two ensemble. Round-three training candidates were worse on development and were rejected before the latest test was opened. Re-saving the same weights produced a different file hash; deployment predictions match the retained incumbent. Both fresh sets are small, nonrandom, related print with non-expert references. They are not book-disjoint tests or estimates of collection-wide error. Sequential tests should not be pooled into a nominally independent benchmark.

## Latest test and error attribution

All five selected lines were detected. Automatic and checked crops gave identical predictions: 26 edits each. There were **2 space-involving edits**, **24 other character edits**, **0 invalid-inventory characters** and **0 UTF-8 failures**. Space mistakes can involve geometry and sequence recognition; this is an operational grouping, not a uniquely identified causal decomposition. No false-positive or recall claim is made for unlabelled page regions.

'''
text+='| Line | Edits | Reference code points | CER |\n|---|---:|---:|---:|\n'
for line in r['lines']:
 m=line['automatic'];text+=f"| {line['id']} | {m['errors']} | {m['reference_characters']} | {100*m['cer']:.2f}% |\n"
counts={k:sum(x['automatic'][k] for x in r['lines']) for k in ['substitute','delete','insert']}
text+=f"\nEdit totals: {counts['substitute']} substitutions, {counts['delete']} deletions, {counts['insert']} insertions; one of five lines exact. Most repeated errors are inserted SA and vowel sign U (two each). Other confusions involve PA/THA, BA/NA, dependent vowels and separators. Full names/code points and alignments are in [round3-report.json](results/round3-report.json).\n\n"
text+='''The primary references remain unchanged. Predeclared alternatives yield **20.16–25.20% CER** (25–31 edits, variant denominators 123–124). This sensitivity range is not a confidence interval or a corrected score. It does not eliminate the recognition gap. Footer/refrain candidates were replaced with numbered body lines before round-three training and prediction inspection because their mappings were unresolved; selection history is recorded explicitly.

On the prior five-line test, all lines were detected, automatic/checked crops matched, and the 17 edits involved no space or encoding failures. BA→NA, BHA→vowel II, and inserted BBA each occurred twice. Predeclared alternatives gave 14.41–16.10%, so the error could not be explained away by those alternatives.

## Experiment ledger

Different reference sets are deliberately separated. Synthetic accuracy does not predict historical-print accuracy. Checkpoint minima are selected on development every 50 steps; full histories contain the worse checkpoints too.

| Experiment | Evaluation denominator | Edits | CER | Decision |
|---|---:|---:|---:|---|
| Synthetic CRNN, 1000 steps | 2910 synthetic | 38 | 1.31% | Pretraining only |
| Synthetic model, raw real lines | 75 real | 36 | 48.00% | Reject raw preprocessing |
| Contrast and text/number chunks | 75 | 22 | 29.33% | Keep |
| Binarization and chunks | 75 | 28 | 37.33% | Reject |
| Initial 15-line adaptation | 75 | 6 | 8.00% | Real labels help |
| Same model, expanded dev | 155 | 13 | 8.39% | Larger comparison |
| 3 / 5 / 10 / 15-line adaptation | 155 | 18 / 19 / 10 / 13 | 11.61 / 12.26 / 6.45 / 8.39% | 10-line model selected initially |
| Initial beam / three-model ensemble | 155 | 11 / 12 | 7.10 / 7.74% | Reject |
| 100-step cloud warm start | 155 | 13 | 8.39% | Reject |
| Original baseline on expanded dev | 405 | 31 | 7.65% | Round-two comparator |
| 300-step control | 405 | 28 | 6.91% | Extend |
| 300-step 40% word recombination | 405 | 35 | 8.64% | Worse |
| Same + bold replay | 405 | 36 | 8.89% | Worse |
| Long control, best 2800/3000 | 405 | 18 | 4.44% | Strong member |
| Long bold, best 2700/3000 | 405 | 19 | 4.69% | Alternative |
| Long 10% words, best 1550/3000 | 405 | 20 | 4.94% | Worse |
| Matched 600-step ordinary control | 405 | 20 | 4.94% | Comparator |
| Packed recurrence, 600-step pilot | 405 | 19 | 4.69% | Extend |
| Ordinary / packed 64-pixel pilots | 405 | 35 / 35 | 8.64 / 8.64% | Reject |
| Five extra non-expert labels | 405 | 23 | 5.68% | Reject |
| Packed recurrence, best 2450/3000 | 405 | 16 | 3.95% | Strong member |
| Word geometry legacy / preserved | 405 | 26 / 23 | 6.42 / 5.68% | Fix helps pair, neither wins |
| Packed model + structural beam | 405 | 15 | 3.70% | Decoder candidate |
| Ensemble greedy | 405 | 17 | 4.20% | Worse than selected decoder |
| Ensemble + beam 10 + structural filter | 405 | 14 | 3.46% | Selected round two |
| Targeted 20-line data, packed pilot | 453 | 24 | 5.30% | Reject against 20-edit incumbent |
| Targeted 20-line data, ordinary pilot | 453 | 26 | 5.74% | Reject |
| Targeted-data ensemble | 453 | 25 | 5.52% | Reject; retain incumbent |

The final targeted pilots each ran 500 steps. The preregistered rule extended a candidate to 3000 only if fixed-decoder development errors were at most incumbent + 2; neither qualified. Several earlier promising trials did run 3000 steps. More training is not assumed to be better without a matched result.

The expanded-repertoire synthetic trial reached 111/3252 (3.41%) at step 300 before interruption. It used a different synthetic set and lacked a comparable real-data evaluation, so it was not selected. More non-expert labels and word recombination failed to consistently help; this does not establish that good diverse labels are harmful.

Two VLM probes failed on an untranscribed line: Gemma3:4b returned repetitive Kannada (37.0 seconds), and Gemma4:e2b returned repetitive Devanagari despite a chart (63.9 seconds). Both exhausted 700 tokens with incomplete JSON: **2/2 wrong-script/format failures**, no defensible CER. Raw outputs are preserved. Qwen3-VL8B did not complete amid local memory exhaustion; no accuracy is attributed to it. Subsequent ML ran in Colab.

## Robustness, throughput and verification

| Final model, 453 development code points | Edits | CER |
|---|---:|---:|
| Unchanged | 20 | 4.42% |
| Half resolution | 36 | 7.95% |
| Gaussian blur radius 0.8 pixels | 18 | 3.97% |
| Rotation 1 degree | 20 | 4.42% |

These probes are descriptive, not another test-set tuning loop. Mild blur helping slightly does not imply arbitrary blur is safe. The prior 405-character probe was 14/31/13/14 edits under the same four conditions. Other printed page 23 and handwritten page 4 have saved unscored output; no CER or handwriting success is claimed. The robust detector found 29 handwriting lines, but detection count alone cannot establish transcription quality.

'''
text+=f'''The final page-20 CLI run took **{seconds:.3f} seconds** on a two-thread Colab CPU, plus **0.044 seconds** model loading, for 16 lines. CPU/GPU text matched exactly. Page processing includes detection, recognition and scan artifacts; import startup/PDF rendering and final HTML serialization are outside this measurement. A previous page-19 run took 3.141 seconds. At the latest observation, 47 comparable printed pages would take roughly **{47*seconds:.0f} seconds**, or **{seconds:.2f}N seconds** for N pages, excluding review and I/O variation. The supplied mix contains handwriting and cannot be promised the same quality.

T4 usage displayed about 1.15 units/hour; 2.91 seconds would correspond to roughly 0.00093 units at that rate, but the measured CPU time is not a GPU billing benchmark. Notebook setup, uploads, browser delays and idle time dominated these tiny experiments. The final observed balance was **167.31**, **0 active sessions**, **0 units/hour**, preserving the 75-unit floor. All cloud output archives were size/SHA-256 checked, ZIP-integrity checked and extracted before runtime termination. Receipts and shutdown screenshots are retained.

Initial annotation authoring/freezing spanned about 14 minutes (09:41–09:56 UTC), plus earlier chart inspection and later review. This is an AI-assisted elapsed window, not expert annotation productivity. Additional references were created in later rounds with ambiguity records. Expert adjudication remains the primary bottleneck.

Fifteen regression tests passed in the final cloud run (3.58 seconds). They cover CER scalar counting, CTC collapse/beam correctness, Unicode diagnostics, PDF input, empty pages, safe review HTML, frozen/disjoint references, faint-line/gap detection, word geometry and packed recurrence. Editable package/console checks passed in the prior evaluation. A lightweight local audit also checks current hashes, aggregate edits and Python syntax without model loading. Browser verification confirmed Khojki font rendering and a separate 16-line correction JSON without altering raw OCR.

Deployment SHA-256: `{h}`. See [MODEL_CARD.md](MODEL_CARD.md), [round3-evaluation-complete.json](results/round3-evaluation-complete.json), [round3-report.json](results/round3-report.json) and the visual comparisons. The final training/decoder selection used development data; fresh test results were not used to choose the deployed candidate.
'''
(ROOT/'RESULTS.md').write_text(text)
card={'model_file':'models/printed-khojki.pt','sha256':h,'bytes':model.stat().st_size,'architecture':'Two CNN-BiLSTM-CTC models, one packed and one ordinary recurrence; log-mean probabilities','parameters_per_member':1060946,'height':48,'layout':'robust','beam_width':10,'structural_filter':True,'actual_training_lines':15,'actual_training_codepoints':384,'round3_candidates_rejected':True,'selected_source_members':['experiments/precision-packed/best.pt','experiments/long-control/best.pt'],'primary_test':s,'limitations':['Non-expert labels','Same-book/page evaluation','Restricted positive training repertoire','Handwriting unvalidated','Development selection overfits small data'],'devices_verified':['cpu','cuda']}
(ROOT/'models/model-card.json').write_text(json.dumps(card,indent=2))
(ROOT/'MODEL_CARD.md').write_text(f'''# Deployment model and provenance

`models/printed-khojki.pt` is the final compact ensemble ({model.stat().st_size:,} bytes), SHA-256 `{h}`. `models/round3-selected.pt` is byte-identical. `models/round2-selected.pt` contains the same retained weights before reserialization. The original 10-line baseline remains in `models/baseline-printed-khojki.pt`.

Each member has 1,060,946 parameters: convolutional features and two bidirectional LSTM layers, trained with CTC. Height 48, robust layout, beam width 10 and structural prefix filtering are saved in metadata. One member uses packed recurrence; the other retains ordinary recurrence. Both were adapted on 15 page-17 lines (384 code points) with 25% synthetic replay. Synthetic rendering uses shaped Noto Khojki fonts; the training scripts record the exact mixtures and seeds.

Round-three candidates used 20 lines but were rejected. Their expanded training-pool error must not be described as the final model's fitted-training error. Selection records and source checkpoint hashes are in `experiments/round2-selection.json` and `experiments/round3-final-selection.json`. Best steps were 2450 for packed recurrence and 2800 for the ordinary member. Complete resumable checkpoints remain in the working project; deployment needs only the compact file.

The latest test is 26/123 = 21.14% CER on five chart-transcribed real lines. Previous fresh testing gave 17/118 = 14.41%. These small related-print sets have non-expert labels and explicit ambiguities. No full-script, book-disjoint, handwriting, or near-perfect claim is justified. Rare output classes lack positive examples. The output alphabet's presence does not demonstrate recognition coverage.

CPU and CUDA inference were checked, including exact page text agreement. MPS for the final packed ensemble was not tested. Use the repository's editable install so bundled assets remain available. Checkpoints are PyTorch artifacts: load only the supplied, hash-verified files or other trusted checkpoints.

The source PDF hash and page provenance are in `data/provenance.json`; frozen test hashes, annotation tokens, alternatives, crop boxes and split changes are in `data/round2/` and `data/round3/`. Original references remain in `data/annotations-v2-frozen.json`. Font licensing is bundled under `assets/OFL.txt`. Exercise scan material is not intended for public redistribution.
''')
print('README, RESULTS and model card written from verified artifacts.')
