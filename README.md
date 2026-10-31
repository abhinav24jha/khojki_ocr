# Khojki OCR take-home

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
