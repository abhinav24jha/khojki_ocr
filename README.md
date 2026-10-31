# Khojki OCR take-home

Page image/PDF → automatic lines → visual sequence recognition → Khojki UTF-8, with a saved model and offline correction interface.

**Latest frozen five-line test: 21/120 = 17.50% CER**, versus the previous model's 21/120 = 17.50%. Development: 14/405 = 3.46%. References are non-expert chart transcriptions; two fresh lines have declared ambiguity. This is a measured printed-page prototype, not universal or near-perfect OCR.

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

Python 3.10+. The default `models/printed-khojki.pt` is byte-identical to `models/round4-final.pt`. It contains 2 member(s) and requires no API key, Ollama or training. The CLI defaults to two CPU threads and sequential pages. CPU/CUDA parity was checked; MPS for the selected model was not verified. Further training belongs in Colab, following the earlier local-memory failure.

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

A confirmed RGB-padding bug now has a fix and regression test. Nine controlled training runs tested padding, normalization, frozen recurrence, a slower learning rate and one isolated training-label correction; promising candidates ran 3000 or 5000 total steps. Optical variants were selected only on development. The final configuration is `incumbent` with views `[[1.0, 0.0]]`. Complete outcomes, including unsuccessful fixes, are in RESULTS.md.

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

All cloud output was size/hash/ZIP-checked before shutdown. Final balance: **166.94 units; zero active sessions and ongoing use**. No repository was published and no email was sent. Font licence: `assets/OFL.txt`.
