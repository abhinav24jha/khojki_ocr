# Khojki OCR

A runnable printed Khojki OCR prototype: page image or PDF → detected lines → visual sequence recognition → UTF-8 text, with saved weights, an offline review interface, and traceable experiments.

**Latest frozen test: 21 character edits over 120 reference code points, or 17.50% CER, on five real lines.** The previous model gets the same result. Development CER is 14/405 = 3.46%, after repeated selection on that small set. The references are AI-assisted, non-expert chart transcriptions of related printed material; this is not a verified accuracy estimate for the whole collection or handwriting. The fourth round improved correctness and auditability but did **not** improve the selected model's accuracy.

## Contents

- [Start here](#start-here)
- [Installation and inference](#installation-and-inference)
- [Command-line options](#command-line-options)
- [Outputs and review](#outputs-and-review)
- [Pipeline and Unicode](#pipeline-and-unicode)
- [Data and reference protocol](#data-and-reference-protocol)
- [Results and experiment history](#results-and-experiment-history)
- [Reproducing evaluation](#reproducing-evaluation)
- [Training and resuming](#training-and-resuming)
- [Colab workflow and resource limits](#colab-workflow-and-resource-limits)
- [Development and tests](#development-and-tests)
- [Repository guide](#repository-guide)
- [Troubleshooting](#troubleshooting)
- [Scope, licensing, and references](#scope-licensing-and-references)

## Start here

| What to inspect | Artifact |
|---|---|
| Assessment retrospective, in plain Markdown | [technical-retro.md](technical-retro.md) |
| Latest measurements and failed experiments | [RESULTS.md](RESULTS.md) |
| Full earlier experimental ledger | [RESULTS_ROUND3.md](RESULTS_ROUND3.md) |
| Design decisions and alternatives | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Checkpoint provenance and limitations | [MODEL_CARD.md](MODEL_CARD.md), [machine-readable model card](models/model-card.json) |
| Five test references and declared ambiguities | [test.json](data/round4/test.json), [freeze record](data/round4/test-freeze.json) |
| Raw latest evaluation | [evaluation.json](results/round4-final-r4/evaluation.json) |
| Saved source/reference/prediction comparison | [round4-comparison.html](results/round4-comparison.html) |
| Saved 16-line page review | [review.html](results/round4-cpu-demo/page-001/review.html) |

The retrospective is the plain Markdown document linked above. Its 575 words were [visually checked to fit one US Letter page at 10-point body text](results/retrospective-format-check.json). Older PDF/preview artifacts are historical; they are not the primary retrospective.

GitHub displays HTML source rather than running the review application. Open generated HTML locally. Scan images are excluded from the public repository because the assessment supplies them for exercise-only use. Historical HTML that references those images needs the private artifacts; a fresh inference run produces its own complete review directory.

## Installation and inference

Use Python 3.10 or newer. Inference requires no API key, hosted model, Ollama server, or retraining. The default device is CPU and the default CPU thread limit is two. CPU and CUDA predictions were compared for the selected model; MPS compatibility is available but has not been verified for that model.

```bash
git clone https://github.com/abhinav24jha/khojki_ocr.git
cd khojki_ocr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# Supply a page image you are authorized to process.
khojki-ocr /path/to/page.png --output runs/example --device cpu
# Equivalent module invocation:
python -m khojki_ocr /path/to/page.png --output runs/example --device cpu
```

An editable installation is convenient for experiments because scripts, annotations and outputs remain together in the checkout. A regular installation (`python -m pip install .`) also includes the deployment checkpoint and review font; the installed inference command can run outside the repository. Training and reproduction scripts are repository tools and should be run from the checkout.

The saved deployment model is approximately 8.5 MB. PyTorch and other dependencies are substantially larger. Page size, line width and intermediate tensors affect memory use; a small weight file does not guarantee a small peak process allocation. Keep training in Colab or another managed GPU environment following the earlier local memory failure.

For PDFs:

```bash
# Render selected pages at 150 DPI.
khojki-ocr /path/to/scans.pdf --pages 19 20 --output runs/pdf

# Preserve native embedded scan pixels when each selected page has one image.
khojki-ocr /path/to/scans.pdf --pages 19 20 --pdf-native --output runs/native

# An explicit assisted region, in pixels of the input/extracted/rendered page.
khojki-ocr /path/to/page.png --region 60 130 710 980 --output runs/region
```

Use a new output directory for each run to keep experiments separate. Native extraction and rasterization can produce different pixel coordinates and scale; the frozen assessment boxes belong to the original embedded scan images.

## Command-line options

`khojki-ocr SOURCE --output DIRECTORY [OPTIONS]`

| Argument | Default | Meaning |
|---|---|---|
| `SOURCE` | Required | One Pillow-readable page image or a PDF. |
| `--output PATH` | Required | Directory receiving `run.json` and one subdirectory per page. |
| `--checkpoint PATH` | Bundled `printed-khojki.pt` | Override the saved model. Use trusted project checkpoints only. |
| `--device cpu\|cuda\|mps` | `cpu` | Computation device; the selected backend must be available. |
| `--threads N` | `2` | Positive PyTorch CPU thread limit. |
| `--pages N [N ...]` | All PDF pages | One-based PDF page numbers. |
| `--pdf-native` | Off | Extract the sole embedded image on each selected PDF page. Errors if that condition is not met; omit to rasterize. |
| `--region LEFT TOP RIGHT BOTTOM` | Automatic region | Nonnegative, ordered pixel coordinates within the page. The same region applies to every selected page. |
| `-h`, `--help` | — | Print usage without running inference. |

The checkpoint stores layout, input height, recurrence modes, beam width, structural filtering, and optical views. Ordinary inference replays those choices; there is no need to reconstruct a list of experimental decoder flags.

## Outputs and review

A single image is numbered `page-001`; PDF directories retain one-based page numbers.

```text
runs/example/
├── run.json
└── page-001/
    ├── transcription.txt
    ├── result.json
    ├── overlay.png
    ├── review.html
    ├── NotoSansKhojki-Regular.ttf
    ├── OFL.txt
    └── p001-01.png                # one crop per detected line
```

- **`transcription.txt`** is raw OCR, one detected line per newline, encoded as UTF-8. Printed Khojki separators remain U+1123A; distant text/number chunks are joined by an ASCII space.
- **`result.json`** records `image_size`, `region`, `manual_region`, `layout`, `line_count`, detection/recognition/processing seconds, the scope statement, and `lines`.
- Each line contains `id`, `box` (`[left, top, right, bottom]`), `crop`, `text`, `recognition_seconds`, `review_required`, and `validation`. Coordinates use the page's pixel coordinate system, with right/bottom bounds exclusive.
- **`validation`** contains `valid_inventory`, `issues`, per-character `codepoints`, `utf8_roundtrip`, `normalization`, and `nfc_changed`. An issue includes an index and diagnostic code, with explanatory fields where applicable. A clean inventory is not proof of visual correctness.
- **`overlay.png`** makes omissions, merged lines and inappropriate page regions inspectable.
- **`run.json`** records source, checkpoint SHA-256, device, model-load time and page summaries. Per-line records live in each page's `result.json`.

Open `review.html`, compare every crop with the source, edit its transcription, then select **Export reviewed transcriptions**. The browser downloads a separate artifact:

```json
{
  "type": "human_review",
  "lines": [{"id": "p001-01", "text": "reviewed Unicode text"}]
}
```

Corrections are held in the browser until exported. They do not overwrite raw predictions or frozen references and are not autosaved. Export before closing or refreshing the page. The interface works offline using adjacent images and the bundled font; no text or scan is sent to a service.

## Pipeline and Unicode

The runtime separates five jobs:

1. **Load one page at a time.** Images respect EXIF orientation; PDFs use native image extraction or rasterization. A 30-megapixel page guard limits downstream processing, but is not a complete memory sandbox for decoding arbitrary files.
2. **Find lines.** Background-relative ink, components, frame heuristics, horizontal peaks and valleys identify the selected printed layout. Detached signs stay with their line. A manual region is an explicit, recorded fallback.
3. **Normalize and split distant columns.** Contrast normalization reduces scan-background variation. Wide gaps separate text from number columns. Images preserve aspect ratio at a height of 48 pixels.
4. **Recognize sequences.** Two CNN–bidirectional-LSTM–CTC recognizers average probabilities. One uses packed recurrence, one ordinary recurrence. The total parameter count is 2,121,892. CTC beam decoding uses width 10 and a conservative structural prefix filter.
5. **Validate and expose evidence.** Raw output, crops, geometry, code points, diagnostics and review files are retained. There is no dictionary, reference lookup, or language-model substitution.

The inventory covers assigned Khojki positions through U+11241, excluding reserved U+11212, plus Gujarati digits U+0AE6–U+0AEF and a small ASCII whitespace/punctuation set. Support in the vocabulary does **not** imply examples or reliable recognition: 34 Khojki classes have no real training example.

Common dependent vowels use U+1122C–U+11233; anusvara is U+11234; independent AA is U+11201; the word separator is U+1123A. Independent vowels and dependent O/AU use their atomic Unicode identities instead of visually assembled substitutes. The validator reports unsupported characters, orphan marks, adjacent vowel signs, non-atomic vowel sequences, NFC changes and UTF-8 round trips. It never silently normalizes or repairs output. Its checks are more extensive than the deliberately narrow beam prefix filter. See [architecture](ARCHITECTURE.md) for why a renderer or encoding verifier cannot certify historical glyph identity.

## Data and reference protocol

The source PDF is **not** distributed here. Its hash is recorded in [data/provenance.json](data/provenance.json). If you have the supplied assessment file, restore page images and frozen line crops:

```bash
python scripts/restore_data.py /path/to/sample_khojki_script_vatiations.pdf
```

The script checks the exact PDF hash, then extracts selected embedded scans and crops using saved coordinates. A different PDF, even a visually similar re-export, is rejected because its pixels and page ordering may differ. It does not change labels or splits. Restoration supports evaluation and data-dependent tests; it does not reconstruct every historical review image or private experiment archive.

Annotation files are UTF-8 JSON arrays. Required evaluation fields are:

| Field | Meaning |
|---|---|
| `id` | Stable line identifier. |
| `page` | One-based source-PDF page number. |
| `image` | Crop path relative to `data/`, such as `lines/p21a-01.png`. |
| `box` | Crop rectangle in native source-page pixels. |
| `split` | `train`, `dev`, or `test`. |
| `text` | Logical-order reference Unicode sequence. |

Later records also include `annotation_tokens`, `annotation_method`, `uncertainty`, and `alternate_references`. Freeze records preserve annotation and crop hashes and whether predictions had been seen. These labels are AI-assisted visual/chart transcriptions, not expert linguistic ground truth.

The deployed model uses **15 page-17 training lines / 384 code points**. Its selection set has **16 lines / 405 code points**. Attempts with expanded data and a proposed training-label correction were rejected; those extra labels were not silently folded into the final weights.

Test-set history matters:

- An initial 10-line test became development after its errors informed changes.
- Later five-line tests from rounds two and three became historical diagnostics after inspection and subsequent planning. Some earlier references were explicitly promoted into rejected expansion experiments.
- The latest five page-21 references were frozen before their predictions were examined. Two ambiguous readings were declared beforehand. They produce the same aggregate CER under the recorded alternatives.
- All material is related print. There is no book-disjoint benchmark, independently adjudicated test set, or scored handwriting set. Do not combine sequential test scores into an allegedly independent pooled benchmark.

Do not run annotation-authoring/freezing scripts over the preserved files merely to reproduce inference. They document how the exercise was conducted. If references need correction, create a separate version, document the evidence, and keep original predictions and scores.

## Results and experiment history

| Measurement | Errors / reference code points | Interpretation |
|---|---:|---|
| Latest frozen test, final model | **21 / 120 (17.50%)** | Five page-21 lines; latest primary result. |
| Same test, previous model | 21 / 120 (17.50%) | No final-round accuracy improvement. |
| Selected development result | 14 / 405 (3.46%) | Repeatedly optimized selection set. |
| Round-two fresh test at that time | 17 / 118 (14.41%) | Now a historical diagnostic. |
| Round-three fresh test at that time | 26 / 123 (21.14%) | Now a historical diagnostic. |
| Initial synthetic validation | 38 / 2,910 (1.31%) | Modern-font data, not real-page accuracy. |
| Initial real-line recognition | 36 / 75 (48.00%) | Synthetic-to-real gap. |
| Initial real lines after normalization/chunking | 22 / 75 (29.33%) | Pre-adaptation preprocessing improvement. |

For the latest five lines, detection found all five and automatic versus manually checked crops gave identical predictions: **21 character edits, zero space-involving edits, zero unsupported-code-point failures, zero UTF-8 failures, and zero exact lines**. The paired crop intervention removed no errors; this does not prove segmentation is perfect elsewhere or causally independent of recognition. Unlabelled page regions and extra detections are unscored.

Three edit alignments pair an inserted PA with independent E read as dependent I. E has no real training example. Visual similarity in a modern font is not a reason to treat those sequences as equivalent Unicode.

Round four ran nine named training studies and 11,400 total optimizer steps:

| Study | Development character errors | Decision |
|---|---:|---|
| Six matched 600-step pilots: control, old/fixed word padding, frozen BatchNorm statistics, combined fix, frozen recurrence | 16–17 | At most two candidates meeting the 16-error rule extended. |
| Old-padding / fixed-padding candidates extended to 3,000 total steps | 14 / 16 | No improvement over incumbent ensemble. |
| Three lower-rate 1,000-step runs: original labels / one proposed correction / correction plus word replay | 16 / 18 / 18 | None met the rule for extension to 5,000; **no 5,000-step follow-up ran**. |
| Five contrast/blur inference variants | 15–18 | Identity view retained. |
| Final retained ensemble | **14** | Same learned weights as the previous deployment. |

The audit found a real bug: Pillow scalar `fill=255` on RGB produced red padding, which became dark after conversion to grayscale. Converting to grayscale before padding fixes the operation. The controlled experiment did not establish an OCR accuracy gain. A failed frozen-recurrence CUDA attempt is also recorded; its cuDNN training-mode issue was fixed before the successful rerun.

A separate image-based reference audit proposed four corrections to the older round-three test. The **same predictions** score 15/128 instead of 26/123 against those proposals. This is a post-hoc, non-expert reference revision, **not improved model accuracy**. Original frozen references remain intact; proposed changes require independent confirmation. See [proposals](data/round4/proposed-reference-revisions.json) and [full ledger](RESULTS.md).

A 16-line printed page took **3.28 seconds on a two-thread Colab CPU**, with approximately 0.045 seconds model-load time. CPU and CUDA full-page text matched. That extrapolates to about **154 seconds for 47 comparable pages**, excluding interpreter startup, PDF rendering, review and final review-file serialization. This is a single-page measurement, not a collection-wide throughput guarantee or GPU cost benchmark.

## Reproducing evaluation

Run these commands after installing the package and restoring the exercise images:

```bash
python scripts/evaluate_pages.py \
  --checkpoint models/printed-khojki.pt \
  --annotations data/round4/test.json --split test \
  --device cpu --output runs/reproduced-test

python scripts/evaluate_pages.py \
  --checkpoint models/printed-khojki.pt \
  --annotations data/round2/train-dev.json --split dev \
  --device cpu --output runs/reproduced-dev
```

`evaluate_pages.py` accepts `--checkpoint`, `--annotations`, `--split {dev,test}`, `--device`, and `--output`. It detects full-page lines, matches unused detections to reference boxes by vertical overlap greater than 0.5, and separately recognizes checked crops. Matching never uses transcript text. Output includes page review directories plus `evaluation.json` with aggregate counts, per-line predictions, alignments/confusions, encoding diagnostics and the crop comparison.

CER is `sum(Levenshtein edits) / sum(reference Unicode scalar values)`. It includes vowel signs, word separators, digits and number-column spaces. It is neither a byte nor a grapheme-cluster metric and is not the mean of per-line percentages. Empty/missed predictions count as deletions. Glyph similarity and a valid encoding do not waive a character error.

For crop-only ablations, `scripts/evaluate.py` accepts one or more `--checkpoint` paths, `--annotations`, `--split {train,dev,test,all}`, `--preprocess {raw,contrast,binary}`, `--no-columns`, `--device`, `--beam`, `--layout {legacy,robust}`, `--constrained`, and `--output`. Omit inference overrides when reproducing the saved deployment configuration. Do not use test output to select another configuration and continue calling it an unseen test.

Saved predictions and logs can be inspected without executing a model. [round4-verification.json](results/round4-verification.json) records the historical deployment checks; these are distinct from the current software test suite.

## Training and resuming

Training is optional. The public checkpoint is an inference ensemble; **it is not a full optimizer-resume checkpoint or a single-model warm start** for `finetune.py`. Full historical `best.pt`/`last.pt` archives remain in the private working artifacts, indexed by [cloud/archive-index.json](cloud/archive-index.json). A public clone contains deployment weights, configurations, histories and logs, but not every intermediate weight or private input archive. Exact historical replay requires those original inputs and compatible dependency versions.

Install training dependencies in the environment that will execute the experiment:

```bash
python -m pip install -e '.[train,test]'
```

A new small study can start with synthetic training, then adapt its **single-model** checkpoint after restoring real crops:

```bash
# Example pilot configuration, not a claim to reproduce the final model.
python scripts/train_synthetic.py --device cuda --steps 500 --batch 12 \
  --repertoire expanded --output my-synthetic-pilot

python scripts/finetune.py \
  --checkpoint experiments/my-synthetic-pilot/best.pt \
  --annotations data/round2/train-dev.json --device cuda \
  --steps 600 --seed 503 --lr 0.00005 --layout robust --packed \
  --synthetic-fraction 0.25 --output my-adaptation-pilot
```

Synthetic data is rendered with HarfBuzz shaping and FreeType glyph rasterization using bundled Noto fonts. Strings stay in logical Unicode order. Font variation and image augmentation supply initial visual coverage; real lines are necessary for historical print. Real-word recombination takes words from the training split only and rejects incompatible separator/crop counts.

`train_synthetic.py` options are `--steps` (2,500), `--device` (`mps`), `--batch` (12), `--output` (`synthetic-v1`), `--repertoire {legacy,expanded}` (`legacy`), and `--resume PATH`. Its `--resume` is **weights-only warm starting**, not optimizer/RNG restoration. It may stop early after reaching its validation threshold. Always pass `--device cuda` explicitly in Colab.

`finetune.py` options:

| Option | Default | Purpose |
|---|---|---|
| `--checkpoint PATH` | Required | Single-model warm-start or resumable checkpoint. |
| `--annotations PATH` | `data/annotations.json` | JSON records containing training/development splits. |
| `--output NAME` | `adapt-v1` | Experiment directory under `experiments/`; use a new name for a new study. |
| `--device` | `mps` | Training device; use `cuda` for Colab GPU. |
| `--steps` | `600` | Final total optimizer step, including steps already completed on resume. |
| `--lr` | `0.0002` | Initial AdamW learning rate for a new run. |
| `--seed` | `31` | Random seed. |
| `--resume` | Off | Restore optimizer, step, history and random states from the checkpoint. |
| `--train-limit` | All training rows | Optional training-line cap for a learning-curve study. |
| `--synthetic-fraction` | `0.25` | Probability of a synthetic replay example. |
| `--word-fraction` | `0` | Probability of training-word recombination. Fractions must leave a nonnegative real-data remainder. |
| `--word-geometry {legacy,preserve}` | `legacy` | Historical or height-preserving word construction. |
| `--bold-replay` | Off | Add the bold Noto font to synthetic replay. |
| `--legacy-rgb-word-padding` | Off | Reproduce the known old padding defect for an ablation only. |
| `--freeze-bn` | Off | Freeze BatchNorm running statistics during adaptation. |
| `--freeze-rnn` | Off | Freeze recurrent parameters and disable recurrent dropout. |
| `--packed` | Off | Use true sequence lengths in recurrent layers. |
| `--height {48,64}` | `48` | Recognizer input height. |

Adaptation writes `config.json`, `history.json`, `last.pt`, and the development-selected `best.pt`; evaluation/checkpointing occurs every 50 steps and at the final step. Checkpoints contain optimizer state, Python/NumPy/PyTorch/device RNG state, settings, annotation hash and history. Snapshot writes replace the destination only after the new checkpoint has been fully written, protecting the previous snapshot from an interrupted write. To continue the example to step 1,200:

```bash
python scripts/finetune.py \
  --checkpoint experiments/my-adaptation-pilot/last.pt --resume \
  --annotations data/round2/train-dev.json --device cuda \
  --steps 1200 --seed 503 --lr 0.00005 --layout robust --packed \
  --synthetic-fraction 0.25 --output my-adaptation-pilot
```

Keep original settings and data on resume. The resume guard also checks the original learning-rate setting; optimizer restoration preserves its stored learning rate. Start a new study rather than using `--lr` to alter a resumed optimizer. Changed labels or mixture/architecture settings require a new named warm start. RNG restoration supports continuity; bitwise equivalence across different library versions, hardware, or nondeterministic kernels is not guaranteed.

## Colab workflow and resource limits

The original experiment workflow was prepared offline before connecting a GPU. [Khojki_round4_complete.ipynb](cloud/Khojki_round4_complete.ipynb) preserves it; [the executed notebook](cloud/Khojki_round4_executed.ipynb), receipts and logs preserve the evidence. Historical notebooks expect their matching private ZIPs and exact hashes. Their bundle builders require source images and original individual checkpoints, so a public clone alone cannot recreate all historical archives.

For a new experiment:

1. Prepare a **new** notebook, parameters and the smallest necessary data/checkpoint bundle before connecting. Do not edit unrelated notebooks.
2. Upload inputs, verify hashes and inspect the current compute balance. Select a GPU only when ready to run.
3. Run a short pilot with a declared development metric and extension rule. Extend only promising configurations. Keep test references out of model selection.
4. Save `last.pt` and `best.pt` together with config, history, logs, data hashes and environment versions. Colab's `/content` storage is temporary; saving a notebook alone does not preserve checkpoint files.
5. Copy or download the result archive, compare byte size and SHA-256, and check ZIP integrity **before** disconnecting. Verify a resumed run skips completed steps and restores the last checkpoint.
6. Disconnect/delete the runtime and verify that no active GPU session remains. Monitor the actual account balance and preserve the requested **75-unit minimum**; there is no automatic billing guarantee in the training scripts.

The round-four shutdown balance was **166.94 units**, with zero active sessions and zero ongoing units/hour in [shutdown evidence](cloud/round4-shutdown.json). These are historical observations, not a statement about the account's present state. Twelve round-four result archives passed size/hash/ZIP checks. Large embedded HTML download payloads were removed after archival to reduce browser memory use. Download files directly rather than keeping many base64 archives in notebook output.

## Development and tests

Install test dependencies and font-rendering dependencies in the checkout:

```bash
python -m pip install -e '.[test,train]'
python -m pytest -m 'not private_data' -q

# Small checks that do not execute PyTorch/model inference:
python -m pytest -m 'not integration and not private_data' -q

# After restoring the supplied PDF:
python -m pytest tests/test_private_data.py -q
```

The public-clone suite uses generated fixtures. Tests marked `integration` execute the real PyTorch/model path on CPU. Tests marked `private_data` require the exercise scans and skip when those files are absent; restore the PDF before running the complete data-dependent checks. Core coverage includes Unicode diagnostics, CTC decoding, padding/geometry, input validation, pipeline artifacts and review escaping, checkpoint handling, and packaging behavior. Tests should assert observable behavior and regressions, rather than merely duplicate implementation formulas.

Historical Colab test results in the results directory describe the source at that run, not the expanded publication-time suite. The [CI workflow](.github/workflows/tests.yml) runs the nonprivate suite on Ubuntu with Python 3.10 and 3.12 and CPU PyTorch, reports coverage, and installs a built wheel outside the checkout for a generated blank-page inference smoke test. See current CI output for the checks on the published revision. Run model tests in an environment with enough memory; do not launch multiple training jobs or large-model imports on a memory-constrained laptop.

### Validation of the publication refactor

The refactored runtime and training paths were exercised on a Colab CPU with Python 3.13.18, including the private assessment images:

| Check | Observed result |
|---|---|
| Full software suite | **157 tests passed in 13.96 seconds**, with **87% package statement coverage**. CLI subprocess tests execute separately and are not included in that coverage measurement. |
| Frozen five-line evaluation | Fresh per-line predictions and detected boxes exactly match the saved baseline; **21/120 edits (17.50% CER)** remains unchanged. |
| Training and resume | A tiny synthetic training smoke test completed; two uninterrupted adaptation steps exactly matched one step plus resume to step two in model, optimizer and RNG state, and common-step history excluding timings. This is a continuity check in one environment, not a cross-device determinism guarantee. |
| Installed wheel | Inference on a generated blank image passed outside the checkout, with the expected output files, bundled font/license and deployment-model hash. |

Evidence: [test and coverage log](results/quality-validation/pytest.log), [prediction/training verification](results/quality-validation/verification.json), and [wheel verification](results/quality-validation/wheel-verification.json). The 61,354,908-byte result archive was saved and SHA-256/ZIP-verified before runtime shutdown; its [receipt](cloud/quality-output-receipt.json) identifies the preserved archive. These checks establish refactor compatibility and operational behavior; they do not demonstrate higher OCR accuracy. GitHub CI results are separate from this completed cloud validation.

Maintain the source's separation of layout, recognition, Unicode, rendering and metrics. Factor out a function when it owns a meaningful operation or reusable invariant, not every statement. Comments should explain a non-obvious constraint—such as atomic vowels, padding color or cuDNN recurrence—not narrate straightforward code. Preserve regression tests and raw measurements when refactoring.

## Repository guide

| Path | Role |
|---|---|
| `src/khojki_ocr/__main__.py` | User-facing inference CLI and run manifest. |
| `pipeline.py` | Page loading, processing and offline review artifacts. |
| `segment.py` | Printed-page region and line detection, overlays. |
| `recognize.py`, `model.py`, `decoding.py` | Preprocessing, CRNNs and CTC inference. |
| `unicode.py`, `vocabulary.py`, `metrics.py` | Encoding diagnostics, fixed character inventory and edit metrics. |
| `resources.py`, `checkpoints.py` | Bundled model/font lookup and training checkpoint loading/writes. |
| `render.py`, `word_augmentation.py` | Shaped synthetic text and training-only word augmentation. |
| `models/` | Compact deployment checkpoints and provenance. |
| `assets/` | Noto fonts and their OFL license. |
| `data/` | Annotation versions, protocols, freeze records and audits; scan images are private. |
| `scripts/` | Training, evaluation, data restoration, report and publication tools. |
| `experiments/` | Named configurations, histories, decisions and logs; large resume checkpoints are private. |
| `results/` | Predictions, evaluations, comparisons and environment snapshots. |
| `cloud/` | Prepared/executed notebooks, builders, manifests and archive receipts. |
| `tests/` | Unit and integration regression tests. |

Historical report/annotation generators are provenance tools, not a mandatory inference pipeline. Review their inputs and destinations before executing them. Curated documentation should not be replaced by an old generated report.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `data/pages` or `data/lines` files | They are intentionally absent from the public repository. Restore the exact assessment PDF, or run inference on your own image. |
| Restoration rejects the PDF | Compare it with `data/provenance.json`; obtain the original exercise file rather than reusing coordinates on a re-export. |
| `--pdf-native` rejects a page | Omit it to rasterize; native mode requires exactly one embedded image. |
| CUDA/MPS unavailable | Use `--device cpu`, or configure a compatible GPU runtime. Device selection does not provision hardware. |
| Tofu/empty boxes instead of Khojki glyphs | Open the generated review HTML with its adjacent Noto font, or install a Khojki font in your editor. UTF-8 storage and font support are separate. |
| Bad line grouping or missing faint text | Inspect `overlay.png`; try an explicit region and record that assistance. Heuristics are tuned to selected print, not arbitrary layouts. |
| Clean Unicode diagnostics but wrong OCR | Review against the crop. Inventory and structural validity cannot identify a visually plausible wrong letter. |
| Training resume rejects changed settings/references | Keep the original settings or create a new named warm start. Do not bypass the provenance guard. |
| Ensemble checkpoint fails in the trainer | Use a single-model synthetic/adaptation checkpoint. Compact inference ensembles do not contain the trainer's full state. |
| Browser edits disappeared | Review edits require explicit JSON export; reload does not restore unexported changes. |
| Memory pressure | Stop training locally, use sequential modest page sizes, reduce batch size for synthetic pilots, and move training to Colab. Avoid large base64 notebook outputs. |

## Scope, licensing, and references

This repository demonstrates the assessment's runnable mini-pipeline, at least five real-line transcriptions, character-level evaluation, saved output, and a one-page-sized Markdown retrospective. It does not establish near-perfect Khojki OCR. The most useful next data collection is an estimated **200–500 independently reviewed printed lines** across books, typefaces, damage and rare characters, with book-disjoint evaluation and a measured learning curve. Handwriting needs different detection assumptions, writer-disjoint evaluation and substantially broader labels; **1,000–3,000 lines** is an initial planning range, not a demonstrated requirement or guarantee.

The supplied scans are for **this exercise only**. Their omission from Git is intentional; do not treat the repository or derived labels as permission to redistribute the source collection. The fonts are distributed under the [SIL Open Font License](assets/OFL.txt). That license covers the font assets, not automatically the project code, model weights or exercise data. No separate general-purpose open-source license is asserted here.

Primary references and implementation background:

- [Assessment's interactive Khojki chart](https://khojki-demo.netlify.app/).
- [Unicode 17, chapter 15: Khojki](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/) and [current Khojki code chart](https://unicode.org/charts/PDF/U11200.pdf).
- [Noto Khojki source](https://github.com/notofonts/khojki) and [font builds](https://notofonts.github.io/khojki/).
- [HarfBuzz shaping documentation](https://harfbuzz.github.io/).
- [CTC explanation](https://distill.pub/2017/ctc/).
- [Historical Khojki encoding proposal](https://www.unicode.org/L2/L2011/11021-khojki.pdf), useful for glyph history; use current Unicode assignments rather than draft assignments.
- [Kraken recognition-training documentation](https://kraken.re/main/user_guide/training_recognition.html), a possible larger-scale training route rather than evidence of an available Khojki model.
