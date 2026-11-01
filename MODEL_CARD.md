# Deployment model card

## Identity and intended use

This is a compact recognizer for a small, related sample of historical **printed Khojki**, supplied as part of an OCR take-home assessment. It supports an auditable line-to-Unicode workflow with mandatory human review. Handwriting, unseen books, unfamiliar typefaces and full-script coverage have not been validated.

| Property | Value |
|---|---|
| Deployment file | `models/printed-khojki.pt` |
| Identical copy | `models/round4-final.pt` |
| File size | 8,523,321 bytes |
| SHA-256 | `b4398167ea03e71277d09eba4cf5a55ef3efac454738f649f01169343ec3abb6` |
| Architecture | Two CNN–BiLSTM–CTC members, probability averaging |
| Parameters | 1,060,946 per member; 2,121,892 total |
| Input | Contrast-normalized grayscale line chunks, aspect ratio preserved, height 48 |
| Recurrence | One packed-sequence member, one ordinary member |
| Layout | `robust` |
| Decoder | CTC beam width 10, conservative structural prefix filter |
| Optical views | `[[1.0, 0.0]]`: identity strength and zero blur |
| Verified devices | CPU and CUDA; selected-model MPS behavior not verified |

There is no dictionary, ground-truth lookup or language-model replacement. The inference checkpoint records the settings needed to replay its behavior. Its serialized metadata changed in round four, but **the learned ensemble weights did not improve or change**. [models/model-card.json](models/model-card.json) is the machine-readable record.

## Training and selection provenance

The selected members derive from shaped Noto synthetic pretraining and the **original 15 page-17 real training lines, containing 384 code points**, with synthetic replay during adaptation. Their source checkpoints were:

- `experiments/precision-packed/best.pt`: packed recurrence, selected at step 2,450.
- `experiments/long-control/best.pt`: ordinary recurrence, selected at step 2,800.

These individual full training checkpoints are preserved in private experiment archives, not the public deployment package. The compact model is sufficient for inference but does not contain the optimizer/RNG state required for exact training continuation, and cannot be passed directly to the single-model fine-tuning script.

Original development has 16 lines / 405 code points. Repeated architecture, preprocessing, decoder and checkpoint selection on that set makes its minimum optimistic. Expanded-data candidates and all round-four candidates were rejected. A corrected-label experiment removed one disputed I sign from the training references (384 → 383 code points); it was **not** adopted into the selected weights.

The round-four selection was `incumbent`. Nine training runs and five optical variants did not beat it. Full settings, data hashes and decisions are preserved in experiment configurations and [the results ledger](RESULTS.md). Legacy RGB-padding experiments intentionally reproduce a known defect; their results are ablations, not a recommendation to use that operation.

## Evaluation

| Set | Character edits | CER | Status |
|---|---:|---:|---|
| Latest five page-21 lines | 21 / 120 | **17.50%** | Frozen before predictions; primary reported test. |
| Same set, previous model | 21 / 120 | 17.50% | No final-round accuracy gain. |
| Original development | 14 / 405 | 3.46% | Repeatedly used for selection. |
| Round-two test | 17 / 118 | 14.41% | Fresh at that time; now historical. |
| Round-three test | 26 / 123 | 21.14% | Fresh at that time; now historical. |

CER is aggregate Unicode-scalar Levenshtein distance, including signs, separators, digits and column spaces. It does not measure glyph similarity, word correctness or semantic recovery. There are zero exact lines on the latest five-line test.

All five latest reference lines were detected. Automatic and checked-crop predictions were identical, with 21 character edits, zero space-related edits, zero unsupported characters and zero UTF-8 failures. These are measurements on selected transcribed regions, not full-page detection recall or collection-wide OCR accuracy. The crop comparison is an intervention, not a complete causal decomposition of segmentation and recognition.

References are AI-assisted, non-expert chart transcriptions from related print. Two latest lines carry predeclared alternatives; these leave aggregate CER at 17.50%. There is no independently adjudicated or book-disjoint test. Sequential historical tests must not be pooled into one fresh benchmark.

Proposed corrections to older references lower the **same** round-three predictions from 26/123 to 15/128 edits. They remain post-hoc and non-expert; that change is not a model gain. Original frozen labels are intact. See [proposed revisions](data/round4/proposed-reference-revisions.json), [latest references](data/round4/test.json), and [freeze record](data/round4/test-freeze.json).

## Known limitations

- Thirty-four supported Khojki classes have no real training examples. Vocabulary membership does not imply recognition competence; independent E is a prominent observed failure.
- Modern Noto rendering differs from historical print. Shaping correctness and synthetic validation accuracy do not establish historical glyph identity.
- Printed-frame, line-spacing and column-gap heuristics can fail on skew, faint text, dense handwriting, multiple columns or decorative material.
- Structural Unicode checks cannot detect plausible wrong letters. Every prediction is flagged for review; there is no calibrated confidence or automatic acceptance threshold.
- The dataset is tiny and references uncertain. The reported result does not support universal, near-perfect or handwritten OCR claims.

## Runtime and reproducibility

A 16-line page took 3.283 seconds on a two-thread Colab CPU, with approximately 0.045 seconds model loading. CPU and CUDA text matched. The page timing excludes interpreter startup, PDF rendering, human review and final review-file serialization. Throughput extrapolation to other layouts and devices is unvalidated.

A fresh evaluation after the publication refactor reproduced the saved five-line predictions and detected boxes exactly. The full cloud suite passed 157 tests; installed-wheel inference also passed outside the checkout. [Verification evidence](results/quality-validation/verification.json) records these compatibility checks, not an additional accuracy gain.

Editable and regular package installations provide the default model and review font. Repository scripts and restored private images are needed for the saved evaluation protocol. [README.md](README.md) gives complete inference, evaluation and training instructions. Load only trusted project checkpoints.

Source scan provenance is in [data/provenance.json](data/provenance.json). The assessment scans are for exercise-only use and are excluded from public Git. The [font license](assets/OFL.txt) covers Noto assets; it does not license the scan collection, all derived artifacts or project code.
