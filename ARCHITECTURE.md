# Pipeline design and engineering decisions

## Problem decomposition

The input is a degraded rendering in a historical typeface; the output is a sequence of abstract Unicode characters. Page geometry, visual recognition and character encoding have different failure modes. The implementation keeps them separate so a valid code point cannot masquerade as evidence that the correct glyph was read.

```text
Page image / PDF
    ↓ native scan extraction or PDF rendering
Page pixels
    ↓ frame/region inference and line detection
Line crops + page coordinates
    ↓ background-relative contrast and distant-column splitting
Normalized text/number chunks
    ↓ two CNN–BiLSTM–CTC members; probability averaging
CTC beam decode with conservative structural filtering
    ↓ join chunks; preserve raw Unicode
UTF-8 + geometry + validation + offline review/export
```

The model does not receive a page number, reference transcription, dictionary or list of expected words. Dataset identifiers exist for provenance and reporting. Manual page regions are supported as explicit assistance and recorded in output.

## Module boundaries

| Module | Responsibility |
|---|---|
| `__main__.py` | Parse/validate inference arguments, select resources and device, write the run manifest. |
| `resources.py` | Resolve model/font assets from an editable checkout or installed package. |
| `pipeline.py` | Load pages, orchestrate one-page inference, write text/JSON/images/review HTML. |
| `segment.py` | Ink masks, printed-frame heuristics, line boxes and overlays. |
| `recognize.py` | Contrast preprocessing, distant-column splitting, checkpoint-driven ensemble inference. |
| `model.py`, `vocabulary.py` | Fixed vocabulary, image tensors, batching, CRNN and greedy CTC collapse. |
| `decoding.py` | CTC prefix beam search. |
| `unicode.py` | Allowed inventory, structural diagnostics and a deliberately narrower decoding filter. |
| `metrics.py` | Unicode-scalar edit distance, alignments and aggregate-score inputs. |
| `checkpoints.py` | Trusted single-model training loads and interruption-safe snapshot replacement. |
| `render.py` | HarfBuzz/FreeType shaping and rasterization. |
| `word_augmentation.py` | Verified training-word extraction and controlled recombination. |

Training and evaluation live in `scripts/`; their command entry points are distinct from reusable inference code. Experiment-specific policies remain named and recorded rather than being folded into runtime defaults. Functions group meaningful stages and invariants. Comments explain decisions such as padding color, atomic Unicode identities and recurrent training behavior.

## Geometry before recognition

Ink is measured relative to the local background. Connected components and density profiles help identify ornamental frames; horizontal peaks and valleys locate line bodies while retaining detached signs. Treating every component as a character would split vowels and dots and merge touching letters. The original threshold/projection version did split detached marks and merge handwriting; adaptive peaks were more suitable for the selected printed pages.

Large horizontal gaps split text and number chunks before recognition. Ignoring tiny speckles while locating these gaps avoids treating noise as a column bridge. One ASCII space joins recognized chunks; the printed word separator remains its own Khojki character.

These are heuristics for selected print, not a general layout model. Headers, skew, faint strokes, dense handwriting and multiple columns can violate their assumptions. Overlays and the checked-crop evaluation expose these errors. The initial test became development before detector changes were selected; its old score is not an independent test of the revised detector.

Images preserve EXIF orientation. PDFs are rendered at 150 DPI unless native extraction is requested; native extraction requires a single embedded image per selected page. Geometry is tied to that resulting pixel coordinate system. Frozen assessment boxes refer to native source scans and cannot be reused on arbitrary rescaled pages.

## Sequence recognition and shaping

Each recognizer has 1,060,946 parameters: a CNN compresses line height, a two-layer bidirectional LSTM supplies visual sequence context, and CTC learns variable-length transcriptions without character boxes. Height is normalized to 48 pixels while preserving width. Two members average probabilities: one reads ordinary padded batches and one uses packed sequence lengths to prevent right padding from affecting recurrent context.

The checkpoint stores vocabulary, height, recurrence mode, layout, beam width, constraints and optical views. The selected beam width is 10. Optical views preserve dimensions and temporal alignment before probability averaging; the selected view is identity, so no test-time blur/contrast ensemble is active. There is no language-model prior that could silently replace a rare name with a common word.

Khojki is an abugida: character/sign combinations are not independent image tiles. HarfBuzz chooses glyphs and positions from logical Unicode strings; FreeType rasterizes those shaped glyph IDs. Bidirectional context helps recognize signs whose visual placement differs from their logical sequence position. Synthetic rendering is a useful starting point, but worn historical print differs from a modern Noto font.

The empirical gap was large: synthetic validation reached 1.31% CER while initial real-line recognition was 48%. Contrast normalization and chunking brought the initial real result to 29.33%; real-line adaptation was still needed. Synthetic success is therefore not a proxy for archival accuracy.

## Data operations and rejected changes

Real adaptation uses frozen training lines with synthetic replay. Word augmentation uses training-only crops and requires agreement between separator-derived crop count and reference word count. That catches some segmentation mistakes, but does not prove every word is correctly bounded or transcribed.

Two bugs/limitations were studied separately. Earlier word resizing reduced ink height; preserving geometry improved its paired result from 26 to 23 development edits, still worse than the incumbent. Later inspection showed that scalar `fill=255` on RGB images produces red padding, becoming dark after grayscale conversion. Converting to grayscale before padding fixes that image contract. The legacy mode remains only for a named reproducibility ablation.

Round four compared six matched 600-step pilots and extended two eligible cases to 3,000 steps. Fixed padding, frozen BatchNorm statistics and frozen recurrent parameters did not improve selection accuracy. Three 1,000-step lower-rate studies tested original references, one isolated training-label proposal and added word replay. None qualified for the conditional 5,000-step extension. Five optical variants also failed to beat the incumbent.

A frozen-recurrence run initially failed because cuDNN backward still required the recurrent module to be in training mode. The corrected configuration freezes parameters and disables recurrent dropout while retaining the necessary mode; failed-run evidence and the completed rerun are preserved. A code fix and a measured OCR gain are separate claims.

## Verification has a bounded role

The validator checks inventory, orphan signs, adjacent vowels, atomic-vowel conventions, NFC changes and UTF-8 round trips. Raw output remains unchanged. The beam filter rejects only a subset of implausible prefixes: orphan marks, adjacent dependent vowels and A followed by the AA sign. It is not the entire validator and not a language model.

A render-and-compare reward could reward modern-font resemblance instead of historical identity, favor a visually similar wrong sequence, or exploit alignment errors. No reinforcement-learning verifier loop was used. Rendering and checks aid transcription/review; they do not decide whether a damaged historical stroke is present.

Unicode decisions follow the [Unicode 17 Khojki specification](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/):

- Reserved U+11212 is excluded; assigned supported Khojki positions extend through U+11241.
- Common dependent vowels are U+1122C–U+11233 and anusvara is U+11234.
- Independent AA is U+11201, not A plus the AA sign. Other atomic independent vowels and dependent O/AU are not replaced by visually assembled sequences.
- The printed word separator is U+1123A. Number-column spacing is ASCII space; observed digits are Gujarati U+0AE6–U+0AEF.
- Encoding, normalization and font display are distinct checks. A warning requests source review, not automatic replacement.

The [current code chart](https://unicode.org/charts/PDF/U11200.pdf) determines assignments. The [2011 proposal](https://www.unicode.org/L2/L2011/11021-khojki.pdf) helps interpret historical shapes, but its draft assignments must not override the current standard. Supporting references: [Noto Khojki](https://github.com/notofonts/khojki), [HarfBuzz](https://harfbuzz.github.io/), and [CTC](https://distill.pub/2017/ctc/).

## Evaluation and reference uncertainty

Whole-page evaluation matches detections to reference boxes geometrically, without looking at OCR text. It also recognizes manually checked crops. A missing matched line yields an empty prediction and deletion errors. Unlabelled regions and extra detections remain unscored. The difference between automatic and checked-crop errors measures an intervention; interactions prevent interpreting it as a universal causal decomposition.

The selected model has 14/405 development edits and **21/120 on the latest frozen five-line test**, identical to the previous model. Automatic and checked crops produce the same five predictions. All 21 edits are character errors rather than spaces; inventory/UTF-8 failures are zero. Three alignments expose independent E confused with PA plus dependent I, and E has no real training example.

References were transcribed with AI/chart assistance, not independently adjudicated by an expert. Later repeated-word image matching identified four proposed corrections to older references. Against the same old predictions those proposals change 26/123 to 15/128 edits. This is post-hoc reference uncertainty, not a model improvement. Original files remain intact, ambiguities are explicit, and the latest test was separately frozen before predictions.

Repeated development selection and sequential reuse of earlier tests limit generalization claims. The final weights still derive from 15 original training lines; rejected expanded/corrected-label studies did not enter deployment. The evidence supports a runnable, measured prototype and an honest failure analysis, not universal or near-perfect OCR. See [RESULTS.md](RESULTS.md) for the complete decision ledger.

## Operational design and next evidence

Inference runs sequentially with a two-thread CPU default, writes raw outputs separately from human corrections, and records the checkpoint hash. A 30-megapixel guard bounds downstream page processing; it does not make arbitrary file decoding memory-safe. Installed packages include the model and review font. Training checkpoints preserve optimizer, history, data hash and random states; compact deployment checkpoints serve inference instead.

Colab experiments should start from a prepared notebook and verified inputs, begin with short pilots, persist checkpoints outside ephemeral runtime storage, verify exported archives, and disconnect after work. Historical full archives are indexed but private; a fresh clone can run inference and generated-fixture tests without them. No accuracy gain is claimed merely because packaging, readability or tests improve.

The next useful experiment is an independently reviewed, book-diverse data collection: an estimated 200–500 printed lines, a double-annotated subset and learning curves at 50/100/200/500 lines, with entire books reserved for testing. Thirty-four supported Khojki classes currently lack real examples, so rare-character coverage must be measured explicitly.

Handwriting needs baseline/polygon detection, writer-disjoint references and broader visual variation; 1,000–3,000 labelled lines is a planning range, not a measured threshold. [Kraken](https://kraken.re/main/user_guide/training_recognition.html) is a possible training framework, but a framework does not supply the missing Khojki labels or establish a pretrained recognizer's suitability.
