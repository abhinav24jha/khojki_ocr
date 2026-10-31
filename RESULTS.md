# Round-four results and audit

The selected configuration is **incumbent**, chosen on the unchanged 16-line / 405-character development set: **14/405 (3.46% CER)**. On five newly frozen page-21 lines it produces **21/120 (17.50% CER)**, versus **21/120 (17.50%)** for the previous model. These are non-expert chart transcriptions, not certified ground truth or collection-wide accuracy.

[Visual comparison](results/round4-comparison.html) · [Source-label audit](data/round4/reference-audit.html) · [Full prior experiment ledger](RESULTS_ROUND3.md)

## What the first-principles audit found

**A real augmentation defect:** Pillow scalar `fill=255` on RGB images creates red padding, not white. Grayscale conversion made those borders dark artificial strokes. `training_words` now converts to grayscale before padding; a regression test checks RGB input and white output margins. A legacy flag reproduces the old behavior for a matched comparison. Both variants reached 16 development edits after 600 steps; through 3000 steps the old artifact reached 14 and the fixed version 16. The fix is correct image handling, but this experiment does **not** establish an accuracy benefit. The default data-generation path is fixed; old trained candidates and their provenance are retained.

**Our AI-authored references contained likely mistakes:** image-only similarity screening found inconsistent labels on near-identical words. Enlarged scan/font comparisons support four historical-test word revisions and one training-only revision (p17r-10, removing an unsupported I sign). Originals remain frozen. Proposed corrections have explicit source evidence and are not expert-certified. They were not silently substituted into the headline test results.

| Same model, different reference version | Original historical test | Proposed source-audited reference |
|---|---:|---:|
| round4 | 26/123 (21.14%) | 15/128 (11.72%) |
| incumbent | 26/123 (21.14%) | 15/128 (11.72%) |

This table separates label effects from model effects. The incumbent's proposed-score reduction from 26/123 to 15/128 is **not an OCR improvement**. The changes were post-hoc, after earlier model outputs had been seen, so the revised score is diagnostic rather than a fresh unbiased estimate. Unresolved p18r-14 and p19r-06 readings remain unresolved. Re-rendering a modern font supports review but is not an oracle for historical glyph identity.

**Coverage remains narrow:** 34 supported Khojki characters have no positive examples in the 15 real training lines; some occur in synthetic replay only. Class coverage is recorded in `data/round4/class-coverage.json`. Output-vocabulary support is not recognition competence. A pixel-similarity audit examined 46 known lines / 190 separator-delimited word crops; three lines failed separator-count checks and were excluded from word crops. Similarity candidates are review prompts, not automatic relabelling.

## Controlled experiments

Nine new named training runs were completed. The initial six share the packed warm start, original labels, seed 503, learning rate 5e-5, 25% synthetic replay, robust layout and bold replay. They isolate no words, legacy red padding, fixed white padding, frozen BatchNorm statistics, fixed words plus frozen BatchNorm, and frozen recurrent weights. The frozen-recurrent pilot initially hit a cuDNN backward constraint; frozen weights now retain training mode with dropout disabled. It was rerun successfully; this failure is documented rather than counted as a completed accuracy result.

At most two 600-step pilots with <=16 edits were extended to 3000; ties chose red and white padding, preserving the paired comparison. Candidate selection also tested a two-new-member ensemble and a strongest-new/old-member ensemble. The single red-padding candidate tied the 14-edit incumbent; its historical regression totals were 17/118 and 26/123.

The follow-up uses seed 607 and a five-times-lower learning rate (1e-5): original labels, one corrected training label, and corrected labels plus 10% fixed word recombination. All first run 1000 steps. Up to two candidates with <=15 development edits extend to 5000, retaining the best checkpoint. The final selection compares these with incumbents and one mixed ensemble, then five fixed optical variants: weaker ink, stronger ink, mild blur, three-strength averaging and original/blur averaging. No new test predictions enter this selection. The small reused development set makes every selected minimum optimistic.


### 600-step pilots

| Candidate | Edits / 405 | CER |
|---|---:|---:|
| incumbent | 14 | 3.46% |
| packed-start | 15 | 3.70% |
| r4-control | 17 | 4.20% |
| r4-word-red | 16 | 3.95% |
| r4-word-white | 16 | 3.95% |
| r4-freezebn | 16 | 3.95% |
| r4-word-white-freezebn | 16 | 3.95% |
| r4-freezernn | 16 | 3.95% |

### After paired 3000-step extensions

| Candidate | Edits / 405 | CER |
|---|---:|---:|
| incumbent | 14 | 3.46% |
| packed-start | 15 | 3.70% |
| r4-control | 17 | 4.20% |
| r4-word-red | 14 | 3.46% |
| r4-word-white | 16 | 3.95% |
| r4-freezebn | 16 | 3.95% |
| r4-word-white-freezebn | 16 | 3.95% |
| r4-freezernn | 16 | 3.95% |
| new-ensemble | 14 | 3.46% |
| mixed-ensemble | 15 | 3.70% |

### 1000-step low-rate pilots

| Candidate | Edits / 405 | CER |
|---|---:|---:|
| r4-low-original | 16 | 3.95% |
| r4-low-corrected | 18 | 4.44% |
| r4-low-white | 18 | 4.44% |

### Final development-only selection

| Candidate | Edits / 405 | CER |
|---|---:|---:|
| incumbent | 14 | 3.46% |
| round4-selected | 14 | 3.46% |
| r4-low-original | 16 | 3.95% |
| r4-low-corrected | 18 | 4.44% |
| r4-low-white | 18 | 4.44% |
| low-mixed | 18 | 4.44% |
| incumbent-weak | 18 | 4.44% |
| incumbent-strong | 16 | 3.95% |
| incumbent-blur | 15 | 3.70% |
| incumbent-ink-average | 16 | 3.95% |
| incumbent-blur-average | 15 | 3.70% |

### Training lengths and saved checkpoints

| Run | Total completed steps | Best greedy checkpoint step | Best greedy dev edits / 405 |
|---|---:|---:|---:|
| r4-control | 600 | 50 | 18 |
| r4-word-red | 3000 | 2250 | 15 |
| r4-word-white | 3000 | 250 | 18 |
| r4-freezebn | 600 | 50 | 18 |
| r4-word-white-freezebn | 600 | 550 | 18 |
| r4-freezernn | 600 | 50 | 16 |
| r4-low-original | 1000 | 950 | 19 |
| r4-low-corrected | 1000 | 750 | 20 |
| r4-low-white | 1000 | 250 | 19 |

Training checkpoint choice uses greedy CER every 50 steps; the tables above re-evaluate best checkpoints with the fixed beam-10 structural decoder. The two criteria are deliberately named, not conflated. Full histories, including later regressions, are retained. Best/last checkpoints include optimizer and RNG state. Longer training was used when pilots met the predeclared rule, rather than assumed to improve quality.

## Fresh evaluation and stage attribution

The first five detected body lines on previously untranscribed page 21 were selected before round-four outputs. All five were retained. Chart-based references and two explicit alternatives were frozen before any recognition output for that page was read (`data/round4/test-freeze.json`). Two lines have unresolved base-letter ambiguity. Predeclared alternatives give **17.50%–17.50% CER**; this is a sensitivity range, not a confidence interval. The fixed primary denominator is 120 scalar values. Do not merge these sequential tests into an independent benchmark.

- Missed selected lines: **0**.
- Automatic versus checked crops: **21 versus 21 edits**; 0 changed predictions.
- Space-involving edits: **0**; other character edits: **21**.
- Invalid-inventory characters: **0**; UTF-8 failures: **0**.
- Exact primary-reference lines: **0/5**.

This is an operational breakdown. Valid-but-wrong code points are recognition errors that an inventory check cannot detect; crop interventions do not uniquely prove causation. Unlabelled page regions are not scored. Three repeated alignments pair inserted PA with independent E-to-vowel-I substitution; a fourth E location is also confused. E has no positive real training example. E and PA+I have similar outlines but remain distinct encodings; this post-evaluation diagnostic does not change references. Detailed alignments and code points are in `results/round4-report.json`.

## Reproducibility, throughput and limits

The final serialized checkpoint reproduced every selected development prediction. All 16 regression tests passed in the final cloud evaluation. A subsequent diagnostic-only Unicode audit added missing atomic I/U/AI/AU checks from Unicode 17 Table 15-4; 14 focused tests passed locally without importing ML libraries. This adds warnings, not text rewriting or decoder changes. CPU and CUDA produced identical full-page-21 text. The CPU CLI processed 16 lines in **3.283 seconds**, plus **0.045 seconds** loading, using two threads. This excludes Python startup/PDF rasterization and manual review. Rough extrapolation: **154 seconds for 47 comparable printed pages**, or 3.28N seconds for N pages. The mixed handwritten collection cannot inherit this quality estimate.

Deployment contains 2 model member(s), 2,121,892 parameters in total; optical views are `[[1.0, 0.0]]`. No API or retraining is required. SHA-256: `b4398167ea03e71277d09eba4cf5a55ef3efac454738f649f01169343ec3abb6`. Model files, all nine resumable experiment archives, manifests, complete logs and outputs were saved and verified before runtime deletion. Final observed balance: **166.94 units**, **0 active sessions**, **0 ongoing usage**, above the 75-unit reserve.

The next high-value work is 200–500 independently adjudicated printed lines spanning books, rare forms and degradation, with book-disjoint tests and a measured learning curve. That is a planning estimate, not a promise. Handwriting needs writer-disjoint labels and a different geometry regime; no handwriting accuracy is claimed. Further tiny-development-set tuning cannot substitute for reference quality or coverage.
