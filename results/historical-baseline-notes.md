# Archived initial-baseline report

This is the report before subsequent rounds. “Final” below refers only to that historical phase. Current results are in ../RESULTS.md.

# Results and experiment ledger

The final model was selected before inspecting test predictions. The reference freeze, model-selection freeze, configuration files, histories, and raw predictions are preserved. CER figures with different denominators below are deliberately separated; synthetic and real development scores are not interchangeable.

## Final test: automatic pages, selected reference lines

| Measure | Result |
|---|---:|
| Transcribed test lines / reference code points | 10 / 250 |
| Automatically detected selected lines | 10 / 10 |
| Automatic-page edits / CER | 21 / **8.40%** |
| Substitutions / deletions / insertions | 14 / 4 / 3 |
| Checked-crop edits / CER | 21 / **8.40%** |
| Exact lines | 1 / 10 |
| Page 19 subset | 8 / 126 = 6.35% |
| Page 20 subset | 13 / 124 = 10.48% |
| Out-of-inventory output characters / UTF-8 failures | 0 / 0 |
| Lines flagged for multiple vowel signs | 1 |

Source: [`test-pages/evaluation.json`](results/test-pages/evaluation.json). The page detector found 16 lines on page 19 and 15 on page 20. Only the ten pre-transcribed lines are scored; extra detections elsewhere are not adjudicated as correct/incorrect by this metric. Visual overlays accompany every page.

The ten references cover a narrow, related printed style. Page 19 was not used for training/development, while page 20 has disjoint development and test lines on the same page. This is a stronger check than scoring training crops, but weaker than a book-disjoint random sample. The AI-assisted references are not independent expert truth. These constraints prevent a credible claim about archive-wide accuracy or a narrow statistical confidence interval.

## Error-stage audit

The operational attribution is **0 missed/merged selected lines; 2 column-layout space deletions; 19 remaining recognition edits; 0 mechanical Unicode-mapping/storage errors**. The two missing spaces correspond to test rows `p19r-01` and `p19r-02`; the gap splitter produced one chunk on each, while the reference includes a separated number column. See [`column-audit.json`](results/column-audit.json) and [`stage-audit.json`](results/stage-audit.json).

Checked crops and automatic crops both total 21 edits, but two predictions differ. One crop changes a vowel mistake and another trades a missing space for a vowel insertion. Therefore a net zero CER difference is **not** proof that geometry has no effect. The attribution above describes observable output failures; it is not a uniquely identifiable causal decomposition. A wrong valid Khojki character is a recognition error here, not a UTF-8 mapping error. The multiple-vowel warning is retained in raw output.

Most frequent errors:

| Reference → prediction | Count |
|---|---:|
| CA U+1120E → GA U+1120A | 3 |
| Column space → deletion | 2 |
| KHA U+11209 → PA U+1121F | 2 |
| PA U+1121F → RA U+11226 | 2 |
| MA U+11224 → TA U+11219 | 2 |
| Anusvara U+11234 → deletion | 2 |

Similar historical letter forms and small detached signs dominate. A code-point allowlist cannot catch these substitutions. The complete alignments and ten source/reference/prediction comparisons are available in the JSON, TSV, and HTML artifacts.

## Iterative experiments

| Experiment | Evaluation set | Errors / characters | CER | Decision |
|---|---|---:|---:|---|
| Synthetic CRNN, 1,000 steps | 120 synthetic lines | 38 / 2,910 | 1.31% | Visual pretraining only |
| Synthetic model, raw whole line | Initial 3 real dev lines | 36 / 75 | 48.00% | Reject preprocessing |
| Synthetic model, contrast + columns | Initial 3 real dev lines | 22 / 75 | 29.33% | Keep preprocessing |
| Synthetic model, binary + columns | Initial 3 real dev lines | 28 / 75 | 37.33% | Reject binarization |
| Adapt-v1, 15 real lines | Initial 3 dev lines | 6 / 75 | 8.00% | Real labels help |
| Adapt-v1, expanded dev | 6 real dev lines | 13 / 155 | 8.39% | Compare on larger dev |
| 3-line adaptation, seed 51 | 6 real dev lines | 18 / 155 | 11.61% | Learning-curve probe |
| 5-line adaptation, seed 51 | 6 real dev lines | 19 / 155 | 12.26% | Learning-curve probe |
| **10-line adaptation, seed 51** | **6 real dev lines** | **10 / 155** | **6.45%** | **Selected; step 300** |
| 15-line adaptation, seed 51 | 6 real dev lines | 13 / 155 | 8.39% | No improvement |
| Prefix beam width 10 | 6 real dev lines | 11 / 155 | 7.10% | Reject; retain greedy |
| Three-checkpoint ensemble | 6 real dev lines | 12 / 155 | 7.74% | Reject |
| T4 warm start, 100 steps, lr 5e-5 | 6 real dev lines | 13 / 155 | 8.39% | Reject; retain original |

The 3/5/10/15-line experiments each ran 500 steps, with checkpoints selected on development CER every 50 steps. Their best steps were 450/300/300/400. Non-monotonic results on six development lines do not show that more labels are harmful; sampling, label ambiguity, and repeated selection on a tiny set confound that conclusion. The 384 training code points describe the full 15-line pool; the selected model's ten-line subset is specified in `experiments/curve-10/config.json`.

The expanded-repertoire synthetic trial reached 111/3,252 = 3.41% on its different synthetic validation set at step 300 before interruption. It did not receive a comparable real-data evaluation and was not selected. Its score is not evidence of superiority or inferiority to the legacy curriculum.

Two VLM smoke tests on one untranscribed print line failed: Gemma3:4b emitted repetitive Kannada (37.0s); Gemma4:e2b, even with a chart image, emitted repetitive Devanagari (63.9s). Both hit their 700-token limit with incomplete JSON. There is no trustworthy CER for those untranscribed runs; report **2/2 wrong-script/format failures**, not an invented accuracy figure. The raw responses are saved. Qwen3-VL8B did not finish: concurrent local VLM inference and training caused memory exhaustion. It has no accuracy result. Heavy local work was stopped; later experiments used Colab only.

## Fixed development robustness probes

| Transform | Errors / 155 | CER |
|---|---:|---:|
| Unchanged | 10 | 6.45% |
| Half resolution | 15 | 9.68% |
| Gaussian blur, radius 0.8 pixels | 10 | 6.45% |
| Rotate 1 degree | 14 | 9.03% |

These probes were prewritten, applied to development crops, and did not trigger another model-selection cycle. They suggest checking scan resolution and skew before deploying on a different collection. They do not establish robustness to severe damage. Page 23 (21 detected lines) and handwritten page 4 (23 detected lines) have saved output but **no CER claim**; inspect them as failure/generalization examples.

## Runtime, cost, and reproducibility

- Cloud CPU, 2 threads, page 19 (739×1200; 16 lines): **1.68s** for page processing and artifact preparation, plus **0.026s** model loading. Python/import startup and PDF rendering are excluded. CPU/GPU page-19 texts matched exactly.
- T4, automatic pages: page 19 **2.03s**, page 20 **0.93s**; the first page includes GPU warm-up. Model startup is separate; PNG/crop output is included in these page timings, while final HTML serialization is just outside the measured block.
- At the CPU observation, 47 comparable printed scans take approximately **79s**, excluding startup/rendering, manual review and I/O variation. For an unknown full collection of N pages, use approximately **1.7N seconds** only as a same-format compute estimate. The supplied 47-page mix includes handwriting, so this is a throughput illustration, not a promise of comparable OCR quality.
- The UI showed about **1.15 compute units/hour** for T4. The observed page times imply roughly 0.0003–0.00065 units/page for sustained processing, excluding setup/idle time. No dollar conversion is claimed. Account balance was 168.49 before cloud work and **168.37 after**, with **zero active sessions** and zero ongoing hourly use verified. The 75-unit reserve was preserved.
- Annotation authoring/freezing spanned roughly 09:41–09:56 UTC (about 14 minutes), with earlier chart inspection and concurrent work. This is an AI-assisted elapsed window, not a measured human annotator rate. Expert review time remains unknown.

Eight tests passed in 3.07s on Colab. The bounded pilot and complete evaluation output archives were downloaded through visible notebook links, checked against their cloud SHA-256, integrity-tested, and extracted locally before shutdown. Checkpoints, logs, model metadata, source hashes, package environment, and browser shutdown evidence are retained. See `cloud/session.json` and `cloud/runtime-disconnected.png`.
