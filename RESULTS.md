# Results and experiment ledger

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

| Line | Edits | Reference code points | CER |
|---|---:|---:|---:|
| p18r-14 | 5 | 23 | 21.74% |
| p18r-15 | 0 | 24 | 0.00% |
| p20r-09 | 5 | 26 | 19.23% |
| p20r-10 | 8 | 26 | 30.77% |
| p20r-11 | 8 | 24 | 33.33% |

Edit totals: 16 substitutions, 1 deletions, 9 insertions; one of five lines exact. Most repeated errors are inserted SA and vowel sign U (two each). Other confusions involve PA/THA, BA/NA, dependent vowels and separators. Full names/code points and alignments are in [round3-report.json](results/round3-report.json).

The primary references remain unchanged. Predeclared alternatives yield **20.16–25.20% CER** (25–31 edits, variant denominators 123–124). This sensitivity range is not a confidence interval or a corrected score. It does not eliminate the recognition gap. Footer/refrain candidates were replaced with numbered body lines before round-three training and prediction inspection because their mappings were unresolved; selection history is recorded explicitly.

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

The final page-20 CLI run took **2.907 seconds** on a two-thread Colab CPU, plus **0.044 seconds** model loading, for 16 lines. CPU/GPU text matched exactly. Page processing includes detection, recognition and scan artifacts; import startup/PDF rendering and final HTML serialization are outside this measurement. A previous page-19 run took 3.141 seconds. At the latest observation, 47 comparable printed pages would take roughly **137 seconds**, or **2.91N seconds** for N pages, excluding review and I/O variation. The supplied mix contains handwriting and cannot be promised the same quality.

T4 usage displayed about 1.15 units/hour; 2.91 seconds would correspond to roughly 0.00093 units at that rate, but the measured CPU time is not a GPU billing benchmark. Notebook setup, uploads, browser delays and idle time dominated these tiny experiments. The final observed balance was **167.31**, **0 active sessions**, **0 units/hour**, preserving the 75-unit floor. All cloud output archives were size/SHA-256 checked, ZIP-integrity checked and extracted before runtime termination. Receipts and shutdown screenshots are retained.

Initial annotation authoring/freezing spanned about 14 minutes (09:41–09:56 UTC), plus earlier chart inspection and later review. This is an AI-assisted elapsed window, not expert annotation productivity. Additional references were created in later rounds with ambiguity records. Expert adjudication remains the primary bottleneck.

Fifteen regression tests passed in the final cloud run (3.58 seconds). They cover CER scalar counting, CTC collapse/beam correctness, Unicode diagnostics, PDF input, empty pages, safe review HTML, frozen/disjoint references, faint-line/gap detection, word geometry and packed recurrence. Editable package/console checks passed in the prior evaluation. A lightweight local audit also checks current hashes, aggregate edits and Python syntax without model loading. Browser verification confirmed Khojki font rendering and a separate 16-line correction JSON without altering raw OCR.

Deployment SHA-256: `6646034c85ad073ce64e7dd1c4efe15ab2dbf68a023e435382eff3f79ff50b5c`. See [MODEL_CARD.md](MODEL_CARD.md), [round3-evaluation-complete.json](results/round3-evaluation-complete.json), [round3-report.json](results/round3-report.json) and the visual comparisons. The final training/decoder selection used development data; fresh test results were not used to choose the deployed candidate.
