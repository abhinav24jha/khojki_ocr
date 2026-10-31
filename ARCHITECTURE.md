# Why this pipeline

The recoverable target is a sequence of abstract Unicode characters, while the input is a degraded rendering in an unfamiliar historical typeface. A good system must separate page geometry, visual recognition, and encoding. A correct code-point inventory alone cannot determine which glyph was printed; a modern-font renderer alone cannot establish equivalence to a historical form.

## Geometry before recognition

The page detector uses background-relative ink, connected components, ornamental-frame density, horizontal ink peaks, and valleys between peaks. A line contains its detached signs as well as its main body. Treating every connected component as a character would split vowels and dots while merging touching letters. The detector takes no page identifier or transcript as input. An explicit manual region is available as an auditable fallback.

The original threshold-projection prototype split detached marks and merged handwriting into large blocks. Adaptive peaks were more suitable for the selected print. This remains a heuristic detector: headers, damaged frames, multiple columns, skew, and close handwriting can invalidate its assumptions. Whole-page overlays expose failures instead of hiding them behind crop-level CER.

Large horizontal gaps are split into text and number chunks before recognition, with one ASCII space between recognized chunks. This reduced the synthetic-to-real mismatch on the initial development set. The first evaluation exposed two missed text/number gaps and one short refrain merged with its following line. The revised detector ignores tiny speckles only for gap finding, lowers the faint-line threshold, and rejects implausible frame crops. Original test lines were explicitly reclassified as development before these changes; five new references were frozen before inspecting their predictions. The earlier 8.4% result remains a historical baseline, never a fresh test of the revised system.

## Learn a visual sequence, not isolated glyph templates

A 1.06M-parameter convolutional network compresses line height; a bidirectional two-layer LSTM provides context; CTC learns variable-length output without character boxes. Images are normalized to 48 pixels in height while retaining width. Decoding is selected on development data; the shipped checkpoint records input height, recurrence mode, beam width, structural filtering, and any ensemble members. Early beam/ensemble trials worsened the original development result, so later decoder changes were measured again against the stronger model. There is no language-model prior that could silently replace a rare name with a more likely word.

Khojki is an abugida: base characters and signs are not independent image tiles. Shaping is therefore part of data generation, with HarfBuzz deciding glyphs, positioning, and combinations and FreeType rasterizing the actual shaped glyph IDs. The labels stay in logical Unicode order. Bidirectional sequence context helps with visually reordered signs; an isolated connected-component classifier would make this substantially harder. A perfectly rendered synthetic line still lacks the worn strokes, spacing, and typeface variation of an old scan, so a small real-data adaptation stage is essential.

The measured synthetic validation CER was 1.31%, but real development CER was initially 48%. Contrast normalization and chunking reduced it to 29.33%; real-line fine-tuning reduced it further. This is direct evidence that modern-font synthetic success is a poor proxy for archival success.

## Controlled architectural and data experiments

Longer ordinary fine-tuning improved the expanded 405-character development set from 7.65% to 4.44% CER. This justified a longer experiment; the 300-step result alone had been 6.91%. Every 50-step checkpoint evaluation is retained, including the many later steps that were worse than the selected minimum. Repeated selection on this tiny development set can overfit it, which is why it cannot stand in for the fresh test.

The bidirectional LSTM originally read right padding past a line's true width. A packed-sequence variant removes that dependency; a regression test verifies invariance to additional padding. Its 600-step pilot reached 4.69% versus a matched control's 4.94%, but did not yet beat the 4.44% incumbent. Both 64-pixel variants reached only 8.64% in their pilots. Changing scale is a new input distribution, and enlarging pixels does not recover lost scan detail.

Five additional, non-expert training transcriptions were tested separately: 5.68% development CER, so more labels were not automatically accepted as an improvement. Separator-delimited real-word recombination initially made performance worse. Inspection found that padding and two resizing stages unintentionally reduced ink height. A subsequent paired experiment isolates that geometry change. Word segmentation is checked against the training reference separator count, but matching counts cannot prove every crop is correct. No development/test words are added to the augmentation bank.

## Verification has a bounded role

The verifier checks the allowed inventory, orphan signs, suspicious vowel sequences, explicit atomic-vowel conventions, and UTF-8 round trips. Reference lines were also rendered beside the scan during annotation. These checks catch wrong-script output and encoding defects cheaply. They cannot distinguish every historical TA from MA or decide whether a damaged stroke exists.

A render-and-compare reward could teach a model to imitate Noto instead of the document, reward an incorrect but similar code-point sequence, or exploit imperfect image alignment. Consequently, there is no reinforcement-learning loop against such a verifier here. It is an annotation/review aid and a diagnostic, not an oracle. All predictions remain available, and the correction UI exports a separate artifact. A separate decoding ablation uses only a conservative prefix filter for orphan marks and adjacent vowel signs; it does not add a dictionary or decide between visually plausible letters. Any such decoder selection is recorded explicitly in model metadata.

## Unicode decisions

- Khojki block identities follow the [Unicode 17 core specification, South/Central Asian scripts](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/). U+11212 is reserved and excluded.
- Common dependent vowels use U+1122C–U+11233; anusvara is U+11234. Independent AA is U+11201, not A plus the AA sign. O/AU signs use U+11232/U+11233, not decomposed AA+E/AI guesses.
- A visible Khojki word separator is U+1123A, not an ASCII colon. Arabic-looking dots are not sufficient evidence for punctuation identity without inspecting the chart/context.
- Visible digits in these lines use Gujarati U+0AE6–U+0AEF. A space marks a distant number column; it is not substituted for a printed word separator.
- UTF-8 storage and font rendering are different checks. NFC is inspected but raw text is not silently normalized or repaired. A validator warning means review, not an automatic replacement.

Fonts and shaping references: [Noto Khojki source](https://github.com/notofonts/khojki), [Noto font builds](https://notofonts.github.io/khojki/), [HarfBuzz documentation](https://harfbuzz.github.io/), and the [CTC explanation](https://distill.pub/2017/ctc/). The [2011 Khojki proposal](https://www.unicode.org/L2/L2011/11021-khojki.pdf) is useful for historical shapes; draft code-point assignments must not replace the current Unicode mapping. [Kraken's recognition-training documentation](https://kraken.re/main/user_guide/training_recognition.html) describes a practical larger-scale HTR route, but using a framework does not supply a pretrained Khojki recognizer.

## Highest-value next experiment

Obtain independently reviewed, diverse real lines, rather than indefinitely tuning a small development set. Start with 200–500 printed lines distributed over books, fonts, damage levels, and rare signs; reserve entire pages/books for validation/test and measure a learning curve at 50/100/200/500 lines. These counts are a planning estimate, not a demonstrated accuracy threshold. Double-annotate a subset to measure reference disagreement before optimizing the model against uncertain labels.

For handwriting, replace the print-frame assumptions with baseline/polygon detection, add writer-disjoint references, and start with roughly 1,000–3,000 diverse labeled lines or a pretrained handwriting backbone. The amount needed depends on writer variation and the accuracy target. Review time and annotation quality are likely to dominate compute cost. Downsampling and skew probes already show why low-resolution pages need a separate benchmark; upscaling cannot restore missing strokes.

## Final experiment outcomes

The longer packed trial reached 16/405 edits; fixed structural beam reduced this to 15, and a two-member packed/ordinary ensemble reached 14/405. The corrected word-height experiment improved its paired result from 26 to 23 edits, but neither word-augmentation configuration beat the incumbent. This separates a real implementation defect from evidence that the augmentation strategy improves deployment.

Fresh testing then exposed 17/118 edits despite the 14/405 development result. A targeted data expansion promoted three of those test lines to training and two to development, adding two previously annotated lines. Both 500-step pilots were worse than the retained ensemble on the expanded 453-character development set (24 and 26 versus 20 errors with fixed decoding). Their ensemble had 25. The preregistered extension threshold was not met, so no longer runs were justified for those candidates.

The retained ensemble's next frozen test produced 26/123 edits, compared with 28/123 for the original baseline. These unfamiliar forms and uncertain labels establish a generalization limitation; repeated development minima are not evidence of near-perfect OCR. No labels were changed to fit predictions. Predeclared alternate readings leave 20.16–25.20% CER, so they do not explain away the gap. The final weights remain trained on 15 page-17 lines. The source/label comparisons, old test results and rejected candidates are all retained.

This submission completes the assessment's runnable pipeline and evaluation requirements. It does not demonstrate universal or near-perfect Khojki recognition. The next substantial improvement needs independent glyph adjudication and broader book-disjoint data, with a measurable coverage target, rather than additional selection on these same few lines.
