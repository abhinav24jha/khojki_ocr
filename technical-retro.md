# Khojki OCR: Technical Retrospective

## Approach and measured result

I built a page/PDF-to-UTF-8 pipeline with automatic line detection, contrast normalization, text/number-column splitting, and two small CNN–BiLSTM–CTC recognizers. I pretrained on shaped Noto text, adapted on 15 printed lines, and saved weights plus an offline correction interface. I used AI-assisted chart transcription; my references are non-expert, with ambiguities recorded before testing.

My latest frozen five-line test produced **21/120 character edits: 17.50% CER**. The previous model scores identically. Development reached 14/405 (3.46%), but repeated selection makes that optimistic. Nine additional training runs did not improve the incumbent, so I retained it. I have not demonstrated near-perfect, book-disjoint or handwritten OCR.

## Error breakdown by stage

I detected all five evaluated lines. Automatic and manually checked crops produced identical predictions: the crop intervention removed **0 errors**. All **21 edits involved characters**, with no space errors; **0 unsupported code points and 0 UTF-8 failures** occurred. Three alignments pair an inserted PA with independent E read as dependent I. E has no real training example. This comparison localizes the observed problem to recognition, without proving segmentation is correct in unlabelled regions.

## Experiments, including failures

I measured 1.31% synthetic-validation CER but 48% on initial real lines; normalization and column splitting reduced the latter to 29.33%. This exposed the modern-font/historical-print gap. On the same 405-character development set:

| Experiment | Character edits |
|---|---:|
| Retained ensemble | **14** |
| Six 600-step padding/normalization/recurrence pilots | 16–17 |
| Old versus fixed padding, extended to 3,000 steps | 14 / 16 |
| 1,000-step slower training: original / corrected label / word replay | 16 / 18 / 18 |
| Five contrast/blur inference variants | 15–18 |

I found that scalar white padding on RGB images became red, then dark in grayscale. I fixed and tested that bug, but the controlled trial showed no accuracy gain. No slower-training candidate qualified for a 5,000-step extension. I preserved histories, resumable checkpoints and failed-run evidence.

## Encoding and validation

I used U+1122C–U+11233 for common dependent vowels, U+11234 for anusvara, U+1123A for word separators and Gujarati U+0AE6–U+0AEF for digits. I used atomic independent vowels and O/AU signs, and ASCII spaces for distant number columns. My CER counts Unicode scalars. My validator reports inventory violations, orphan/adjacent signs, non-atomic vowels, NFC changes and UTF-8 failures without rewriting text; valid encoding cannot certify a glyph reading.

## Bottleneck, scale and handwriting

My highest-value next step is **200–500 independently reviewed printed lines**, spanning books, typefaces, damage and rare signs, with book-disjoint testing. This is a planning estimate: 34 supported Khojki classes lack real training examples. My initial 31-line AI-assisted annotation/freezing took about 14 elapsed minutes, excluding prior chart study and later review; this is not expert annotation productivity.

I measured **3.28 seconds for a 16-line page on a two-thread Colab CPU**, plus 0.045 seconds model loading. CPU/GPU text matched. That suggests approximately 154 seconds for 47 comparable pages, excluding startup, PDF rendering, review and final review-file serialization. For handwriting, I would change detection, reserve unseen writers and initially plan 1,000–3,000 diverse labelled lines; accuracy remains unvalidated.

## Where AI was wrong

My earlier AI-assisted references likely confused U with II and BA with NA. I caught inconsistencies through repeated-word image matching and enlarged scan/chart comparisons. Four proposed corrections change the **same older predictions** from 26/123 to 15/128 edits. I preserved the original labels and report this as a post-hoc, non-expert reference revision requiring confirmation—not an OCR improvement.
