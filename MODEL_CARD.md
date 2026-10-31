# Deployment model and provenance

`models/printed-khojki.pt` is byte-identical to `models/round4-final.pt`: 8,523,321 bytes, SHA-256 `b4398167ea03e71277d09eba4cf5a55ef3efac454738f649f01169343ec3abb6`.

Development-selected configuration: **incumbent**. Source checkpoints: `['models/round3-selected.pt']`. Optical views are (ink-strength multiplier, Gaussian-blur radius): `[[1.0, 0.0]]`. 2 CRNN member(s), 2,121,892 total parameters; height 48, robust layout, beam 10, conservative structural filtering. Metadata preserves all inference choices. There is no dictionary, reference lookup, or language-model text replacement.

All selected members derive from the original 15 page-17 real lines plus shaped Noto synthetic replay. Follow-up corrected-label experiments change only the disputed p17r-10 I sign (384 → 383 real code points). Exact configuration, data hashes and source checkpoints are in `experiments/round4-followup-selection.json` and each named experiment directory. Original 10-line baseline and previous compact deployments remain available. A legacy RGB-padding ablation deliberately preserves an old artificial-border defect; this is recorded in its training config, not hidden.

Latest five-line test: 21/120 = 17.50% CER, previous model 21/120 = 17.50%. Predeclared alternatives: 17.50%–17.50%. Development 14/405 is a repeatedly selected minimum, not unseen accuracy. References are non-expert and related print; no full-script, book-disjoint, handwriting or near-perfect claim is supported. Thirty-four Khojki classes lack real training examples.

The final serialized model reproduces selected development predictions; CPU and CUDA full-page text match. Two-thread CPU page time: 3.283s for 16 lines. MPS was not verified. Use an editable install so bundled fonts/assets remain available; load only trusted checkpoint files.

Frozen new references, alternatives and image hashes: `data/round4/test.json` and `test-freeze.json`. Proposed historical-reference corrections remain separate in `data/round4/proposed-reference-revisions.json`; lower scores from those corrections are not model improvements. Source PDF provenance is in `data/provenance.json`. The scans are for this private exercise; font licensing is bundled.
