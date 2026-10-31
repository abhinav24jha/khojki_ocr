# Deployment model and provenance

`models/printed-khojki.pt` is the final compact ensemble (8,523,551 bytes), SHA-256 `6646034c85ad073ce64e7dd1c4efe15ab2dbf68a023e435382eff3f79ff50b5c`. `models/round3-selected.pt` is byte-identical. `models/round2-selected.pt` contains the same retained weights before reserialization. The original 10-line baseline remains in `models/baseline-printed-khojki.pt`.

Each member has 1,060,946 parameters: convolutional features and two bidirectional LSTM layers, trained with CTC. Height 48, robust layout, beam width 10 and structural prefix filtering are saved in metadata. One member uses packed recurrence; the other retains ordinary recurrence. Both were adapted on 15 page-17 lines (384 code points) with 25% synthetic replay. Synthetic rendering uses shaped Noto Khojki fonts; the training scripts record the exact mixtures and seeds.

Round-three candidates used 20 lines but were rejected. Their expanded training-pool error must not be described as the final model's fitted-training error. Selection records and source checkpoint hashes are in `experiments/round2-selection.json` and `experiments/round3-final-selection.json`. Best steps were 2450 for packed recurrence and 2800 for the ordinary member. Complete resumable checkpoints remain in the working project; deployment needs only the compact file.

The latest test is 26/123 = 21.14% CER on five chart-transcribed real lines. Previous fresh testing gave 17/118 = 14.41%. These small related-print sets have non-expert labels and explicit ambiguities. No full-script, book-disjoint, handwriting, or near-perfect claim is justified. Rare output classes lack positive examples. The output alphabet's presence does not demonstrate recognition coverage.

CPU and CUDA inference were checked, including exact page text agreement. MPS for the final packed ensemble was not tested. Use the repository's editable install so bundled assets remain available. Checkpoints are PyTorch artifacts: load only the supplied, hash-verified files or other trusted checkpoints.

The source PDF hash and page provenance are in `data/provenance.json`; frozen test hashes, annotation tokens, alternatives, crop boxes and split changes are in `data/round2/` and `data/round3/`. Original references remain in `data/annotations-v2-frozen.json`. Font licensing is bundled under `assets/OFL.txt`. Exercise scan material is not intended for public redistribution.
