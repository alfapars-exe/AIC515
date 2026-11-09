### 6.6-bis The green marker: a pipeline artefact that every model exploits (exploratory probe)
This probe was not pre-registered; it was designed after the artefact was noticed during the ground-truth verification (Sec. 3.3-vii) and is reported as exploratory. In the validation split 368 of 487 pairs (75.6 %; Gemini 82.3 %, Ideogram 2.4 %) carry the residue. Table 14 and Fig. 10 give the four conditions of Sec. 5.7-5 on these 368 images and, for comparison, on the 119 residue-free images.

Three results follow. **(a) The marker alone triggers the box.** On authentic images onto which only the marker pixels were transferred — no character was changed — the box-trained model reaches a per-image AP of 0.906 [0.887, 0.924] against the (hypothetical) box ground truth, covers 80 % of the box at its calibrated threshold and raises its image-level score from 0.10 to 0.53; on the same authentic images without the marker AP is 0.178 and coverage 0.04. **(b) Removing the marker removes a quarter of the performance.** Repairing the residue in real forgeries (content unchanged) lowers the box model's AP from 0.988 to 0.763 (paired Δ −0.225 [−0.262, −0.189]) and its box coverage from 0.94 to 0.59; on residue-free forgeries the box model scores 0.747 — essentially the repaired level. The "generator gap" of Secs. 6.4 and 6.7 is therefore largely a marker gap. **(c) The pixel and glyph models inherit the cue.** Because the marker line differs from the authentic image, it is part of the derived pixel-change ground truth (median 10 % of the positive pixels of `pixel_8_morph` in affected pairs, p75 19 %); the pixel model's AP under its own ground truth changes by −0.266 [−0.295, −0.238] when the marker is repaired, and the marker transferred onto authentic images raises its box-AP from 0.14 to 0.64 and the glyph model's from 0.13 to 0.47. The effect is weaker than for the box model, which not only detects the line but completes the rectangle (IoU between prediction and box 0.74 vs 0.39 and 0.24).

Two further checks bound the finding. A hand-written rule that detects saturated-green line segments anywhere in the image and predicts their bounding box fires on 53 % of test images and reaches a mean F1 of only 0.05 (Table 15): the residue typically marks one or two edges of the field, so the box cannot be read off the marker without learning; the networks combine the line with the appearance of the adjacent text field. The detector used to flag the residue has a false-alarm rate of 0/600 on authentic images, and the residue is present in 604 of the 812 official test forgeries (74.4 %; Gemini 82.8 %), so the test numbers of Sec. 6.5 are obtained on a split in which three quarters of the images carry the cue. Finally, this explains why the pre-registered boundary ablation (Sec. 6.6) missed the shortcut: a Gaussian blur of σ = 1.5 smears a 1–3 px saturated-green line but does not remove its chroma, whereas the placebo and interior conditions never touched it.

**Table 14.** Green-marker probe on the 368 residue-carrying validation pairs (three-seed means, group-bootstrap 95 % CI; "box GT" for authentic inputs is the box of the paired forgery).

{{T:green}}

**Table 14b.** Paired contrasts (residue images).

{{T:green_contrasts}}

**Table 15.** No-learning green-line rule on the official test split and on authentic images.

| Input | n | Fires | Mean F1 (box GT) | Mean IoU | IoU ≥ 0.5 |
|---|---|---|---|---|---|
| Test forgeries, all | 812 | 52.8 % | 0.054 | 0.040 | 3.0 % |
| Test forgeries, Gemini | 727 | 58.9 % | 0.061 | 0.045 | — |
| Test forgeries, Ideogram | 85 | 1.2 % | 0.000 | 0.000 | — |
| Authentic (validation sources) | 600 | 0.0 % | — | — | — |

![Fig. 10. Green-marker probe: per-image AP against the box, box coverage at τ* and image-level score for forged, marker-repaired forged, authentic and authentic-plus-marker inputs (368 residue pairs; 3 seeds; group-bootstrap CI).](phase2/experiments/figures/fig_green_marker.png)
