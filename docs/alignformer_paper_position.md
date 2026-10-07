# AlignFormer: what is new against prior work, and whether it is a CVPR paper

Written 2026-10-07, the day the consensus-floor arm was selected on V2X-Real
val and before its test run had finished. The numbers quoted here are the
ones in [alignformer_v2xreal.md](alignformer_v2xreal.md),
[alignformer_v2xreal_closing_the_gap.md](alignformer_v2xreal_closing_the_gap.md)
and [alignformer_p2.md](alignformer_p2.md); this file only positions them.
Every baseline number is a local run under this evaluator (see "Why no
published number is quoted" in the P2 document), FreeAlign included.

## The prior work this sits against

| method | venue | how it handles the sender's pose error | fusion level |
|---|---|---|---|
| V2VNet-robust (Vadivelu et al.) | CoRL 2020 | regresses a pose correction from fused BEV features, smooths with a Markov random field | intermediate |
| CoAlign (Lu et al.) | ICRA 2023 | agent-object pose graph optimised over detected boxes | intermediate |
| CoBEVFlow (Wei et al.) | NeurIPS 2023 | BEV flow to compensate asynchrony (delay), not pose | intermediate |
| MRCNet (Hong et al.) | CVPR 2024 | motion-aware robust communication for pose error and delay | intermediate |
| FreeAlign (Lei et al., arXiv 2405.02965) | ICRA 2024 | drops localization: relative-distance graph matching over boxes, RANSAC pose | late, boxes only |
| RoCo (Huang et al.) | ACM MM 2024 | alternates object matching and pose adjustment | late / hybrid |
| CoLoca-QuA | this repository's starting point | learned pose regression from quantized features; does not converge as published ([coloca_qua_baseline.md](coloca_qua_baseline.md)) | intermediate |

FreeAlign is the only one that sends what we send (the 7-float box, 218
bytes per agent-frame on V2X-Real) and needs no localization at all, so it
is the comparison that matters; the intermediate-fusion methods are the
comparison the project exists to make on bytes.

## What is new, as the measurements support it

1. **A learned, pose-aware association that answers where graph matching
   cannot.** FreeAlign's relative-distance graph is degenerate on a pair
   that shares one or two objects, so it declines 84 % of those pairs on
   OPV2V and gains +0.011 AP@0.7 on them over doing nothing. AlignFormer's
   Sinkhorn association answers 93 % of them and gains +0.184. This is the
   one place the method is structurally different rather than better tuned.
2. **Two estimators and one decision.** The soft weighted-Kabsch fit from
   the learned mixture proposes; an exact RANSAC re-solve over the learned
   correspondences, with a 6 / 3 / 1.5 / 0.75 / 0.5 m gate schedule,
   verifies; a Wald test on the per-pair uncertainty, an agreement rule
   between the two fits at 1.0 m and a consensus floor of four hard pairs
   decide whether the pair is corrected at all. Every piece is textbook
   (SuperGlue-style matching, Kabsch, RANSAC, Huber IRLS, a Wald test). The
   composition is what took a 0.031 AP@0.7 deficit to FreeAlign on V2X-Real
   test to a lead of +0.0096 on val, at no new bytes and no retraining.
3. **FreeAlign's bytes.** The message is the box, byte for byte FreeAlign's,
   1,021x to 4,254x below the intermediate-fusion methods on OPV2V. The
   communication-efficiency goal is met by construction; there is nothing
   left for a codebook to quantize.
4. **Two findings a reviewer will not have seen.**
   - FreeAlign calibrated on mean pose error, the natural criterion for a
     pose estimator, under-reports itself by 0.061 AP@0.7 on V2X-Real test
     against the same method calibrated on val AP. A fair comparison needs
     the baseline tuned on the reported metric.
   - Appearance does not help association on either dataset, and the
     reason is measured: geometry already associates 99.4 % of objects
     (stage-1 Top-1), so a DINOv2 head that separates cross-agent identities
     at AUC 0.885 moves the matcher by -0.011 AP@0.7 on the sweep mean. The
     remaining gap to the oracle is solver precision on dense pairs, not
     association.

## What a reviewer will object to, honestly

- **The headline margin.** +0.0096 AP@0.7 over FreeAlign on the V2X-Real
  val sweep mean, even seed-paired (SE 0.0003) and ahead at every sigma,
  reads as "within noise" at CVPR. The stronger cells are the clean case
  (+0.03), AP@0.5 and AP@0.3 (+0.01 to +0.04), the sparse bucket and the
  delay rows. The paper has to be framed on those and on the decision
  rule, not on AP@0.7 against FreeAlign.
- **Every component is known.** The conceptual hook has to be "learned
  proposal, geometric verification, calibrated abstention for cooperative
  fusion", with the sparse-pair result as the evidence that the learned
  part matters.
- **Late fusion is unfashionable.** The OPV2V head-to-head against seven
  intermediate-fusion methods helps (tie at sigma 0, lead from 0.4 m); it
  does not exist on V2X-Real.
- **One dataset carries the final pipeline.** OPV2V still shows the
  pre-re-solve numbers, where FreeAlign leads AP@0.7 by 0.015 to 0.049.
- **Every baseline is a reimplementation.** Defensible under one evaluator
  and one sweep, but it needs a paragraph and ideally one sanity check
  against a published number.

## What would make it credible, in priority order

| item | cost on this machine | status |
|---|---|---|
| 1. Run the final pipeline on OPV2V (FreeAlign re-selected by val AP there too) | one chain, no new code, about a day | **running** (2026-10-07, chain O; results land in [alignformer_p2.md](alignformer_p2.md)) |
| 2. Intermediate-fusion baselines on V2X-Real under the same sweep | about a week of GPU time; OpenCOOD has the V2X-Real configs | not started |
| 3. A second real dataset (V2V4Real is the natural one; it has GPS noise of its own) | several weeks: detector, cache, pair index, calibration | not started |
| 4. Seed-paired confidence intervals in every table | already computed; a rendering change | partly done (sweep-mean tables carry mean +/- SE) |

## Feasibility verdict

The CVPR 2027 deadline is normally mid-November, about five weeks from the
date above. Items 1 and 4 fit; item 2 fits if it starts within the week;
item 3 does not fit. With items 1, 2 and 4 done and the framing above, the
estimate for the main track is roughly one in three. The same paper is a
solid ICRA / IROS submission and a plausible WACV or RA-L paper, and a CVPR
workshop paper is a near-certain acceptance if the main track is too tight.
Spending the five weeks on a codebook, on appearance, or on a bigger
backbone would not move any of these estimates; the bytes are already
minimal and the appearance question is closed with a mechanism.
