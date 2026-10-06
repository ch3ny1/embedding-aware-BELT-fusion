# AlignFormer on V2X-Real: three message trunks against FreeAlign on real traffic

> Companion to [alignformer_p2.md](alignformer_p2.md), whose OPV2V measurements
> retired the per-object embedding ("The message: the boxes-only trunk through
> the deployed pipeline"). This document repeats the trunk comparison on
> **V2X-Real**, real traffic with real localization, and adds the one trunk
> OPV2V could not motivate: LiDAR + camera. Everything is trained from
> scratch on this machine; no published V2X-Real number is quoted beside ours.

## Verdict

- **FreeAlign, calibrated on the metric this report quotes, beats every
  trunk under localization noise on V2X-Real test.** With its edge threshold
  selected by val AP (1.0 m) instead of by mean pose error (0.3 m), FreeAlign
  leads at AP@0.7 at every sigma from 0.2 up (by 0.012 at 0.2, 0.073 at 2.0)
  and on the sweep mean by 0.032-0.045 against all three trunks at both
  training seeds, and at AP@0.5 from sigma 0.6 up (0.006-0.021 on the sweep
  mean). The trunks keep the clean case at every threshold (+0.015-0.027
  AP@0.7 at sigma 0) and AP@0.3 on the sweep mean (+0.006-0.016). Under 100
  to 400 ms of delay the split is the same: ours at sigma 0 and at AP@0.3,
  FreeAlign at AP@0.7 and AP@0.5 once localization error is injected.
- **The verdict this document first carried, "every trunk beats FreeAlign at
  every cell", was against an under-calibrated FreeAlign.** The mean-error
  criterion put its edge threshold at 0.3 m (29 % coverage); twenty times
  the calibration data did not move it, and widening the grid made it pick
  thresholds at which FreeAlign answers 1-5 % of pairs. Selecting the
  threshold by FreeAlign's own val AP gives 1.0 m (60 % coverage) and
  +0.061 AP@0.7 on test. Every table below carries both columns; quote the
  1.0 m one. "Re-calibrating FreeAlign on the reported metric" has the
  whole story.
- **An exact re-solve after the soft association closes most of the gap,
  at no new bytes and no training** (2026-10-02, test, 5 seeds): the AP@0.7
  gap to FreeAlign 1.0 m goes from -0.031 to -0.005 (-0.001 with an
  agreement rule between the soft and the exact fit), AP@0.5 and AP@0.3
  reverse (+0.010 / +0.019), the clean case widens to +0.032; FreeAlign
  keeps AP@0.7 from 0.8 m up. Under 100 / 200 / 400 ms of delay the same
  holds (3 seeds): AP@0.7 a draw on the sweep mean (-0.003 / -0.000 /
  -0.003), AP@0.5 and AP@0.3 ours by 0.009-0.020 and 0.025-0.040. The loss
  had been solver precision on the dense pairs (a Sinkhorn mixture blurs
  the fit); see
  [alignformer_v2xreal_closing_the_gap.md](alignformer_v2xreal_closing_the_gap.md).
- **Giving AlignFormer the same treatment does not close the gap.** Its
  per-pair decision-rule family, selected on val AP exactly as FreeAlign's
  threshold was, picks a hard Wald threshold at level 0.2 that gains
  +0.002 AP@0.7 over the deployed rule on val and +0.002 on test, and
  widens the clean-case lead over FreeAlign to 0.023-0.035. FreeAlign still
  leads AP@0.7 by 0.031 and AP@0.5 by 0.007 on the test sweep mean. The gap
  is structural: FreeAlign's association is pose-invariant and ours is not,
  and FreeAlign itself is 0.08 under the oracle. "Giving AlignFormer the
  same treatment" has the tables and the next step.
- **The LiDAR embedding's validation lead over boxes-only (+0.012 AP@0.7,
  replicated at a second training seed) is a wash on test**: +0.001 at seed
  0 and -0.0005 at seed 1 on the sweep mean. Seed 0's clean-case lead
  (+0.012 at sigma 0) and its AP@0.3 lead at all eight cells are +0.001 and
  +0.0006 at seed 1. This is the same val-to-test shrinkage OPV2V showed.
- **The frozen camera feature never beats the LiDAR embedding** whose bytes
  it shares, and on test it costs 0.012 (seed 0) and 0.005 (seed 1) AP@0.7
  against boxes-only.
- Message: boxes-only is FreeAlign's 218 B per agent-frame; either embedding
  trunk is 4,197 B.

## The question, and why V2X-Real

OPV2V put the embedding's value at zero three times over, and the closing
argument of P2 named the one place the verdict could still turn: OPV2V is
simulated, its detections are clean enough that nearest-centre association
already scores Top-1 1.0000, and no ego object has a competitor within 2 m.
Real traffic supplies the competitors, real detectors supply the noise, and
real cameras supply an appearance cue that CARLA's rendering did not. So the
same three-way comparison is run again where it could come out differently:

| trunk | what the CAV transmits per object | message |
|---|---|---|
| boxes only | the 7-float box | FreeAlign's message, byte for byte |
| boxes + LiDAR embedding | box + 128 floats from the LiDAR ROI feature | the P2 "deployed" message |
| boxes + LiDAR + camera embedding | box + 128 floats from LiDAR ROI ‖ camera ROI | same bytes as the row above |

The camera trunk's 128 floats are the same size as the LiDAR trunk's: the
camera feature is concatenated to the LiDAR feature *before* the embedding
head, so the wire cost of "adding the camera" is zero and the comparison is
purely about what the head can extract.

## Protocol

**Dataset.** V2X-Real, official `train` / `val` / `test` splits (48 / 6 / 14
scenarios), two infrastructure agents and two vehicles per scenario, up to
four cameras per vehicle. Three files that failed their CRC were deleted
before any of this ran (memory `v2x-real-dataset-state`); one zero-byte test
image is dropped by the loader with a warning and is scored as "no camera"
for that object. Ground-truth range ±102.4 × ±51.2 m, communication range
70 m, one vehicle class {Car, PoliceCar, LongVehicle}. Frame rate 10 Hz, so
`--delay-frames 1 / 2 / 4` below is 100 / 200 / 400 ms as on OPV2V.

**Detector.** PointPillar late fusion, trained from scratch for 15 epochs
(`configs/v2xreal_detector.yaml`). The checkpoint is the one with the lowest
validation loss, **epoch 13** (0.873, against 0.893 at epoch 15 and 0.894 at
epoch 11); the OPV2V precedent of taking the schedule's last epoch was not
the best one here and is not followed. Single anchor class, so the absolute
AP is not comparable to the paper's three-class figures; every comparison
below is internal to this detector.

**Cache.** The detector's boxes, the 384-channel rotated-ROI LiDAR feature
and, for every box that projects into one of the agent's cameras, a frozen
ImageNet ResNet-18 `layer3` feature pooled with `roi_align` over the
projected 2-D box (`alignformer/camera_features.py`; projection
`K @ inv(extrinsic)`, the convention verified by eye). One cache, three
trunks: 22,936 / 2,770 / 6,850 agent-frames (train / val / test), 2.58 GB.
Camera coverage, the fraction of cached objects with a visible 2-D box in at
least one camera: **80.1 % / 83.4 % / 89.6 %**. The camera trunk is not
starved.

**Calibration is on the official `val` split, nowhere else.** OPV2V has no
usable validation directory (its `validate/` is a symlink to `test/`, refused
in code), so P2 carved validation scenarios out of train. V2X-Real ships a
real one, and everything that is chosen -- the detector epoch, the
correspondence-variance fit, each trunk's shrinkage tau, FreeAlign's six free
parameters -- is chosen on it. `test` is read once per trunk, for the report.

**What is transferred from OPV2V unchanged, and stated as such.** The model
and every training hyper-parameter (`configs/alignformer_v2xreal.yaml`
differs from `alignformer_r140.yaml` only in the four data paths, the
`camera_dim` switch and the four fitted variance numbers;
`tests/test_alignformer_v2xreal_configs.py` pins that list). The IRLS constants: Huber, 2 iterations, evidence guard 3.0. The
per-pair abstention arm. None of these was re-tuned; re-tuning them on
V2X-Real val would have been legitimate and was not done, so if they are
suboptimal here, that cost is paid by all three trunks equally and by
FreeAlign not at all.

**FreeAlign is recalibrated.** Its six parameters (edge feature, edge
threshold, anchor limit, epsilon power and offset, robust estimator) were
re-chosen on V2X-Real val at stride 4 and sigma 1.0 m by
`scripts/calibrate_freealign.py`, and the evaluator reads the whole selected
block through `--freealign-calibration`. FreeAlign is therefore tuned to
this dataset by the same procedure as on OPV2V, not run with OPV2V's
constants. EdgeGAT is still not ported (see P2's "What is ported"). That
procedure's criterion -- mean translation error over all pairs at sigma 1 m
-- turned out to under-calibrate FreeAlign here; "Re-calibrating FreeAlign
on the reported metric" re-selects the edge threshold on val AP, and every
test table carries both the 0.3 m and the 1.0 m column.

## Stage 1: association on real traffic is still not the bottleneck

Matching trained for 20 epochs per trunk on 42,036 train pairs, validated on
4,334 val pairs (11,652 countable objects). Best validation Top-1:

| trunk | best Top-1 | epoch | nearest-centre Top-1 | chance |
|---|---:|---:|---:|---:|
| zeroed embedding (boxes-only control) | **0.9942** | 19 | 0.9997 | 0.165 |
| LiDAR embedding | 0.9912 | 15 | 0.9997 | 0.165 |
| LiDAR + camera embedding | 0.9818 | 13 | 0.9997 | 0.165 |

Same ordering as OPV2V (0.9972 / 0.9983 there, nearest-centre 0.9998), with
the gaps a little wider. The hypothesis that dispatched the V2X-Real run was
that real traffic would supply near competitors that a descriptor could
separate; nearest-centre Top-1 of **0.9997** says it does not, at least not
within the 70 m communication range and after the detector's own filtering.
The camera trunk associates *worst* by 0.012, which on 11,652 objects is
about 140 more mistakes than the control; the frozen ImageNet feature adds
noise the head does not learn to discount in 20 epochs.

## The correspondence-variance fit: confidence again, range now worth nothing

Fitted on val at sigma 0 as in P2 (`scripts/fit_correspondence_variance.py`,
`outputs/v2xreal/variance_fit_result.json`), 11,652 correspondences over
4,334 pairs. The additive form was fixed in advance; the other four are
reported so "nothing else fits materially better" is measured.

| Gamma deviance improvement over constant | translation | heading |
|---|---:|---:|
| range | 0.000 (range scale → ∞) | 0.000 |
| score_additive (**selected**) | 0.023 | 0.066 |
| score_min | 0.022 | 0.082 |

| | OPV2V (`r140`) | V2X-Real |
|---|---:|---:|
| RMS translation disagreement per correspondence | 0.33 m | **0.44 m** |
| RMS yaw disagreement | 6.7 deg | **10.0 deg** |
| sigma_translation_m / exponent | 0.2516 / 1.128 | 0.2887 / 0.493 |
| sigma_yaw_deg / exponent | 4.639 / 1.932 | 6.014 / 1.031 |
| corr(range, min score) | -0.352 | -0.142 |
| range's deviance improvement, translation | 0.020 | 0.000 |

Two things differ from OPV2V and both are about the data, not the model.
Range explains **nothing** on V2X-Real: the RMS translation disagreement is
flat across range bins (0.44, 0.46, 0.41, 0.41 m for 0-20, 20-35, 35-50,
50-70 m) and yaw disagreement *falls* with range (11.0 → 8.4 deg), so the
fitted range scale runs off to infinity in both channels. Detection score
still carries signal (the lowest score decile is at 0.50 m / 11.3 deg), but
the exponents are half what they were: the confidence-to-noise relation is
weaker on a real detector, and the residual is larger everywhere. That
residual -- 0.44 m and 10 deg between two agents' detections of the same
object at sigma 0 -- is the number that sets everything below. It contains
the detector's own error *and* the dataset's own localization error, and the
fit cannot separate the two.

## Stage 2 and shrinkage: the correction is nearly untrusted at sigma 0

Head B, scalar inverse-variance weighting, 30 epochs with training sigma
ramped 0 → 2 m, warm-started from each trunk's own stage-1 checkpoint; then
one shrinkage calibration per trunk on val at sigma 0 with the deployed IRLS
settings.

| trunk | tau_translation | tau_yaw | residual at sigma 0, translation / yaw MAE |
|---|---:|---:|---:|
| boxes only | 4.786 m | 6.50 deg | 1.244 m / 2.32 deg |
| boxes + LiDAR embedding | 2.895 m | 5.61 deg | 1.068 m / 2.05 deg |
| boxes + LiDAR + camera | 3.018 m | 5.49 deg | 1.097 m / 2.13 deg |
| OPV2V `r140`, boxes + embedding, for scale | 0.166 m | 0.29 deg | 0.119 m / 0.16 deg |

The tau is **17-29x** OPV2V's for translation and **19-22x** for yaw, because
the residual it is fitted to is 9-10x and 12-14x larger. The boxes-only trunk's
residual is the largest of the three (1.24 m against 1.07 and 1.10 m), the
first sign that on real traffic the embedding is doing something. On the last training epoch of
the LiDAR trunk, validation translation MAE is 1.84 m against predict-zero's
2.79 m at 2 m injected noise, and yaw MAE is **2.59 deg against
predict-zero's 1.60 deg**: the head's yaw estimate is worse than doing
nothing, which on OPV2V was head A's failure mode and here is the data's.
The corner loss, which is what AP sees, still improves 45 % (7.57 m against
13.80 m). 588 of the 4,334 val pairs are unalignable (fewer than the
evidence guard's three usable correspondences); the fallback fires on 89 %
of them.

The shape to expect in the sweeps follows from this: at sigma 0 the
shrinkage rule and the per-pair abstention will keep nearly every correction
off the boxes, so all three trunks should sit close to the uncorrected
baseline and to FreeAlign; the trunks can only separate from each other and
from FreeAlign once the injected error is comparable to the 2.9 m tau.

## The message on V2X-Real

Measured on the 6,850 test agent-frames (`outputs/v2xreal/bandwidth_result.json`,
`bandwidth.alignformer_message_bytes` and `alignment_overhead_bytes`, float32):
**7.77 objects per agent-frame** (median 7, max 29), against OPV2V's 15.69.

| bytes / frame / agent | shared box payload | alignment overhead | total |
|---|---:|---:|---:|
| FreeAlign | 218 | **0** | 218 |
| AlignFormer, boxes only | 218 | **0** | 218 |
| AlignFormer, boxes + LiDAR embedding | 218 | **3,980** | 4,197 |
| AlignFormer, boxes + LiDAR + camera | 218 | **3,980** | 4,197 |

The ratio is 19.3x by construction (128 embedding floats against 7 box
floats); the absolute numbers halve because real traffic within 70 m holds
half as many detected vehicles as the simulator's.

## Results on validation: the embedding leads on the calibration split

Official val split, 717 frames, three paired noise seeds shared across the
three trunks and FreeAlign (FreeAlign's rows are bit-identical across the
three files, which is the check that the pairing is real). Every arm is the
deployed one: IRLS Huber n = 2 with guard 3.0, per-pair abstention, each
trunk's own tau. Rendered from `outputs/v2xreal/trunk_comparison_val_result.json`.
FreeAlign in this section is the MAE-calibrated 0.3 m configuration the
sweeps were first run with; at the AP-calibrated 1.0 m it scores 0.370
AP@0.7 on this split against 0.344 / 0.356 / 0.339 for the three trunks
(see "Re-calibrating FreeAlign on the reported metric").

| AP@0.7, val | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncorrected (= oracle at 0) | **.4517** | | | | | | | .1952 | |
| boxes only | .4216 | .3754 | .3491 | .3412 | .3330 | .3251 | .3146 | .2922 | .3440 |
| boxes + LiDAR embedding | .4319 | **.3870** | **.3584** | **.3474** | **.3438** | **.3356** | **.3304** | **.3156** | **.3563** |
| boxes + LiDAR + camera | .4188 | .3726 | .3406 | .3298 | .3223 | .3183 | .3103 | .2960 | .3386 |
| FreeAlign | .4248 | .3723 | .3227 | .2997 | .2899 | .2852 | .2831 | .2829 | .3201 |
| LiDAR - boxes | +.0103 | +.0116 | +.0092 | +.0062 | +.0108 | +.0106 | +.0159 | +.0234 | **+.0123 ±.0006** |
| camera - boxes | -.0028 | -.0028 | -.0085 | -.0114 | -.0106 | -.0068 | -.0043 | +.0038 | **-.0054 ±.0004** |
| LiDAR - FreeAlign | +.0071 | +.0146 | +.0356 | +.0477 | +.0539 | +.0505 | +.0474 | +.0327 | **+.0362** |
| boxes - FreeAlign | -.0032 | +.0031 | +.0264 | +.0415 | +.0431 | +.0399 | +.0315 | +.0093 | **+.0239** |

| val, sweep mean | boxes only | LiDAR | camera | LiDAR - boxes | camera - boxes | boxes - FreeAlign | LiDAR - FreeAlign |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP@0.5 | .5084 | **.5177** | .5113 | **+.0093** (all 8 cells) | +.0029 (mixed) | +.0328 | +.0421 |
| AP@0.3 | .5636 | **.5707** | .5698 | **+.0071** (all 8 cells) | +.0062 | +.0208 | +.0278 |

**The LiDAR embedding is ahead of boxes-only at all 24 cells**, by 0.012 at
AP@0.7 on the sweep mean with a seed spread of 0.0006, and its lead grows
with the injected error (+0.010 at sigma 0, +0.023 at sigma 2). This is the
opposite of OPV2V, where the same comparison sat within ±0.005 with no stable
sign on either split. Its pose error says the same thing more loudly:
translation MAE 0.73 m against 1.38 m at sigma 0 and 1.93 m against 2.70 m
at sigma 2 (predict-zero 2.70 m: the boxes-only trunk is barely beating doing
nothing on pose there, and the LiDAR trunk is beating it by 0.8 m). Since
association Top-1 is 0.99 for both, the embedding is not buying
correspondences; it is buying *weights* -- which correspondences to trust in
the solve -- and on a real detector with 0.44 m of per-correspondence
disagreement that is worth something it was not worth on CARLA's 0.33 m.

**The frozen camera feature costs AP@0.7** (-0.005 against boxes-only,
seven of eight cells, and -0.018 against the LiDAR trunk whose message it
shares byte for byte) while being a wash at AP@0.3 and AP@0.5. Its stage-1
Top-1 was the worst of the three and its tau and residual sit between the
other two. Caveat 5 applies: this is a null on ImageNet features pooled over
projected LiDAR boxes, not on cameras.

**Against the MAE-calibrated FreeAlign, all three trunks win the sweep mean at every threshold,
and FreeAlign wins the clean end.** At sigma 0 FreeAlign is ahead of every
trunk at AP@0.3 (by 0.010-0.018) and AP@0.5 (by 0.014-0.023), and ahead of
boxes-only and camera at AP@0.7; the LiDAR trunk holds AP@0.7 at sigma 0
(+0.007) and from sigma 0.2 up leads at every threshold. From sigma 0.6 up
every trunk leads FreeAlign at every threshold, by 0.03-0.05 AP@0.7 and
0.04-0.08 AP@0.5. The shape is the OPV2V one shifted: there the crossover
was at sigma 0.2, here it is at 0.4-0.6, because the clean-case regression
is larger.

**The clean case is the cost, and it is 2-3x OPV2V's.** Uncorrected AP@0.7
at sigma 0 is 0.4517; the per-pair arm gives 0.4216 (boxes), 0.4319 (LiDAR),
0.4188 (camera): a regression of 0.020-0.033 where OPV2V's was 0.0014 for
the shipped trunk. The mechanism is visible in coverage: the per-pair rule
still corrects **42-47 %** of pairs at sigma 0 (OPV2V: 43-45 %, but there
the pose MAE at sigma 0 was 0.03 m; here it is 0.73-1.38 m, because the
detector disagreement that the Wald statistic reads as "the poses differ" is
three times larger). FreeAlign's always-correct policy at 27 % coverage
emits 0.43 m and regresses 0.027 -- about the same cost by a different
route. The shrinkage-only arm (IRLS without the per-pair rule, `alignformer_irls`
in the files) regresses only 0.009 at sigma 0, but its tau of 3-5 m shrinks
every correction to nothing under noise: at sigma 2 it scores 0.195,
identical to uncorrected. On V2X-Real the per-pair rule is not a refinement
of shrinkage but the only arm that works at all, and the sigma-0 cost is its
price. Re-tuning the rule's threshold on V2X-Real val (caveat 4) is the
obvious lever and was not pulled.

## A second training seed on validation: the LiDAR lead survives, the camera verdict does not

Everything above is one checkpoint per trunk. To put a number on
training-seed variance, all three trunks were retrained with
`training.seed: 1` (stage 1, stage 2 and shrinkage calibration; nothing else
changed) and swept on val under the shipped evaluation configs, so the noise
draws are the same three seeds as the seed-0 sweeps and every cell is paired
across both training seeds. One gotcha is recorded here because it cost an
afternoon: the config's `training.seed` also seeds the AP sweep's noise
draws, so a sweep run *with* the seed-1 config is not pairable with the
seed-0 files; the sweep must be run with the shipped config and only the
checkpoint swapped. Rendered from
`outputs/v2xreal/trunk_comparison_seed1_val_result.json` and the three
`seed_replicate_B_*_val_result.json` files.

| val, training seed 1 | Top-1 | tau (m / deg) | residual (m / deg) | AP@0.7 sweep mean | AP@0.5 | AP@0.3 |
|---|---:|---:|---:|---:|---:|---:|
| boxes only | .9911 | 2.48 / 5.58 | 1.00 / 2.14 | .3367 | .5057 | .5642 |
| boxes + LiDAR embedding | .9894 | 3.00 / 6.34 | 1.08 / 2.18 | **.3530** | **.5157** | **.5699** |
| boxes + LiDAR + camera | .9889 | 2.67 / 5.25 | 1.04 / 2.08 | .3470 | .5135 | .5689 |

| AP@0.7 sweep mean, val | training seed 0 | training seed 1 | seed 1 - seed 0 |
|---|---:|---:|---:|
| boxes only | .3440 | .3367 | -.0073 ±.0001 (all 8 cells) |
| boxes + LiDAR embedding | .3563 | .3530 | -.0033 ±.0005 (all 8 cells) |
| boxes + LiDAR + camera | .3386 | .3470 | +.0084 ±.0002 (all 8 cells) |
| LiDAR - boxes | **+.0123 ±.0006** | **+.0163 ±.0002** | |
| camera - boxes | -.0054 ±.0004 | +.0104 ±.0003 | |
| LiDAR - FreeAlign | +.0362 | +.0329 | |
| boxes - FreeAlign | +.0239 | +.0166 | |

**Training-seed variance is 0.003-0.008 AP@0.7 per trunk on the sweep
mean**, ten times the noise-seed spread (0.0002-0.0015) the error bars
above report, and it is a whole-checkpoint shift: within one trunk the seed
difference has the same sign at all eight sigmas. It is also visible in the
calibration: the boxes-only tau moved from 4.79 m to 2.48 m between seeds
while the residual moved only 1.24 m to 1.00 m, so tau is the number to
trust least from a single training run.

**The LiDAR embedding's lead is larger than that variance and has the same
sign at both training seeds**: +0.012 and +0.016 AP@0.7, +0.009 and +0.010
AP@0.5, +0.007 and +0.006 AP@0.3, in every case at all eight sigmas. At seed
1 it also holds the clean end against FreeAlign at AP@0.7 (+0.002) as it did
at seed 0 (+0.007). This is the finding on the calibration split and it
replicates there; it does not survive the move to test (the seed-1 test
replicate at the end of the next section).

**The camera verdict is inside training-seed variance and flips sign**:
-0.005 at seed 0, +0.010 at seed 1 against boxes-only; against the LiDAR
trunk it is -0.018 and -0.006. Two seeds cannot say whether the frozen
camera feature costs or buys 0.005 over boxes; they can say it does not
overtake the LiDAR embedding at either seed while sharing its bytes. The
sentence to quote is "the camera feature adds nothing over the LiDAR
embedding", not "the camera feature hurts".

**Against the MAE-calibrated FreeAlign, both seeds agree on the shape**: every trunk wins the
sweep mean at every threshold (boxes-only by 0.017-0.024 AP@0.7, LiDAR by
0.033-0.036), FreeAlign wins sigma 0 and 0.2 at AP@0.3 and AP@0.5 by
0.01-0.02, and the crossover sits at sigma 0.4-0.6.

## Re-calibrating FreeAlign on the reported metric

The first version of this document asked, in its caveat 8, for a FreeAlign
calibration with more than six val scenarios behind it. That was done, and
the data was not the problem.

**More calibration data does not move the selection.** The original grid
(192 configurations, scored on translation MAE over all pairs at sigma 1 m
with an abstention counted as the uncorrected pose) was re-run on all 717
val frames (975 pairs, four times the original 245) and on val plus the 44
train scenarios that share no recording clip with test (50 scenarios, 4,852
pairs at stride 2). Both pick the same six values as the 245-pair run:
`distance_yaw`, 0.3 m, gamma 2, offset 100, power 1, RANSAC (the power ties
with 3, because an offset of 100 makes the subgraph score depend on its size
alone). Files: `freealign_calibration_val_stride1_result.json`,
`freealign_calibration_valplus_result.json`.

**Widening the grid makes the criterion worse, not the baseline better.**
Every run had chosen 0.3 m, the smallest threshold in the grid, so the grid
was extended down to 0.1 m. The MAE criterion then picks 0.15 m on val
(4.8 % coverage) and 0.1 m on val plus train (0.6 %): with heavy-tailed
answered errors and an uncorrected error of only 1.3 m at sigma 1, a mean
over all pairs that charges nothing for an abstention is minimised by
abstaining. The calibrator's docstring anticipated this at sigma 0 and
guarded against it by refusing sigma 0; at sigma 1 on this dataset it
happens anyway (`*_wide_result.json`).

**Selecting the edge threshold by FreeAlign's own AP on val reverses the
picture.** The other five parameters were flat under every criterion and
stay at the MAE selection; the threshold was swept through the deployed
pipeline on val (three paired noise seeds, the boxes-only trunk beside it;
`B_boxes_only_fathr*_val_result.json`):

| FreeAlign edge threshold, val | AP@0.7 sweep mean | AP@0.5 | AP@0.3 | AP@0.7 at sigma 0 | emitted at sigma 0 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 m | .258 | .410 | .500 | .449 | 0.00 m | 1 % |
| 0.15 m | .266 | .419 | .506 | .444 | 0.01 m | 5 % |
| 0.2 m | .282 | .437 | .518 | .436 | 0.11 m | 12 % |
| 0.3 m (MAE selection) | .320 | .476 | .543 | .425 | 0.43 m | 27 % |
| 0.5 m | .353 | .504 | .558 | .421 | 0.98 m | 40 % |
| 1.0 m | .3702 | **.516** | **.560** | .414 | 3.9 m | 56 % |
| 1.5 m (AP@0.7 argmax) | **.3703** | .514 | .556 | .408 | 6.4 m | 65 % |
| boxes-only AlignFormer, same files | .344 | .508 | .564 | .422 | | |
| LiDAR AlignFormer | .356 | .518 | .571 | .432 | | |

AP rises monotonically with the threshold up to 1.0 m and is flat from
there, while the emitted correction at sigma 0 rises from 0.4 m to 6.4 m.
FreeAlign's mean pose error and its AP move in opposite directions: AP is
scored on the fused boxes and does not care how far the worst answered pairs
are thrown, only that the bulk of them are corrected, and the mean-error
criterion optimised the wrong summary of the same distribution. At 1.0 m
FreeAlign's val AP@0.7 is above the boxes-only trunk by 0.026 and above the
LiDAR trunk by 0.014.

**1.0 m and 1.5 m tie at AP@0.7** (0.0001 apart, below the noise-seed
spread); 1.0 m wins AP@0.5, AP@0.3 and the clean case. Rather than break the
tie by hand, both were swept on test and both are reported, with 1.0 m the
column to quote (`freealign_calibration_valap_thr{1,1.5}_result.json`).
FreeAlign's row depends on the detections and the noise draws and not on any
checkpoint, so one boxes-only re-run per threshold supplies the column for
every trunk and training seed (`summarize_v2xreal_trunks.py
--freealign-from`); the AlignFormer rows of the re-runs reproduce the
original files to the last digit.

**What was and was not re-tuned.** FreeAlign now has a seven-point search
on the reported metric behind its one live parameter. AlignFormer's
abstention constants and tau are still the residual-fitted values (caveat
4); the symmetric treatment would select them on val AP too, and was not
done here because it is a method change. The asymmetry runs in FreeAlign's
favour, which is the direction a fairness check should err in.

## Results on test: the trunk verdict does not transfer, and the FreeAlign verdict turns on its calibration

Official test split, 14 scenarios, 2,172 frames, five paired noise seeds,
the seed-0 checkpoints and calibrations from validation, nothing re-chosen
on test. FreeAlign appears twice: at the MAE-calibrated 0.3 m the sweeps
were first run with, and at the AP-calibrated 1.0 m (1.5 m in the text).
Bold marks the best trunk. Rendered from
`outputs/v2xreal/trunk_comparison_test_result.json` and
`trunk_comparison_seed0_faap1_test_result.json`.

| AP@0.7, test | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncorrected (= oracle at 0) | **.4223** | .3185 | .2155 | .1771 | .1621 | .1580 | .1584 | .1632 | .2219 |
| boxes only | .3855 | .3447 | **.3168** | **.3036** | **.2950** | .2904 | **.2739** | **.2601** | .3086 |
| boxes + LiDAR embedding | **.3970** | **.3486** | .3155 | .3017 | .2931 | **.2910** | .2733 | .2595 | **.3095** |
| boxes + LiDAR + camera | .3882 | .3371 | .3005 | .2882 | .2751 | .2754 | .2599 | .2503 | .2967 |
| FreeAlign, 0.3 m (MAE-calibrated) | .3700 | .3274 | .2829 | .2628 | .2542 | .2499 | .2485 | .2493 | .2806 |
| FreeAlign, 1.0 m (val-AP-calibrated) | .3702 | .3577 | .3420 | .3362 | .3331 | .3321 | .3303 | .3301 | .3415 |
| FreeAlign, 1.5 m | .3617 | .3516 | .3399 | .3356 | .3328 | .3322 | .3307 | .3304 | .3393 |
| LiDAR - boxes | +.0115 | +.0041 | -.0010 | -.0011 | -.0041 | -.0026 | -.0009 | +.0014 | **+.0009 ±.0002** |
| camera - boxes | +.0027 | -.0086 | -.0173 | -.0168 | -.0197 | -.0156 | -.0130 | -.0069 | **-.0119 ±.0002** |
| boxes - FreeAlign 0.3 m | +.0155 | +.0176 | +.0332 | +.0389 | +.0433 | +.0418 | +.0268 | +.0074 | **+.0281 ±.0002** |
| LiDAR - FreeAlign 0.3 m | +.0270 | +.0217 | +.0322 | +.0377 | +.0392 | +.0393 | +.0259 | +.0088 | **+.0290 ±.0003** |
| boxes - FreeAlign 1.0 m | +.0154 | -.0119 | -.0270 | -.0349 | -.0360 | -.0399 | -.0548 | -.0733 | **-.0328 ±.0003** |
| LiDAR - FreeAlign 1.0 m | +.0268 | -.0079 | -.0280 | -.0360 | -.0401 | -.0425 | -.0557 | -.0718 | **-.0319 ±.0004** |

| test, sweep mean | boxes only | LiDAR | camera | LiDAR - boxes | camera - boxes | boxes - FreeAlign 0.3 m | LiDAR - FreeAlign 0.3 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP@0.5 | .4831 | **.4878** | .4723 | +.0047 (sigma 0 / 0.2: +.012 / +.011) | -.0108 (7 of 8) | **+.0564** (all 8) | **+.0610** (all 8) |
| AP@0.3 | .5469 | **.5544** | .5442 | **+.0075** (all 8) | -.0027 | **+.0540** (all 8) | **+.0615** (all 8) |

| test, sweep mean | FreeAlign 1.0 m | boxes - FreeAlign 1.0 m | LiDAR - FreeAlign 1.0 m | camera - FreeAlign 1.0 m |
|---|---:|---:|---:|---:|
| AP@0.7 | .3415 | -.0328 ±.0003 (ours at sigma 0 only, +.015) | -.0319 ±.0004 (sigma 0: +.027) | -.0447 ±.0003 (sigma 0: +.018) |
| AP@0.5 | .4933 | -.0102 ±.0005 (ours at sigma 0-0.4) | -.0055 ±.0003 (ours at sigma 0-0.4) | -.0210 ±.0003 (ours at sigma 0-0.2) |
| AP@0.3 | .5390 | **+.0084 ±.0004** (ours at sigma 0-1.0) | **+.0159 ±.0002** (ours at sigma 0-1.5) | **+.0057 ±.0004** (ours at sigma 0-1.0) |

**Against the MAE-calibrated FreeAlign every trunk wins all 24 cells;
against the AP-calibrated one the split is by sigma and by threshold.** At
0.3 m FreeAlign's val parameters transferred badly -- 0.43 m of emitted
correction at sigma 0 on val became 2.68 m on test at the same 29 % coverage
-- and the trunks lead by +0.028 / +0.056 / +0.054 AP@0.7 / 0.5 / 0.3 on the
sweep mean. At 1.0 m FreeAlign emits 4.82 m at sigma 0 (60 % coverage) and
still scores 0.370 AP@0.7 there, the same as at 0.3 m; from sigma 0.2 up its
AP@0.7 barely moves (0.358 to 0.330) while the trunks fall from 0.345 to
0.260, because FreeAlign's correction comes from the box graphs and never
sees the pose, so the injected error costs it only what it costs the
uncorrected boxes it leaves alone. Result: **the trunks hold sigma 0 at
every threshold (+0.015-0.027 AP@0.7) and AP@0.3 on the sweep mean
(+0.006-0.016, ours at every sigma up to 1.0); FreeAlign holds AP@0.7 from
sigma 0.2 up and AP@0.5 from 0.6 up, by 0.032-0.045 and 0.006-0.021 on the
sweep means.** At 1.5 m FreeAlign is 0.002-0.008 below its 1.0 m self on
every sweep mean and the verdict is unchanged (boxes-only: -0.031 / -0.006 /
+0.016). The protocol was the same for both sides -- calibrate on val,
report on test, never re-choose on test -- and the 1.0 m threshold was fixed
on val AP before any test sweep with it ran.

**The LiDAR embedding's lead over boxes-only does not transfer at AP@0.7.**
+0.012 on val, replicated at +0.016 at a second training seed, becomes
+0.0009 ±0.0002 on test: the embedding trunk leads at sigma 0 and 0.2
(+0.012, +0.004) and is level or 0.001-0.004 behind from 0.4 up. At AP@0.3
it still leads at all eight cells (+0.0075) and at AP@0.5 on the sweep mean
(+0.005, carried by the clean end). The pose diagnostic that made the val
case -- 0.73 m against 1.38 m at sigma 0 -- reverses on test: 0.64 m against
0.55 m. The val split is six scenarios and 717 frames, and the OPV2V
comparison showed exactly this pattern (val +0.0045, test -0.0019), so the
honest reading is that **the embedding helps the clean case and the loose
threshold on V2X-Real, and is a wash at AP@0.7 under localization noise**.
Not zero, as on OPV2V; not the +0.012 that val promised either. The seed-1
checkpoints, swept on test below, take even the clean-case and AP@0.3 gains
away.

**The camera trunk costs 0.012 AP@0.7 on test**, negative at seven of eight
cells against boxes-only and at all eight against the LiDAR trunk (-0.013),
and 0.011 at AP@0.5. Val's sign flip across training seeds means the size is
uncertain; test's sign is not (the seed-1 test sweep below puts the cost at
0.005). The line to quote stands: frozen ImageNet
features pooled over projected LiDAR boxes never beat the LiDAR embedding
they are concatenated to, and on test they are worse than sending nothing.

**Coverage and the IRLS-only arm behave as on val.** The per-pair rule
corrects 45-52 % of pairs at sigma 0 and 77-81 % at sigma 2; the IRLS-only
arm corrects 5-8 % and scores 0.4138 at sigma 0 (-0.0085 against uncorrected)
and 0.1630 at sigma 2, identical to uncorrected. The per-pair rule is the
whole method on this dataset.

### The second training seed on test: the LiDAR lead is gone, the FreeAlign split repeats

Same protocol, the seed-1 checkpoints and their val calibrations, the same
five paired noise seeds. Rendered from
`outputs/v2xreal/trunk_comparison_seed1_test_result.json` and the three
`seed_replicate_B_*_test_result.json` files.

| test, sweep mean | AP@0.7 seed 0 | seed 1 | AP@0.5 seed 0 | seed 1 | AP@0.3 seed 0 | seed 1 |
|---|---:|---:|---:|---:|---:|---:|
| boxes only | .3086 | .3034 | .4831 | .4805 | .5469 | .5494 |
| boxes + LiDAR embedding | .3095 | .3030 | .4878 | .4813 | .5544 | .5500 |
| boxes + LiDAR + camera | .2967 | .2984 | .4723 | .4762 | .5442 | .5468 |
| LiDAR - boxes | +.0009 ±.0002 | -.0005 ±.0001 | +.0047 | +.0008 ±.0002 | +.0075 (all 8) | +.0006 ±.0002 |
| camera - boxes | -.0119 ±.0002 | -.0050 ±.0005 | -.0108 | -.0044 | -.0027 | -.0026 |
| boxes - FreeAlign 0.3 m | +.0281 | +.0229 | +.0564 | +.0538 | +.0540 | +.0566 |
| LiDAR - FreeAlign 0.3 m | +.0290 | +.0224 | +.0610 | +.0546 | +.0615 | +.0571 |
| camera - FreeAlign 0.3 m | +.0162 | +.0178 | +.0455 | +.0494 | +.0513 | +.0539 |
| boxes - FreeAlign 1.0 m | -.0328 | -.0380 | -.0102 | -.0128 | +.0084 | +.0109 |
| LiDAR - FreeAlign 1.0 m | -.0319 | -.0385 | -.0055 | -.0120 | +.0159 | +.0115 |
| camera - FreeAlign 1.0 m | -.0447 | -.0431 | -.0210 | -.0171 | +.0057 | +.0083 |

| AP@0.7, test | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boxes only, seed 1 - seed 0 | +.0050 | -.0007 | -.0057 | -.0065 | -.0080 | -.0074 | -.0096 | -.0087 | -.0052 ±.0002 |
| LiDAR, seed 1 - seed 0 | -.0055 | -.0058 | -.0092 | -.0091 | -.0074 | -.0059 | -.0069 | -.0028 | -.0066 ±.0003 |
| camera, seed 1 - seed 0 | +.0014 | +.0043 | +.0029 | +.0039 | +.0041 | -.0001 | -.0000 | -.0031 | +.0017 ±.0003 |
| LiDAR - boxes, seed 0 | +.0115 | +.0041 | -.0010 | -.0011 | -.0041 | -.0026 | -.0009 | +.0014 | +.0009 ±.0002 |
| LiDAR - boxes, seed 1 | +.0010 | -.0011 | -.0045 | -.0037 | -.0035 | -.0011 | +.0018 | +.0074 | -.0005 ±.0001 |
| boxes - FreeAlign 0.3 m, seed 1 | +.0205 | +.0170 | +.0275 | +.0323 | +.0353 | +.0344 | +.0172 | -.0014 | +.0229 ±.0002 |
| boxes - FreeAlign 1.0 m, seed 1 | +.0203 | -.0126 | -.0327 | -.0414 | -.0440 | -.0473 | -.0643 | -.0820 | -.0380 ±.0003 |

**The FreeAlign verdict repeats at the second seed, at both calibrations.**
Against the 0.3 m column seed 1 keeps 70 of the 72 test cells (boxes-only
and the camera trunk give up sigma 2 at AP@0.7 by 0.0014 and 0.0026) with
+0.023 / +0.054 / +0.057 on the sweep means. Against the 1.0 m column
(`trunk_comparison_seed1_faap1_test_result.json`) seed 1 is -0.038 / -0.013
/ +0.011 for boxes-only and -0.039 / -0.012 / +0.012 for LiDAR at AP@0.7 /
0.5 / 0.3, with the clean case still ours at every threshold (+0.020-0.021
AP@0.7 at sigma 0). The same split as seed 0, 0.005 wider.

**The LiDAR embedding's lead does not exist at the second seed.** Seed 0's
+0.0009 on the AP@0.7 sweep mean is -0.0005 at seed 1; the clean-case lead
of +0.012 is +0.001; the AP@0.3 lead at all eight cells is +0.0006 on the
mean, behind at sigma 0-0.8 and ahead from sigma 1 up. Nor do the seeds
agree on where a gain would sit: at seed 0 the embedding led at the clean
end and trailed under noise, at seed 1 it trails at sigma 0.2-1.0 and leads
at sigma 1.5 and 2 (+0.002, +0.007). Two seeds that disagree on the shape
and agree only that the mean is within 0.001 of zero is a wash. The
clean-case and loose-threshold gains the paragraph above held on to are
seed-0 properties, and the Verdict now says so.

**Training-seed variance on test is 0.002-0.007 AP@0.7 per trunk, with the
same sign per trunk as on val**: boxes-only -0.005 (val -0.007), LiDAR
-0.007 (val -0.003), camera +0.002 (val +0.008). The shift is
whole-checkpoint and consistent across splits while the trunk *ordering* is
not, which is the argument for reading any single-checkpoint trunk
difference under 0.01 as noise on this dataset.

**The camera cost is 0.005 at seed 1 against 0.012 at seed 0**, negative
against boxes-only at seven of eight sigmas at seed 0 and all eight at seed
1, and against the LiDAR trunk at both (-0.013, -0.005). Sign replicates;
size does not.

## Results under communication delay on test

Constant delay of 1, 2 and 4 frames (100 / 200 / 400 ms at 10 Hz) on the
other agent's message, localization error swept at sigma 0 / 0.4 / 1 / 2,
three paired noise seeds, test split. Rendered from
`outputs/v2xreal/trunk_comparison_delay{1,2,4}_test_result.json` (FreeAlign
at 0.3 m) and `trunk_comparison_delay{1,2,4}_faap1_test_result.json` (1.0 m).

| test, sweep mean over sigma 0 / 0.4 / 1 / 2 | oracle | boxes only | LiDAR | camera | FreeAlign 0.3 m | boxes - FreeAlign 0.3 m | LiDAR - boxes | camera - boxes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 ms, AP@0.7 | .3420 | .2560 | **.2567** | .2478 | .2422 | **+.0139** (all 4) | +.0006 | -.0082 |
| 100 ms, AP@0.5 | .5588 | .4354 | **.4407** | .4280 | .3872 | **+.0482** (all 4) | +.0053 | -.0074 |
| 100 ms, AP@0.3 | .6075 | .5301 | **.5375** | .5277 | .4660 | **+.0641** (all 4) | +.0074 | -.0024 |
| 200 ms, AP@0.7 | .3187 | **.2349** | .2338 | .2275 | .2265 | **+.0086** (sigma 2: -.0006) | -.0011 | -.0074 |
| 200 ms, AP@0.5 | .4814 | .3778 | **.3812** | .3711 | .3438 | **+.0327** (all 4) | +.0034 | -.0067 |
| 200 ms, AP@0.3 | .5673 | .4818 | **.4886** | .4810 | .4253 | **+.0571** (all 4) | +.0067 | -.0008 |
| 400 ms, AP@0.7 | .3136 | **.2289** | .2276 | .2227 | .2268 | +.0021 (sigma 0: -.003, sigma 2: -.005) | -.0013 | -.0062 |
| 400 ms, AP@0.5 | .4464 | .3482 | **.3491** | .3418 | .3309 | **+.0161** (sigma 0: -.0006) | +.0010 | -.0064 |
| 400 ms, AP@0.3 | .4881 | .4226 | **.4260** | .4199 | .3925 | **+.0330** (all 4) | +.0034 | -.0026 |

| test, sweep mean over sigma 0 / 0.4 / 1 / 2 | FreeAlign 1.0 m | boxes - FreeAlign 1.0 m at sigma 0 / 0.4 / 1 / 2 | sweep mean | LiDAR - FreeAlign 1.0 m | camera - FreeAlign 1.0 m |
|---|---:|---|---:|---:|---:|
| 100 ms, AP@0.7 | .2841 | +.010 / -.027 / -.039 / -.058 | -.0282 | -.0277 | -.0366 |
| 100 ms, AP@0.5 | .4425 | +.034 / +.007 / -.016 / -.056 | -.0077 | -.0023 | -.0150 |
| 100 ms, AP@0.3 | .5126 | +.035 / +.031 / +.021 / -.016 | **+.0176** | **+.0250** | **+.0152** |
| 200 ms, AP@0.7 | .2572 | +.015 / -.023 / -.034 / -.049 | -.0227 | -.0238 | -.0301 |
| 200 ms, AP@0.5 | .3838 | +.030 / +.005 / -.017 / -.046 | -.0070 | -.0035 | -.0137 |
| 200 ms, AP@0.3 | .4544 | +.054 / +.042 / +.024 / -.010 | **+.0275** | **+.0343** | **+.0267** |
| 400 ms, AP@0.7 | .2529 | +.012 / -.025 / -.036 / -.048 | -.0243 | -.0256 | -.0305 |
| 400 ms, AP@0.5 | .3602 | +.024 / -.000 / -.026 / -.048 | -.0126 | -.0116 | -.0190 |
| 400 ms, AP@0.3 | .4045 | +.041 / +.035 / +.013 / -.015 | **+.0183** | **+.0217** | **+.0157** |

The oracle row is the ceiling with perfect poses and a stale message: it
falls from 0.4223 to 0.3420 / 0.3187 / 0.3136 AP@0.7, so most of what delay
costs is the other agent's boxes being where the objects were, which no
pose correction touches. Within what is left:

- **Trained cross-view appearance works, on the sparse pairs too once
  stopped early and pooled over the agent's own frames** (2026-10-05,
  val): zero-shot DINOv2 is a null like colour (AUC 0.68 against the 0.80
  bar); a projection head trained on the dataset's own cross-agent pairs
  reaches 0.83-0.85 but is at chance (0.54) on pairs sharing one or two
  objects, the matcher's loss bucket, when trained 20 epochs; selected on
  held-out train at epoch 3 and pooled over two neighbouring frames each
  side it reaches 0.885 there (n 374) and 0.900 on dense pairs, both bars
  passed. Not yet in stage 1; see the appearance section of
  [alignformer_v2xreal_closing_the_gap.md](alignformer_v2xreal_closing_the_gap.md).
- **Against the MAE-calibrated FreeAlign** the trunks win every cell at
  100 ms, every cell but sigma 2 at AP@0.7 at 200 ms, and at 400 ms draw
  AP@0.7 (+0.002) while keeping AP@0.5 and AP@0.3 by 0.016 and 0.033.
- **Against the AP-calibrated FreeAlign the split is the undelayed one, at
  every delay**: ours at sigma 0 at every threshold (+0.010-0.015 AP@0.7,
  +0.024-0.034 AP@0.5, +0.035-0.054 AP@0.3), FreeAlign at AP@0.7 from sigma
  0.4 up by 0.023-0.058 and at AP@0.5 from sigma 1 up, ours at AP@0.3 up to
  sigma 1 and on its sweep mean (+0.015-0.034). Delay does not move who wins
  where; it lowers both sides by about the same amount (FreeAlign 0.342 ->
  0.284 / 0.257 / 0.253 AP@0.7, boxes-only 0.309 -> 0.256 / 0.235 / 0.229).
  At 1.5 m FreeAlign is within 0.006 of its 1.0 m self at every delay and
  threshold, and no verdict changes
  (`trunk_comparison_delay{1,2,4}_faap1.5_test_result.json`).
- **The trunks do not separate under delay at AP@0.7**: LiDAR minus boxes
  is +0.0006 / -0.0011 / -0.0013, inside the seed bars, with a +0.003-0.007
  edge at AP@0.3. The camera trunk is 0.006-0.008 behind at AP@0.7 at every
  delay. Delay is not where the embedding was going to earn its bytes.
- **With the exact re-solve the AP@0.7 deficit under delay closes to a
  draw** (`B_boxes_only_icp_delay{1,2,4}_test_result.json`, 3 seeds): the
  re-solved boxes-only trunk is .2813 / .2576 / .2502 AP@0.7 on the sweep
  mean against FreeAlign's .2841 / .2572 / .2529 (-0.003 / -0.000 /
  -0.003), and ahead at AP@0.5 (+0.020 / +0.017 / +0.009) and AP@0.3
  (+0.031 / +0.040 / +0.029). FreeAlign keeps AP@0.7 from sigma 0.4 up by
  0.008-0.016; tables and the bucket split are in the companion file.

## Giving AlignFormer the same treatment, and what is left

The AlignFormer-side follow-up to the re-calibration is in
[alignformer_v2xreal_closing_the_gap.md](alignformer_v2xreal_closing_the_gap.md):
the decision rule selected on val AP (a hard Wald threshold at 0.2, +0.002
over the deployed rule on val and on test; the verdict does not move), the
attribution of the gap by shared-object bucket (two thirds of the frames
share three or more objects and lose 0.10 AP@0.7 to FreeAlign at sigma 2 on
solver precision, not association; the clean-case loss is the 1-2 bucket
answering with wrong matches), the exact re-solve built on that finding, and
the camera-colour probe.

## Caveats, in descending order of how much they could matter

1. **One detector, one class, trained here.** The absolute AP is this
   detector's and is not the paper's three-class number. Every comparison is
   paired on identical detections, noise draws and frames, so the *orderings*
   do not depend on the detector being strong, but a stronger detector could
   change the residual that sets tau and with it the regime where the trunks
   separate.
2. **The ground truth carries the dataset's own localization error.** The
   correction is scored against V2X-Real's recorded poses; a correction that
   is right about the world and wrong about the recording is penalised. This
   is the same for every arm, including FreeAlign, and cannot be removed
   without a better ground truth.
3. **Two training seeds per trunk.** Training-seed variance is 0.003-0.008
   AP@0.7 per trunk on val and 0.002-0.007 on test, ten times the noise-seed
   spread the error bars report. The LiDAR-versus-boxes lead is larger than
   it on val and replicates there; on test it is +0.001 and -0.0005 at the
   two seeds. The camera-versus-boxes difference flips sign on val and is
   negative at both seeds on test. Two seeds bound the variance; they do not
   estimate it well.
4. **IRLS constants transferred from OPV2V**, not re-chosen on V2X-Real
   val. The decision rule *has* now been selected on val AP (section above)
   and moves the result by 0.002; the IRLS guard and iteration count have
   not, and remain the one asymmetry left against FreeAlign's AP-selected
   threshold. Legitimate to re-tune; expected to matter about as much.
5. **Frozen ImageNet camera features.** A null on the camera trunk is a null
   about frozen features pooled over projected LiDAR boxes, not about cameras.
6. **FreeAlign without EdgeGAT**, as on OPV2V.
7. **Delay is constant `sim` mode** with the clamp at scenario start, no
   motion model on either side.
8. **FreeAlign's calibration criterion, not its data, set the first
   verdict.** The mean-error criterion put the edge threshold at 0.3 m and
   did not move with twenty times the pairs; selected on val AP the
   threshold is 1.0 m and the verdict under noise reverses. The grid now
   runs to 0.1 m and the threshold is selected on the reported metric; the
   other five parameters are still the MAE selection, which was flat across
   them. A residual fit of the same family set AlignFormer's tau and
   abstention constants; the decision rule was then re-selected on AP
   (section above) and did not change the verdict, the IRLS constants were
   not (caveat 4). The OPV2V FreeAlign
   column in [alignformer_p2.md](alignformer_p2.md) was calibrated by the
   same MAE criterion and has not been re-checked.
9. **Val is six scenarios.** The trunk ordering at AP@0.7 did not transfer
   from val to test (+0.012 and +0.016 at two seeds on val; +0.001 and
   -0.0005 on test), nor did the pose-MAE ordering. Quote
   test; use val only for what it was used for, choosing tau and the
   FreeAlign parameters.

## Reproducing

The full command sequence, from the cache build to the FreeAlign re-calibration
and the paired summaries, is in
[alignformer_v2xreal_reproducing.md](alignformer_v2xreal_reproducing.md).
