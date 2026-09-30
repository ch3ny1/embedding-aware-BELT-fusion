# AlignFormer on V2X-Real: three message trunks against FreeAlign on real traffic

> Companion to [alignformer_p2.md](alignformer_p2.md), whose OPV2V measurements
> retired the per-object embedding ("The message: the boxes-only trunk through
> the deployed pipeline"). This document repeats the trunk comparison on
> **V2X-Real**, real traffic with real localization, and adds the one trunk
> OPV2V could not motivate: LiDAR + camera. Everything is trained from
> scratch on this machine; no published V2X-Real number is quoted beside ours.

## Verdict

- **Against FreeAlign on V2X-Real test, every trunk wins every cell**: 8
  sigmas x 3 thresholds, including the clean case, by +0.028 AP@0.7 /
  +0.056 AP@0.5 / +0.054 AP@0.3 on the sweep mean for the boxes-only trunk.
  Under 100 and 200 ms of delay the same holds at every cell but one; at
  400 ms AP@0.7 is a draw and AP@0.5 / AP@0.3 stay ours.
- **The LiDAR embedding's validation lead over boxes-only (+0.012 AP@0.7,
  replicated at a second training seed) shrinks to +0.001 on test**, where it
  survives only in the clean case (+0.012 at sigma 0) and at AP@0.3 (all
  eight cells). This is the same val-to-test shrinkage OPV2V showed.
- **The frozen camera feature never beats the LiDAR embedding** whose bytes
  it shares, and on test it costs 0.012 AP@0.7 against boxes-only.
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
constants. EdgeGAT is still not ported (see P2's "What is ported").

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

**Against FreeAlign, all three trunks win the sweep mean at every threshold,
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

## A second training seed: the LiDAR lead survives, the camera verdict does not

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
at seed 0 (+0.007). This is the finding of the comparison and it replicates.

**The camera verdict is inside training-seed variance and flips sign**:
-0.005 at seed 0, +0.010 at seed 1 against boxes-only; against the LiDAR
trunk it is -0.018 and -0.006. Two seeds cannot say whether the frozen
camera feature costs or buys 0.005 over boxes; they can say it does not
overtake the LiDAR embedding at either seed while sharing its bytes. The
sentence to quote is "the camera feature adds nothing over the LiDAR
embedding", not "the camera feature hurts".

**Against FreeAlign, both seeds agree on the shape**: every trunk wins the
sweep mean at every threshold (boxes-only by 0.017-0.024 AP@0.7, LiDAR by
0.033-0.036), FreeAlign wins sigma 0 and 0.2 at AP@0.3 and AP@0.5 by
0.01-0.02, and the crossover sits at sigma 0.4-0.6.

## Results on test: the FreeAlign verdict holds everywhere, the trunk verdict shrinks

Official test split, 14 scenarios, 2,172 frames, five paired noise seeds,
the seed-0 checkpoints and calibrations from validation, nothing re-chosen.
Rendered from `outputs/v2xreal/trunk_comparison_test_result.json`.

| AP@0.7, test | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncorrected (= oracle at 0) | **.4223** | .3185 | .2155 | .1771 | .1621 | .1580 | .1584 | .1632 | .2219 |
| boxes only | .3855 | .3447 | **.3168** | **.3036** | **.2950** | .2904 | **.2739** | **.2601** | .3086 |
| boxes + LiDAR embedding | **.3970** | **.3486** | .3155 | .3017 | .2931 | **.2910** | .2733 | .2595 | **.3095** |
| boxes + LiDAR + camera | .3882 | .3371 | .3005 | .2882 | .2751 | .2754 | .2599 | .2503 | .2967 |
| FreeAlign | .3700 | .3274 | .2829 | .2628 | .2542 | .2499 | .2485 | .2493 | .2806 |
| LiDAR - boxes | +.0115 | +.0041 | -.0010 | -.0011 | -.0041 | -.0026 | -.0009 | +.0014 | **+.0009 ±.0002** |
| camera - boxes | +.0027 | -.0086 | -.0173 | -.0168 | -.0197 | -.0156 | -.0130 | -.0069 | **-.0119 ±.0002** |
| boxes - FreeAlign | +.0155 | +.0176 | +.0332 | +.0389 | +.0433 | +.0418 | +.0268 | +.0074 | **+.0281 ±.0002** |
| LiDAR - FreeAlign | +.0270 | +.0217 | +.0322 | +.0377 | +.0392 | +.0393 | +.0259 | +.0088 | **+.0290 ±.0003** |

| test, sweep mean | boxes only | LiDAR | camera | LiDAR - boxes | camera - boxes | boxes - FreeAlign | LiDAR - FreeAlign |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP@0.5 | .4831 | **.4878** | .4723 | +.0047 (sigma 0 / 0.2: +.012 / +.011) | -.0108 (7 of 8) | **+.0564** (all 8) | **+.0610** (all 8) |
| AP@0.3 | .5469 | **.5544** | .5442 | **+.0075** (all 8) | -.0027 | **+.0540** (all 8) | **+.0615** (all 8) |

**Every trunk beats FreeAlign at every one of the 24 cells**, and the margin
is larger than on validation: +0.028 / +0.056 / +0.054 for boxes-only at
AP@0.7 / 0.5 / 0.3 on the sweep mean (val: +0.024 / +0.033 / +0.021). The
clean end, which FreeAlign held on validation at AP@0.3 and AP@0.5, is ours on
test at every threshold (+0.016 / +0.015 / +0.021 for boxes-only). The reason
is in the pose diagnostics rather than in our trunks: FreeAlign's six
parameters, chosen on val where they emitted 0.43 m of correction at sigma 0,
emit **2.68 m** at sigma 0 on test (29 % coverage, unchanged), a regression
against uncorrected of 0.052 AP@0.7 where ours is 0.025-0.037. Both sides
were calibrated on val and neither was looked at on test, so this is the
comparison as protocol defines it; caveat 8 records that the transfer went
worse for FreeAlign than for us and that a re-tuned FreeAlign is the fair
follow-up, not a re-tuned AlignFormer.

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
checkpoints are being swept on test to put a training-seed bar on this
number (`seed1_*_test_result.json`, to follow).

**The camera trunk costs 0.012 AP@0.7 on test**, negative at seven of eight
cells against boxes-only and at all eight against the LiDAR trunk (-0.013),
and 0.011 at AP@0.5. Val's sign flip across training seeds means the size is
uncertain; test's sign is not. The line to quote stands: frozen ImageNet
features pooled over projected LiDAR boxes never beat the LiDAR embedding
they are concatenated to, and on test they are worse than sending nothing.

**Coverage and the IRLS-only arm behave as on val.** The per-pair rule
corrects 45-52 % of pairs at sigma 0 and 77-81 % at sigma 2; the IRLS-only
arm corrects 5-8 % and scores 0.4138 at sigma 0 (-0.0085 against uncorrected)
and 0.1630 at sigma 2, identical to uncorrected. The per-pair rule is the
whole method on this dataset.

## Results under communication delay on test

Constant delay of 1, 2 and 4 frames (100 / 200 / 400 ms at 10 Hz) on the
other agent's message, localization error swept at sigma 0 / 0.4 / 1 / 2,
three paired noise seeds, test split. Rendered from
`outputs/v2xreal/trunk_comparison_delay{1,2,4}_test_result.json`.

| test, sweep mean over sigma 0 / 0.4 / 1 / 2 | oracle | boxes only | LiDAR | camera | FreeAlign | boxes - FreeAlign | LiDAR - boxes | camera - boxes |
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

The oracle row is the ceiling with perfect poses and a stale message: it
falls from 0.4223 to 0.3420 / 0.3187 / 0.3136 AP@0.7, so most of what delay
costs is the other agent's boxes being where the objects were, which no
pose correction touches. Within what is left:

- **100 ms: ours at every cell and every threshold** for boxes-only and
  LiDAR; the camera trunk drops one cell (sigma 2, AP@0.7, -0.004).
- **200 ms: ours at every cell except sigma 2 at AP@0.7**, a draw for
  boxes-only (-0.0006) and a loss for LiDAR (-0.004); AP@0.5 and AP@0.3 at
  every cell by 0.03-0.06.
- **400 ms: AP@0.7 is a draw** on the sweep mean (+0.002, losing sigma 0 by
  0.003 and sigma 2 by 0.005, winning 0.4 and 1 by 0.007-0.010); AP@0.5 and
  AP@0.3 are ours by 0.016 and 0.033 with one cell lost (sigma 0, AP@0.5,
  -0.0006). On OPV2V the 400 ms AP@0.7 cell went to FreeAlign by 0.050; here
  it does not, because FreeAlign's emitted correction under 400 ms is 3.3 m
  at sigma 0 (ours 0.76 m) -- the always-correct policy is punished by the
  same miscalibration as in the clean case.
- **The trunks do not separate under delay at AP@0.7**: LiDAR minus boxes
  is +0.0006 / -0.0011 / -0.0013, inside the seed bars, with a +0.003-0.007
  edge at AP@0.3. The camera trunk is 0.006-0.008 behind at AP@0.7 at every
  delay. Delay is not where the embedding was going to earn its bytes.

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
3. **Two training seeds per trunk, on val only.** Training-seed variance is
   0.003-0.008 AP@0.7 per trunk, ten times the noise-seed spread the error
   bars report; the LiDAR-versus-boxes lead is larger than it and replicates,
   the camera-versus-boxes difference is not and flips sign. The test-split
   numbers are the seed-0 checkpoints alone.
4. **IRLS and abstention constants transferred from OPV2V**, not re-chosen
   on V2X-Real val. Legitimate to re-tune; not done.
5. **Frozen ImageNet camera features.** A null on the camera trunk is a null
   about frozen features pooled over projected LiDAR boxes, not about cameras.
6. **FreeAlign without EdgeGAT**, as on OPV2V.
7. **Delay is constant `sim` mode** with the clamp at scenario start, no
   motion model on either side.
8. **FreeAlign's val calibration transferred worse than ours.** Its emitted
   correction at sigma 0 went from 0.43 m on val to 2.68 m on test while its
   coverage stayed at 29 %, and that is most of why the clean case flips to
   ours on test. Protocol was the same for both sides (calibrate on val,
   report on test, never look at test to re-choose), so the comparison
   stands as run; the fair follow-up is a FreeAlign calibration with more
   than six val scenarios behind it, not a re-tuned AlignFormer.
9. **Val is six scenarios.** The trunk ordering at AP@0.7 did not transfer
   from val to test (+0.012 to +0.001), nor did the pose-MAE ordering. Quote
   test; use val only for what it was used for, choosing tau and the
   FreeAlign parameters.

## Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
D=/media/chenyi/basement2/dataset/v2x-real; V=outputs/v2xreal
AF=configs/alignformer_v2xreal.yaml; AFC=configs/alignformer_v2xreal_camera.yaml
DET=configs/v2xreal_detector.yaml
ROBUST="--robust-solve huber --robust-iterations 2 --robust-min-evidence 3.0"

# Detector (15 epochs; checkpoint epoch chosen by val loss, named in $DET).
# Cache with camera features, all three splits (~40 min; resumes).
python -m embedding_aware_belt_fusion.alignformer.cache --config $DET \
  --splits $D/train $D/val $D/test \
  --cache-root /media/chenyi/basement2/cache/alignformer_v2xreal --camera

# Stage 1, three trunks (~17 min each on an RTX 4090).
python -m embedding_aware_belt_fusion.alignformer.train --config $AF --stage 1 --output-dir $V/stage1
python -m embedding_aware_belt_fusion.alignformer.train --config $AF --stage 1 --zero-embeddings --output-dir $V/stage1_zero_embeddings
python -m embedding_aware_belt_fusion.alignformer.train --config $AFC --stage 1 --output-dir $V/stage1_camera

# Correspondence variance on val; the four numbers are pasted into both configs.
python scripts/fit_correspondence_variance.py --config $AF --output $V/variance_fit_result.json

# Stage 2 (~28 min each) and shrinkage on val, per trunk.
for spec in "boxes+embeddings|$AF|stage1|B_boxes+embeddings" \
            "boxes_only|$AF|stage1_zero_embeddings|B_boxes_only" \
            "boxes+embeddings+camera|$AFC|stage1_camera|B_camera"; do
  IFS='|' read -r content cfg s1 tag <<<"$spec"
  python -m embedding_aware_belt_fusion.alignformer.train --config $cfg --stage 2 --head B \
    --message-content $content --stage1-checkpoint $V/$s1/best.pth \
    --variance-weighting scalar --output-dir $V/stage2_$tag
  python -m embedding_aware_belt_fusion.alignformer.evaluate --config $cfg --metric shrinkage \
    --checkpoint $V/stage2_$tag/best.pth $ROBUST --output $V/shrinkage_${tag}_calibration_result.json
done

# FreeAlign's six parameters on val.
python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF \
  --split $D/val --stride 4 --sigma 1.0 --output $V/freealign_calibration_result.json
FA="--freealign --freealign-calibration $V/freealign_calibration_result.json"

# Sweeps: val 3 seeds (pipeline check), test 5 seeds (report), delay 1/2/4 on test 3 seeds.
sweep() { local tag=$1 cfg=$2 split=$3 seeds=$4; shift 4
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $cfg --split $D/$split --checkpoint $V/stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds $seeds "$@"; }
for spec in "B_boxes_only|$AF" "B_boxes+embeddings|$AF" "B_camera|$AFC"; do
  IFS='|' read -r tag cfg <<<"$spec"
  sweep $tag $cfg val 3 --output $V/${tag}_val_result.json
  sweep $tag $cfg test 5 --output $V/${tag}_test_result.json
  for d in 1 2 4; do
    sweep $tag $cfg test 3 --sweep 0 0.4 1.0 2.0 --delay-frames $d --output $V/${tag}_delay${d}_test_result.json
  done
done

# Message bytes (structural; the camera trunk shares the LiDAR trunk's bytes).
python - <<'PY'
import json; from pathlib import Path
from embedding_aware_belt_fusion.alignformer.bandwidth import alignformer_message_bytes, alignment_overhead_bytes
m = alignformer_message_bytes(Path("/media/chenyi/basement2/cache/alignformer_v2xreal/test"), 128, 64)
n = m["objects_per_agent_mean"]
Path("outputs/v2xreal/bandwidth_result.json").write_text(json.dumps({
  "method": "v2xreal_bandwidth", "metric": "bytes_per_frame_per_agent", "precision": "float32",
  "alignformer": m, "alignment_overhead": {k: alignment_overhead_bytes(n, e) for k, e in
  [("freealign", 0), ("alignformer_boxes_only", 0), ("alignformer_boxes_embedding", 128), ("alignformer_boxes_embedding_camera", 128)]}}, indent=2))
PY

# Training-seed replicate: seed-1 copies of the configs for TRAINING only;
# the sweep runs on the shipped config (whose training.seed also seeds the
# noise draws) with the seed-1 checkpoint and shrinkage swapped in.
sed 's/^  seed: 0$/  seed: 1/' $AF > $V/alignformer_v2xreal_seed1.yaml
sed 's/^  seed: 0$/  seed: 1/' $AFC > $V/alignformer_v2xreal_camera_seed1.yaml
for spec in "boxes_only|$V/alignformer_v2xreal_seed1.yaml|$AF|--zero-embeddings|B_boxes_only" \
            "boxes+embeddings|$V/alignformer_v2xreal_seed1.yaml|$AF||B_boxes+embeddings" \
            "boxes+embeddings+camera|$V/alignformer_v2xreal_camera_seed1.yaml|$AFC||B_camera"; do
  IFS='|' read -r content tcfg ecfg zero tag <<<"$spec"
  python -m embedding_aware_belt_fusion.alignformer.train --config $tcfg --stage 1 $zero --output-dir $V/seed1_stage1_$tag
  python -m embedding_aware_belt_fusion.alignformer.train --config $tcfg --stage 2 --head B \
    --message-content $content --stage1-checkpoint $V/seed1_stage1_$tag/best.pth \
    --variance-weighting scalar --output-dir $V/seed1_stage2_$tag
  python -m embedding_aware_belt_fusion.alignformer.evaluate --config $ecfg --metric shrinkage \
    --checkpoint $V/seed1_stage2_$tag/best.pth $ROBUST --output $V/shrinkage_seed1_${tag}_calibration_result.json
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $ecfg --split $D/val --checkpoint $V/seed1_stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_seed1_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds 3 --output $V/seed1_${tag}_val_result.json
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $ecfg --split $D/test --checkpoint $V/seed1_stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_seed1_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds 5 --output $V/seed1_${tag}_test_result.json
done

# Every table under "Results" is rendered from those JSONs.
for split in val test delay1_test delay2_test delay4_test; do
  python scripts/summarize_v2xreal_trunks.py --reference boxes_only \
    --trunk boxes_only=$V/B_boxes_only_${split}_result.json \
    --trunk lidar=$V/B_boxes+embeddings_${split}_result.json \
    --trunk lidar_camera=$V/B_camera_${split}_result.json \
    --output $V/trunk_comparison_${split}_result.json
done
python scripts/summarize_v2xreal_trunks.py --reference boxes_only \
  --trunk boxes_only=$V/seed1_B_boxes_only_val_result.json \
  --trunk lidar=$V/seed1_B_boxes+embeddings_val_result.json \
  --trunk lidar_camera=$V/seed1_B_camera_val_result.json \
  --output $V/trunk_comparison_seed1_val_result.json
for tag in B_boxes_only B_boxes+embeddings B_camera; do
  python scripts/summarize_v2xreal_trunks.py --reference seed0 \
    --trunk seed0=$V/${tag}_val_result.json --trunk seed1=$V/seed1_${tag}_val_result.json \
    --output $V/seed_replicate_${tag}_val_result.json
done
```
