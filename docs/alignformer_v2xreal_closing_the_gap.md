# V2X-Real: closing the gap to FreeAlign and to the oracle

Companion to [alignformer_v2xreal.md](alignformer_v2xreal.md), which carries
the three-trunk comparison, the FreeAlign re-calibration and the test and
delay tables. This file holds the AlignFormer-side work that followed the
re-calibration: the decision rule given the same treatment, the attribution
of the remaining gap by shared-object bucket, the exact re-solve that the
attribution points at, and the camera-colour probe. Commands are in
[alignformer_v2xreal_reproducing.md](alignformer_v2xreal_reproducing.md).

## Giving AlignFormer the same treatment: its decision rule selected on val AP

The FreeAlign reversal came from selecting its one open parameter on val AP
instead of mean pose error (previous section). AlignFormer's deployed
inference rule was set on OPV2V by residual-fit criteria of the same family,
so the comparison was asymmetric (caveat 4). The per-pair decision-rule
family was therefore swept on the official val split through the deployed
pipeline, boxes-only trunk, 3 noise seeds, the same paired draws, and ranked
by the criterion that picked FreeAlign's threshold: AP@0.7 sweep mean. The
family: the global threshold `alignformer` (tau), the per-pair James-Stein
shrinkage `per_pair` that ships, a hard Wald threshold `abstain:L` at levels
0.5 / 0.2 / 0.1 / 0.05 / 0.01 (full correction above, abstention below), and
threshold-then-shrink `both:L` at 0.5 / 0.2 / 0.05. The IRLS solve is on in
every arm, so `alignformer_irls` equals `alignformer` here. The directional
rule needs the fit's precision matrix, which this estimate does not carry,
and was dropped. Selection by `scripts/select_alignformer_arm.py`; result
`alignformer_arm_selection_val_result.json`.

| val, 3 seeds, sweep mean | AP@0.7 | AP@0.5 | AP@0.3 | sigma 0 AP@0.7 | sigma 2 AP@0.7 | sigma 0 MAE / cov | sigma 2 cov |
|---|---|---|---|---|---|---|---|
| per_pair (deployed) | .3436 | .5080 | .5628 | .4216 | .2893 | 1.38 m / 47 % | 78 % |
| **abstain 0.2 (selected)** | **.3456** | .5117 | .5663 | .4230 | .2942 | 1.07 m / 24 % | 69 % |
| both 0.2 | .3445 | .5112 | .5678 | .4305 | .2896 | 1.01 m / 24 % | 69 % |
| abstain 0.1 | .3441 | .5122 | .5686 | .4302 | .2936 | 0.89 m / 16 % | 63 % |
| abstain 0.05 | .3419 | .5119 | .5693 | .4361 | .2934 | 0.69 m / 8 % | 59 % |
| abstain 0.01 | .3336 | .5056 | .5669 | **.4441** | .2888 | 0.43 m / 2 % | 47 % |
| abstain 0.5 | .3428 | .5053 | .5587 | .3973 | .2942 | 1.62 m / 49 % | 78 % |
| alignformer (= irls) | .2534 | .3996 | .4875 | .4423 | .1948 | 1.23 m / 8 % | 13 % |
| FreeAlign 1.0 m | **.3702** | **.5160** | .5599 | .4144 | **.3505** | 3.91 m / 56 % | 56 % |

Every thresholded arm sits within 0.012 AP@0.7 of every other, and the best
of them is 0.025 under FreeAlign on the selection metric. The family beats
FreeAlign at sigma 0 at every level and on AP@0.3 (.569 against .560); it
loses at sigma 2 by 0.056 while *emitting more corrections* (69 % against
56 %). The corrections AlignFormer emits at high sigma are worse than
FreeAlign's; a decision rule chooses when to act and cannot repair the
estimate it acts on.

**On test** (2,172 frames, 5 seeds; deployed, selected and the two
runners-up in one file, `B_boxes_only_arms_test_result.json`, FreeAlign 1.0 m
beside them):

| test AP@0.7 | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | sweep mean |
|---|---|---|---|---|---|---|---|---|---|
| per_pair (deployed) | .3855 | .3447 | .3168 | .3036 | .2950 | .2904 | .2739 | .2601 | .3088 |
| abstain 0.2 (selected) | .3932 | .3404 | .3100 | .3032 | .2993 | .2940 | .2797 | .2661 | .3107 |
| both 0.2 | .4029 | .3477 | .3152 | .3022 | .2933 | .2890 | .2734 | .2594 | .3104 |
| abstain 0.1 | .4049 | .3406 | .3040 | .2978 | .2929 | .2906 | .2767 | .2644 | .3090 |
| FreeAlign 1.0 m | .3702 | .3577 | .3420 | .3362 | .3331 | .3321 | .3303 | .3301 | .3415 |

| test, sweep mean, paired by seed | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| selected minus deployed | +0.0021 +/- 0.0001 | +0.0027 +/- 0.0002 | +0.0026 +/- 0.0001 |
| selected minus FreeAlign 1.0 m | -0.0307 +/- 0.0003 | -0.0074 +/- 0.0005 | **+0.0110 +/- 0.0003** |
| deployed minus FreeAlign 1.0 m | -0.0328 +/- 0.0003 | -0.0102 +/- 0.0005 | +0.0084 +/- 0.0004 |

Selected-arm pose on test: sigma 0 MAE 0.48 m at 32 % coverage (deployed
0.56 m at 51 %), sigma 2 1.61 m at 71 % (deployed 1.61 m at 80 %); FreeAlign
4.8 to 5.9 m at 60 % throughout.

- **The verdict does not move.** The selected rule gains 0.002-0.003 over the
  deployed one at every threshold, the size of a selection effect, and
  FreeAlign still takes AP@0.7 and AP@0.5 under noise. The transfer from val
  to test is clean: +0.002 on val, +0.002 on test.
- **The clean case widens in our favour.** Every thresholded arm beats the
  deployed rule at sigma 0 (.393 to .405 against .386) by answering fewer
  pairs; the strictest arm in the test set, `abstain 0.1`, is 0.017 under
  the oracle there and 0.035 above FreeAlign.
- **The remaining gap is structural**, not a calibration artefact. FreeAlign
  matches on distances between boxes of the same agent, which do not move
  when that agent's pose is wrong; its coverage is 60 % at every sigma and
  its AP@0.7 falls 0.04 across the sweep. AlignFormer's stage-1 tokens are
  the sender's boxes projected into the ego frame through the noisy pose,
  so at sigma 2 every token is displaced by about 2 m and 2 deg before the
  transformer sees it. The oracle is .4223 at every sigma, so FreeAlign is
  0.052 / 0.092 / 0.081 (sigma 0 / sigma 2 / sweep mean) under the ceiling
  and is not the bound. The next step is a pose-invariant stage 1
  (FreeAlign's association, or a learned EdgeGAT-style matcher over
  intra-agent pairwise geometry) in front of our weighted solve, variance
  model and Wald abstention, which is what holds the clean case.

### Where the gap lives: the sweep split by shared objects

Every sweep file buckets each ego-CAV pair by how many ground-truth objects
the two agents share (`ap_by_shared_objects`, `pose_by_condition`). Read
that way the 0.03 sweep-mean gap to FreeAlign is two different problems.

| test AP@0.7 by bucket (share of frames) | sigma | uncorrected | per_pair | abstain 0.2 | FreeAlign 1.0 m | oracle |
|---|---|---|---|---|---|---|
| 3+ shared (1,093 frames, 67 %) | 0 | .437 | .418 | .412 | .399 | .437 |
| | 0.4 | .223 | .356 | .348 | .397 | .437 |
| | 1.0 | .166 | .333 | .337 | .395 | .437 |
| | 2.0 | .173 | .289 | .298 | **.395** | .437 |
| 1-2 shared (373 frames, 23 %) | 0 | .426 | **.337** | .380 | .376 | .426 |
| | 0.4 | .248 | .273 | .267 | .246 | .426 |
| | 1.0 | .198 | .250 | .248 | .207 | .426 |
| | 2.0 | .191 | **.243** | .240 | .200 | .426 |
| 0 shared (156 frames, 10 %) | 0 | .327 | .308 | .308 | .327 | .327 |
| | 2.0 | .214 | .212 | .212 | .214 | .327 |

Pose error on the answered pairs (test, boxes-only `per_pair`): in the 3+
bucket 0.37 m at sigma 0 and 0.79 m at sigma 2 at 95 % coverage; FreeAlign
answers 93 % there with a 6.1 m *mean* error at every sigma and sits at 90 %
of the oracle, which is only possible if most of its corrections are
essentially exact and a few are wild. In the 1-2 bucket we answer 606 of
1,131 pairs at sigma 0 with a 2.2 m error against a truth of zero, and
FreeAlign answers 185 with a 27 m error.

- **The large loss is solver precision on dense pairs, not association.**
  Two thirds of the frames share three or more objects; both methods answer
  over 90 % of them; we lose 0.10 AP@0.7 to FreeAlign at sigma 2 there. A
  0.5-0.8 m residual is fatal at IoU 0.7 on a car. The soft Sinkhorn row
  blends neighbours into one virtual point as the tokens get noisier and the
  weighted solve returns a blurred answer; FreeAlign *decides* the
  correspondence and fits exactly. The remedy is a hard re-solve over the
  inlier set after Sinkhorn (`alignformer.refine`, next section), at zero
  extra bytes.
- **The clean-case loss is the 1-2 bucket answering with wrong matches**:
  -0.09 AP@0.7 on 23 % of frames, half of it recovered by the Wald threshold
  at 0.2 (.380). The same bucket is where we beat FreeAlign at sigma 2
  (+0.05 over doing nothing, FreeAlign abstains) and where an identity cue
  would have a home: with one or two shared objects there is no geometric
  context, and whether the single match is right is the whole question.
- **FreeAlign is not the ceiling.** It is 0.04 under the oracle in the 3+
  bucket at every sigma and at the uncorrected line in the other two; the
  oracle is .4223 at every sigma, 0.08 above its sweep mean.

**Under delay and on the LiDAR trunk the selected rule is the deployed
rule** (files `B_boxes_only_arms_delay{1,2,4}_test_result.json`,
`B_boxes+embeddings_arms_test_result.json`). Delay, 3 seeds, sweep mean
over sigma 0 / 0.4 / 1 / 2, selected minus deployed: +0.001 / +0.002 /
+0.004 AP@0.7 at 100 / 200 / 400 ms, +0.002 to +0.004 at the other two
thresholds; selected minus FreeAlign 1.0 m: -0.027 / -0.021 / -0.021 AP@0.7,
-0.004 to -0.009 AP@0.5, +0.020 to +0.030 AP@0.3. LiDAR trunk, 5 seeds:
selected minus deployed +0.001 / -0.002 / -0.001, FreeAlign still +0.031
ahead at AP@0.7. The decision rule was never the lever; the next section
is.

## The exact re-solve

`alignformer.refine` is the remedy the attribution points at: plain ICP
initialised by the soft estimate. Move the CAV boxes by the IRLS ``(psi,
t)``, take the mutually-nearest ego/CAV centre pairs within a gate, solve
SE(2) exactly over them (unweighted, heading-augmented, heading-folded as
head B does), repeat with a tighter gate (2.0 / 1.0 / 0.5 m by default).
It engages only above an evidence floor of three hard pairs, because the
1-2 bucket is where the soft fit beats FreeAlign and a nearest-neighbour
step with two points will snap to the wrong neighbour. The refined arm keeps
the soft fit's Wald statistic, so every decision rule applies to it
unchanged: `alignformer_icp` beside `alignformer_irls`, `<arm>_icp` beside
each decision arm, each pair differing in the solve alone. Gate schedule and
evidence floor are selected on val (three settings, chain K: default gates
2.0 / 1.0 / 0.5 m with three hard pairs, wide 3.0 / 1.5 / 0.75 / 0.5, and the
default gates with two pairs).

**Validation, default gates** (717 frames, 3 paired noise seeds, boxes-only
trunk; file `B_boxes_only_icp_default_val_result.json`, selection
`alignformer_resolve_selection_default_val_result.json`):

| val AP@0.7 | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | sweep mean |
|---|---|---|---|---|---|---|---|---|---|
| abstain 0.2 (selected rule) | .4230 | .3699 | .3473 | .3397 | .3414 | .3335 | .3154 | .2942 | .3456 |
| abstain 0.2, re-solve | **.4312** | .3807 | .3626 | .3572 | **.3621** | **.3590** | **.3523** | .3456 | .3688 |
| abstain 0.2, re-solve, agree 0.5 | .4129 | .3928 | .3750 | .3643 | .3564 | .3592 | .3381 | .3242 | .3653 |
| abstain 0.2, re-solve, agree 1.0 | .4107 | .3931 | **.3791** | **.3715** | .3666 | .3663 | .3552 | .3399 | **.3728** |
| per_pair (deployed) | .4216 | .3730 | .3503 | .3402 | .3359 | .3286 | .3100 | .2893 | .3436 |
| per_pair, re-solve | .4289 | .3819 | .3664 | .3543 | .3550 | .3520 | .3425 | .3266 | .3635 |
| FreeAlign 1.0 m | .4144 | **.3940** | .3735 | .3633 | .3564 | .3577 | .3519 | **.3505** | .3702 |

| val sweep means | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| abstain 0.2 | .3456 | .5117 | .5663 |
| abstain 0.2, re-solve | .3688 | .5280 | **.5755** |
| abstain 0.2, re-solve, agree 1.0 | **.3728** | **.5295** | .5741 |
| FreeAlign 1.0 m | .3702 | .5160 | .5599 |

| paired by seed, sweep mean | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| re-solve minus the rule it refines | +0.0222 +/- 0.0007 | +0.0158 +/- 0.0006 | +0.0087 +/- 0.0003 |
| re-solve minus FreeAlign | -0.0019 +/- 0.0004 | +0.0110 +/- 0.0007 | +0.0146 +/- 0.0006 |
| re-solve + agree 1.0 minus FreeAlign | **+0.0022 +/- 0.0004** | **+0.0124 +/- 0.0006** | **+0.0131 +/- 0.0006** |
| agree 1.0 minus re-solve | +0.0041 +/- 0.0007 | +0.0014 +/- 0.0002 | -0.0015 +/- 0.0003 |

- **The re-solve does what the attribution said it would.** On the dense
  pairs at sigma 2 AP@0.7 goes from .419 to .531 (FreeAlign .562) and the
  answered error from 0.52 m to 0.38 m; the sparse bucket is untouched by
  construction (.125 at sigma 2 either way). +0.022 AP@0.7 on the sweep
  mean, every cell, with no change to the clean case except a gain (.423 to
  .431: the exact fit on correct pairs at sigma 0 returns a smaller
  correction than the soft mixture did).
- **It ties FreeAlign on AP@0.7 and beats it on everything else.** -0.002
  on AP@0.7, +0.011 AP@0.5, +0.015 AP@0.3, +0.017 at sigma 0; FreeAlign
  keeps sigma 2 by 0.005 and the 1.5 m cell.
- **The agreement rule at 1.0 m is the first AlignFormer arm ahead of
  FreeAlign on all three sweep means**, by +0.002 / +0.012 / +0.013. It pays
  in the clean case (.411 against the re-solve's .431; it answers 67 % of
  pairs at sigma 0 with a 1.75 m mean error against the Wald rule's 24 %) and
  earns it from 0.2 m to 1.0 m, where answering with the exact fit beats
  abstaining. The selection criterion (AP@0.7 sweep mean) picks it; both go
  to test.
- **What is left on dense pairs is 0.03 at sigma 2** (.531 against .562,
  oracle .60 on this split). The sparse bucket is where the remaining 0.08 to
  the oracle lives and is the pose graph's job.

**Test** (2,172 frames, 5 paired noise seeds, boxes-only trunk, default
gates, the deployed inference constants unchanged; file
`B_boxes_only_icp_test_result.json`):

| test AP@0.7 | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | sweep mean |
|---|---|---|---|---|---|---|---|---|---|
| per_pair (deployed) | .3855 | .3447 | .3168 | .3036 | .2950 | .2904 | .2739 | .2601 | .3088 |
| abstain 0.2 (selected rule) | .3932 | .3404 | .3100 | .3032 | .2993 | .2940 | .2797 | .2661 | .3107 |
| abstain 0.2, re-solve | **.4024** | .3519 | .3269 | .3237 | .3236 | .3239 | .3220 | .3188 | .3366 |
| abstain 0.2, re-solve, agree 1.0 | .3868 | **.3664** | **.3463** | **.3404** | .3330 | .3280 | .3146 | .3057 | .3401 |
| per_pair, re-solve | .3981 | .3554 | .3308 | .3201 | .3187 | .3157 | .3065 | .2999 | .3306 |
| FreeAlign 1.0 m | .3702 | .3577 | .3420 | .3362 | **.3331** | **.3321** | **.3303** | **.3301** | **.3415** |
| uncorrected (= oracle at 0) | .4223 | .3185 | .2155 | .1771 | .1621 | .1580 | .1584 | .1632 | .2219 |

| test sweep means | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| abstain 0.2 | .3107 | .4854 | .5502 |
| abstain 0.2, re-solve | .3366 | **.5038** | **.5583** |
| abstain 0.2, re-solve, agree 1.0 | .3401 | .5011 | .5520 |
| FreeAlign 1.0 m | **.3415** | .4933 | .5390 |

| paired by seed, sweep mean (5 seeds) | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| re-solve minus the rule it refines | +0.0256 +/- 0.0003 | +0.0177 +/- 0.0003 | +0.0084 +/- 0.0002 |
| re-solve on the deployed rule minus deployed | +0.0220 +/- 0.0001 | +0.0180 +/- 0.0003 | +0.0086 +/- 0.0002 |
| re-solve minus FreeAlign | -0.0051 +/- 0.0003 | **+0.0103 +/- 0.0005** | **+0.0194 +/- 0.0004** |
| re-solve + agree 1.0 minus FreeAlign | -0.0014 +/- 0.0002 | **+0.0082 +/- 0.0004** | **+0.0137 +/- 0.0003** |

Per sigma, agree 1.0 minus FreeAlign at AP@0.7: +0.017, +0.010, +0.004,
+0.001 through 0.6 m, then -0.001, -0.004, -0.011, -0.027. The re-solve
alone: +0.032 at sigma 0, -0.004 to -0.016 from 0.2 m up.

| test AP@0.7 by bucket | sigma | abstain 0.2 | re-solve | re-solve + agree 1.0 | FreeAlign | oracle |
|---|---|---|---|---|---|---|
| 3+ shared (67 % of frames) | 0 | .412 | **.424** | .405 | .399 | .437 |
| | 1.0 | .337 | .376 | .385 | **.395** | .437 |
| | 2.0 | .298 | .370 | .354 | **.395** | .437 |
| 1-2 shared (23 %) | 0 | **.380** | .380 | .371 | .376 | .426 |
| | 2.0 | .240 | .243 | **.244** | .200 | .426 |
| 0 shared (10 %) | any | as uncorrected | same | same | as uncorrected | .327 |

Dense-pair answered error at sigma 2: 0.65 m (abstain 0.2), 0.55 m
(re-solve), 0.53 m (agree 1.0); FreeAlign 6.1 m mean.

- **The re-solve transfers from val to test and then some**: +0.026 /
  +0.018 / +0.008 AP@0.7 / 0.5 / 0.3 over the rule it refines (val +0.022 /
  +0.016 / +0.009), +0.022 over the deployed rule, at every one of the 24
  cells, with no new bytes, no training and the deployed constants.
- **The AP@0.7 gap to FreeAlign goes from 0.031 to 0.005**, and to 0.001
  with the agreement rule; **AP@0.5 and AP@0.3 reverse** (+0.010 and +0.019,
  five standard errors and more); **the clean case widens to +0.032**.
  FreeAlign keeps AP@0.7 from 0.8 m up, by 0.001 to 0.027, which is the
  dense bucket at large noise: .370 against .395 at sigma 2, with the
  re-solve's answered residual still 0.55 m there against the 0.38 m it
  reached on val.
- **The val selection picked the agreement rule and test agrees on the
  ordering at AP@0.7** (agree 1.0 above the re-solve by 0.004, on val by
  0.004) but not at the other two thresholds, where the re-solve alone is
  better by 0.002 and 0.006; and agree 1.0's val lead over FreeAlign
  (+0.002) is -0.001 on test. The two arms are a trade: agree 1.0 answers
  twice as many pairs in the clean case and wins 0.2-0.6 m; the re-solve
  wins sigma 0 and the loose thresholds.
- **The sparse buckets are untouched by construction**, which is where the
  remaining 0.09 to the oracle sits, and where the pose graph was a null.

**The other two gate settings** (same split, seeds and arms):

| val, sweep mean AP@0.7 | re-solve | re-solve + agree 1.0 | paired vs default |
|---|---|---|---|
| default: gates 2.0 / 1.0 / 0.5 m, 3 hard pairs | .3688 | .3728 | |
| wide: 3.0 / 1.5 / 0.75 / 0.5 m, 3 pairs | .3690 | .3718 | -0.0000 / -0.0012 (+/- 0.0002) |
| 2 hard pairs, default gates | .3699 | .3744 | +0.0009 / +0.0017 (+/- 0.0001) |

The setting barely matters: the gain is the hard re-solve itself, not its
gate. Two hard pairs would have been selected over three by 0.001-0.002,
below the 0.002-0.008 training-seed variance; the default, the
pre-registered setting, is the one on test, and the two-pair setting is
recorded as a sensitivity rather than re-run.

## The agreement rule: two estimators, one decision

With the re-solve every pair has two estimates of its correction, the soft
weighted fit and the hard exact fit, reached by different routes from the
same detections. `alignformer.refine.agree` answers with the exact fit where
the two agree within a tolerance (translation gap plus `heading_lambda`
times the heading gap, in metres), abstains where they disagree, and hands
back the decision arm it wraps where the re-solve did not engage. It is
aimed at the clean-case loss, pairs answered with a wrong match. Arms
`<decision arm>_agree_<tolerance>`, tolerances 0.3 / 0.5 / 1.0 m on val
(chain K). On the 170-frame smoke (198 pairs, one seed) the 0.5 m rule
answered 77 % of pairs at sigma 0 with a 0.43 m mean error against the
Wald rule's 33 % at 0.75 m, for the same AP@0.7; full validation decides.

## The pose graph: a sparse pair solved through a third agent

Every val frame and 58 % of test frames carry four agents, and 45 % of
pairs have a fixed infrastructure unit as the noised partner.
`alignformer.posegraph` solves every CAV's ego-frame correction jointly
from the frame's pairwise estimates: the ego's estimate of CAV `j`
measures `C_j` directly; CAV `i`'s estimate of `j`, from projecting `j`'s
boxes into `i`'s frame with both noisy poses (one extra model pass per
unordered CAV pair), measures `T_i<-ego C_j C_i^-1 T_ego<-i`. Measurements
are weighted by the fits' own precision matrices; Gauss-Newton in float64
with autograd Jacobians; a weak prior at the identity pins a CAV nobody
measured. The convention is derived in the module docstring and pinned by a
test against the projection itself.

Two modes. `joint` gives every CAV the joint solution; on the smoke it
moved dense pairs the ego had already solved well (dense answered error
0.32 m to 0.49 m) and lost 0.017 AP@0.7 at sigma 2 against the re-solve.
`fill` keeps every answered ego estimate as the same object and fills only
the CAVs the ego abstained on, each carrying the graph's marginal precision
and Wald statistic with the base arm's decision rule applied at the
chi-square limit; on the smoke it held sigma 2 at the re-solve's .394 with
98 % coverage against 90 %, and cost 0.014 at sigma 0 by filling sparse
pairs with 1.9 m errors the gate let through: the composed precision is
optimistic, as the per-pair Wald precision already was.

**Validation (chain L): a null, in both modes.** Same split, seeds and
arms as above, graph built on `abstain 0.2, re-solve`, files
`B_boxes_only_graph_{fill,joint}_val_result.json`.

| val, sweep mean, paired vs the re-solve it builds on | AP@0.7 | AP@0.5 | AP@0.3 | sigma 0 | sigma 2 |
|---|---|---|---|---|---|
| re-solve (abstain 0.2) | .3688 | .5280 | .5755 | .4312 | .3456 |
| graph, fill | .3680 (-0.0005 +/- 0.0003) | .5262 (-0.0017) | .5716 (-0.0037) | .4205 | .3480 |
| graph, joint | .3642 (-0.0044 +/- 0.0001) | .5230 (-0.0047) | .5697 (-0.0055) | .4164 | .3363 |

- **The filled pairs are not solved, they are answered.** Fill mode raises
  coverage of the 1-2-shared bucket from 59 % to 84 % at sigma 2 and of the
  0-shared bucket from 8 % to 40 %, but the filled corrections carry 6-8 m
  mean error and the buckets' AP does not move (.125 to .124 and .085 to
  .087 at sigma 2). The third agent's measurement of a sparse pair is as
  unreliable as the ego's: in a frame where the ego shares one object with a
  CAV, the other CAVs usually share few with it too, and the CAV-CAV
  estimates come with the same optimistic precision.
- **The gate did not protect the clean case.** Fill costs 0.011 at sigma 0
  and 0.2 and gains 0.002-0.003 from 0.6 m up, which nets to nothing; a
  stricter level would only hand the arm back to the re-solve.
- **Joint mode moves the dense pairs** (answered error 0.38 m to 0.54 m at
  sigma 2) exactly as the smoke said.

Verdict: in this form the pose graph is not taken to test. What it needs
is better cross measurements, not a better solver: either a quality gate on
the CAV-CAV estimates (only measurements whose re-solve engaged on three or
more hard pairs), or anchors whose pose carries no error, which is the
infrastructure-unit protocol question for the group (45 % of pairs). Both
are cheap to try once the GPU is free; neither is a one-day certainty.

## Camera colour on real traffic: coverage passes, signal does not

`scripts/analyze_v2xreal_colour_separability.py` is the OPV2V probe moved to
V2X-Real's cameras (two per agent, 1920x1080, the projection verified by eye
on 2026-09-29) with one addition: every statistic is also split by the
shared-object bucket, because the 1-2 bucket is where an identity cue has a
home here. Same pre-registered bars (ambiguous-subset AUC 0.80, coverage
0.40, 40-70 m coverage 0.25), same shuffle control. 400 agent pairs of the
official val split, 1,979 shared objects, 926 usable in both agents.
Result file `colour_separability_val_result.json`.

| val, cross-agent, paired AUC (true partner vs nearest competitor) | hue-saturation | mean colour | n |
|---|---|---|---|
| all shared objects | 0.602 | 0.608 | 924 |
| ambiguous subset (competitor within 8 m) | **0.632** | 0.612 | 502 |
| ambiguous, 3+ shared | 0.634 | 0.613 | 481 |
| ambiguous, 1-2 shared | 0.595 | 0.571 | 21 |
| same-agent bound (second camera, same instant) | 0.421 | 0.558 | 95 |
| shuffle control | 0.485 | 0.482 | 924 |
| coverage: overall / 40-70 m / 70 m+ | 47 % / 36 % / 32 % | | |

- **Coverage passes, and that is new.** On OPV2V it was 39 % overall and
  21 % at 40-70 m; here 47 % and 36 %. V2X-Real's cameras have four times
  the pixels. The failure is not a lack of pixels.
- **Signal fails, at the same level as the LiDAR embedding.** 0.63 on the
  ambiguous subset against the 0.80 bar, and against 0.627 for the OPV2V
  LiDAR embedding that bought +0.0001 on association. The shuffle control is
  0.49, so the number is a measurement.
- **The same-agent bound is below the cross-agent number.** A second camera
  on the same agent at the same instant matches the vehicle *worse* (0.42
  on hue-saturation) than the other agent's camera does. The raw descriptor
  is dominated by which camera took the picture (exposure, white balance,
  the infrastructure units' mounting) rather than by what it shows. n = 95,
  so this is a bound on the probe, not a precise number; it is enough to say
  that no hand-set colour descriptor is an identity cue across these
  cameras.
- **The 1-2 bucket has 21 ambiguous objects in 400 pairs.** Too few to
  measure a cue on; the bucket is 23 % of test frames but its objects are
  rarely in both cameras. Any appearance cue aimed at it must be measured on
  far more pairs than this probe reads.

Verdict: colour as a simple attribute is closed on V2X-Real as it was on
OPV2V, for a different reason (camera, not pixels). Vehicle type as an
attribute is already on the wire: the 7-float box carries length, width and
height. The only route left for an appearance cue is a learned embedding
trained on cross-agent correspondences (42,036 train pairs supply the
positives) with camera-invariance as an explicit objective, quantized, and
it has to clear the boxes-only control on the 1-2 bucket before it is worth
a byte.
