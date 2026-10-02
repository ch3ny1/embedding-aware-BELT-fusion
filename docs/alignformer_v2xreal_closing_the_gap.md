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

The delay re-run (100 / 200 / 400 ms) and the LiDAR-trunk re-run with the
selected arm are in flight; files `B_boxes_only_arms_delay{1,2,4}_test_result.json`
and `B_boxes+embeddings_arms_test_result.json`.

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
evidence floor are selected on val (three settings, chain J); results follow
in this section when the sweeps land.

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
optimistic, as the per-pair Wald precision already was. Validation (chain
L, after chain K) decides the mode and whether the gate needs a stricter
level than the base arm's; result files
`B_boxes_only_graph_{fill,joint}_val_result.json`.

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
