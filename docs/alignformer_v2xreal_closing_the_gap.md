# V2X-Real: closing the gap to FreeAlign and to the oracle

Where this work stands against prior work and as a paper is in
[alignformer_paper_position.md](alignformer_paper_position.md).
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

**Under communication delay the ordering holds** (`B_boxes_only_icp_delay{1,2,4}_test_result.json`;
constant 1 / 2 / 4 frames = 100 / 200 / 400 ms on the other agent's message,
sigma 0 / 0.4 / 1 / 2, 3 paired seeds, 2,172 frames; same arms, same
constants). Sweep means:

| test, sweep mean | oracle | abstain 0.2 | re-solve | re-solve + agree 1.0 | FreeAlign 1.0 m | re-solve - FreeAlign | agree 1.0 - FreeAlign |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 ms, AP@0.7 | .3420 | .2572 | .2813 | .2820 | **.2841** | -.0034 | -.0029 |
| 100 ms, AP@0.5 | .5588 | .4390 | **.4631** | .4595 | .4425 | **+.0199** | **+.0153** |
| 100 ms, AP@0.3 | .6075 | .5330 | **.5433** | .5376 | .5126 | **+.0307** | **+.0236** |
| 200 ms, AP@0.7 | .3187 | .2364 | .2576 | **.2579** | .2572 | -.0003 | +.0001 |
| 200 ms, AP@0.5 | .4814 | .3786 | **.4009** | .3997 | .3838 | **+.0166** | **+.0151** |
| 200 ms, AP@0.3 | .5673 | .4844 | **.4947** | .4876 | .4544 | **+.0401** | **+.0325** |
| 400 ms, AP@0.7 | .3136 | .2325 | .2502 | .2520 | **.2529** | -.0031 | -.0013 |
| 400 ms, AP@0.5 | .4464 | .3507 | .3685 | **.3693** | .3602 | **+.0092** | **+.0093** |
| 400 ms, AP@0.3 | .4881 | .4244 | **.4326** | .4293 | .4045 | **+.0288** | **+.0246** |

Paired standard errors over the three seeds are 0.0002-0.0011. Per sigma,
re-solve minus FreeAlign at AP@0.7: +.025 / -.016 / -.010 / -.012 at 100 ms,
+.031 / -.015 / -.009 / -.008 at 200 ms, +.032 / -.016 / -.015 / -.013 at
400 ms; agree 1.0 minus FreeAlign: +.013 / +.002 / -.005 / -.022,
+.018 / +.004 / -.004 / -.018, +.025 / +.001 / -.009 / -.022.

- **The re-solve's gain over the rule it refines survives delay but
  shrinks with it**: +0.024 / +0.021 / +0.018 AP@0.7 on the sweep mean at
  100 / 200 / 400 ms (undelayed +0.026), growing with sigma at every delay
  (+0.009 at sigma 0 to +0.042 / +0.036 / +0.031 at sigma 2). A stale
  message moves the other agent's boxes with the objects, so the hard
  pairs the re-solve fits are themselves displaced: the dense-pair
  answered residual at sigma 2 is 0.62 / 0.74 / 0.82 m against 0.55 m
  undelayed.
- **Against FreeAlign the undelayed verdict carries over at every delay**:
  AP@0.7 is a draw on the sweep mean (-0.003 / -0.000 / -0.003 for the
  re-solve, -0.003 / +0.000 / -0.001 with agreement; the selected-rule gap
  had been -0.027 / -0.021 / -0.021), AP@0.5 and AP@0.3 are ours by
  0.009-0.020 and 0.025-0.040, and the clean case is ours by 0.025-0.032.
  FreeAlign keeps AP@0.7 from sigma 0.4 up for the re-solve alone, and
  from sigma 1 up with the agreement rule, which again holds sigma 0.4
  (+0.001 to +0.004).
- **Where FreeAlign's AP@0.7 lead lives under delay is the dense bucket
  at large noise, as before**: at sigma 2, 3+ shared objects, .289 / .261 /
  .248 for the re-solve against .309 / .276 / .272 for FreeAlign (oracle
  .339 / .313 / .314). The 1-2 bucket is ours at every delay (.215 / .202 /
  .206 against .187 / .184 / .199), and the 0 bucket is the uncorrected
  pose for everyone.

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

## Appearance, round three: a foundation model, zero-shot and then trained

The camera trunk's null was about a frozen ImageNet ResNet-18 pooled over
projected boxes, and the colour probe's about paint; neither said
appearance cannot carry cross-agent identity. Two further questions were
asked on 2026-10-05, through the same pre-registered probe (same 400 val
pairs, gates, nearest-neighbour distractor within 8 m, shuffle control,
bars AUC 0.80 / coverage 0.40 / far-field 0.25). Code:
`alignformer/foundation_features.py` (frozen DINOv2, timm, 224-px
letterboxed crop with 10 % context; descriptors = pooled class token and
the silhouette-masked mean of patch tokens), `alignformer/appearance_head.py`
and `scripts/{cache_v2xreal_appearance_features,train_v2xreal_appearance_head}.py`.

**Zero-shot DINOv2 is a null** (`appearance_separability_{colour_dinov2_small,dinov2_base}_val_result.json`;
colour reproduced alongside, 0.632 / 0.612):

| val, 400 pairs, ambiguous subset (n 502) | AUC | same-agent bound (n 95) |
|---|---:|---:|
| colour, hue-saturation | .632 | .421 |
| DINOv2 ViT-S/14, masked patch mean | .653 | .611 |
| DINOv2 ViT-B/14, masked patch mean | .681 | .653 |
| DINOv2 ViT-B/14, class token | .645 | .726 |

Coverage 0.468 and far-field 0.361 pass; shuffle controls 0.49-0.51. The
same-agent bound is the telling number: two cameras of ONE vehicle at ONE
instant barely separate it from its neighbour, so viewpoint dominates the
raw feature and a bigger frozen backbone is not the lever.

**A projection head trained for viewpoint invariance clears the signal bar
but not where it would pay.** Supervision comes from the dataset itself:
every vehicle annotated in two agents' frames at one timestamp is a
cross-view positive, the nearest other vehicles in the partner frame are
hard negatives (InfoNCE, symmetric, in-batch false negatives of the same
object masked), a two-layer head over the frozen 1536-d ViT-B descriptors.
Train split at every second frame (11,509 agent-frames, 76,330 pairs), 20
fixed epochs, val scored along the way but not selected on
(`appearance_head_base_val_result.json`; the checkpoint is not committed).

| val ambiguous AUC | 400 probe pairs (n 502) | every val pair (n 9,801) | shared 1-2 (n 21 / 374) | shared 3+ (n 481 / 9,427) | 40-70 m | 70 m+ |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot ViT-B patch mean | .681 | .694 | .476 / .468 | .690 / .703 | .648 / .698 | .667 / .610 |
| untrained head (random projection, 5 inits) | .657 | .696 | - / .607 | - / .700 | - / .713 | - / .659 |
| control: trained on shuffled identities | .627 | - | .381 / - | .638 / - | .600 / - | .556 / - |
| **trained head** | **.831** | **.845** | **.286 / .537** | **.854 / .857** | **.752 / .804** | **.444 / .700** |
| trained head, train-split probe | 1.000 | | | | | |

(`appearance_head_base_allpairs_val_result.json` is the every-pair
diagnostic: the same split, more pairs, no selection.)

- **Signal bar met**: 0.831 on the pre-registered pairs, 0.845 on all of
  val, against 0.80; +0.15 over zero-shot and over an untrained head. A
  head trained on cross-agent pairs does make DINOv2 features carry
  identity across the viewpoint change. That is the answer to "should we
  train for view invariance": yes, it works, in 1.3 minutes of training.
- **The control bar was mis-specified and fails as written.** It asked a
  head trained on shuffled identities to score within 0.05 of 0.5; it
  scores 0.627. But any projection of a feature whose zero-shot AUC is 0.68
  keeps part of that geometry: an untrained head scores 0.657, and the
  control sits below both. The right reading of the control is "no higher
  than zero-shot", which it meets; the script's literal verdict
  (`worth_building: false`) is recorded as is, with this note. The
  shuffled-pair control of the probe itself is 0.50 for every column.
- **It overfits**: train-probe AUC 1.000, val 0.857 at epoch 10 and 0.831
  at 20 with the loss still falling. A held-out slice of train for early
  stopping, dropout or a smaller head would recover some of 0.03; none of
  that was selected on val.
- **On the pairs that share one or two objects it is at chance**: 0.537
  on 374 ambiguous objects across all of val (0.286 on the 21 in the probe
  sample), with zero-shot below chance there (0.47). That bucket is where
  the matcher loses in the clean case (0.09 AP@0.7 on 23 % of test frames)
  and the whole reason an identity cue was wanted. Where the head works
  (3+ shared objects, 0.857) the exact re-solve already closes the gap by
  geometry. The sparse pairs are the far, oblique, often rear-versus-front
  views that make up few training pairs and the hardest ones; a head
  trained to the average pair does not reach them.

**Where this left appearance, before the sparse-pair round.** Trained
cross-view appearance is real and cheap, which colour and frozen features
could not show; but a head trained to the average pair did not reach the
sparse pairs, so it was not carried into the matcher. The next round went
after those pairs directly.

### Round four: training toward the sparse pairs (2026-10-05, evening)

The cache diagnostic first: sparse (<= 2 shared) anchors are far (median
51 m in train, 61 m in val, 41-49 % beyond 70 m), zero-shot is at chance
on them in train too (0.50), the train split holds only 280 such ambiguous
objects while dense pairs carry 17,000 anchors beyond 40 m, and in val
87 % of the sparse-bucket distractors are themselves seen by the ego: the
question is "two far cars 8 m apart, both seen by both agents, which is
which". Range, not the bucket, is the difficulty.

Protocol, fixed before the first number (`scripts/train_v2xreal_appearance_head_sparse.py`):
the last fifth of train scenarios held out whole; every configuration keeps
the epoch with the best held-out far-range AUC (mean over 40-70 m and
70 m+); val read once, every val pair (8,016 ordered pairs, 374 ambiguous
sparse objects), for the selected configuration. Bars: shared-1-2
ambiguous AUC >= 0.80 and shared-3+ >= 0.80. Levers tried: draws weighted
toward far anchors (x4, x16) and sparse pairs (x10); the partner agent's
neighbouring cached frames as further positives (x3 pairs); dropout 0.2;
colour (hue-saturation + mean colour, 195-d) concatenated to the DINOv2
descriptor; and each agent pooling its own object's embedding over its
neighbouring frames (``--track-window``, cached steps each side; the
ground-truth id stands in for the agent's own tracker, and only the
agent's own frames are pooled, so nothing crosses agents).

| val, every pair, ambiguous AUC | selected (held-out far AUC) | shared 1-2 (n 374) | shared 3+ (n 9,427) | 40-70 m | 70 m+ | verdict |
|---|---|---:|---:|---:|---:|---|
| first head, 20 epochs, no selection | - | .537 | .857 | .804 | .700 | |
| DINOv2 only, 6 configs | far x16 + sparse x10 + offsets + dropout, ep 20 (.747) | .623 | .892 | .863 | .667 | fail |
| + colour | baseline, ep 20 (.768) | .463 | .887 | .853 | .707 | fail |
| + colour, track window 2 | far x16 + sparse x10 + offsets + dropout, ep 18 (.824) | .679 | .910 | .856 | .750 | fail |
| **DINOv2 only, track window 2** | **baseline, ep 3 (.820)** | **.885** | **.900** | **.897** | **.838** | **pass** |

Files `appearance_head_sparse{,_colour,_colour_track2,_track2}_val_result.json`.
Window sensitivity for the passing head, eval-only on every val pair
(`appearance_head_track_window_sensitivity_val_result.json`; window 2 was
the single value chosen in advance):

| window (cached steps each side; 1 step = 0.2 s) | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| selected head (baseline, ep 3), shared 1-2 | .807 | .882 | .885 | .904 | .906 |
| selected head, all ambiguous | .876 | .891 | .899 | .902 | .906 |
| first head (20 epochs), shared 1-2 | .537 | .516 | .516 | .471 | .463 |
| zero-shot patch mean, shared 1-2 | .468 | .468 | .471 | .468 | .465 |

- **The two levers that worked were not the ones the round was named
  for.** Weighting the draws toward far or sparse rows never moved the
  held-out far AUC (0.72-0.75 whichever way), and colour hurt the sparse
  bucket (0.46 alone, 0.68 with the window). What worked: **stopping
  early** (the held-out selection with the window picked the plain
  configuration at epoch 3; that head alone, no pooling, is at 0.807 on
  the sparse bucket, where the 20-epoch heads are at 0.54-0.62: they
  overfit the dense pairs and lose the far ones), and **each agent pooling
  its own object over a few frames** (+0.08 on the sparse bucket at two
  steps, saturating by three; pooling does nothing for zero-shot and
  nothing for the overfit head, so it is cleaning a signal that has to be
  there first).
- **Both bars pass**: 0.885 on the sparse bucket (n 374), 0.900 on the
  dense one, 0.838 beyond 70 m, 0.899 over all 9,801 ambiguous objects.
  Selection never touched val; the probe's own 400 pairs give 0.908 /
  0.905 (n 21) for the same head.
- **What it costs on the wire**: a 128-d unit vector per box, before
  quantization; the pooling happens on the sending agent.

**Where this left appearance after round four.** A per-object appearance
embedding that separates a vehicle from its nearest neighbour across
agents on the sparse pairs existed, on annotated boxes with ground-truth
tracks. The next step was the one every earlier appearance result failed
to reach: into the matcher.

### Into the matcher: the DINO-head trunk (2026-10-06)

`alignformer/dino_features.py` carries the head through the existing
LiDAR+camera path: per-detection crop descriptors (frozen DINOv2-base,
camera chosen by visible area, the probe's occlusion gate) through the
head, pooled CAUSALLY over up to four earlier frames of the same agent
(mutual-nearest world-frame centre, gate 2 m + 1.5 m per frame of gap; only
the sender's own frames and pose). The cache keeps the pooled vector for
training and the per-frame one beside it; evaluation pools the live frame
with its cached predecessors the same way, and reproduces the cached
vectors to float16. `configs/alignformer_v2xreal_dino{,_det}.yaml` differ
from the LiDAR config in `camera_dim` 128, `camera_source` and the cache
root only (pinned). Stage 1 and 2 as every other trunk; val sweep, 3 seeds,
the chain-M arms, against the boxes-only trunk on the same seeds
(`B_dino_icp_val_result.json`, `B_dino_det_icp_val_result.json`).

Two heads were tried. The round-four head, trained on annotated boxes,
scores only AUC 0.73 on DETECTOR boxes (0.70 unpooled; detector position
and heading error misalign the silhouette, and the probe's occlusion gate
was not yet in the matcher path). A head trained on the detections
themselves, with the id of the annotation each detection matched as its
label (`--detection-cache`; same held-out selection, same bars;
`appearance_head_detections_val_result.json`), scores 0.835 on them
(0.803 at 40-70 m). On real detections the shared-1-2 bucket nearly
vanishes: 9 ambiguous objects in all of val against 374 on annotations,
because the far objects that define those pairs are rarely detected by
both agents with a camera view.

| val, 3 seeds, AP@0.7 | boxes-only + re-solve | + DINO head (annotations) | + DINO head (detections) | FreeAlign 1.0 m |
|---|---:|---:|---:|---:|
| sigma 0 | .4312 | .4401 | .4375 | .4144 |
| shared 1-2 at sigma 0 (39 frames) | .200 | .254 | .249 | .241 |
| shared 3+ at sigma 0 (243 frames) | .616 | .613 | .604 | .575 |
| shared 3+ at sigma 2, soft arm (Wald 0.2) | .419 | .397 | .387 | .562 |
| sweep mean, re-solve arm | .3688 | .3553 | .3577 | .3702 |
| sweep mean, agreement arm | .3728 | .3691 | .3699 | |
| sweep mean, Wald 0.2 (no re-solve) | .3456 | .3177 | .3168 | |
| stage-1 val Top-1 | .994 | .970 | .976 | |

Paired on the seeds, the detection-trained trunk minus boxes-only on the
sweep mean: -0.011 +/- 0.000 AP@0.7 with the re-solve, -0.003 with the
agreement rule, -0.030 for the soft arm alone; the annotation-trained one
-0.012 / -0.003 / -0.028.

- **The clean case and the sparse bucket move the right way, by the same
  amount with either head**: +0.006 to +0.009 at sigma 0, +0.05 on the
  shared-1-2 bucket (39 frames, so roughly +0.01 on the full split).
- **Under any localization noise the soft association is worse with the
  cue than without it**, by 0.02-0.03 from sigma 0.2 up; the re-solve
  recovers most of that and the agreement rule nearly all, which is why the
  arms that re-fit by geometry lose little. Stage-1 Top-1 says the same:
  geometry alone associates 99.4 % of val pairs at the training noise;
  adding a 0.84-AUC cue can only perturb an association that is already
  right where the overlap is dense, and the transformer learns to lean on
  it anyway.
- **Where the cue could pay, the detector does not deliver the input**: the
  sparse pairs' far objects are rarely detected by both agents with a
  camera view, so the bucket is 9 objects on detections.

**Verdict.** Not shipped. On V2X-Real the matcher's association is
geometric where it can be, and appearance has no room there; where it
would have room, detection coverage takes it away. The appearance work
leaves behind a trained cross-view head that works (0.835 on detections,
0.885 on annotated sparse pairs), the track-pooled camera path, and the
cue-agnostic probe, all reproducible; and a sharper picture of the
remaining gap to FreeAlign, which is still the dense bucket at large noise
and belongs to the solver, not to association.

## The dense bucket: a hard fit over the learned correspondences (2026-10-06 to 07)

Everything above left one cell to FreeAlign: pairs sharing three or more
objects at large noise, two thirds of the frames. The re-solve had taken
the deployed arm from -0.031 to -0.005 AP@0.7 there, and the question was
whether the rest was association (which the appearance work had just
ruled out) or the solver. A diagnostic on the 194 dense val pairs at
sigma 2 (`scripts/diagnose_v2xreal_dense_resolve.py`, stride 3, one
seed; `dense_resolve_diagnosis{,_icpr,_search}_val_result.json`) settled
it.

**Diagnosis.** The re-solve's final correspondences are 95 % precise and
98 % complete against the ground-truth identities. Its least-squares fit
over them nevertheless has a mean answered residual of 0.60 m (median
0.25, 13 % over 1 m). The same fit over only the correct pairs is 0.28 m;
a RANSAC fit over the re-solve's own pairs 0.27 m; the fit over oracle
correspondences 0.27 m; FreeAlign 4.31 m mean (12 % over 1 m, a tail of
wrong answers behind a 0.26 m median). So the residual is not the
correspondences but the fit: a handful of wrong pairs with full weight
pull a least-squares answer, and a consensus fit over the same pairs does
not let them. Nineteen of the 194 pairs never engaged the re-solve at all
(the first gate found fewer than three pairs under a soft estimate that
was 3.4 m off on average); a candidate search around the soft estimate
engaged every one of them at 0.23 m. With both, the mean residual on the
bucket is 0.28 m and 2 % of pairs are over 1 m.

**The RANSAC re-solve** (`refine.py`, mode `icp_ransac`, suffix `_icpr`).
Each ICP stage fits SE(2) by two-point RANSAC instead of least squares:
every pair of correspondences proposes a transform (exhaustively up to 16
pairs, 512 fixed draws beyond), the transform with the largest consensus
at `inlier_m` 1.0 wins, and the heading-augmented exact fit over its
inliers is the stage's answer. The final stage's inlier count is exported
as the estimate's `consensus`. A `candidate_search` (`--refine-search-m`)
samples poses around the soft estimate when the first gate finds fewer
than `min_pairs`. Val, 3 seeds, Wald 0.2, the chain-M arms
(`B_boxes_only_icpr*_val_result.json`; FreeAlign 1.0 m .3702 / .5160 /
.5599):

| val sweep mean, re-solve arm | AP@0.7 | AP@0.5 | AP@0.3 | vs FreeAlign (paired) | dense sigma 2 |
|---|---:|---:|---:|---:|---:|
| ICP least squares, gates 2 / 1 / 0.5 (deployed) | .3688 | .5280 | .5755 | -0.0019 +/- 0.0004 | .531 |
| RANSAC, inlier 1.0, same gates | .3712 | .5304 | .5770 | +0.0004 +/- 0.0005 | .539 |
| RANSAC, inlier 0.5 | .3720 | .5302 | .5772 | +0.0011 +/- 0.0006 | .538 |
| RANSAC, gates 4 / 2 / 1 / 0.5 | .3725 | .5325 | .5788 | +0.0016 +/- 0.0006 | .548 |
| **RANSAC, gates 6 / 3 / 1.5 / 0.75 / 0.5** | **.3730** | **.5326** | **.5789** | **+0.0021 +/- 0.0006** | **.548** |
| + candidate search 8 m / 12 m | .3724 | .5323 | .5788 | +0.0015 +/- 0.0005 | .545 |

The hard fit is worth +0.002 to +0.004 over least squares and a wide first
gate another +0.002: a 6 m gate engages the pairs the search was built
for, and the search on top of it adds nothing. Inlier 0.5 and 1.0 are a
tie; 1.0, the pre-registered value, stays. FreeAlign still holds the dense
bucket at sigma 2 (.548 against .562).

**The consensus floor** (`AgreementConfig.consensus_floor`, arm suffix
`_c<floor>`). The agreement rule answers only when the soft and the exact
fit agree within the tolerance; on the dense pairs at sigma 2 that is
exactly backwards, because the pairs the re-solve fixes are the ones where
the soft estimate was metres off while the exact fit rests on a consensus
of five or more. The rule now also answers when the exact fit's consensus
is at least `floor` hard pairs. Floors 4 and 6 were pre-registered and
swept on val with tolerances 0.5 and 1.0 m
(`B_boxes_only_icpr_cons_val_result.json`):

| val sweep mean, 6 m schedule | AP@0.7 | AP@0.5 | AP@0.3 | vs FreeAlign (paired) | dense sigma 2 | sigma 0 |
|---|---:|---:|---:|---:|---:|---:|
| re-solve, no agreement | .3730 | .5326 | .5789 | +0.0021 +/- 0.0006 | .548 | .4329 |
| agree 1.0 | .3756 | .5326 | .5766 | +0.0049 +/- 0.0003 | .522 | .4139 |
| agree 0.5 | .3678 | .5212 | .5681 | -0.0029 +/- 0.0006 | .482 | .4174 |
| **agree 1.0, floor 4** | **.3803** | **.5389** | **.5821** | **+0.0096 +/- 0.0003** | **.570** | .4140 |
| agree 1.0, floor 6 | .3774 | .5351 | .5792 | +0.0067 +/- 0.0003 | .555 | .4139 |
| agree 0.5, floor 4 | .3787 | .5358 | .5799 | +0.0081 +/- 0.0002 | .563 | .4162 |
| per_pair, agree 1.0, floor 4 | .3788 | .5346 | .5766 | +0.0081 +/- 0.0003 | .570 | .4093 |
| FreeAlign 1.0 m | .3702 | .5160 | .5599 | | .562 | .4144 |

Agreement at 1.0 m with a floor of four is ahead of FreeAlign at every
sigma from 0.2 (by 0.004 at 0.2, 0.011 at 2.0; .4140 against .4144 at 0)
and takes the dense bucket at sigma 2 for the first time (.570 against
.562). Coverage at sigma 2 goes 0.64 -> 0.70 at an unchanged 0.25 m
answered error on the dense pairs: the floor admits exactly the pairs the
plain rule threw away. Selected before test: Wald 0.2, RANSAC re-solve,
inlier 1.0, gates 6 / 3 / 1.5 / 0.75 / 0.5, agreement 1.0 m, floor 4.

**Test** (2,172 frames, 5 paired seeds, every arm above;
`B_boxes_only_icpr_test_result.json`):

| test AP@0.7 | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | sweep mean |
|---|---|---|---|---|---|---|---|---|---|
| abstain 0.2 (soft) | .3932 | .3404 | .3100 | .3032 | .2993 | .2940 | .2797 | .2661 | .3107 |
| abstain 0.2, RANSAC re-solve | **.4057** | .3554 | .3288 | .3265 | .3259 | .3265 | .3274 | .3267 | .3404 |
| abstain 0.2, re-solve, agree 1.0 | .3902 | .3695 | .3478 | .3413 | .3321 | .3278 | .3149 | .3053 | .3411 |
| **abstain 0.2, re-solve, agree 1.0, floor 4** | .3893 | **.3721** | **.3553** | **.3501** | **.3456** | **.3428** | **.3385** | **.3349** | **.3536** |
| abstain 0.2, re-solve, agree 1.0, floor 6 | .3903 | .3701 | .3495 | .3432 | .3353 | .3317 | .3235 | .3185 | .3453 |
| FreeAlign 1.0 m | .3702 | .3577 | .3420 | .3362 | .3331 | .3321 | .3303 | .3301 | .3415 |

| test, paired by seed vs FreeAlign (5 seeds) | AP@0.7 | AP@0.5 | AP@0.3 |
|---|---|---|---|
| abstain 0.2 (soft) | -0.0307 +/- 0.0003 | -0.0074 +/- 0.0005 | +0.0110 +/- 0.0003 |
| re-solve (RANSAC) | -0.0011 +/- 0.0002 | +0.0165 +/- 0.0004 | +0.0252 +/- 0.0003 |
| re-solve, agree 1.0 | -0.0005 +/- 0.0002 | +0.0097 +/- 0.0003 | +0.0153 +/- 0.0003 |
| **re-solve, agree 1.0, floor 4** | **+0.0120 +/- 0.0001** | **+0.0262 +/- 0.0002** | **+0.0284 +/- 0.0003** |
| re-solve, agree 1.0, floor 6 | +0.0037 +/- 0.0002 | +0.0151 +/- 0.0002 | +0.0201 +/- 0.0003 |
| re-solve, agree 0.5, floor 4 | +0.0084 +/- 0.0001 | +0.0213 +/- 0.0003 | +0.0254 +/- 0.0003 |

Sweep means: selected arm .3536 / .5196 / .5671 against FreeAlign .3415 /
.4933 / .5390. The selected arm is ahead at every sigma including 0
(.3893 against .3702) and at every sigma at AP@0.5 (.561 -> .479 against
.528 -> .466) and AP@0.3 (.594 -> .522 against .560 -> .506). The test
gain is larger than val's (+0.012 against +0.010) and its order among the
arms is val's. By bucket (AP@0.7, uncorrected | selected | re-solve only |
FreeAlign | oracle):

| test | sigma | uncorrected | selected | re-solve | FreeAlign | oracle |
|---|---|---|---|---|---|---|
| 3+ shared (1,093 frames) | 0 | .437 | .405 | .425 | .399 | .437 |
| | 0.4 | .223 | .401 | .370 | .396 | .437 |
| | 1.0 | .166 | .397 | .377 | .395 | .437 |
| | 2.0 | .173 | .389 | .380 | **.395** | .437 |
| 1-2 shared (373 frames) | 0 | .426 | .368 | .380 | .376 | .426 |
| | 1.0 | .198 | .252 | .249 | .207 | .426 |
| | 2.0 | .191 | .244 | .241 | .200 | .426 |
| 0 shared (156 frames) | 2.0 | .214 | .212 | .212 | .214 | .327 |

The dense bucket is ours at 0.4 and 1.0 and FreeAlign's by 0.006 at
2.0 (val had it ours at 2.0 by 0.008); the sparse bucket is ours by
0.04-0.05 at every noise level. At sigma 2 the selected arm answers 72 %
of pairs with a 0.93 m mean answered translation error (0.41 m on the
dense pairs) against FreeAlign's 60 % and 8.0 m (6.1 m on the dense
pairs, the tail behind its sharp median).

**Verdict.** The gap to FreeAlign on V2X-Real is closed on test at every
sigma and every IoU threshold, with FreeAlign's message and no training:
+0.012 / +0.026 / +0.028 AP@0.7 / 0.5 / 0.3 on the sweep mean, paired
standard errors 0.0001-0.0003. What did it was a hard consensus fit over
the learned correspondences and a decision rule that trusts that fit's
own evidence.

**Under communication delay** (`B_boxes_only_icpr_delay{1,2,4}_test_result.json`;
constant 1 / 2 / 4 frames = 100 / 200 / 400 ms on the other agent's message,
sigma 0 / 0.4 / 1 / 2, 3 paired seeds, 2,172 frames, same arms and
constants) the selected arm is ahead of FreeAlign at every delay, every
sigma and every threshold, by more than without delay. Sweep means:

| test, sweep mean | oracle | soft (Wald 0.2) | re-solve | selected (agree 1.0, floor 4) | FreeAlign 1.0 m | selected - FreeAlign (paired) |
|---|---:|---:|---:|---:|---:|---:|
| 100 ms, AP@0.7 | .3420 | .2572 | .2880 | **.2963** | .2841 | **+.0118 +/- .0002** |
| 100 ms, AP@0.5 | .5588 | .4390 | .4680 | **.4731** | .4425 | **+.0305 +/- .0002** |
| 100 ms, AP@0.3 | .6075 | .5330 | .5490 | **.5513** | .5126 | **+.0388 +/- .0006** |
| 200 ms, AP@0.7 | .3187 | .2364 | .2655 | **.2728** | .2572 | **+.0153 +/- .0002** |
| 200 ms, AP@0.5 | .4814 | .3786 | .4101 | **.4168** | .3838 | **+.0328 +/- .0003** |
| 200 ms, AP@0.3 | .5673 | .4844 | **.4987** | .4974 | .4544 | **+.0434 +/- .0004** |
| 400 ms, AP@0.7 | .3136 | .2325 | .2578 | **.2702** | .2529 | **+.0168 +/- .0003** |
| 400 ms, AP@0.5 | .4464 | .3507 | .3767 | **.3886** | .3602 | **+.0280 +/- .0003** |
| 400 ms, AP@0.3 | .4881 | .4244 | .4394 | **.4440** | .4045 | **+.0388 +/- .0005** |

Per sigma at AP@0.7 the selected arm leads by 0.019 / 0.013 / 0.011 /
0.005 at 100 ms (sigma 0 / 0.4 / 1 / 2), 0.027 / 0.016 / 0.013 / 0.007 at
200 ms and 0.033 / 0.017 / 0.013 / 0.007 at 400 ms. The dense bucket at
sigma 2 is a tie at 100 ms (.308 against .309) and ours at 200 / 400 ms
(.278 / .278 against .276 / .272); the sparse bucket is ours by 0.02-0.03
at every delay. The re-solve alone already leads FreeAlign at AP@0.7 under
delay (+0.003 / +0.008 / +0.004), where the least-squares re-solve had been a
draw; the consensus floor adds 0.008-0.012 on top.
