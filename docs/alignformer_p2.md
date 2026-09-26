# AlignFormer P2: stage-2 pose training, the boxes-only ablation, and fused AP under localization error

> **Current: the `r140` detector.** The headline numbers in this document are
> from the per-agent detector running at the **full `cav_lidar_range` of
> +/-140.8 m** -- the range every intermediate-fusion baseline uses and the
> range the evaluated `GT_RANGE` (+/-140 m) covers. It replaces an earlier
> detector at OpenCOOD's stock +/-70.4 m, which no agent could report the 9.9%
> of evaluated ground truth beyond |x| = 70.4 m from, and whose measured clean
> recall at IoU 0.7 was therefore capped at 0.894 against the new 0.916.
> The +/-70.4 m results are **kept**, clearly labelled `r70`, because the
> difference between the two ranges is itself a result; see
> [The detector range fix](#the-detector-range-fix-704-m---1408-m). Tables that
> have not been re-run at +/-140.8 m -- the head A / boxes-only / `match_weight
> 0` ablations and the per-epoch curve -- say so where they appear.
>
> Both ranges were measured after the two defects P2 originally surfaced were
> fixed: the heading channel's pi ambiguity (diagnosed in
> [alignformer_pose_floor.md](alignformer_pose_floor.md)) and the
> communication-range mismatch between the pair index and the evaluation
> protocol. The four `r70` configurations -- head A, head B, the boxes-only
> ablation and the `match_weight 0` control -- were retrained from scratch on
> the corrected pair index and re-measured together, so the comparisons among
> them are like-for-like. (P2's fifth configuration, head A / boxes_only, was
> dropped: head A collapses to predict-zero either way, so ablating its message
> content measures nothing.)

Stage 1 established that the two agents' object sets can be put into
correspondence (P1: cross-agent Top-1 0.9961 at 70 m; 0.9983 once the
detector was widened to +/-140.8 m). It also established,
through the association diagnostic, that the *embedding* contributes nothing to
that correspondence on OPV2V -- a finding that still holds, and is re-checked
below at the pose and AP levels. P2 asks the question the method actually
exists for:

> Given a correspondence, can the sender's SE(2) localization error be recovered
> well enough that correcting it before fusion improves mAP under localization
> error?

Two gates are reported here. The **P2 gate** as the plan wrote it -- head B's
yaw MAE strictly below the predict-zero baseline at every sigma, the precise
thing [CoLoca-QuA failed](coloca_qua_baseline.md) -- and, because the gate is a
proxy, the **number the project needs**: fused AP on the OPV2V test split for
vanilla late fusion, AlignFormer-corrected late fusion, and the true-pose
oracle, across a localization-noise sweep.

## Results in one place

| Question | Answer |
|---|---|
| **P2 gate** (head B yaw MAE below predict-zero at every non-zero sigma) | **PASS, 7 of 7 on validation**, definition unmodified, at both detector ranges (6262 pairs out to 70 m). On the **test** split the same criterion still **fails at sigma = 0.2 and 0.4 m** and passes from 0.6 m up -- and at +/-140.8 m the *translation* criterion now fails at 0.2 m too (0.3043 against 0.2797 m), where at +/-70.4 m it passed everywhere. The gate is a validation statement and is not a claim about test; see [below](#p2-gate). |
| Head A vs head B | **Decisive for B, at both ranges.** At `r70` head A's translation MAE tracks predict-zero to within 1-3% at every sigma (2.688 against 2.734 m at sigma = 2) and its **yaw MAE is *worse* than predict-zero** (1.643 against 1.600 deg). At `r140` its translation pulls 14-20% ahead of predict-zero but its **yaw is worse than predict-zero at every sigma** (0.811 against 0.800 at sigma = 1; 1.603 against 1.600 at sigma = 2), and its error still grows almost linearly in the injected noise -- the signature of a head sitting at the conditional mean. Exactly [CoLoca-QuA's failure](coloca_qua_baseline.md) on the same data. Head B is 8.0x better on yaw and 17x on translation at `r70` (8.9x and 16x at `r140`), with an error essentially flat in sigma. |
| Does the embedding help? | **Barely, and the headline "nothing" still stands where it was measured.** On *association* it is still worth nothing: the boxes-only stage 1 reaches Top-1 **0.9976** against boxes+embeddings' 0.9961. On *fused AP* it is worth **+0.000 to +0.021 AP@0.7** (mean +0.011), the same marginal amount measured before the fixes (+0.010 to +0.016). What did change is the *pose* metric, where the gap widened from ~3% to 12-15% and boxes-only would now **fail** the P2 gate at sigma = 0.2 m (0.1662 against 0.1600) where boxes+embeddings passes. See the caveat below: one seed, and separate warm starts. |
| Does anything in the message help? | **Yes: the matching supervision, still more than the embedding.** Dropping `match_nll` (`match_weight 0`) costs head B 24% of its corner loss (0.919 -> 1.138 m) and 9-32% of its yaw accuracy. The value is in learning a correspondence from *geometry*, which the auxiliary loss supervises. |
| **mAP under localization error** (the number the project needs) | **AlignFormer recovers 78-84% of the oracle-vs-vanilla gap at every sigma from 0.4 to 2.0 m**, worth **+0.46 to +0.60 AP@0.7** on the 2170-frame test split, and **+0.18 at sigma = 0.2 m**. At sigma = 0 it still costs **0.055** (it cost 0.070 at +/-70.4 m: the regression shrank by a fifth but did **not** close). |
| **Inverse-variance correspondence weighting** (deployed, task 19) | **Better pose everywhere, better AP at every non-zero sigma, 0.011 worse at sigma = 0.** Weighting each correspondence by its fitted inverse variance cuts translation MAE 13-16% and yaw MAE 12-14% at every sigma, lifts AP@0.7 by +0.003 to +0.016 from sigma = 0.2 up, and costs 0.0105 at sigma = 0. It also **fixes the test-split translation gate failure at sigma = 0.2 m** (0.2599 against predict-zero's 0.2797, where the unweighted fit gave 0.3043). The variance is driven by **detection score, not range** -- see [below](#the-correspondence-variance-model). |
| **Head-to-head against FreeAlign** (the closest published competitor, reimplemented onto our late fusion) | **Not ordered: FreeAlign wins AP@0.7 at every sigma (by 0.015 to 0.049), AlignFormer wins AP@0.3 from 0.2 m up (by up to 0.020) and AP@0.5 through the middle of the sweep.** FreeAlign's pose error has a slightly BETTER median (0.109 m against 0.116 m) and a 5.7x worse mean, because 2.55% of its corrections are wrong by more than 3 m against AlignFormer's 0.93%. On the 8.7% of pairs sharing one or two objects -- where a relative-distance graph is structurally degenerate -- FreeAlign declines 84% and gains +0.011 AP@0.7 over doing nothing while AlignFormer answers 93% and gains +0.184. The row is a **reimplementation**, never the authors' code. See [below](#freealign-the-boxes-only-competitor-reimplemented-onto-our-late-fusion). |
| **Head-to-head against intermediate fusion** (the claim the project exists to make) | **Ties the strongest baseline on the clean row, loses one row at 0.2 m, and leads the whole field from 0.4 m, at 1,021x to 4,254x fewer bytes.** At sigma = 0 AlignFormer is above V2X-ViT, Where2comm and F-Cooper and below CoAlign, CoBEVT, AttFuse and V2VAM; at 0.2 m six of seven are ahead of it; from 0.4 m it leads every one, by +0.030 to +0.040 over V2X-ViT and +0.16 to +0.33 over the rest. All seven were run locally under this sweep and this evaluator -- no published number is quoted. See [below](#head-to-head-against-intermediate-fusion). |

### The headline

On the official 2170-frame OPV2V test split, global-sorted AP@0.7, with the
detector at **+/-140.8 m** (`r140`):

| sigma (m) | Vanilla late fusion | AlignFormer | Oracle (true pose) | Gain | Gap recovered |
|---|---:|---:|---:|---:|---:|
| 0 | **0.8964** | 0.8415 | 0.8964 | -0.0549 | -- |
| 0.2 | 0.5978 | **0.7743** | 0.8964 | **+0.1765** | 59.1% |
| 0.4 | 0.3158 | **0.7717** | 0.8964 | **+0.4559** | 78.5% |
| 0.6 | 0.2154 | **0.7717** | 0.8964 | **+0.5564** | 81.7% |
| 0.8 | 0.1844 | **0.7804** | 0.8964 | **+0.5961** | 83.7% |
| 1.0 | 0.1763 | **0.7771** | 0.8964 | **+0.6008** | 83.4% |
| 1.5 | 0.1799 | **0.7728** | 0.8964 | **+0.5928** | 82.7% |
| 2.0 | 0.1898 | **0.7441** | 0.8964 | **+0.5542** | 78.4% |

The table above is the **unweighted** Procrustes fit, kept as the reference
configuration. The **deployed** configuration additionally weights each
correspondence by its fitted inverse variance (task 19). Both are published,
because neither dominates:

| sigma (m) | Vanilla late fusion | AlignFormer (unweighted) | **AlignFormer (IVW, deployed)** | IVW is worth | Oracle |
|---|---:|---:|---:|---:|---:|
| 0 | **0.8964** | 0.8415 | 0.8310 | **-0.0105** | 0.8964 |
| 0.2 | 0.5978 | 0.7743 | **0.7841** | +0.0098 | 0.8964 |
| 0.4 | 0.3158 | 0.7717 | **0.7835** | +0.0118 | 0.8964 |
| 0.6 | 0.2154 | 0.7717 | **0.7819** | +0.0102 | 0.8964 |
| 0.8 | 0.1844 | 0.7804 | **0.7850** | +0.0046 | 0.8964 |
| 1.0 | 0.1763 | 0.7771 | **0.7851** | +0.0080 | 0.8964 |
| 1.5 | 0.1799 | 0.7728 | **0.7756** | +0.0028 | 0.8964 |
| 2.0 | 0.1898 | 0.7441 | **0.7600** | +0.0159 | 0.8964 |

The pose metric it was designed to improve moves further than the AP does:

| sigma (m) | translation MAE (unweighted -> IVW) | yaw MAE (unweighted -> IVW) | predict-zero t / yaw |
|---|---:|---:|---:|
| 0 | 0.1856 -> **0.1610 m** (-13%) | 0.3440 -> **0.2962 deg** (-14%) | 0 / 0 |
| 0.2 | 0.3043 -> **0.2599 m** (-15%) | 0.4308 -> **0.3772 deg** (-12%) | 0.2797 / 0.1610 |
| 0.4 | 0.3045 -> **0.2562 m** (-16%) | 0.4550 -> **0.3803 deg** (-16%) | 0.5450 / 0.3241 |
| 0.6 | 0.3063 -> **0.2626 m** (-14%) | 0.4550 -> **0.3844 deg** (-16%) | 0.8140 / 0.4865 |
| 0.8 | 0.3161 -> **0.2675 m** (-15%) | 0.4777 -> **0.3948 deg** (-17%) | 1.0904 / 0.6308 |
| 1.0 | 0.3216 -> **0.2649 m** (-18%) | 0.4825 -> **0.3979 deg** (-18%) | 1.3459 / 0.7892 |
| 1.5 | 0.3408 -> **0.2988 m** (-12%) | 0.5099 -> **0.4324 deg** (-15%) | 2.0195 / 1.1962 |
| 2.0 | 0.3628 -> **0.3291 m** (-9%) | 0.5114 -> **0.4500 deg** (-12%) | 2.6933 / 1.6053 |

Two consequences worth stating plainly. IVW **closes the test-split translation
gate failure at sigma = 0.2 m**: 0.2599 m now beats predict-zero's 0.2797 m,
where the unweighted fit's 0.3043 m did not. And it **does not close the yaw
failure** at sigma = 0.2 and 0.4 m (0.3772 against 0.1610, 0.3803 against
0.3241), which remains the one criterion that passes on validation and fails on
test.

The `sigma = 0` cost is the same trade the whole method makes, one notch
sharper: a better-conditioned estimator moves boxes more confidently, which is
exactly wrong when there was nothing to correct. Turning shrinkage off makes
this explicit -- IVW without shrinkage scores 0.7978 at sigma = 0 (worse) but
0.7941 at sigma = 0.2 (better than either shrunk configuration).

And the same table at the superseded **+/-70.4 m** range (`r70`), kept because
the difference between the two is a result in its own right:

| sigma (m) | Vanilla late fusion | AlignFormer | Oracle (true pose) | Gain | Gap recovered |
|---|---:|---:|---:|---:|---:|
| 0 | **0.8764** | 0.8068 | 0.8764 | -0.0696 | -- |
| 0.2 | 0.5846 | **0.7488** | 0.8764 | **+0.1641** | 56.3% |
| 0.4 | 0.3069 | **0.7410** | 0.8764 | **+0.4341** | 76.2% |
| 0.6 | 0.2106 | **0.7412** | 0.8764 | **+0.5306** | 79.7% |
| 0.8 | 0.1785 | **0.7404** | 0.8764 | **+0.5619** | 80.5% |
| 1.0 | 0.1690 | **0.7386** | 0.8764 | **+0.5695** | 80.5% |
| 1.5 | 0.1737 | **0.7326** | 0.8764 | **+0.5588** | 79.5% |
| 2.0 | 0.1831 | **0.7046** | 0.8764 | **+0.5215** | 75.2% |

Both tables are on **the same 33,089 ground-truth boxes**: OpenCOOD filters
test-time ground truth by the fixed `GT_RANGE`, not by `cav_lidar_range`
(`base_postprocessor.generate_object_center` uses `anchor_args` only when
`train=True`), so the two ranges differ in what the detector can *find*, never
in what it is *scored against*. Verified frame by frame on 120 test frames:
zero frames where the two ground-truth sets differ.

Three things are visible at a glance.

**Vanilla late fusion collapses under localization error** -- 0.8964 -> 0.1763
AP@0.7 by sigma = 1 m, an 80% relative loss. That collapse is the problem
AlignFormer exists to solve, and it is severe.

**AlignFormer is almost flat in sigma** -- 0.8415 at sigma = 0 down to 0.7441 at
sigma = 2 m. That flatness is the closed-form solver working: it removes
essentially all of the *injected* error, leaving a residual set by the detector
rather than by the noise. It is also why the method wins by more as conditions
get worse, which is the right direction for a robustness method.

**The remaining cost is at sigma = 0 only, and it is now 0.055** (0.070 at
+/-70.4 m). Where
localization is already perfect, moving boxes by an imperfect estimate can only
lose AP, and shrinkage suppresses the correction entirely on 74% of test pairs
there rather than on all of them. The stated target -- AP@0.7 at or above
uncorrected at **every** sigma including 0 -- is therefore still not met at
sigma = 0, and that is the one row that fails it. Widening the detector shrank
that regression by a fifth; it did not close it, and nothing in this document
claims it did. At every other sigma the target is met by a wide margin.

### How the four fixes accumulate

Same evaluator, same 2170-frame test split, AP@0.7. The first four columns are
at +/-70.4 m; the last is the widened detector, whose uncorrected and oracle
rows move too (shown in the `r140 oracle` column).

| sigma (m) | r70 uncorrected | P2 as first measured | + heading fold & shrinkage | + 70 m pair index | **+ 140.8 m detector** | r140 oracle |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.8764 | 0.5424 | 0.7665 | 0.8068 | **0.8415** | 0.8964 |
| 0.2 | 0.5846 | 0.5419 | 0.7096 | 0.7488 | **0.7743** | 0.8964 |
| 0.4 | 0.3069 | 0.5398 | 0.6963 | 0.7410 | **0.7717** | 0.8964 |
| 0.6 | 0.2106 | 0.5368 | 0.6948 | 0.7412 | **0.7717** | 0.8964 |
| 0.8 | 0.1785 | 0.5320 | 0.6971 | 0.7404 | **0.7804** | 0.8964 |
| 1.0 | 0.1690 | 0.5303 | 0.6957 | 0.7386 | **0.7771** | 0.8964 |
| 1.5 | 0.1737 | 0.5198 | 0.6845 | 0.7326 | **0.7728** | 0.8964 |
| 2.0 | 0.1831 | 0.4961 | 0.6707 | 0.7046 | **0.7441** | 0.8964 |

### What shrinkage is still buying

Same checkpoint, same sweep, `--shrinkage` on and off, AP@0.7. **This table is
`r70` and has not been re-run at +/-140.8 m**; `tau` was recalibrated for the
widened detector (0.1806 m / 0.2670 deg against 0.1590 m / 0.2468 deg) and the
sigma = 0 exact-fallback rate rose from 66.1% to 74.2% accordingly, so the
trade below is expected to be larger, not smaller, at the wider range.

| sigma (m) | Uncorrected | Shrinkage off | **Shrinkage on** | Shrinkage buys |
|---|---:|---:|---:|---:|
| 0 | 0.8764 | 0.7569 | **0.8068** | +0.0500 |
| 0.2 | 0.5846 | 0.7512 | **0.7488** | -0.0025 |
| 0.4 | 0.3069 | 0.7489 | **0.7410** | -0.0079 |
| 0.6 | 0.2106 | 0.7461 | **0.7412** | -0.0049 |
| 0.8 | 0.1785 | 0.7450 | **0.7404** | -0.0046 |
| 1 | 0.1690 | 0.7401 | **0.7386** | -0.0015 |
| 1.5 | 0.1737 | 0.7360 | **0.7326** | -0.0034 |
| 2 | 0.1831 | 0.7060 | **0.7046** | -0.0013 |

Unchanged in character from before the pair-index fix: shrinkage buys the clean
case -- **+0.050 at sigma = 0** -- for 0.001 to 0.008 everywhere else. That is
the trade it exists to make, and it is what keeps the sigma = 0 cost at 0.070
instead of 0.120.

The heading fold is the large one (+0.167 to +0.177 everywhere). Matching the
pair index to the evaluation protocol is the second (+0.034 to +0.048), and it
is the only one that helps the clean case materially (+0.040 at sigma = 0).
Widening the detector is the third (+0.026 to +0.040 on AlignFormer's own rows,
and +0.020 on the oracle it is measured against). None of the four is a
modelling change: every one is a defect in what the model was shown or in what
the detector was allowed to see.

## The detector range fix (+/-70.4 m -> +/-140.8 m)

### What was wrong

`configs/alignformer_detector.yaml` was a copy of OpenCOOD's stock
`point_pillar_late_fusion.yaml`, whose `cav_lidar_range` is
`[-70.4, -40, -3, 70.4, 40, 1]`. Every intermediate-fusion baseline in the
head-to-head below runs at `[-140.8, -40, -3, 140.8, 40, 1]`, and the evaluated
`GT_RANGE` is `[-140, -40, -3, 140, 40, 1]` for all of them. **9.6% of the
evaluated ground truth lies beyond |x| = 70.4 m** (176 of 1824 boxes over 120
sampled test frames; task 17 measured 10.0% over a different sample), and the
narrow detector could not produce a box there at all. Its recall was capped
near 90% in every published row.

The consequence was a recall cap, and it was partial rather than total: late
fusion merges several agents, so a box 90 m from the ego can be 30 m from a CAV
and still be found. Measured, the +/-70.4 m detector's clean recall at IoU 0.7
was **0.894** over the whole test split, against **0.916** for the widened one
-- see [Where the AP came from](#where-the-ap-came-from-and-what-it-cost).

`configs/alignformer_detector_r140.yaml` is the same detector at the wider
range. Only `cav_lidar_range` and the output paths differ --
`tests/test_alignformer_range.py` asserts that the optimizer, scheduler,
augmentation, anchors, NMS and the **voxel budget** are identical. The voxel
budget needed no change, which is worth stating because it is the opposite of
what the doubled grid suggests: over 400 randomly sampled OPV2V train
agent-frames the occupied-pillar count is 7,481 mean / 11,885 max at
+/-140.8 m against 7,235 / 11,535 at +/-70.4 m, both far below
`max_voxel_train = 16000`. OPV2V's LiDAR simply returns few points past 70 m.
The extra cost is the BEV grid (704 x 200 cells against 352 x 200): 2h19m to
train against 2h02m.

The ROI cache **had** to be rebuilt rather than inherited.
`embedding.rotated_roi_align` maps each box's world coordinates into normalized
BEV grid coordinates through `lidar_range`, so the same box pools different
features at the two ranges (pinned by
`test_roi_features_depend_on_the_lidar_range`). The new cache is 5.12 GB at
`/media/chenyi/basement2/cache/alignformer_r140`; the +/-70.4 m one is kept.
Reproducibility was re-verified the way it was before -- six cached frames
re-derived live through `cache._cache_one_frame` and compared with
`np.array_equal` on boxes, scores, gt_ids and ROI: all identical. The pair
index is pose-derived and range-independent; the 70 m index was reused
unchanged, and rebuilding it from scratch reproduces it element for element
(35,298 train pairs, 10,790 test).

### The P0 clean gate

Clean late fusion, 2170-frame official test split, global-sorted, on the same
33,089 ground-truth boxes:

| Metric | r70 (+/-70.4 m) | **r140 (+/-140.8 m)** | Delta |
|---|---:|---:|---:|
| AP@0.3 | 0.9284 | **0.9552** | +0.0268 |
| AP@0.5 | 0.9251 | **0.9502** | +0.0251 |
| **AP@0.7** | **0.8764** | **0.8964** | **+0.0200** |

**0.8964 is the new oracle and the new `uncorrected` sigma = 0 row**, and it
replaces 0.8764 as the number the project's late-fusion ceiling is quoted
against.

The `frame_order` variant moves the other way (AP@0.7 0.8180 -> 0.7325). That
is not the reported metric -- `fusion.average_precision` documents per-frame
accumulation as silently inflating AP, and every number in this project is
`global_sorted` -- but the direction is informative: the wide detector emits
more low-confidence far-field detections, which cost per-frame precision on
frames with few ground-truth boxes while helping the dataset-level curve.

### Where the AP came from, and what it cost

**The gain is a recall gain, and it is concentrated exactly where the handicap
was.** Clean late fusion, per-ground-truth-box recall at IoU 0.7, over all
33,089 boxes of the test split:

| ego-frame \|x\| (m) | GT boxes | r70 recall | **r140 recall** | recovered |
|---|---:|---:|---:|---:|
| 0 - 35 | 19,189 | 0.966 | **0.969** | +51 boxes |
| 35 - 70.4 | 10,619 | 0.889 | **0.895** | +68 |
| 70.4 - 105 | 2,737 | 0.541 | **0.721** | **+493** |
| 105 - 140 | 544 | 0.243 | **0.421** | **+97** |
| **all** | **33,089** | **0.894** | **0.916** | **+709** |

Overall recall goes 0.894 -> 0.916, which is the +0.020 AP@0.7 almost exactly,
and **83% of the recovered boxes are beyond |x| = 70.4 m**.

Two things in that table correct earlier statements in this project.

**The narrow detector was not blind past 70.4 m -- it was partially sighted.**
Task 17 described the 10% of ground truth beyond |x| = 70.4 m as "invisible to
our detector". It is not: late fusion merges several agents, and a box 90 m from
the ego can be 30 m from a CAV, so the +/-70.4 m detector still recalled **54%**
of the 70.4-105 m bucket and **24%** of the 105-140 m one. The recall ceiling it
imposed was about 0.894, not 0.90 exactly, and the headroom the range fix could
possibly recover was 3,281 boxes rather than all of them. That the fix recovered
590 of them -- 18% of the far-field population, not 100% -- is the honest size of
the effect.

**The far field is genuinely harder, not merely out of range.** Even at
+/-140.8 m, recall is 0.721 and 0.421 in the two far buckets against 0.969 near
the ego. OPV2V's LiDAR returns few points there (which is also why the pillar
count barely moved), and low-point-density objects are hard to detect and, as
the next table shows, hard to localize.

The cost lands on the pose estimate. Cross-agent detection disagreement per
correspondence, measured with the **true** pose (so it is detector disagreement,
not pose error), over all 6262 validation pairs:

| ego-frame \|x\| (m) | r70 correspondences | r70 mean \|dx\|,\|dy\| (m) | r140 correspondences | r140 mean \|dx\|,\|dy\| (m) | r140 mean \|dyaw\| (deg) |
|---|---:|---:|---:|---:|---:|
| 0 - 35 | 46,550 | 0.151 | 46,901 | 0.154 | 3.69 |
| 35 - 70.4 | 14,940 | 0.175 | 15,721 | 0.179 | 4.49 |
| **70.4 - 140.8** | **0** | -- | **1,366** | **0.230** | **6.65** |
| all | 61,490 | 0.157 | 63,988 | 0.162 | -- |

The far-field correspondences the wide detector adds are **52% noisier in
translation and 80% noisier in heading** than the near-field ones. They are
only 2.1% of the population, but they enter the Procrustes fit with the same
weight as everything else, and they raise the mean per-correspondence
disagreement from 0.157 m to 0.162 m.

**That is measurable in the pose metric, and it is a regression.** On the test
split, AlignFormer's translation MAE at sigma = 0.2 m goes 0.2509 -> 0.3043 m
and its yaw MAE 0.3605 -> 0.4308 deg; on validation, translation MAE at
sigma = 1 m goes 0.1288 -> 0.1373 m (yaw improves there, 0.1726 -> 0.1632).
**Pose estimation got worse and fused AP got better**, because IoU is far more
sensitive to a missing box than to a few centimetres of residual centre error,
and the recall the wider range buys is worth more than the precision the
far-field correspondences cost. This is reported rather than averaged away, and
nothing was tuned to recover it.

## Reproducing

### The +/-140.8 m pipeline (`r140`, the headline numbers)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
O=outputs/alignformer/r140
DET=configs/alignformer_detector_r140.yaml
AF=configs/alignformer_r140.yaml
TEST=/media/chenyi/Elements1/Dataset/OPV2V/test

# The per-agent detector, 15 epochs, 2h19m on an RTX 4090. --split-root points
# the train/val symlink split at the ext4 disk: the repository's own volume is
# NTFS and no longer round-trips symlinks.
python -u scripts/train_late_fusion.py --hypes_yaml $DET \
  --split-root /media/chenyi/basement2/cache/opv2v_splits \
  --output-dir outputs/alignformer/point_pillar_late_fusion_r140

# P0 clean gate (~22 min)
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config $DET --split $TEST --metric ap --output $O/p0_r140_gate.json

# The ROI cache, rebuilt because rotated_roi_align is range-dependent
# (28.6 min train + 10.8 min test, 5.12 GB)
python -m embedding_aware_belt_fusion.alignformer.cache --config $DET \
  --splits /media/chenyi/Elements1/Dataset/OPV2V/{train,test} \
  --cache-root /media/chenyi/basement2/cache/alignformer_r140

# Stage 1, stage 2 head B, shrinkage recalibration, the P2 gate, the sweep
python -m embedding_aware_belt_fusion.alignformer.train \
  --config $AF --stage 1 --output-dir $O/stage1
python -m embedding_aware_belt_fusion.alignformer.train \
  --config $AF --stage 2 --head B --message-content boxes+embeddings \
  --stage1-checkpoint $O/stage1/best.pth \
  --output-dir $O/stage2_B_boxes+embeddings

E="python -m embedding_aware_belt_fusion.alignformer.evaluate"
$E --config $AF --metric shrinkage \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
  --output $O/shrinkage_B_calibration_result.json
$E --config $AF --metric pose \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --seeds 3 \
  --shrinkage $O/shrinkage_B_calibration_result.json --output $O/p2_r140_gate.json
$E --config $DET --split $TEST --metric noisy_ap --alignformer-config $AF \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
  --shrinkage $O/shrinkage_B_calibration_result.json \
  --output $O/p2_r140_noisy_ap_result.json

# Bytes per frame per agent, re-measured: the wider detector transmits more
# objects (15.69 per agent against 13.06), so AlignFormer's own side moves.
python -m embedding_aware_belt_fusion.alignformer.bandwidth --frames 100 \
  --alignformer-cache /media/chenyi/basement2/cache/alignformer_r140/test \
  --output $O/bandwidth_r140_result.json

python scripts/summarize_alignformer_p2.py \
  --pose $O/p2_r140_gate.json --noisy-ap $O/p2_r140_noisy_ap_result.json \
  --history $O/stage2_B_boxes+embeddings/history.json
```

### The +/-70.4 m pipeline (`r70`, the ablations and the range comparison)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
O=outputs/alignformer/r70            # "r70": the 70 m pair index

# Stage 1, twice: the boxes-only ablation must warm-start from a trunk that
# never saw an embedding (~15 min each on an RTX 4090).
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 1 --output-dir $O/stage1
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 1 --zero-embeddings \
  --output-dir $O/stage1_zero_embeddings

# Four stage-2 configurations (~22 min each alone; they are independent, so
# running all four at once takes ~35 min total on one 4090).
T="python -m embedding_aware_belt_fusion.alignformer.train \
     --config configs/alignformer.yaml --stage 2"
$T --head B --message-content boxes+embeddings \
   --stage1-checkpoint $O/stage1/best.pth \
   --output-dir $O/stage2_B_boxes+embeddings
$T --head B --message-content boxes_only \
   --stage1-checkpoint $O/stage1_zero_embeddings/best.pth \
   --output-dir $O/stage2_B_boxes_only
$T --head A --message-content boxes+embeddings \
   --stage1-checkpoint $O/stage1/best.pth \
   --output-dir $O/stage2_A_boxes+embeddings
# Fairness control: head B's architecture without head B's extra supervision.
$T --head B --message-content boxes+embeddings --match-weight 0 \
   --stage1-checkpoint $O/stage1/best.pth \
   --output-dir $O/stage2_B_boxes+embeddings_nomatch

E="python -m embedding_aware_belt_fusion.alignformer.evaluate"

# Shrinkage: validation only, sigma = 0 only, one calibration per DEPLOYED
# configuration (one tau cannot describe two different estimators).
$E --config configs/alignformer.yaml --metric shrinkage \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth --output $O/shrinkage_B_calibration_result.json
$E --config configs/alignformer.yaml --metric shrinkage \
  --checkpoint $O/stage2_B_boxes_only/best.pth --output $O/shrinkage_B_boxes_only_calibration_result.json

# The P2 gate, definition unmodified, on the deployed configuration
$E --config configs/alignformer.yaml --metric pose \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --seeds 3 \
  --shrinkage $O/shrinkage_B_calibration_result.json --output $O/p2_r70_gate.json

# The ablation table: every configuration, paired noise draws, NO shrinkage
$E --config configs/alignformer.yaml --metric pose \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
               $O/stage2_B_boxes_only/best.pth \
               $O/stage2_A_boxes+embeddings/best.pth \
               $O/stage2_B_boxes+embeddings_nomatch/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --seeds 3 \
  --output $O/p2_r70_ablations_result.json

# Fused AP under localization error, official 2170-frame test split (~15 min)
$E --config configs/alignformer_detector.yaml \
  --split /media/chenyi/Elements1/Dataset/OPV2V/test \
  --metric noisy_ap --alignformer-config configs/alignformer.yaml \
  --checkpoint $O/stage2_B_boxes+embeddings/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
  --shrinkage $O/shrinkage_B_calibration_result.json --output $O/p2_r70_noisy_ap_result.json
# ... and the same with stage2_B_boxes_only + shrinkage_B_boxes_only_calibration_result.json for the
# ablation, and without --shrinkage for the shrinkage-off condition.

# Every table below is rendered from those JSONs
python scripts/summarize_alignformer_p2.py \
  --pose $O/p2_r70_gate.json --noisy-ap $O/p2_r70_noisy_ap_result.json \
  --history $O/stage2_B_boxes+embeddings/history.json
```

## Method

### What stage 2 trains

The embedding head is **frozen** at its stage-1 weights and kept in eval mode;
stage 2 trains the pose head and the shared trunk on

```
corner_loss + match_weight * match_nll
```

`corner_loss` (`losses.py`) is the mean L1 displacement of the CAV boxes' four
BEV corners between the predicted and the true correction. It is in metres
throughout -- unlike CoLoca-QuA's weighted `(2, 2, 1)` MSE on `(dx, dy, dpsi)`,
which mixes metres and radians under arbitrary weights -- it weights a yaw error
by how far away the object is, and it optimizes exactly what IoU and AP measure.

The noise curriculum ramps sigma linearly from 0 to `train.max_xy_std = 2.0` m
over 30 epochs (yaw noise in degrees equals xy noise in metres, this
repository's existing convention). Validation is pinned at the curriculum
maximum with deterministic per-sample noise, so the per-epoch curve is
interpretable rather than reporting the easiest condition at epoch 0.

### Warm start, and why the ablation gets its own

Each configuration warm-starts from a stage-1 checkpoint. Head B is the same
module stage 1 trained, so it loads whole; head A shares only the trunk, so it
inherits the trunk and starts its regression MLP fresh. That makes an A-vs-B
comparison a comparison of *heads* rather than of how much trunk training each
one happened to get.

`boxes_only` warm-starts from `stage1_zero_embeddings/best.pth`, **not** from
the full stage-1 run. Warm-starting the ablation from a trunk that was fitted
*with* embeddings would leak the embedding into the condition that is supposed
to be without it.

That correctness has a cost worth naming, because it is the main confound on
the embedding ablation: the two stage-1 runs are separate fits and selected
their best checkpoints at different epochs (14 at Top-1 0.9961 for the full
run, 9 at 0.9976 for the zero-embedding one). So the boxes_only stage-2 run
differs from boxes+embeddings in *two* ways -- no embedding, and a different
warm start -- and only the first is the thing under test. Repeating both over
several seeds is the fix, and is not done here.

### The ablation is a message content, not a second model

`boxes_only` runs the identical forward pass with `train.zero_embeddings`
applied to the enriched batch, so nothing but the embedding can differ between
the two conditions. `zero_embeddings` returns a **new** dict and leaves geometry
untouched; a second code path for the ablation is how the two conditions drift
apart without anyone noticing.

### Checkpoint selection

Best-by **validation corner loss** -- the training objective, measured on
held-out scenarios. Deliberately *not* by yaw MAE: yaw MAE is the P2 gate, and
selecting on the gate would make the gate a statement about checkpoint selection
rather than about the method.

The consequence is visible in the results and is reported rather than tuned
away: validation is pinned at sigma = 2 m, so selection favours the high-noise
end of the curriculum.

### Gradient finiteness is asserted every step

Stage 1 drove head B's match temperature to `tau = 0.038` (best checkpoint;
0.033 at its final epoch), below the ~0.043 at which
[design spec 3.4](superpowers/specs/2026-09-17-alignformer-design.md) places the
`atan2(0, 0)` gradient singularity in `AlignFormerB._soft_correspondence` as
*reachable*. `match_nll` never backpropagated through that branch. `corner_loss`
does, and stage 2 starts from exactly that checkpoint.

A finite loss is not evidence of a healthy step -- an earlier bug in this
project produced perfectly finite losses while poisoning every parameter with
NaN gradients. `stage2._assert_finite_gradients` therefore checks the
**gradients**, on every step, before `optimizer.step`. It does so via the total
norm `clip_grad_norm_` returns: that total is non-finite if and only if some
gradient is, so one device synchronization covers every parameter, which is what
makes a per-step check affordable. On failure it names the offending tensors and
reports the current temperature rather than clamping past the problem.

**Result: no non-finite gradient occurred in any of the six runs**, and the
check is now doing more work than it was. On the 70 m index `tau` does *not*
stay put: it falls from 0.029 to **0.021** over head B's 30 epochs, and to
**0.0106** in the `match_weight 0` control -- half and a quarter of the ~0.043
the spec placed the singularity at, where the original runs sat at 0.037. The
guard was never triggered even so. The spec's reachability analysis is
therefore more stressed than before and still holds, but the margin it was
relying on is gone, and this check should not be removed.

### The fused-AP sweep

`noisy_fusion.py` fuses three conditions from **one** set of detections per
frame, so nothing but the correction differs between them:

| Condition | Correction applied to each CAV's boxes |
|---|---|
| `oracle` | the true relative pose. Sigma-independent; the ceiling. Identical to the P0 pipeline. |
| `uncorrected` | the *noisy* relative pose -- what plain late fusion does when the pose it is given is wrong. The floor. |
| `alignformer` | the noisy relative pose, then AlignFormer's estimated residual SE(2) on top. |

The detector runs once per agent per frame and every condition and sigma reuses
those detections; without that the sweep would be a detector benchmark. The
object sets AlignFormer scores are built exactly as `alignformer.dataset` builds
them for training -- top-64 by score, ROI features sampled in each agent's own
frame and round-tripped through float16 as the cache stores them, CAV boxes
projected into the ego frame with the noisy pose -- because a mismatch there
would be a train/test skew that looks like a modelling result.

**Consistency check, and it holds exactly:** at sigma = 0 the noisy transform
*is* the true transform, so `uncorrected_sigma_0m` must equal `oracle` to the
last digit. It does, at all three IoU thresholds.

AP is reported global-sorted over the whole 2170-frame official test split, from
the same `fusion.average_precision` the P0 gate used.

## Findings that are not the gate

### The low-overlap fallback fires, but its threshold is half what it documents, and the pairs it misses are corrupted

The brief flagged a structural risk: ego-CAV pairs that share no object at all
cannot be aligned, and applying a garbage SE(2) to them would actively corrupt a
share of every fused frame. `procrustes.MIN_MATCH_MASS = 1.0` is the guard meant
to make those fall back to the uncorrected relative pose. Four measurements,
each of which changes what that risk means.

> **Population note.** Findings 1, 2 and 4 below were counted on the **40 m**
> pair index and have not been recounted on the 70 m one; only finding 5, on
> the test split, is post-fix. The widened index admits further-apart pairs
> that share fewer objects (4.60 matched objects per pair beyond 40 m in the
> train split, against 8.84 within it), so the unalignable *rate* is certainly
> higher now than the 4.55% below. The finding these support -- that the guard
> fires where it should and that the ~6% slipping through are badly corrupted
> -- is about the guard's mechanism, not about the rate, and is unchanged. The
> recount is carried as an open item.


**1. The 18.5% figure does not reproduce.** Counted over the actual object sets
stage 2 trains on, pairs sharing no *detected* object are **4.55%** of the
18,496 training pairs and **0 of the 4,176** scenario-disjoint validation pairs.
The claim appeared only in a docstring in `metrics.py` with no supporting
measurement anywhere in the repository; both docstrings are corrected. The
test-split rate is in the sweep's pose table below.

**2. The guard does fire, and is well-calibrated.** The correction is the exact
identity for **94.1% of the unalignable pairs** and for only **0.27% of the
alignable ones**. It is neither inert nor trigger-happy. *Caveat on where this
is measured:* the validation split contains no unalignable pairs at all, so
findings 2 and 4 are measured on the 18,496 **training** pairs at sigma = 2 m --
scenarios the model fitted. They therefore describe the guard's behaviour, not
its generalization. The held-out equivalent is the test-split pose table below,
which counts unalignable pairs and fallbacks on data nothing in the pipeline has
seen.

**3. Its effective threshold is half its documented one.** `AlignFormerB` hands
`weighted_se2_kabsch` the *heading-augmented* weight vector `cat([mass, mass])`,
whose total is twice the `PoseEstimate.confidence` reported alongside it. The
solver gates on that augmented total, so the floor actually enforced on
`confidence` is `MIN_MATCH_MASS / 2` -- **half** an effective matched object,
not the "one effective matched object" the constant's docstring states. A pair
with confidence in `[0.5, 1.0)` looks suppressed by the constant and is
nevertheless corrected. Pinned in
`tests/test_alignformer_procrustes.py::test_the_heading_augmentation_halves_the_effective_match_mass_floor`,
and the reason every fallback statistic here is read off the emitted `(psi, t)`
(`stage2.is_fallback`) rather than inferred from the constant.

**4. The ~6% of unalignable pairs that slip through are corrupted badly.** On
the unalignable subset the model's yaw MAE is **2.525 deg against a predict-zero
1.544**, and its corner error **15.50 m against 13.52**. Since 94.1% of that
subset is the exact identity and therefore scores exactly predict-zero, the
remaining 5.9% must average roughly **18 deg of yaw error and ~46 m of corner
displacement**. That is the corruption the brief anticipated, now quantified:
severe per pair, but confined to about **0.27% of all pairs** (5.9% of 4.55%).

**5. On the held-out test split the problem is small.** The fused-AP sweep counts
**37 unalignable pairs out of 3,445 (1.07%)** and a fallback rate of 1.2-1.5%,
i.e. the guard fires on essentially exactly the unalignable set and almost
nothing else. Whatever corruption remains is confined to about 1% of pairs on
the data the AP numbers are computed from, so it is not what limits the AP
result.

The concrete consequence: raising the gate to its documented intent -- i.e.
requiring `confidence >= MIN_MATCH_MASS` rather than `>= MIN_MATCH_MASS / 2` --
would catch more of the 5.9% that slip through. That change is **not** made
here. It alters trained behaviour and would require re-running every
configuration, and this task's job is to measure, not to tune; it is recorded as
the first thing to try in P3.

### The P2 gate passes, and the floor that used to fail it is gone

P2 as first measured had head B's yaw MAE **flat at 0.2995-0.3010 deg from
sigma = 0 to 1.0 m** -- a sigma-independent floor larger than the error being
corrected below sigma = 0.38 m, so predicting zero won there and the gate
failed at sigma = 0.2 m. Design spec section 8, Risk 4 had anticipated exactly
that.

Two defects produced that floor, and both were in what the model was shown
rather than in the model:

1. **The heading channel took a direction the detector never estimates.**
   `configs/alignformer_detector.yaml` declares no `dir_args` head, so 20.3% of
   cross-agent detections of the same object disagree by ~180 deg, and each one
   displaced a heading virtual point by `2 * heading_lambda = 4 m`. Folded in
   `procrustes.heading_orientation`; full diagnosis in
   [alignformer_pose_floor.md](alignformer_pose_floor.md).
2. **The pair index was built at the wrong communication range.** It used
   `comm_range_m = 40` (CoLoca-QuA's paper value) while OpenCOOD's
   `LateFusionDataset` -- the dataset the AP numbers are computed on -- admits
   every CAV within `COM_RANGE = 70` m. A third of the evaluated pairs were
   therefore out of distribution by construction, and the same filter hid 35.8%
   of the training pairs. The two are now pinned together by a test.

What is left is genuine estimator noise, and it is shrunk at inference by the
positive-part empirical-Bayes rule in `alignformer/shrinkage.py`, calibrated
once on validation at sigma = 0 (`tau_t = 0.1590 m`, `tau_yaw = 0.2468 deg`,
6262 pairs) and used unchanged at every sigma and on the test split.

Head B's yaw MAE is **0.1489-0.1942 deg** at +/-140.8 m (0.1473-0.2065 at
+/-70.4 m), strictly below the empirical predict-zero value at all seven
non-zero sigmas at both ranges. **The gate passes, with its definition
unmodified** -- and on a harder population than it was originally posed on,
since the widened index adds 2086 further-apart validation pairs that share
fewer objects (7.61 matched objects per pair beyond 40 m against 10.92
within it).

### The clean case still costs 0.055 AP@0.7

This is the one target not met. At sigma = 0 the true correction is exactly the
identity, so any correction at all can only lose AP; the method scores 0.8415
against the oracle's 0.8964.

It is much smaller than it was -- P2 measured -0.334, the heading fold and
shrinkage brought it to -0.110, matching the pair index to the protocol brought
it to -0.070, and widening the detector brings it to **-0.055** -- but it is
not zero, and the honest statement is that AlignFormer is still not
unconditionally free to leave on. The widened range shrank the regression by a
fifth; it did not close it.

Shrinkage is what keeps it this small: at sigma = 0 it suppresses the
correction **entirely** on 74.2% of test pairs (the fallback fraction in the
test-split pose table below is the exact-identity rate), against 24.6% at
sigma = 0.2 and 2% at sigma >= 1. On the pairs it does not fully suppress, the
residual it lets through is 0.1856 m and 0.3440 deg -- enough to drop a box
below the 0.7 IoU threshold that was above it. AP@0.3, which tolerates that
residual, is 0.9357 against an oracle 0.9552, i.e. at the looser threshold the
clean-case cost is only 0.020.

(The paragraphs above are `r140`. The numbers quoted in the sub-sections that
follow -- the embedding ablation, the `match_weight 0` control and the
unalignable-pair counts -- are `r70` and say so.)

Two routes remain and neither is taken here, because both are tuning against
the measurement this document exists to take: gate the correction on an
estimate of the localization error, or train with a loss that pins the identity
at sigma = 0.

### The correspondence variance model

Inverse-variance weighting needs a model of how noisy each correspondence is.
Fitted on 63,988 correspondences over the 6,262 scenario-disjoint validation
pairs at sigma = 0, so the residual is detector noise alone with no injected
pose error in it.

The form was **fixed in advance**: additive, because a correspondence's
disagreement is the difference of two independent detections and its variance
is therefore the sum of theirs. Five candidate forms per channel were fitted
anyway, so that "nothing else fits materially better" is a measurement rather
than an assertion:

| Candidate | Deviance improvement over constant |
|---|---:|
| constant | 0.000 |
| range | 0.020 |
| score_additive (**selected**) | 0.129 |
| score_min | 0.136 |

**Detection score beats range by roughly 7x**, and this refuted the hypothesis
that dispatched the task -- task 18's far-vs-near split had implicated *range*,
and range is largely a proxy. The two are only weakly correlated (Pearson
-0.352), so they are not the same variable, and score wins on its own terms.
The mechanism is straightforward once measured: distant objects carry fewer
LiDAR points, so they score lower, and it is the point count that sets the
localization noise. The lowest score decile has RMS translation disagreement
0.532 m and RMS yaw 13.56 deg; the range deciles span only 0.272 to ~0.43 m.

The fit also answered a question posed without a prior: translation and yaw
need **different exponents**, 1.128 against 1.932. Centres and heading virtual
points should not carry the same weight, and forcing them to would have thrown
away most of the yaw gain.

Fitted parameters (`outputs/alignformer/r140/correspondence_variance_result.json`):

```
sigma_translation_m   0.2516   translation_exponent  1.128
sigma_yaw_deg         4.639    yaw_exponent          1.932
score_reference       0.4
```

**Two weighting modes were trained from those same parameters, and the fit file's
own `mode: split` is not what ships.** `split` carries the translation and yaw
variances through the Kabsch solve separately; `scalar` reduces them to one
weight per correspondence. On validation `scalar` wins -- translation MAE
**0.1418 m against 0.1488 m**, yaw **0.1693 deg against 0.1868 deg** -- so
`stage2_B_ivw_scalar` is the deployed checkpoint and every IVW number in this
document comes from it. The `split` checkpoint exists at
`outputs/alignformer/r140/stage2_B_ivw_split/` but was never carried through the
AP sweep, so it appears in no reported figure. Quoting the fit file's `mode`
as the deployed configuration would be wrong; this was caught in task 21.

### The oracle-correspondence ceiling: how much is matching worth at all?

Replacing the learned Sinkhorn assignment with the ground-truth one, built from
the `gt_ids` the dataset already carries, bounds what *any* matching improvement
can contribute -- a camera-augmented embedding, a larger trunk, a different
matcher. The decision rule was pre-registered before the run: mean AP@0.7 delta
over sigma in [0.2, 2.0] on validation, above 0.02 means headroom, below 0.01
means matching is closed.

**The rule returned `ambiguous` by 0.0001** -- mean delta **+0.0101** against a
0.0100 boundary, from one noise seed per cell. It should be read as
indistinguishable from closed, and the mean badly misdescribes the shape:

| sigma (m) | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation delta | +0.0070 | +0.0059 | +0.0049 | +0.0068 | +0.0036 | +0.0145 | +0.0282 |

At every sigma a vehicle with working localization actually sees (<= 1.0 m) the
delta is 0.0036-0.0070, inside "closed". The mean clears the line only at 1.5 and
2.0 m, where the *learned* matcher starts to fail while the oracle's stays flat.

**The error decomposition refutes the premise that matching is the lever.**

| Split | sigma | deployed | oracle floor | residual | reachable by matching |
|---|---|---:|---:|---:|---:|
| validation | 0 | 0.0375 m | 0.0436 m | **-0.0060** | **-16.0%** |
| validation | 1 | 0.1281 m | 0.1309 m | **-0.0027** | **-2.1%** |
| test | 0 | 0.1610 m | 0.1378 m | +0.0232 | +14.4% |
| test | 1 | 0.2649 m | 0.2386 m | +0.0262 | +9.9% |

On validation the residual is **negative** below sigma 1.5: the learned soft
correspondence produces a *better* pose than the ground-truth hard assignment,
so perfect matching is not even an upper bound there. On test, **86-90% of the
deployed translation error is unreachable by any matching improvement.**

**The two splits disagree, and that is the finding.** Test gives mean delta
**+0.0454**, 4.5x validation, above "headroom" at every sigma. The mechanism was
measured rather than assumed (`scripts/correspondence_evidence_budget.py`): test
carries **6x more sparse pairs** -- 12.8% with three or fewer true
correspondences against validation's 2.1% -- and on its sparsest pairs the
learned matcher gives 1.98 m where the oracle holds 1.05 m. A validation-only
rule was the wrong instrument here, because the quantity it measures is one of
the quantities that differs most between the two splits. The rule's output is
reported unchanged rather than re-chosen after the fact.

**Weighting beats matching.** The oracle correspondence with *uniform* weights
scores **-0.0026 on validation** -- worse than the shipped model -- against
IVW's +0.0101. Inverse-variance weighting alone is worth more than a perfect
correspondence on validation, and over a third of the ceiling on test. That
continues the task 19 result rather than contradicting it.

Two qualifiers travel with this ceiling. It is **not a strict upper bound**:
17.5-17.7% of ego detections carry no `gt_id` (they fail IoU >= 0.3 against
their own agent's local ground truth) and get zero mass under the oracle,
although the Sinkhorn matcher can and does use them -- a matcher that recovered
identity for those could exceed this "ceiling". And hard one-to-one assignment
discards the soft matcher's averaging, which is why the oracle loses outright on
sparse pairs.

The substitution is verified structurally, not by inspection: `model.py` was
refactored into `reduce_correspondence` + `solve_pose`, which both
`AlignFormerB.forward` and the new `alignformer/oracle.py` call, so only the
(B, M, N) matrix differs. The test-split `alignformer`, `uncorrected` and
`oracle` rows come back **bit-identical** to the published sweep, max absolute
difference 0.0.

**What this means for the camera contingency.** Do not start a camera pipeline
for OPV2V association. The validation case for more matching capacity is
confined to sigma >= 1.5 m; the test case is confined to the 12.8% sparse-pair
tail, which is a generalization failure of the *existing* matcher reachable by
harder training data and by refusing to correct at low match mass; and 86-90% of
the residual pose error is not matching at all.

### The embedding: still nothing on association, marginal on AP, larger on pose

**This sub-section is `r70` throughout**: the boxes-only ablation was not
retrained at +/-140.8 m, so comparing it against an `r140` deployed
configuration would confound the message content with the detector range.

This is the project's most-cited finding, so it was re-run in full rather than
assumed. The pre-registered association diagnostic ruled that the appearance
embedding contributes nothing on OPV2V (hard-subset delta +0.0001, 95% CI
[-0.0013, +0.0014]; true-partner separability AUC 0.560 on raw ROI features;
competing vehicles differ by a median 0.071 m in width). Post-fix:

| Level | boxes+embeddings | boxes_only | Verdict |
|---|---:|---:|---|
| Stage-1 association Top-1 (validation, sigma 0.5 m) | 0.9961 | **0.9976** | the embedding is worth **nothing**, and is marginally behind |
| Fused AP@0.7 (test split, sigma 0.2-2.0 m) | 0.7046-0.7488 | 0.7023-0.7280 | **+0.000 to +0.021**, mean +0.011 |
| Validation translation MAE at sigma 0 (no shrinkage) | 0.1264 m | 0.1418 m | +12% for the embedding |
| Validation yaw MAE at sigma 0 (no shrinkage) | 0.1670 deg | 0.1928 deg | +15% for the embedding |
| P2 gate at sigma = 0.2 m | 0.1473 < 0.1600 **pass** | 0.1662 > 0.1600 **fail** | the embedding is the difference |

Fused AP@0.7 in full, each configuration under **its own**
validation-calibrated shrinkage, on the 2170-frame test split:

| sigma (m) | Uncorrected | boxes+embeddings | boxes_only | Embedding is worth |
|---|---:|---:|---:|---:|
| 0 | 0.8764 | **0.8068** | 0.8064 | +0.0004 |
| 0.2 | 0.5846 | **0.7488** | 0.7280 | +0.0208 |
| 0.4 | 0.3069 | **0.7410** | 0.7199 | +0.0211 |
| 0.6 | 0.2106 | **0.7412** | 0.7236 | +0.0176 |
| 0.8 | 0.1785 | **0.7404** | 0.7290 | +0.0113 |
| 1 | 0.1690 | **0.7386** | 0.7291 | +0.0095 |
| 1.5 | 0.1737 | **0.7326** | 0.7230 | +0.0096 |
| 2 | 0.1831 | **0.7046** | 0.7023 | +0.0024 |

**The headline claim survives where it was made.** On association -- which is
what the pre-registered diagnostic measured -- the embedding is still worth
nothing, and the boxes-only stage 1 is in fact marginally *better*. On fused
AP, which is what the project is judged on, it is worth +0.011 on average, the
same marginal amount as the +0.010 to +0.016 measured before the fixes. That is
one order smaller than the range fix (+0.04) and two orders smaller than the
heading fold (+0.17).

**What did change is the pose metric.** The boxes+embeddings-vs-boxes_only gap
there went from ~3% before the fixes to 12-15% after, and it is now the
difference between passing and failing the P2 gate at sigma = 0.2 m. Three
reasons to treat that as suggestive rather than established:

- **One seed each.** Nothing here is repeated, and a 12% difference in
  validation MAE between two 30-epoch runs is within the range a seed can move.
- **Separate warm starts.** `boxes_only` warm-starts from its own stage-1 run
  (correctly -- see below), and those two stage-1 runs selected their best
  checkpoints at different epochs (14 and 9) with different Top-1. Part of the
  gap could be the warm start rather than the embedding.
- **It does not carry to AP.** A 12% pose improvement that buys 0.011 AP@0.7 is
  a pose improvement below the level fused AP can resolve.

The claim that should be published is therefore the narrow one, unchanged:
**the appearance embedding does not carry the method.** The wider claim, that
it contributes literally nothing anywhere, is no longer exactly right at the
pose level and should be stated with the numbers above.

#### The matched `r140` rerun removes both confounds, and the null holds

The two caveats above -- one seed, and warm starts that selected best
checkpoints at different epochs with different Top-1 -- were removable, so they
were removed. Both arms were retrained at +/-140.8 m to **epoch 30**, each from
**its own** stage-1 lineage, so message content is the only difference between
them:

| sigma (m) | boxes+embeddings | boxes_only | Embedding is worth |
|---|---:|---:|---:|
| 0 | 0.8425 | **0.8452** | -0.0028 |
| 0.2 | **0.7899** | 0.7874 | +0.0025 |
| 0.4 | **0.7851** | 0.7821 | +0.0029 |
| 0.6 | **0.7892** | 0.7848 | +0.0043 |
| 0.8 | **0.7920** | 0.7881 | +0.0039 |
| 1.0 | **0.7906** | 0.7903 | +0.0003 |
| 1.5 | 0.7791 | **0.7804** | -0.0014 |
| 2.0 | 0.7584 | **0.7629** | -0.0045 |

**Mean +0.0007, range -0.0045 to +0.0043, sign flipping three times.** The
+0.011 mean measured at `r70` was the confound, not the embedding: once the two
arms are matched on epoch and lineage, the fused-AP benefit disappears into the
noise, and boxes-only is ahead on three of eight rows including the two
hardest.

This supersedes the "marginal on AP" half of this sub-section's title. The
honest summary across every level at which the embedding has been measured:

| Level | Verdict |
|---|---|
| Association Top-1 | **nothing** (boxes-only marginally ahead) |
| Fused AP@0.7, matched arms | **nothing** (+0.0007, sign flips) |
| Fused AP@0.7, mismatched arms (`r70`) | +0.011 -- **an artifact of the mismatch** |
| Pose MAE at sigma = 0, mismatched arms (`r70`) | +12-15%, never re-tested matched |

The claim to publish is the strong one: **on OPV2V the appearance embedding
contributes nothing that geometry does not already supply.** Three independent
reasons, all measured: no appearance signal exists to extract (true-partner
separability AUC 0.560 on raw ROI features; all 65,774 CAV box widths are
2.014 +/- 0.081 m, because CARLA reuses a small vehicle asset library), the
0.8 m/cell BEV resolution cannot resolve instance identity anyway, and -- the
binding constraint -- **no ego object in the whole validation split has a
competitor within 2 m**, so there is nothing for any descriptor to
disambiguate. The nearest-centre association baseline scores Top-1 **1.0000**
at sigma = 0.

That last reason is geometric, not perceptual, which is why the camera
contingency in the design spec **cannot rescue association here** however good
the features are. It remains open on a real-traffic dataset such as V2X-Real,
where competitors within 2 m are common.

## Tables

All of the following are rendered by `scripts/summarize_alignformer_p2.py`.
The gate, pose and fused-AP tables come from
`outputs/alignformer/r140/p2_r140_gate.json` and
`outputs/alignformer/r140/p2_r140_noisy_ap_result.json` unless a heading says
`r70`; the four-configuration ablation comparison
(`outputs/alignformer/r70/p2_r70_ablations_result.json`, **without** shrinkage,
since one calibration cannot describe four different estimators) and the
per-epoch curve are `r70` and have not been re-run at +/-140.8 m.

## P2 gate

Gate configuration: **B / boxes+embeddings** -- **PASS, 7 of 7** (`r140`)

| sigma | Yaw MAE (deg) | Predict-zero yaw (deg) | Below? | r70 yaw MAE (deg) |
|---|---:|---:|---|---:|
| sigma_0.2m | 0.1489 | 0.1600 | yes | 0.1473 |
| sigma_0.4m | 0.1624 | 0.3199 | yes | 0.1654 |
| sigma_0.6m | 0.1629 | 0.4799 | yes | 0.1690 |
| sigma_0.8m | 0.1626 | 0.6398 | yes | 0.1708 |
| sigma_1m | 0.1632 | 0.7998 | yes | 0.1726 |
| sigma_1.5m | 0.1702 | 1.1996 | yes | 0.1816 |
| sigma_2m | 0.1942 | 1.5995 | yes | 0.2065 |

Measured on the scenario-disjoint validation split (6262 pairs, `val_scenario_fraction` 0.15, `split_seed` 0), mean of 3 independent noise draws, with the validation-calibrated shrinkage applied. The population is identical at both ranges: the pair index is pose-derived and was not rebuilt. `sigma = 0` is excluded from the gate because the predict-zero baseline is exactly 0 there; it is reported in the pose sweep below as the clean-case diagnostic, at **0.0327 m / 0.0497 deg** (`r70`: 0.0320 m / 0.0509 deg).

Yaw is better at the wider range at every sigma from 0.4 m up. Translation is
**worse** at every sigma (0.1575 against 0.1265 m at sigma = 0.2, 0.1373
against 0.1288 at sigma = 1), which is the far-field correspondences entering
the fit -- see [Where the AP came from, and what it
cost](#where-the-ap-came-from-and-what-it-cost). Translation is not part of the
gate's definition and the gate was not changed to accommodate it.

## Pose sweep

**This table is `r70` and covers all four configurations.** The `r140` run
retrained only the deployed configuration (B / boxes+embeddings); its numbers
are in the P2 gate table above and the test-split table below.


| sigma (m) | Configuration | Translation MAE (m) | Predict-zero translation (m) | Yaw MAE (deg) | Predict-zero yaw (deg) | Analytic predict-zero yaw (deg) |
|---|---|---:|---:|---:|---:|---:|
| 0 | B / boxes+embeddings | 0.1264 | 0.0000 | 0.1670 | 0.0000 | 0.0000 |
| 0 | B / boxes_only | 0.1418 | 0.0000 | 0.1928 | 0.0000 | 0.0000 |
| 0 | A / boxes+embeddings (match_weight 0) | 0.1355 | 0.0000 | 0.3362 | 0.0000 | 0.0000 |
| 0 | B / boxes+embeddings (match_weight 0) | 0.1384 | 0.0000 | 0.1892 | 0.0000 | 0.0000 |
| 0.2 | B / boxes+embeddings | 0.1265 | 0.2734 | 0.1674 | 0.1600 | 0.1596 |
| 0.2 | B / boxes_only | 0.1418 | 0.2734 | 0.1931 | 0.1600 | 0.1596 |
| 0.2 | A / boxes+embeddings (match_weight 0) | 0.3023 | 0.2734 | 0.3751 | 0.1600 | 0.1596 |
| 0.2 | B / boxes+embeddings (match_weight 0) | 0.1388 | 0.2734 | 0.1895 | 0.1600 | 0.1596 |
| 0.4 | B / boxes+embeddings | 0.1268 | 0.5468 | 0.1680 | 0.3199 | 0.3192 |
| 0.4 | B / boxes_only | 0.1419 | 0.5468 | 0.1935 | 0.3199 | 0.3192 |
| 0.4 | A / boxes+embeddings (match_weight 0) | 0.5512 | 0.5468 | 0.4687 | 0.3199 | 0.3192 |
| 0.4 | B / boxes+embeddings (match_weight 0) | 0.1391 | 0.5468 | 0.1901 | 0.3199 | 0.3192 |
| 0.6 | B / boxes+embeddings | 0.1271 | 0.8201 | 0.1688 | 0.4799 | 0.4787 |
| 0.6 | B / boxes_only | 0.1428 | 0.8201 | 0.1952 | 0.4799 | 0.4787 |
| 0.6 | A / boxes+embeddings (match_weight 0) | 0.8114 | 0.8201 | 0.5928 | 0.4799 | 0.4787 |
| 0.6 | B / boxes+embeddings (match_weight 0) | 0.1395 | 0.8201 | 0.1909 | 0.4799 | 0.4787 |
| 0.8 | B / boxes+embeddings | 0.1279 | 1.0935 | 0.1699 | 0.6398 | 0.6383 |
| 0.8 | B / boxes_only | 0.1444 | 1.0935 | 0.1981 | 0.6398 | 0.6383 |
| 0.8 | A / boxes+embeddings (match_weight 0) | 1.0758 | 1.0935 | 0.7316 | 0.6398 | 0.6383 |
| 0.8 | B / boxes+embeddings (match_weight 0) | 0.1404 | 1.0935 | 0.1923 | 0.6398 | 0.6383 |
| 1 | B / boxes+embeddings | 0.1288 | 1.3669 | 0.1717 | 0.7998 | 0.7979 |
| 1 | B / boxes_only | 0.1459 | 1.3669 | 0.2012 | 0.7998 | 0.7979 |
| 1 | A / boxes+embeddings (match_weight 0) | 1.3422 | 1.3669 | 0.8770 | 0.7998 | 0.7979 |
| 1 | B / boxes+embeddings (match_weight 0) | 0.1422 | 1.3669 | 0.1959 | 0.7998 | 0.7979 |
| 1.5 | B / boxes+embeddings | 0.1345 | 2.0503 | 0.1806 | 1.1996 | 1.1968 |
| 1.5 | B / boxes_only | 0.1536 | 2.0503 | 0.2136 | 1.1996 | 1.1968 |
| 1.5 | A / boxes+embeddings (match_weight 0) | 2.0128 | 2.0503 | 1.2540 | 1.1996 | 1.1968 |
| 1.5 | B / boxes+embeddings (match_weight 0) | 0.1540 | 2.0503 | 0.2149 | 1.1996 | 1.1968 |
| 2 | B / boxes+embeddings | 0.1574 | 2.7338 | 0.2055 | 1.5995 | 1.5958 |
| 2 | B / boxes_only | 0.1804 | 2.7338 | 0.2454 | 1.5995 | 1.5958 |
| 2 | A / boxes+embeddings (match_weight 0) | 2.6876 | 2.7338 | 1.6428 | 1.5995 | 1.5958 |
| 2 | B / boxes+embeddings (match_weight 0) | 0.1975 | 2.7338 | 0.2721 | 1.5995 | 1.5958 |

No shrinkage in this table: it is calibrated per estimator, so applying one configuration's `tau` to another's output would compare calibrations rather than heads. All four configurations see byte-identical perturbations, so the comparison is paired.

## Fused AP under localization error

`r140`, and the `r70` AP@0.7 alongside:

| sigma (m) | Condition | AP@0.3 | AP@0.5 | AP@0.7 | r70 AP@0.7 |
|---|---|---:|---:|---:|---:|
| -- | oracle (true pose) | 0.9552 | 0.9502 | 0.8964 | 0.8764 |
| 0 | vanilla late fusion (uncorrected) | 0.9552 | 0.9502 | 0.8964 | 0.8764 |
| 0 | AlignFormer-corrected | 0.9357 | 0.9190 | 0.8415 | 0.8068 |
| 0.2 | vanilla late fusion (uncorrected) | 0.9544 | 0.9234 | 0.5978 | 0.5846 |
| 0.2 | AlignFormer-corrected | 0.9365 | 0.9158 | 0.7743 | 0.7488 |
| 0.4 | vanilla late fusion (uncorrected) | 0.9193 | 0.6914 | 0.3158 | 0.3069 |
| 0.4 | AlignFormer-corrected | 0.9357 | 0.9117 | 0.7717 | 0.7410 |
| 0.6 | vanilla late fusion (uncorrected) | 0.7816 | 0.4656 | 0.2154 | 0.2106 |
| 0.6 | AlignFormer-corrected | 0.9337 | 0.9045 | 0.7717 | 0.7412 |
| 0.8 | vanilla late fusion (uncorrected) | 0.6473 | 0.3613 | 0.1844 | 0.1785 |
| 0.8 | AlignFormer-corrected | 0.9304 | 0.9064 | 0.7804 | 0.7404 |
| 1 | vanilla late fusion (uncorrected) | 0.5501 | 0.3117 | 0.1763 | 0.1690 |
| 1 | AlignFormer-corrected | 0.9301 | 0.9020 | 0.7771 | 0.7386 |
| 1.5 | vanilla late fusion (uncorrected) | 0.4062 | 0.2608 | 0.1799 | 0.1737 |
| 1.5 | AlignFormer-corrected | 0.9258 | 0.8970 | 0.7728 | 0.7326 |
| 2 | vanilla late fusion (uncorrected) | 0.3524 | 0.2533 | 0.1898 | 0.1831 |
| 2 | AlignFormer-corrected | 0.9157 | 0.8798 | 0.7441 | 0.7046 |

Every AlignFormer row rose (+0.026 to +0.040 AP@0.7), and so did every
uncorrected and oracle row. The consistency check still holds exactly: at
sigma = 0 the noisy transform *is* the true transform, so `uncorrected_sigma_0m`
equals `oracle` to the last digit, at all three IoU thresholds.

## Gap recovered at AP@0.7

| sigma (m) | Vanilla AP@0.7 | AlignFormer AP@0.7 | Oracle AP@0.7 | Gain over vanilla | Gap recovered | r70 gap recovered |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.8964 | 0.8415 | 0.8964 | -0.0549 | -- | -- |
| 0.2 | 0.5978 | 0.7743 | 0.8964 | +0.1765 | 59.1% | 56.3% |
| 0.4 | 0.3158 | 0.7717 | 0.8964 | +0.4559 | 78.5% | 76.2% |
| 0.6 | 0.2154 | 0.7717 | 0.8964 | +0.5564 | 81.7% | 79.7% |
| 0.8 | 0.1844 | 0.7804 | 0.8964 | +0.5961 | 83.7% | 80.5% |
| 1 | 0.1763 | 0.7771 | 0.8964 | +0.6008 | 83.4% | 80.5% |
| 1.5 | 0.1799 | 0.7728 | 0.8964 | +0.5928 | 82.7% | 79.5% |
| 2 | 0.1898 | 0.7441 | 0.8964 | +0.5542 | 78.4% | 75.2% |

## Pose error on the test split

`r140`, with the `r70` columns beside the two the gate criterion reads:

| sigma (m) | Pairs | Translation MAE (m) | Predict-zero (m) | Yaw MAE (deg) | Predict-zero (deg) | Unalignable pairs | Fell back | r70 translation | r70 yaw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3445 | 0.1856 | 0.0000 | 0.3440 | 0.0000 | 34 | 0.742 | 0.1453 | 0.2692 |
| 0.2 | 3445 | 0.3043 | 0.2797 | 0.4308 | 0.1610 | 34 | 0.246 | 0.2509 | 0.3605 |
| 0.4 | 3445 | 0.3045 | 0.5450 | 0.4550 | 0.3241 | 34 | 0.074 | 0.2539 | 0.3817 |
| 0.6 | 3445 | 0.3063 | 0.8140 | 0.4550 | 0.4865 | 34 | 0.039 | 0.2646 | 0.3945 |
| 0.8 | 3445 | 0.3161 | 1.0904 | 0.4777 | 0.6308 | 34 | 0.025 | 0.2615 | 0.3880 |
| 1 | 3445 | 0.3216 | 1.3459 | 0.4825 | 0.7892 | 34 | 0.021 | 0.2707 | 0.4084 |
| 1.5 | 3445 | 0.3408 | 2.0195 | 0.5099 | 1.1962 | 34 | 0.017 | 0.2880 | 0.4358 |
| 2 | 3445 | 0.3628 | 2.6933 | 0.5114 | 1.6053 | 34 | 0.019 | 0.3262 | 0.4447 |

**Read this table against the gate.** The P2 gate is a *validation*
measurement, and on validation head B's yaw MAE is below predict-zero at all
seven non-zero sigmas at both ranges. On the **test** split the same criterion
still **fails at sigma = 0.2 and 0.4 m** -- 0.4308 against 0.1610 and 0.4550
against 0.3241 -- and passes from 0.6 m up. That qualifier survives the range
fix unchanged.

What the range fix *did* change here is translation: at +/-70.4 m the
translation criterion passed everywhere (0.2509 against 0.2797 at sigma = 0.2);
at +/-140.8 m it **fails at sigma = 0.2** (0.3043 against 0.2797). Every
test-split pose number is worse at the wider range, by 0.04 to 0.06 m and 0.06
to 0.09 deg. The mechanism is measured in [Where the AP came from, and what it
cost](#where-the-ap-came-from-and-what-it-cost): the far-field correspondences
the wide detector adds disagree across agents by 0.230 m against the near
field's 0.151 m, and they enter the Procrustes fit unweighted. It compounds the
validation-vs-test gap already diagnosed in
[alignformer_pose_floor.md](alignformer_pose_floor.md) section 6 -- the test
split is harder and `tau`, calibrated on validation, is a little small there.

It does not change the AP result -- fused AP@0.7 at sigma = 0.2 is 0.7743
against the uncorrected 0.5978, better than the +/-70.4 m pair on both sides --
because IoU is far more sensitive to a *missing* box than to a few centimetres
of residual centre error. But the gate should not be read as a claim about the
test split, and it is not one.

The fallback column is the *exact-identity* rate, which at sigma = 0 is
shrinkage doing its job: **74.2%** of test pairs receive no correction at all
there (66.1% at +/-70.4 m, consistent with the larger recalibrated `tau`),
against 24.6% at sigma = 0.2 and ~2% above 1 m. The 34 structurally unalignable
pairs (0.99%, down from 37) are a separate, much smaller population.

## Training curve: stage2_B_boxes+embeddings

**`r70`.** The `r140` stage-2 run reached its best validation corner loss of
0.9789 m at epoch 26 of 30 (against `r70`'s 0.9194 m at epoch 28); its stage-1
warm start reached validation Top-1 **0.9983** at epoch 10 (against 0.9961 at
epoch 14). Its per-epoch curve is in
`outputs/alignformer/r140/stage2_B_boxes+embeddings/history.json`.

| Epoch | sigma (m) | Train loss | Train corner (m) | Val corner (m) | Val translation MAE (m) | Val yaw MAE (deg) | Match temperature |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 1.4754 | 1.4288 | 4.3165 | 0.7470 | 0.9387 | 0.02940 |
| 2 | 0.069 | 2.3225 | 2.1702 | 5.5132 | 0.9604 | 1.1367 | 0.02565 |
| 3 | 0.138 | 2.1344 | 2.0051 | 6.5854 | 1.1339 | 1.2810 | 0.02256 |
| 4 | 0.207 | 2.2226 | 2.0951 | 3.8747 | 0.7336 | 0.8394 | 0.02135 |
| 5 | 0.276 | 2.0459 | 1.9413 | 4.0699 | 0.7333 | 0.9259 | 0.02048 |
| 6 | 0.345 | 1.9438 | 1.8438 | 4.5385 | 0.7658 | 1.0367 | 0.02027 |
| 7 | 0.414 | 1.8756 | 1.7830 | 3.7557 | 0.7130 | 0.8673 | 0.01926 |
| 8 | 0.483 | 1.9079 | 1.8089 | 3.7092 | 0.6706 | 0.7961 | 0.01948 |
| 9 | 0.552 | 1.8128 | 1.7192 | 3.2820 | 0.5812 | 0.6999 | 0.01944 |
| 10 | 0.621 | 1.8625 | 1.7805 | 3.1540 | 0.6174 | 0.7069 | 0.01966 |
| 11 | 0.690 | 1.8246 | 1.7442 | 2.7566 | 0.5140 | 0.5832 | 0.01960 |
| 12 | 0.759 | 1.9804 | 1.8990 | 2.2021 | 0.4009 | 0.4759 | 0.01979 |
| 13 | 0.828 | 1.9207 | 1.8452 | 2.1560 | 0.4056 | 0.4813 | 0.02003 |
| 14 | 0.897 | 1.9233 | 1.8508 | 2.4174 | 0.4349 | 0.5668 | 0.02035 |
| 15 | 0.966 | 1.8833 | 1.8152 | 1.7860 | 0.3095 | 0.3972 | 0.02039 |
| 16 | 1.034 | 1.9246 | 1.8590 | 1.7587 | 0.3060 | 0.4037 | 0.02023 |
| 17 | 1.103 | 1.9448 | 1.8779 | 1.5048 | 0.2609 | 0.3299 | 0.02050 |
| 18 | 1.172 | 1.8827 | 1.8192 | 1.4782 | 0.2592 | 0.3193 | 0.02057 |
| 19 | 1.241 | 1.8735 | 1.8130 | 1.2761 | 0.2152 | 0.2815 | 0.02062 |
| 20 | 1.310 | 1.9432 | 1.8860 | 1.2670 | 0.2207 | 0.2793 | 0.02091 |
| 21 | 1.379 | 1.8939 | 1.8386 | 1.2025 | 0.1972 | 0.2734 | 0.02094 |
| 22 | 1.448 | 1.8963 | 1.8425 | 1.1401 | 0.1949 | 0.2461 | 0.02101 |
| 23 | 1.517 | 1.9518 | 1.8991 | 1.0577 | 0.1796 | 0.2278 | 0.02112 |
| 24 | 1.586 | 1.9559 | 1.9030 | 1.0140 | 0.1704 | 0.2180 | 0.02114 |
| 25 | 1.655 | 1.9711 | 1.9202 | 0.9671 | 0.1614 | 0.2094 | 0.02119 |
| 26 | 1.724 | 2.0042 | 1.9531 | 0.9507 | 0.1593 | 0.2087 | 0.02121 |
| 27 | 1.793 | 2.0396 | 1.9889 | 0.9497 | 0.1587 | 0.2073 | 0.02125 |
| 28 | 1.862 | 2.1067 | 2.0542 | 0.9194 | 0.1538 | 0.2016 | 0.02125 |
| 29 | 1.931 | 2.1561 | 2.1025 | 0.9353 | 0.1557 | 0.2040 | 0.02125 |
| 30 | 2.000 | 2.1706 | 2.1166 | 0.9357 | 0.1558 | 0.2040 | 0.02125 |

## Head-to-head against intermediate fusion

This is the comparison the project exists to produce, and the claim it tests is
**robustness per byte**: under one noise sweep, one evaluator and one ground
truth, AlignFormer's late fusion should retain more AP@0.7 than
intermediate-fusion methods that transmit dense BEV feature maps. It is
explicitly **not** a claim about clean AP; on the clean row AlignFormer is
level with the two strongest-clean methods it beats and below the four that
beat it.

**Everything here was run locally.** Published numbers are not quoted beside
ours anywhere: different detectors, splits and noise conventions make that
meaningless, and [the design spec](superpowers/specs/2026-09-17-alignformer-design.md)
forbids it. Seven trained intermediate-fusion checkpoints already on this
machine were re-run on the same 2170-frame OPV2V test split, under the same
sigma sweep with `sigma_yaw(deg) = sigma_xy(m)` on the CAV pose only, and
scored by the same `alignformer/fusion.py::average_precision`.

The perturbation is not merely the same *distribution* -- it is the same
*draw*. `alignformer/baselines.py` takes each CAV's displacement from
`noisy_fusion._sweep_rng(seed, sigma, frame, agent)` with this project's seed
and AlignFormer's own agent ordering, so on any given frame a baseline sees the
metre-for-metre displacement AlignFormer saw. The comparison is paired.

### AP@0.7 under the sweep

AlignFormer is `r140`; the seven baselines are unchanged from task 17 -- they
already ran at +/-140.8 m, which is the whole point of this re-measurement --
and they are scored against the same 33,089 ground-truth boxes AlignFormer is.

| sigma (m) | **AlignFormer** | late fusion (uncorrected) | V2X-ViT | CoAlign (fusion only) | CoBEVT | AttFuse | Where2comm | F-Cooper | V2VAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.8415** | 0.8964 | 0.8174 | 0.8831 | 0.8773 | 0.8638 | 0.7849 | 0.8402 | 0.8975 |
| 0.2 | **0.7743** | 0.5978 | 0.7894 | 0.7889 | 0.7824 | 0.7758 | 0.7117 | 0.7117 | 0.8006 |
| 0.4 | **0.7717** | 0.3158 | 0.7319 | 0.6150 | 0.5377 | 0.5983 | 0.5851 | 0.4445 | 0.5702 |
| 0.6 | **0.7717** | 0.2154 | 0.6732 | 0.5324 | 0.3807 | 0.5002 | 0.4695 | 0.2858 | 0.4109 |
| 0.8 | **0.7804** | 0.1844 | 0.6250 | 0.5090 | 0.2983 | 0.4648 | 0.4106 | 0.2037 | 0.3385 |
| 1 | **0.7771** | 0.1763 | 0.6027 | 0.5028 | 0.2746 | 0.4533 | 0.3894 | 0.1788 | 0.3181 |
| 1.5 | **0.7728** | 0.1799 | 0.5675 | 0.5023 | 0.2538 | 0.4447 | 0.3851 | 0.1506 | 0.3410 |
| 2 | **0.7441** | 0.1898 | 0.5515 | 0.5057 | 0.2544 | 0.4481 | 0.4106 | 0.1469 | 0.3741 |

| Baseline | sigma where AlignFormer overtakes it | was, at +/-70.4 m | sigmas where the baseline wins |
|---|---|---|---|
| V2X-ViT | **0 m** (not robust; see below) | 0.4 m | 0.2 |
| CoAlign (fusion only) | 0.4 m | 0.4 m | 0, 0.2 |
| CoBEVT | 0.4 m | 0.4 m | 0, 0.2 |
| AttFuse | 0.4 m | 0.4 m | 0, 0.2 |
| Where2comm | 0 m | 0 m | none |
| F-Cooper | **0 m** (not robust; see below) | 0.2 m | none |
| V2VAM | 0.4 m | 0.4 m | 0, 0.2 |

Three things this says, in the order they matter.

**AlignFormer is still beaten on the clean and near-clean rows, but by fewer
methods.** At sigma = 0 four of the seven are above it (V2VAM 0.8975, CoAlign
0.8831, CoBEVT 0.8773, AttFuse 0.8638 against AlignFormer's 0.8415); V2X-ViT,
Where2comm and F-Cooper are now below. At sigma = 0.2 m six of seven are still
ahead. That is the expected shape -- late fusion does not beat intermediate
fusion on clean AP, and this work does not claim it does -- and it is still
compounded by AlignFormer's own sigma = 0 regression against uncorrected late
fusion (0.8415 against 0.8964), which is carried into this table rather than
hidden from it.

**From sigma = 0.4 m upward AlignFormer leads every baseline, and that lead is
now robust to the ground-truth convention.** Against V2X-ViT the 0.4 m margin
is **+0.0398** under the late-fusion convention and **+0.0297** under the
intermediate one -- at +/-70.4 m it was +0.009 and -0.0005, i.e. a tie. The
0.4 m crossover has therefore changed status: it is no longer a coin flip. From
0.6 m the margin over V2X-ViT is +0.099, then +0.174 at 1.0 m and +0.193 at
2.0 m; against the other six the 0.4 m lead runs +0.157 (CoAlign) to +0.327
(F-Cooper).

**The two thin margins have moved to sigma = 0, and they are the ones to read
as ties.** Against V2X-ViT at sigma = 0, AlignFormer leads by +0.0241 under the
late-fusion ground truth but only **+0.0082** under the intermediate one --
inside the convention band. Against F-Cooper at sigma = 0 the margin is
**+0.0013** under one convention and **-0.0137** under the other, i.e. it
**flips**. **The defensible claim is: AlignFormer ties V2X-ViT and F-Cooper on
the clean row, loses to V2X-ViT at 0.2 m, and leads the entire field from
0.4 m upward.** Where2comm is the one baseline it beats outright at every
sigma.

**V2X-ViT is a real robustness baseline and the others are not close to it.**
It retains 67% of its clean AP@0.7 at sigma = 2 m where CoBEVT retains 29% and
F-Cooper 17%. That reproduces, locally and under our own convention, the thing
V2X-ViT is known for. AlignFormer retains 88%.

### Retention: AP@0.7 as a fraction of each method's own clean score

| sigma (m) | **AlignFormer** | V2X-ViT | CoAlign (fusion only) | CoBEVT | AttFuse | Where2comm | F-Cooper | V2VAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | **92%** | 97% | 89% | 89% | 90% | 91% | 85% | 89% |
| 0.4 | **92%** | 90% | 70% | 61% | 69% | 75% | 53% | 64% |
| 0.6 | **92%** | 82% | 60% | 43% | 58% | 60% | 34% | 46% |
| 0.8 | **93%** | 76% | 58% | 34% | 54% | 52% | 24% | 38% |
| 1 | **92%** | 74% | 57% | 31% | 52% | 50% | 21% | 35% |
| 1.5 | **92%** | 69% | 57% | 29% | 51% | 49% | 18% | 38% |
| 2 | **88%** | 67% | 57% | 29% | 52% | 52% | 17% | 42% |

Retention is the fairer axis for a robustness claim, because it quotients out
the clean-AP advantage the baselines start with. On it, AlignFormer is ahead of
every baseline at every sigma from 0.4 m up, and the only method in the same
region is V2X-ViT. It is computed against each method's own clean score, so
AlignFormer's column barely moves even though every absolute number rose: the
denominator rose too.

### Bytes per frame per agent

Measured, not asserted, on both sides. For each baseline a forward hook on the
model's own fusion site records the per-agent tensor the fusion stage consumes;
for AlignFormer the object count is read off the cached detections the sweep
actually transmits (mean **15.69** objects per agent at +/-140.8 m against
13.06 at +/-70.4 m, never truncated at the 64-object budget), times the 7-float
box plus 128-float embedding. Both at float32, which is what the tensors are at
runtime; neither side is credited with a quantizer that is not in the measured
pipeline.

**AlignFormer's own side is the part that moved**: the wider detector finds
more objects, so it transmits 20% more bytes. The baselines' figures reproduce
the task-17 measurement to the byte, which is also the check that re-running
the measurement changed nothing on their side.

| Method | What crosses the wire | Bytes / frame / agent | vs AlignFormer | vs AlignFormer at +/-70.4 m |
|---|---|---:|---:|---:|
| **AlignFormer** | 15.69 boxes+embeddings x 540 B (7 box floats + 128 embedding floats, float32) | **8,474** | 1x | (7,051) |
| AlignFormer, boxes only | 15.69 boxes x 28 B | 439 | 0.05x | (366) |
| V2X-ViT | BEV feature 256x48x176 float32 | 8,650,752 | 1,021x | 1,227x |
| CoAlign (fusion only) | BEV feature 64x100x352 float32 | 9,011,200 | 1,063x | 1,278x |
| CoBEVT | BEV feature 256x96x352 float32 | 34,603,008 | 4,083x | 4,908x |
| AttFuse | BEV feature 64x100x352 float32 | 9,011,200 | 1,063x | 1,278x |
| Where2comm | BEV feature 64x192x704 float32 | 34,603,008 | 4,083x | 4,908x |
| Where2comm, after its own communication mask | selected cells only (rate 0.211) | 7,300,874 | 862x | 1,035x |
| F-Cooper | BEV feature 256x100x352 float32 | 36,044,800 | 4,254x | 5,112x |
| V2VAM | BEV feature 256x50x176 float32 | 9,011,200 | 1,063x | 1,278x |

The baselines are charged the *smallest* tensor from which the ego could
reconstruct their contribution -- every one of these models shares one encoder,
so a receiver holding the first tensor the fusion stage consumes can run the
rest of the chain itself. Charging them for every fused scale would have
inflated the ratio. As an external cross-check,
`/media/chenyi/Elements1/models/opv2v/feature_stats_bwinf.txt`, written by an
earlier unrelated Where2comm experiment on this machine, records 221.15 KB per
agent for its *selected* features; our Where2comm-after-mask figure is 7.30 MB
because that earlier measurement counts non-zero elements rather than the dense
tensor that must actually be addressed and sent.

**The claim as it stands: three orders of magnitude fewer bytes, and more
AP@0.7 than any of them at every sigma at or above 0.4 m.** Against the
robustness SOTA specifically: 1,021x fewer bytes, level on the clean row, and
ahead from 0.4 m upward by a margin larger than any measurement artefact
identified below.

### All three IoU thresholds

AlignFormer's rows are `r140`; the baselines' are unchanged.

| Method | sigma (m) | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|---:|
| AlignFormer | 0 | 0.9357 | 0.9190 | 0.8415 |
| AlignFormer | 0.2 | 0.9365 | 0.9158 | 0.7743 |
| AlignFormer | 0.4 | 0.9357 | 0.9117 | 0.7717 |
| AlignFormer | 0.6 | 0.9337 | 0.9045 | 0.7717 |
| AlignFormer | 0.8 | 0.9304 | 0.9064 | 0.7804 |
| AlignFormer | 1 | 0.9301 | 0.9020 | 0.7771 |
| AlignFormer | 1.5 | 0.9258 | 0.8970 | 0.7728 |
| AlignFormer | 2 | 0.9157 | 0.8798 | 0.7441 |
| V2X-ViT | 0 | 0.9199 | 0.9125 | 0.8174 |
| V2X-ViT | 0.2 | 0.9162 | 0.9035 | 0.7894 |
| V2X-ViT | 0.4 | 0.9104 | 0.8883 | 0.7319 |
| V2X-ViT | 0.6 | 0.8905 | 0.8552 | 0.6732 |
| V2X-ViT | 0.8 | 0.8696 | 0.8264 | 0.6250 |
| V2X-ViT | 1 | 0.8436 | 0.8031 | 0.6027 |
| V2X-ViT | 1.5 | 0.7854 | 0.7513 | 0.5675 |
| V2X-ViT | 2 | 0.7448 | 0.7149 | 0.5515 |
| CoAlign (fusion only) | 0 | 0.9295 | 0.9260 | 0.8831 |
| CoAlign (fusion only) | 0.2 | 0.9284 | 0.9200 | 0.7889 |
| CoAlign (fusion only) | 0.4 | 0.9195 | 0.8687 | 0.6150 |
| CoAlign (fusion only) | 0.6 | 0.8881 | 0.7899 | 0.5324 |
| CoAlign (fusion only) | 0.8 | 0.8412 | 0.7287 | 0.5090 |
| CoAlign (fusion only) | 1 | 0.7988 | 0.6920 | 0.5028 |
| CoAlign (fusion only) | 1.5 | 0.7166 | 0.6400 | 0.5023 |
| CoAlign (fusion only) | 2 | 0.6729 | 0.6118 | 0.5057 |
| CoBEVT | 0 | 0.9112 | 0.9091 | 0.8773 |
| CoBEVT | 0.2 | 0.9068 | 0.8997 | 0.7824 |
| CoBEVT | 0.4 | 0.8795 | 0.8205 | 0.5377 |
| CoBEVT | 0.6 | 0.7926 | 0.6610 | 0.3807 |
| CoBEVT | 0.8 | 0.6720 | 0.5182 | 0.2983 |
| CoBEVT | 1 | 0.5809 | 0.4438 | 0.2746 |
| CoBEVT | 1.5 | 0.4427 | 0.3522 | 0.2538 |
| CoBEVT | 2 | 0.3855 | 0.3221 | 0.2544 |
| AttFuse | 0 | 0.9228 | 0.9182 | 0.8638 |
| AttFuse | 0.2 | 0.9211 | 0.9105 | 0.7758 |
| AttFuse | 0.4 | 0.9106 | 0.8616 | 0.5983 |
| AttFuse | 0.6 | 0.8752 | 0.7774 | 0.5002 |
| AttFuse | 0.8 | 0.8189 | 0.7034 | 0.4648 |
| AttFuse | 1 | 0.7699 | 0.6619 | 0.4533 |
| AttFuse | 1.5 | 0.6789 | 0.5974 | 0.4447 |
| AttFuse | 2 | 0.6359 | 0.5727 | 0.4481 |
| Where2comm | 0 | 0.9176 | 0.9087 | 0.7849 |
| Where2comm | 0.2 | 0.9157 | 0.9000 | 0.7117 |
| Where2comm | 0.4 | 0.9117 | 0.8703 | 0.5851 |
| Where2comm | 0.6 | 0.8925 | 0.8112 | 0.4695 |
| Where2comm | 0.8 | 0.8662 | 0.7579 | 0.4106 |
| Where2comm | 1 | 0.8415 | 0.7282 | 0.3894 |
| Where2comm | 1.5 | 0.7966 | 0.6976 | 0.3851 |
| Where2comm | 2 | 0.7756 | 0.6945 | 0.4106 |
| F-Cooper | 0 | 0.9032 | 0.9001 | 0.8402 |
| F-Cooper | 0.2 | 0.9015 | 0.8918 | 0.7117 |
| F-Cooper | 0.4 | 0.8886 | 0.8259 | 0.4445 |
| F-Cooper | 0.6 | 0.8372 | 0.6837 | 0.2858 |
| F-Cooper | 0.8 | 0.7461 | 0.5457 | 0.2037 |
| F-Cooper | 1 | 0.6640 | 0.4681 | 0.1788 |
| F-Cooper | 1.5 | 0.4895 | 0.3463 | 0.1506 |
| F-Cooper | 2 | 0.3987 | 0.2934 | 0.1469 |
| V2VAM | 0 | 0.9426 | 0.9383 | 0.8975 |
| V2VAM | 0.2 | 0.9426 | 0.9320 | 0.8006 |
| V2VAM | 0.4 | 0.9342 | 0.8711 | 0.5702 |
| V2VAM | 0.6 | 0.8953 | 0.7530 | 0.4109 |
| V2VAM | 0.8 | 0.8316 | 0.6480 | 0.3385 |
| V2VAM | 1 | 0.7725 | 0.5923 | 0.3181 |
| V2VAM | 1.5 | 0.6853 | 0.5388 | 0.3410 |
| V2VAM | 2 | 0.6553 | 0.5422 | 0.3741 |

### Caveats, in descending order of how much they could matter

**1. RESOLVED. AlignFormer's detector used to be short-sighted where the
baselines are not.** Until task 18 AlignFormer's late-fusion detector ran at
`cav_lidar_range = [-70.4, -40, -3, 70.4, 40, 1]` while every baseline ran at
+/-140.8 m in x, against a shared evaluated `GT_RANGE` of +/-140 m. 9.9% of the
evaluated ground truth lies beyond |x| = 70.4 m, and no agent could report a box
past that range in its own frame; measured, clean recall at IoU 0.7 was capped
at **0.894** against the widened detector's **0.916**. **The detector was retrained at +/-140.8 m and every AlignFormer row
in this section is from that run**; see [The detector range
fix](#the-detector-range-fix-704-m---1408-m). The handicap is gone, and the
clean-AP deficit that remains (0.8415 against CoAlign's 0.8831, CoBEVT's
0.8773, AttFuse's 0.8638 and V2VAM's 0.8975) is now a property of late fusion
and of AlignFormer's own sigma = 0 correction cost, not of a configuration
mismatch.

**2. Every baseline was evaluated under noise it may not have been trained
under -- which is the standard benchmark protocol, and is still worth saying.**
V2X-ViT's and Where2comm's stored configs carry
`wild_setting: {loc_err: true, xyz_std: 0.2, ryp_std: 0.2, async: true}`, so
those two were plausibly trained (and certainly evaluated by their authors)
with localization noise; CoAlign, CoBEVT, AttFuse, F-Cooper and V2VAM carry no
noise setting at all. That setting is **disabled** here for every method, so
the only noise in the sweep is ours. The consequence is asymmetric and runs
against the baselines: a method trained clean and tested noisy is not being
shown at its best. V2X-ViT's strong showing is consistent with it being the one
that was.

**3. "CoAlign" here is CoAlign's fusion, not CoAlign's robustness mechanism.**
`external/OpenCOOD/opencood/models/point_pillar_coalign.py` states in its own
header that it contains the multiscale intermediate feature fusion **only**,
and that the agent-object pose graph -- which is the part of CoAlign that
corrects pose error -- is not included. This row should not be read as a
measurement of CoAlign's published method. It is the most direct competitor on
paper and the one this comparison is least able to represent.

**4. The two fusion families have different ground-truth conventions, and one
had to be chosen.** `generate_object_center` filters objects against
`GT_RANGE` in the *reference agent's* frame and in 3D, so late fusion (each CAV
referenced to itself) and intermediate fusion (every CAV referenced to the ego)
admit different object sets wherever agents differ in heading or elevation --
on **363 of 2170 frames**, 33,089 boxes against 32,604. Every number in the
tables above uses the **late-fusion** set, which is the convention every AP in
this project has always been measured on, and
`baselines.late_fusion_convention_ground_truth` was verified to reproduce
`LateFusionDataset`'s own ground truth exactly on 61 sampled frames. The size
of the thumb this puts on the scale is measured rather than assumed:

| Baseline | frames where the two GT sets differ | AP@0.7 sigma 0 (late-fusion GT) | AP@0.7 sigma 0 (native GT) | AP@0.7 sigma 1 (late-fusion GT) | AP@0.7 sigma 1 (native GT) |
|---|---:|---:|---:|---:|---:|
| V2X-ViT | 363 | 0.8174 | 0.8340 | 0.6027 | 0.6150 |
| CoAlign (fusion only) | 363 | 0.8831 | 0.9012 | 0.5028 | 0.5108 |
| CoBEVT | 363 | 0.8773 | 0.8960 | 0.2746 | 0.2799 |
| AttFuse | 363 | 0.8638 | 0.8812 | 0.4533 | 0.4609 |
| Where2comm | 363 | 0.7849 | 0.8003 | 0.3894 | 0.3973 |
| F-Cooper | 363 | 0.8402 | 0.8559 | 0.1788 | 0.1816 |
| V2VAM | 363 | 0.8975 | 0.9132 | 0.3181 | 0.3233 |

Under the intermediate convention every baseline gains 0.008 to 0.019
AP@0.7 while AlignFormer gains only 0.001 to 0.007, so the choice made here is
worth about **0.010 in AlignFormer's favour**. At +/-70.4 m that was enough to
decide exactly one crossover -- sigma = 0.4 m against V2X-ViT. At +/-140.8 m it
decides two different ones, both at sigma = 0. The same sweep, emitted under
both conventions:

| sigma (m) | AlignFormer (late-fusion GT) | AlignFormer (intermediate GT) | V2X-ViT (late-fusion GT) | V2X-ViT (intermediate GT) | F-Cooper (late-fusion GT) | F-Cooper (intermediate GT) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.8415 | 0.8422 | 0.8174 | 0.8340 | 0.8402 | 0.8559 |
| 0.2 | 0.7743 | 0.7770 | 0.7894 | 0.8054 | 0.7117 | -- |
| 0.4 | 0.7717 | 0.7764 | 0.7319 | 0.7467 | 0.4445 | -- |
| 0.6 | 0.7717 | 0.7770 | 0.6732 | 0.6872 | 0.2858 | -- |
| 0.8 | 0.7804 | 0.7858 | 0.6250 | 0.6378 | 0.2037 | -- |
| 1 | 0.7771 | 0.7830 | 0.6027 | 0.6150 | 0.1788 | 0.1816 |
| 1.5 | 0.7728 | 0.7783 | 0.5675 | 0.5788 | 0.1506 | -- |
| 2 | 0.7441 | 0.7495 | 0.5515 | 0.5623 | 0.1469 | -- |

**What the convention decides, and what it does not.**

- **sigma = 0 against V2X-ViT: not robust.** +0.0241 under the late-fusion
  ground truth, +0.0082 under the intermediate one. Inside the band. Read it as
  a tie.
- **sigma = 0 against F-Cooper: not robust, and it flips.** +0.0013 under the
  late-fusion ground truth, **-0.0137** under the intermediate one. Read it as
  a tie.
- **sigma = 0.4 m against V2X-ViT: now robust.** +0.0398 and +0.0297, both well
  outside the band -- at +/-70.4 m this was the coin-flip row (+0.009 / -0.0005)
  and the reason the previous version of this document quoted 0.6 m. It no
  longer needs that qualifier.
- Every sigma from 0.6 m up against every baseline: margins of +0.099 upward,
  never in question.

**So the claim that survives both conventions is: AlignFormer ties V2X-ViT and
F-Cooper on the clean row, loses to V2X-ViT at sigma = 0.2 m, and leads the
entire field from sigma = 0.4 m upward.** The earlier "ties V2X-ViT at 0.4 m,
leads the field from 0.6 m" is superseded, and only by the detector range fix
-- no model, loss or hyperparameter changed.

**5. Checkpoint provenance cannot be fully verified.** These are third-party
weights already present on this machine. Several of their stored `config.yaml`
files name the *test* split as `root_dir`, which is what OpenCOOD's own
`inference.py` writes back, but it means training provenance cannot be
established from the files alone. The clean-AP column should be read with that
in mind; the robustness column, which is about degradation from each method's
own clean score, is less exposed to it.

**6. One noise seed.** Each cell is a single draw, as in the rest of this
document's AP tables. The pose sweep uses three; the AP sweeps use one.

### Provenance

| Method | Checkpoint | Trained with pose noise? | Notes |
|---|---|---|---|
| V2X-ViT | `net_epoch60.pth` | yes -- {'loc_err': True, 'xyz_std': 0.2, 'ryp_std': 0.2} | OpenCOOD point_pillar_transformer (V2X-ViT). Its own config carries wild_setting async=True loc_err=True xyz_std=0.2 ryp_std=0.2, i.e. it was trained/evaluated under OpenCOOD's noisy setting; that setting is disabled for this sweep. |
| CoAlign (fusion only) | `net_epoch15.pth` | no | OpenCOOD point_pillar_coalign. The model file states in its header that it contains CoAlign's multiscale intermediate feature fusion ONLY, not the agent-object pose graph that is CoAlign's pose-error correction. This row is therefore CoAlign's fusion, not CoAlign's robustness mechanism. |
| CoBEVT | `net_epoch19.pth` | no | OpenCOOD point_pillar_cobevt, no-compression variant. |
| AttFuse | `latest.pth` | no | OpenCOOD point_pillar_intermediate (AttFuse / the OPV2V paper's attentive fusion). |
| Where2comm | `net_epoch50.pth` | yes -- {'loc_err': True, 'xyz_std': 0.2, 'ryp_std': 0.2} | OpenCOOD point_pillar_where2comm. Its config carries wild_setting async=True loc_err=True; disabled for this sweep. |
| F-Cooper | `latest.pth` | no | OpenCOOD point_pillar_fcooper (maxout spatial fusion). |
| V2VAM | `latest.pth` | no | OpenCOOD point_pillar_intermediate_V2VAM, no-compression. |

### Baselines that could not be run, and why

Three rows are missing, and the reason is the same in each case: the model code
is not in this repository's OpenCOOD submodule, and `external/OpenCOOD` is not
ours to modify.

| Checkpoint | `model.core_method` | Why not run |
|---|---|---|
| `ermvp` | `point_pillar_ermvp` | No `opencood/models/point_pillar_ermvp.py` in this fork. Its config also sets `max_cav: 2`, below the 5 in-range CAVs this split reaches. |
| `comamba` | `point_pillar_opv2v_comamba` | No such module in this fork. |
| `pointpillar_mamba` | `point_pillar_mamba_simple` | No such module in this fork. |

V2VAM **is** in the table, but only after two dead imports in
`point_pillar_intermediate_V2VAM.py` were satisfied in memory:
`opencood.models.sub_modules.noise` does not exist in this fork and
`fuse_modules.self_attn` has been trimmed of `regroup`. Neither name is used
anywhere below its import line, so `baselines._shim_missing_v2vam_import` binds
stubs that **raise if called**, and the model's computation is untouched. The
result JSON records the shim.

### Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD

# One sweep per baseline (~35 min each alone on an RTX 4090; they are
# independent and were run concurrently).
for b in v2xvit coalign cobevt attfuse where2comm fcooper v2vam; do
  python -m embedding_aware_belt_fusion.alignformer.baselines --baseline $b \
    --output outputs/alignformer/baselines/${b}_result.json
done

# Bytes per frame per agent, at the +/-140.8 m detector. The baselines'
# figures are structural (tensor shapes) and reproduce task 17's to the byte;
# AlignFormer's side moves, because the wider detector transmits more objects.
python -m embedding_aware_belt_fusion.alignformer.bandwidth --frames 100 \
  --alignformer-cache /media/chenyi/basement2/cache/alignformer_r140/test \
  --output outputs/alignformer/r140/bandwidth_r140_result.json

# AlignFormer's own sweep already emits BOTH ground-truth conventions, so
# caveat 4 is measured rather than argued -- p2_r140_noisy_ap_result.json
# carries `ap` (late-fusion convention, the headline) and
# `ap_intermediate_convention_gt` side by side, and is passed to the
# summarizer for both roles.
O=outputs/alignformer/r140

# Every table above is rendered from those JSONs
python scripts/summarize_alignformer_baselines.py \
  --alignformer $O/p2_r140_noisy_ap_result.json \
  --baselines outputs/alignformer/baselines/{v2xvit,coalign,cobevt,attfuse,where2comm,fcooper,v2vam}_result.json \
  --bandwidth $O/bandwidth_r140_result.json \
  --alignformer-gt-conventions $O/p2_r140_noisy_ap_result.json
```

## FreeAlign: the boxes-only competitor, reimplemented onto our late fusion

FreeAlign (Lei, Ni, Han, Tang, Wang, Feng, Chen, Wang; ICRA 2024;
arXiv 2405.02965) is the closest published competitor: it aligns agents from
**boxes only, with no localization prior**, by building a salient-object graph
per agent -- nodes are detected boxes, edges are relative distances, invariant
to the viewer's pose -- and matching common subgraphs. It became the most
important baseline once the association diagnostic established that
AlignFormer's appearance embedding contributes nothing to matching on OPV2V,
leaving object-level **geometry** as what both methods actually use.

**Every number below is a reimplementation of the published method, not a run
of the authors' code, and must never be quoted as theirs.** Their repository is
built on CoAlign's OpenCOOD fork with a different dataset, postprocessor and AP
implementation; it needs CoAlign's uncertainty detector (absent here), a
stage-1 precalc pass, source-built `g2opy` and Baidu-Drive checkpoints. Running
it end to end would mean training a detector from scratch and would still not
be apples-to-apples. `alignformer/freealign.py` ports the algorithm onto **our**
detections, **our** sweep and **our** evaluator instead, which isolates the
alignment algorithm -- the thing under test -- from detector and evaluator
differences. Both rows below come out of **one** `--metric noisy_ap`
invocation per split, so the detections, the noise draws, the fusion and the AP
are identical and the only thing that differs is the alignment algorithm.

**The pairing measured here is the one their paper does not report.** Every
FreeAlign result there sits on an intermediate-fusion backbone --
CoAlign+FreeAlign, V2X-ViT+FreeAlign, Where2comm+FreeAlign -- so the system
still transmits feature maps. AlignFormer's claim is robustness at *late-fusion*
bandwidth, so the honest comparison holds bytes fixed, and the row that does
that is **late fusion + FreeAlign**.

### The headline: FreeAlign wins AP@0.7 and AlignFormer wins AP@0.3

Test split, 2170 frames, global-sorted AP, shrinkage as deployed on the
AlignFormer arm and none on FreeAlign (it has no such step).

| sigma (m) | uncorrected | AlignFormer | FreeAlign | AP@0.7 delta |
|---|---:|---:|---:|---:|
| 0 | 0.8964 | 0.8310 | **0.8461** | -0.0151 |
| 0.2 | 0.5978 | 0.7841 | **0.8334** | -0.0492 |
| 0.4 | 0.3158 | 0.7835 | **0.8164** | -0.0329 |
| 0.6 | 0.2154 | 0.7819 | **0.8094** | -0.0276 |
| 0.8 | 0.1844 | 0.7850 | **0.8069** | -0.0219 |
| 1.0 | 0.1763 | 0.7851 | **0.8054** | -0.0203 |
| 1.5 | 0.1799 | 0.7756 | **0.8048** | -0.0291 |
| 2.0 | 0.1898 | 0.7600 | **0.8047** | -0.0448 |

Negative delta means FreeAlign is ahead. **It is ahead at every sigma on
AP@0.7, by 0.015 to 0.049**, and it is essentially flat across the sweep
(0.8047-0.8461), which is exactly the pose-prior independence its paper claims.
That is reported first because it is the result that costs this project the
most.

At the thresholds FreeAlign's own paper reports, the picture reverses:

| sigma (m) | AP@0.3 AlignFormer | AP@0.3 FreeAlign | AP@0.5 AlignFormer | AP@0.5 FreeAlign |
|---|---:|---:|---:|---:|
| 0 | 0.9373 | 0.9377 | 0.9163 | **0.9287** |
| 0.2 | **0.9379** | 0.9376 | 0.9139 | **0.9277** |
| 0.4 | **0.9376** | 0.9368 | 0.9122 | **0.9169** |
| 0.6 | **0.9349** | 0.9309 | **0.9077** | 0.9043 |
| 0.8 | **0.9328** | 0.9233 | **0.9054** | 0.8983 |
| 1.0 | **0.9337** | 0.9163 | **0.9059** | 0.8929 |
| 1.5 | **0.9275** | 0.9074 | **0.8959** | 0.8879 |
| 2.0 | **0.9202** | 0.8999 | 0.8841 | **0.8857** |

**AlignFormer leads AP@0.3 at every sigma from 0.2 m up, by as much as 0.020,
and AP@0.5 through the middle of the sweep.** The two methods are not ordered:
which one wins depends on the IoU threshold, and any single-threshold headline
would be a choice of framing rather than a finding.

On validation FreeAlign's lead at AP@0.7 is wider and flatter still (0.8957
-0.8962 at every sigma against AlignFormer's 0.8506-0.8985), and AlignFormer
wins only the clean row.

### Why a method with five times the mean pose error wins AP@0.7

The pose numbers look incompatible with the AP table until the distribution is
looked at. Test split, sigma = 1.0 m, stride-8 diagnostic, 431 pairs, the same
detections for both (`scripts/calibrate_freealign.py --selected-only`):

| | FreeAlign | AlignFormer |
|---|---:|---:|
| translation **median** (m) | **0.109** | 0.116 |
| translation mean (m) | 1.317 | **0.233** |
| yaw **median** (deg) | **0.092** | 0.119 |
| yaw mean (deg) | 2.351 | **0.378** |
| coverage (pairs corrected at all) | 0.907 | **0.988** |
| error rate, \|dt\| > 3 m (their Table II metric) | 2.55% | **0.93%** |

**FreeAlign's median pose error is slightly better than AlignFormer's; its mean
is 5.7x worse, entirely because of a 2.55% catastrophic tail.** AP@0.7 responds
to the typical pair and rewards FreeAlign's noise-independence; MAE is dragged
by the tail and rewards AlignFormer. Both summaries are true and they point
opposite ways, which is why both are reported.

### The structural experiment: shared-object count

FreeAlign's evidence is pairwise *distances*, so one shared object is a 1-node
graph with **no edge**, two give a single scalar that fixes neither rotation nor
the reflection, and three are needed before a distance graph rigidly determines
SE(2). AlignFormer augments every object with heading virtual points
(`heading_lambda = 2.0` m), so a *single* matched object determines the full
SE(2). Averaged over the split that difference is diluted, so it is reported as
its own slice. Test split: 34 pairs share no object (1.0%), 300 share one or two
(8.7%), 3111 share three or more (90.3%); no frame straddles two slices.

AP@0.7, sigma = 1.0 m:

| slice | frames | pairs | uncorrected | AlignFormer | FreeAlign | AF coverage | FA coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 shared | 30 | 34 | 0.1173 | 0.1188 | 0.1173 | 0.118 | 0.059 |
| 1-2 shared | 290 | 300 | 0.3348 | **0.5185** | 0.3456 | 0.930 | 0.163 |
| 3+ shared | 1731 | 3111 | 0.1521 | 0.8119 | **0.8513** | 0.997 | 0.989 |

**The prediction holds, and the mechanism is coverage.** On the 1-2 slice
FreeAlign declines 84% of pairs -- its own safety rule firing exactly where the
spec said it would -- and gains +0.011 AP@0.7 over doing nothing, while
AlignFormer answers 93% of them and gains +0.184. On the 3+ slice, which is 90%
of pairs and therefore the aggregate, FreeAlign wins.

The pose diagnostic says the same thing more sharply. Of the sparse pairs
FreeAlign *does* answer, **60% are wrong by more than 3 m** (median 3.32 m)
against AlignFormer's 10.3% (median 0.454 m); of the no-shared-object pairs it
answers (25% of them) the mean error is **113 m**, while AlignFormer declines
all of them. On the 3+ slice the two are level on the median (0.099 m against
0.103 m) and FreeAlign's tail is 1.29% against AlignFormer's 0.00%.

### The port is not a strawman, and here is the evidence

A reimplementation that loses can always be dismissed, so the port was verified
where it *should* succeed before any loss was reported.

- **Validation, 234 pairs, sigma = 1.0 m: 0.0994 m mean / 0.0818 m median
  translation, 0.113 deg mean yaw, coverage 1.000, error rate 0.00%.** The
  paper reports 0.266 m / 0.017 deg / 0.56% on OPV2V for the full model and
  0.283 m / 0.029 deg / 0.78% for the anchor-based matching *without* GNN edge
  features, which is the path ported here. The port's translation is better
  than either; its yaw is worse; its error rate is zero on this slice.
- **Test, 3+ shared objects: error rate 1.29% against their no-GNN 0.78%** --
  a factor of 1.65 on a harder pair population, which is the right regime.
- On validation it also **beats the deployed AlignFormer** on the same pairs
  (0.0994 m against 0.124 m mean, 0.082 m against 0.094 m median).
- The parameters the paper leaves open were chosen by a 192-point grid on the
  validation slice, and **two of them came back as the authors' own shipped
  values** (`max_error = 0.5` m, `min_nodes = 3`).

### What is ported, what is not, and every deviation

| Their component | Here |
|---|---|
| Salient-object graph, fully connected, one node per box (Section IV-A) | ported |
| Edge feature = EdgeGAT over the relative-distance matrix, contrastive loss (eq. 2) | **not ported.** Their Table VI prices anchor-based matching without GNN features on OPV2V at 0.029 deg / 0.283 m / 0.78% against 0.017 / 0.266 / 0.56% with them, and their released configs set `gnn: false`, so this is the path their own code ships |
| MASS, all four steps: n x m anchor init, anchor-list expansion to gamma, subgraph growth, minimal-eps selection (Section IV-B) | ported |
| Relative pose by RANSAC or LMedS over the matched point sets (Section IV-C) | ported. The port carries its own unweighted least-squares core so that `MIN_MATCH_MASS`, heading virtual points and the inverse-variance weighting cannot leak into the competitor; a test pins that this core agrees with `procrustes.weighted_se2_kabsch` at uniform weights |
| Discard the message below a minimum common-subgraph node count | ported; a discarded message is fused uncorrected, exactly as our own fallback does, which is what makes the two coverage figures comparable |
| Clock-deviation estimation (their title's "and Clock Devices") | out of scope: this sweep perturbs pose only and the frames are synchronous. **Their problem is strictly larger than ours** |

Deviations beyond the two above, in full:

- **The paper and the authors' code disagree on the edge attribute, and the
  port follows the paper -- which is also the better of the two here.** Section
  V-C says that absent the GNN "edge matching is determined by the relative
  distance between two nodes"; their `greedy_match.py` builds a 2-channel edge
  of relative distance *and relative yaw*. Both were measured on validation:
  distance alone gives 0.0994 m, distance-and-yaw 0.2903 m. The detector here
  reports a box's *axis*, not its direction -- 20.3% of cross-agent detections
  of the same object disagree by ~180 degrees -- so a yaw channel imports
  exactly the ambiguity a distance-only graph is immune to. **That immunity is a
  real advantage of their design** and it is kept.
- **The selection score's constant is load-bearing and is taken from their
  code.** The paper's step (iv) is `eps = (1/r^p) sum_e eps_e` with `p` "a
  tunable hyperparameter"; their `get_best_match` seeds the accumulator at 100
  and divides by the match count, making the selection "largest common subgraph
  first, discrepancy as the tie-break" -- which is also what Section IV-B's own
  phrase "approximate **maximum** common subgraph" says. With the constant at
  zero, a three-node coincidence outscores the true eleven-node subgraph and the
  port degrades to 19 m of error; that was observed here before the constant was
  restored, and it is the single likeliest way a naive port turns into a
  strawman.
- Relative yaw in the 2-channel variant is wrapped to `(-pi, pi]`; the shipped
  code subtracts raw yaws, which is not invariant across the branch cut.
- RANSAC/LMedS enumerate **all** minimal (2-correspondence) samples rather than
  drawing random ones, capped at 512 by even subsampling. At the subgraph sizes
  seen here that is exhaustive, hence stronger than random sampling, and it is
  deterministic.
- Ties in the greedy steps break by smallest discrepancy, then by index; the
  paper specifies no order.
- `gamma` (3), the epsilon exponent (1.0) and the RANSAC inlier threshold
  (1.0 m) have no published value and were chosen on validation.

### Two facts the earlier spec asserted that this measurement corrects

- The spec records "**18.5% of ego-CAV pairs share no object**" and "~7.8% share
  exactly one or two", measured over the ROI cache. On the evaluated test split
  those figures are **1.0% and 8.7%**, and on validation **0.0% and 1.8%**. The
  sparse slice is real and it is where AlignFormer's structural advantage shows
  up, but the no-shared-object population is an order of magnitude smaller than
  the spec claimed and the argument should not lean on it.
- The spec notes that FreeAlign does not use heading and calls that a genuine
  advantage. That is confirmed and quantified above: adding the heading channel
  the authors' own code carries makes the port **three times worse** on this
  detector.

### Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
O=outputs/alignformer/r140
DET=configs/alignformer_detector_r140.yaml
AF=configs/alignformer_r140.yaml
VAL=/media/chenyi/basement2/cache/opv2v_splits/val
TEST=/media/chenyi/Elements1/Dataset/OPV2V/test
CKPT=$O/stage2_B_ivw_scalar/best.pth
SHRINK=$O/shrinkage_ivw_scalar_calibration_result.json
E="python -m embedding_aware_belt_fusion.alignformer.evaluate"

# 1. Choose the port's free parameters on VALIDATION only (~25 min).
python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF \
  --split $VAL --stride 8 --sigma 1.0 \
  --output $O/freealign_calibration_result.json

# 2. One evaluator pass per split produces BOTH rows (~25 min val, ~95 min test).
for SPLIT_NAME in val test; do
  [ $SPLIT_NAME = val ] && SPLIT=$VAL || SPLIT=$TEST
  $E --config $DET --split $SPLIT --metric noisy_ap --alignformer-config $AF \
     --checkpoint $CKPT --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
     --shrinkage $SHRINK --freealign \
     --output $O/freealign_${SPLIT_NAME}_result.json
done

# 3. Median and their own >3 m error rate, paired, to explain the AP/MAE
#    inversion. This is a MEASUREMENT, so it may point at test (~6 min each).
for SPLIT_NAME in val test; do
  [ $SPLIT_NAME = val ] && SPLIT=$VAL || SPLIT=$TEST
  python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF \
    --split $SPLIT --stride 8 --sigma 1.0 --selected-only \
    --checkpoint $CKPT --shrinkage $SHRINK \
    --output $O/freealign_pose_diagnostic_${SPLIT_NAME}_result.json
done

# 4. Every table above is rendered from those JSONs
python scripts/summarize_freealign.py \
  --test $O/freealign_test_result.json --validation $O/freealign_val_result.json \
  --calibration $O/freealign_calibration_result.json \
  --output $O/freealign_result.json
```

The AlignFormer, uncorrected and oracle rows of this run reproduce
`p2_r140_ivw_scalar_noisy_ap_result.json` exactly at every sigma and every IoU
threshold, so adding the condition changed nothing about the deployed arm.

## The robust re-weighted solve: closing the AP@0.7 gap to FreeAlign

Task 20 left the comparison unordered: FreeAlign won AP@0.7 at every sigma and
AlignFormer won AP@0.3, and the pose diagnostic said why. **It was not the
tail.** On the 90.3% of test pairs that share three or more objects -- the
slice that drives aggregate AP -- FreeAlign's answered translation median was
0.0984 m against AlignFormer's 0.1024 m, and its yaw median 0.0747 deg against
0.1007. AlignFormer owned the mean (0.2203 m against 1.2984 m), the tail (0.93%
beyond 3 m against 2.55%) and the coverage (0.988 against 0.907), but AP@0.7
rewards the *typical* pair and the typical pair went to FreeAlign.

The only structural difference in that regime is how the pose is solved.
FreeAlign fits **hard**: LMedS over a matched subgraph, where an outlier
correspondence is discarded. Head B fits **once**, by weighted least squares
over soft Sinkhorn mass times inverse-variance precision -- and least squares
has no redescending influence function, so a wrong correspondence with
small-but-nonzero mass biases the fit in proportion to its residual and nothing
ever removes it.

`alignformer/robust.py` is the intervention that difference suggests: solve
exactly as now, compute per-point residuals over the 2M heading-augmented
points, re-weight, re-solve, a bounded number of times. It **wraps**
`weighted_se2_kabsch`; it does not replace it. It is applied at **inference
only**, on the deployed `stage2_B_ivw_scalar/best.pth`; nothing was retrained.

Four things about the design are load-bearing.

**The scale is fitted per sample, not fixed in metres.** A fixed threshold
cannot transfer across a sweep whose pose error spans 0 to 2 m. It is the
mass-weighted median residual divided by `sqrt(2 ln 2)` (the Rayleigh
median/sigma ratio, so Huber's 1.345 keeps its textbook meaning), floored at
0.05 m -- well below the detector's own 0.2516 m per-detection RMS, so nothing
real is trimmed by the floor. The weighted median is the same statistic
FreeAlign's LMedS selects on. It is re-estimated from the **original** weights
every iteration; re-estimating from the already-trimmed ones makes each round
trim harder than the last.

**The guard.** Below `min_evidence` effective correspondences the loop is
skipped and the single solve is returned untouched, per sample. AlignFormer's
structural advantage over a distance-graph method lives in the pairs sharing
one or two objects, and a robust loop handed three effective points will trim
its way to nonsense. The threshold was chosen on validation and is 3.0 -- the
count below which a relative-distance graph is degenerate, i.e. the boundary
the `shared_1_2` slice is drawn at.

**The weights are renormalized** to the total the input vector had, exactly as
`variance.augmented_weights` does, so `MIN_MATCH_MASS` fires on exactly the
pairs it fired on before and the loop cannot change *which* pairs are answered.

**The residual weights are detached**, so the autograd path is still one
weighted Kabsch rather than an unrolled fixed point. Pinned by gradient
equality, not by inspection.

### Choosing psi, n_irls and the guard -- validation only

A full sweep costs ~1.5 h per configuration, so a 30-point grid was run on pose
error instead, at four sigmas on the scenario-disjoint validation slice, stride
8, 234 pairs (`scripts/calibrate_robust_solve.py`). Mean change in translation
median, best rows:

| psi | n_irls | guard | mean delta median (m) |
|---|---:|---:|---:|
| **Huber** | **2** | **3.0** | **-0.0095** |
| Huber | 3 | 3.0 | -0.0094 |
| Huber | 1 | 3.0 | -0.0080 |
| Geman-McClure | 1 | 3.0 | -0.0054 |
| Geman-McClure | 3 | 0.0 | +0.0010 |

Huber beats Geman-McClure on every axis, and Geman-McClure at three iterations
is *worse than doing nothing*: it is redescending, so it trims harder, and the
bounded-influence function is the right one on this data. `huber`,
`n_irls = 2`, `min_evidence = 3.0` was frozen before either AP sweep ran.

### The headline: AP@0.7 improves at every sigma, on both splits

Test split, 2170 frames, global-sorted AP, shrinkage as deployed on both
AlignFormer arms and none on FreeAlign.

| sigma (m) | uncorrected | AlignFormer | **+ robust solve** | FreeAlign | gain |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8964 | 0.8310 | **0.8512** | 0.8461 | +0.0202 |
| 0.2 | 0.5978 | 0.7841 | 0.8060 | **0.8334** | +0.0219 |
| 0.4 | 0.3158 | 0.7835 | 0.8114 | **0.8164** | +0.0280 |
| 0.6 | 0.2154 | 0.7819 | **0.8099** | 0.8094 | +0.0281 |
| 0.8 | 0.1844 | 0.7850 | **0.8151** | 0.8069 | +0.0301 |
| 1.0 | 0.1763 | 0.7851 | **0.8164** | 0.8054 | +0.0313 |
| 1.5 | 0.1799 | 0.7756 | **0.8106** | 0.8048 | +0.0350 |
| 2.0 | 0.1898 | 0.7600 | 0.7995 | **0.8047** | +0.0396 |

Mean gain **+0.0293**, at 8 of 8 sigmas. **The AP@0.7 head-to-head against
FreeAlign goes from 0-8 to 5-3.** At AP@0.3 it is now **8-0** in our favour and
at AP@0.5 **6-2**, so task 20's "which one wins is a choice of IoU threshold"
no longer holds in FreeAlign's favour. The three cells FreeAlign still takes
are sigma 0.2 (by 0.027) and 0.4 and 2.0 (by 0.005 each) -- and 0.005 is below
what one noise seed per AP cell resolves, so only the 0.2 m cell is a real
ordering.

Validation, 958 frames: the gain is +0.0126 to +0.0360, again at 8 of 8 sigmas,
mean +0.0224.

**Nothing else moved.** The `uncorrected`, `alignformer`, `freealign` and
`oracle` rows of both runs are bit-identical to the published
`freealign_{val,test}_result.json` at all three IoU thresholds -- 75 cells per
split, max |difference| **0.0**, on both splits.

### The mechanism, measured: the dense-regime deficit is closed and reversed

Paired diagnostic, test, 431 pairs, sigma = 1.0 m, identical detections:

| | FreeAlign | AlignFormer | **+ robust solve** |
|---|---:|---:|---:|
| answered translation median | **0.0992 m** | 0.1147 m | 0.1013 m |
| answered translation mean | 1.2984 m | 0.2203 m | **0.2064 m** |
| **3+ slice answered median** | 0.0984 m | 0.1024 m | **0.0932 m** |
| 3+ slice answered yaw median | **0.0747 deg** | 0.1007 deg | 0.0793 deg |
| >3 m error rate | 2.55% | **0.93%** | **0.93%** |
| coverage | 0.907 | **0.988** | **0.988** |

**On the slice that drives aggregate AP we are now the more precise method in
the typical case**, which is the exact quantity that cost us AP@0.7, and we get
there without giving up the tail or the coverage. The overall answered median
is still marginally behind FreeAlign's, because FreeAlign declines the sparse
pairs that drag ours; conditioned on the pairs both answer, we are ahead.

### The guard works, and it is selective rather than global

A guard set too high would disable the loop everywhere, reproduce the deployed
arm exactly, and read as a clean null. It does not:

| slice | pairs | loop engaged | pose actually moved |
|---|---:|---:|---:|
| all | 431 | 88.2% | 87.9% |
| 1-2 shared | 39 | **5.1%** | 5.1% |
| 3+ shared | 388 | **97.4%** | 97.2% |

AP@0.7 by slice, sigma = 1.0 m, test:

| slice | frames | pairs | uncorrected | AlignFormer | **+ robust** | FreeAlign |
|---|---:|---:|---:|---:|---:|---:|
| 0 shared | 30 | 34 | 0.1173 | 0.1188 | 0.1188 | 0.1173 |
| 1-2 shared | 290 | 300 | 0.3348 | 0.5185 | **0.5191** | 0.3456 |
| 3+ shared | 1731 | 3111 | 0.1521 | 0.8119 | **0.8478** | 0.8513 |

The `shared_0` slice is **exactly unchanged at every sigma** -- the loop never
engages where there is no evidence. The sparse slice moves by +0.0007, +0.0000,
+0.0006, -0.0001, +0.0012, +0.0006, +0.0012, +0.0012 across the sweep, and its
pose MAE is unchanged to three decimals. **The structural advantage on sparse
pairs is preserved, not traded away.**

### The clean case

The sigma = 0 regression (ruling R44) is reduced by a third without being
targeted:

| | AP@0.7 at sigma = 0 | damage against uncorrected |
|---|---:|---:|
| uncorrected | 0.8964 | -- |
| AlignFormer | 0.8310 | -0.0654 |
| FreeAlign | 0.8461 | -0.0503 |
| **AlignFormer + robust solve** | **0.8512** | **-0.0452** |

We now damage the clean case **less than FreeAlign does**, which was not true
before. On validation the damage falls from -0.0214 to -0.0089. The mechanism is
visible in the pose: the sigma = 0 residual, which *is* the error there, falls
from 0.1610 to 0.1485 m overall and from 0.0555 to 0.0416 m on the dense slice.
This is a side effect, not a fix; R44's recommendation of a pose-uncertainty
gate stands.

### Caveats

- **One noise seed per AP cell.** The robust-minus-deployed deltas
  (0.020-0.040) clear that comfortably; the robust-minus-FreeAlign deltas at
  sigma 0, 0.4, 0.6 and 2.0 (0.0005-0.0052) do **not** and must not be quoted
  as orderings.
- `min_evidence` is in units of Sinkhorn soft-match mass, not the number of
  *true* shared objects, which is why 5% of the `shared_1_2` slice still
  engages. The guard and the slice boundary are not the same cut.
- Huber and Geman-McClure are separated by 0.004 m of median on 234 validation
  pairs. The choice is supported, not decisive.
- The robust arm is scored through shrinkage calibrated on the **deployed**
  arm's residuals. A recalibration would probably help it further and was
  deliberately not tried, because it would be a second intervention.
- FreeAlign is still ported without EdgeGAT, which their own ablation prices at
  ~1.4x on the error rate, so a full-model FreeAlign would narrow these margins
  by an unmeasured amount.

### Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
O=outputs/alignformer/r140
DET=configs/alignformer_detector_r140.yaml
AF=configs/alignformer_r140.yaml
VAL=/media/chenyi/basement2/cache/opv2v_splits/val
TEST=/media/chenyi/Elements1/Dataset/OPV2V/test
CKPT=$O/stage2_B_ivw_scalar/best.pth
SHRINK=$O/shrinkage_ivw_scalar_calibration_result.json
E="python -m embedding_aware_belt_fusion.alignformer.evaluate"

# 1. Choose psi, n_irls and the guard on VALIDATION only (~40 min).
python scripts/calibrate_robust_solve.py --config $DET --alignformer-config $AF   --split $VAL --checkpoint $CKPT --shrinkage $SHRINK   --stride 8 --sigma 0 0.4 1.0 2.0   --output $O/robust_solve_calibration_result.json

# 2. One evaluator pass per split produces ALL five arms (~1.5 h val, ~3.5 h test).
for SPLIT_NAME in val test; do
  [ $SPLIT_NAME = val ] && SPLIT=$VAL || SPLIT=$TEST
  $E --config $DET --split $SPLIT --metric noisy_ap --alignformer-config $AF      --checkpoint $CKPT --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0      --shrinkage $SHRINK --freealign      --robust-solve huber --robust-iterations 2 --robust-min-evidence 3.0      --output $O/robust_solve_${SPLIT_NAME}_result.json
done

# 3. Paired medians and engagement. A MEASUREMENT, so it may point at test.
for SPLIT_NAME in val test; do
  [ $SPLIT_NAME = val ] && SPLIT=$VAL || SPLIT=$TEST
  python scripts/calibrate_robust_solve.py --config $DET --alignformer-config $AF     --split $SPLIT --checkpoint $CKPT --shrinkage $SHRINK     --stride 8 --sigma 1.0 --measure-only     --mode huber --iterations 2 --min-evidence 3.0     --output $O/robust_solve_pose_diagnostic_${SPLIT_NAME}_result.json
done

# 4. The pre-registered decision rule and every table above.
python scripts/summarize_robust_solve.py   --validation $O/robust_solve_val_result.json   --test $O/robust_solve_test_result.json   --deployed $O/freealign_test_result.json   --deployed-validation $O/freealign_val_result.json   --calibration $O/robust_solve_calibration_result.json   --output $O/robust_solve_result.json
```

## Error bars: what the AP head-to-head actually supports

Every AP cell above this section was **one noise seed**. That was never a
problem while the differences read off them were 0.020-0.040 wide -- the IRLS
arm's gain over the deployed arm, for instance. It became the blocking problem
for the comparison against FreeAlign, which at AP@0.7 turns on differences of
0.0005-0.011. Task 22's own limitations section said so.

The sweep was re-run under **five independent noise seeds** per sigma, on both
splits, all five conditions, on the full eight-sigma grid.

**The comparison is paired, and that is not a detail.** Every condition inside
one draw sees the *same* perturbed poses: `draws.sweep_noisy_poses` is called
once per (sigma, seed, frame) and its result reaches `uncorrected` and every
estimator through one `noisy_detections` object (`draws.draw_aligned`). The
statistic is the **per-seed difference**; its standard error is the sample
standard deviation of those differences over `sqrt(5)`.

**A difference no larger than its own standard error is a draw** -- written as
one in the direction that flatters this project as readily as in the one that
does not. `alignformer/seedstats.py` is the single place that rule lives.

**sigma = 0 has no draw in it.** It perturbs nothing, so every seed would
produce byte-identical predictions; it is run once and reported as
`deterministic`. A standard deviation of 0.0 there would be a fabricated error
bar on the one cell the clean-case damage is read from.

**One standard error is a weak bar, and these tables say which bar they used.**
With five seeds, "exceeds its own standard error" is |t| > 1 on 4 degrees of
freedom -- roughly p = 0.19 two-sided, not significance at any conventional
level. A 95% interval would need |t| > 2.78. Every per-seed value is in the
result JSON so a stricter reading can be applied without re-running anything.
Where a verdict below rests on |t| barely above 1, it is flagged.

**How much the pairing bought, measured rather than assumed.** The premise is
that the conditions are strongly correlated through the shared draw. On
validation AP@0.7 that is true of our own two arms -- the naive independent
standard error is 2.0-3.1x the paired one at sigma 0.2-0.8, because the two
AlignFormer arms share a checkpoint, a correspondence and every weight but the
solve. It is **false** of the comparison against the competitor: FreeAlign's AP
is almost invariant to the draw (sd 0.0001-0.0002 against 0.0014-0.0031 for
ours), so there is no covariance to recover and pairing buys 0.96-1.12x. It
changes no verdict, but "pairing tightens the FreeAlign comparison" would have
been false and is not claimed.

### What the error bars support, and what they retract

Both splits reproduce the published single-seed runs exactly at seed 0 --
**198 cells each, max |difference| = 0** -- so what follows is a sharper
reading of the same experiment, not a different one.

**Survives, overwhelmingly.** The IRLS loop's win over the deployed arm, at
every threshold on both splits, 7 of 7 drawn sigmas each:

| sweep-mean paired difference | validation | test |
|---|---:|---:|
| AP@0.3 | +0.0008 ±0.0001 | +0.0034 ±0.0001 |
| AP@0.5 | +0.0054 ±0.0002 | +0.0117 ±0.0001 |
| AP@0.7 | +0.0228 ±0.0002 | +0.0299 ±0.0004 |

**Survives.** AP@0.3 and AP@0.5 against FreeAlign are ours: +0.0026 ±0.0001 and
+0.0032 ±0.0001 on validation, +0.0123 ±0.0001 and +0.0116 ±0.0002 on test.

**Does not survive: "the robust solve wins the AP@0.7 head-to-head."** The
section above reports that the IRLS loop "turns the AP@0.7 head-to-head from
0-8 into 5-3" on test. The **cell count reproduces exactly**, and the cell
count is the wrong statistic. Test AP@0.7, IRLS minus FreeAlign:

| sigma | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| delta | +0.0051 | **-0.0235** | -0.0076 | +0.0030 | +0.0071 | +0.0097 | +0.0065 | -0.0036 |

Five cells ours worth **+0.0314**; three cells theirs worth **-0.0347**. The
sigma = 0.2 cell alone outweighs every cell we win. On the sweep mean the sign
is negative on **both** splits -- **-0.0035 ±0.0003 on validation (|t| = 11.4)
and -0.0004 ±0.0002 on test (|t| = 2.4)**. The two splits were thought to
disagree; with error bars they agree in sign and differ only in size.

The test margin is one to flag rather than lean on: |t| = 2.4 clears the
one-standard-error bar this document uses but not a 95% interval. The honest
statement is that **at AP@0.7 the loop closes almost all of a gap that was 0-8
against us and wins the middle of the sweep outright (sigma 0.6-1.5, four
cells, |t| = 3.3 to 13.9) -- and does not win the head-to-head.**

**Also retracted.** FreeAlign against the *deployed* AlignFormer at AP@0.5 on
test is **+0.0001 ±0.0002, |t| = 0.7 -- a draw**, 3-4 per sigma. The earlier
section quotes it as an ordering; it is not one.

**Where the structural advantage does survive, enormously.** Test, the 1-2
shared-object slice, AP@0.7, IRLS minus FreeAlign: **+0.1651 ±0.0032** at
sigma = 1.0 and **+0.1509 ±0.0035** at sigma = 2.0. A relative-distance graph is
degenerate below three shared objects; heading virtual points are not. The
slice is 9.7% of pairs, so it is invisible in the aggregate. At sigma = 0 the
same slice runs **-0.2042** the other way, which is ruling R44's clean-case
regression concentrated exactly where the evidence is thinnest.

### Shrinkage for the IRLS arm: the refit works and is still declined

`tau` was re-measured by the same procedure, on validation at sigma = 0, with
the IRLS solve in the loop: **0.1225 m** against the deployed **0.1510 m**
(yaw 0.1638 deg against 0.2212). The arm's residuals really are ~19% tighter,
so the deployed tau over-shrinks it.

It does what it was predicted to do, largest exactly where predicted -- AP@0.7,
validation, refitted minus deployed tau: **+0.0087 ±0.0005** at sigma = 0.2,
+0.0068 at 0.4, decaying to +0.0002 at 2.0; sweep mean **+0.0029 ±0.0002**.

And it is **not shipped**, because two of the rule's three criteria fail, on
two different splits. AP@0.7 at sigma = 0 regresses by **-0.0012**, larger than
any error bar in that comparison; and test's 1-2 shared slice regresses at
**all eight sigmas**, beyond one paired standard error at six of them, worst
-0.0064. A smaller tau shrinks less everywhere: at sigma = 0 the true
correction is the identity, so shrinking less does more damage, and the sparse
slice is where the evidence is thinnest and the shrinkage was doing the most
work. That is the trade the rule was written to refuse.

Shipping it would have shrunk the validation AP@0.7 deficit to FreeAlign from
-0.0035 to **-0.0006** -- a 5.5x improvement that is *still* a deficit at 2.9
standard errors. **The rule's output does not change the head-to-head verdict
on either split**, which is worth knowing about a rule that could otherwise be
suspected of deciding it.

The follow-up is not a third tau. The cost is confined to sigma = 0 and the
benefit to the low-noise end, which is the shape ruling R44 already identified:
a pose-uncertainty gate that declines to correct when there is nothing to
correct would keep both ends. That needs its own pre-registration.

```bash
# Task 23 in full. Roughly 4.5 h on an RTX 4090; the ordering matters and is
# not an accident -- every validation-side input to the part B decision exists
# before test is read at all.
O=outputs/alignformer/r140
DET=configs/alignformer_detector_r140.yaml
AF=configs/alignformer_r140.yaml
VAL=/media/chenyi/basement2/cache/opv2v_splits/val
TEST=/media/chenyi/Elements1/Dataset/OPV2V/test
CKPT=$O/stage2_B_ivw_scalar/best.pth
TAU=$O/shrinkage_ivw_scalar_calibration_result.json
E="python -m embedding_aware_belt_fusion.alignformer.evaluate"
R="--robust-solve huber --robust-iterations 2 --robust-min-evidence 3.0"

# 1. Part A, validation (~55 min).
$E --metric noisy_ap --config $DET --alignformer-config $AF --split $VAL   --checkpoint $CKPT --shrinkage $TAU --freealign $R --ap-seeds 5   --output $O/seeds_val_result.json

# 2. Part B's tau, VALIDATION ONLY, before any test number exists (~2 min).
$E --metric shrinkage --config $AF --checkpoint $CKPT $R   --output $O/shrinkage_irls_calibration_result.json
REFIT=$O/shrinkage_irls_calibration_result.json

# 3. Part B, validation (~60 min). No --freealign: FreeAlign takes no
#    shrinkage, so its rows are bit-identical to step 1's under the same draws.
$E --metric noisy_ap --config $DET --alignformer-config $AF --split $VAL   --checkpoint $CKPT --shrinkage $REFIT $R --ap-seeds 5   --output $O/seeds_refit_val_result.json

# 4-5. Test, read once, after everything on validation is settled (~2.5 h, ~2 h).
$E --metric noisy_ap --config $DET --alignformer-config $AF --split $TEST   --checkpoint $CKPT --shrinkage $TAU --freealign $R --ap-seeds 5   --output $O/seeds_test_result.json
$E --metric noisy_ap --config $DET --alignformer-config $AF --split $TEST   --checkpoint $CKPT --shrinkage $REFIT $R --ap-seeds 5   --output $O/seeds_refit_test_result.json

# 6. Every table above, plus the pre-registered rule. Refuses to be handed one
#    split twice, and records which file each side came from.
python scripts/summarize_seed_error_bars.py   --validation $O/seeds_val_result.json --test $O/seeds_test_result.json   --single-seed-validation $O/robust_solve_val_result.json   --single-seed-test $O/robust_solve_test_result.json   --refit-validation $O/seeds_refit_val_result.json   --refit-test $O/seeds_refit_test_result.json   --refit-calibration $REFIT   --output $O/seed_error_bars_result.json
```

### The AP scorer had to be made affordable first

At five seeds the *scoring* became longer than the sweep.
`fusion.average_precision` re-ran the rotated-IoU **matching** for each of three
IoU thresholds, each of two sort orders, and again for each shared-object slice
-- eighteen passes over the same polygons per condition. The matching depends on
none of those things, so `fusion.match_frames` now does it once and
`average_precision_from_matches` replays the greedy assignment from it;
`average_precision` is a thin wrapper over the two so the definition and the
fast path cannot drift. Published numbers were taken on this path, so it is
pinned by exact-equality tests and was verified on real data to change nothing:
**11,448 cells -- AP, the three slices, the intermediate-convention GT and every
pose statistic -- max |difference| = 0.**

## Per-pair abstention: the rule that fixes the clean case, and what it cost

The section above leaves one gap, and task 23 measured its shape exactly: the
AP@0.7 deficit to FreeAlign is not spread across the sweep. On test, sigma = 0.2
alone (-0.0235) outweighs every cell the IRLS arm wins (+0.0314 combined), and
the same defect costs **-0.2042** on the 1-2 shared slice at sigma = 0. Both are
one thing: at low sigma the correct action is to not correct, and head B
corrects anyway. Task 23 also closed the obvious escape -- a refitted *global*
shrinkage tau buys the low-noise end and gives the clean case back, because a
smaller scalar shrinks less everywhere.

`alignformer/abstain.py` replaces the global scalar with a per-pair decision
taken from the weighted solve's **own** covariance. Nothing is learned and
nothing new is fitted: for parameters `(t_x, t_y, psi)` the residual Jacobian is
`J_n = [I | perp(R(psi) q_n)]`, and the Wald statistic of the emitted correction
against "no correction is needed" is `T^2 = (A theta)^T B^-1 (A theta) / tau^2`.

**The covariance is a sandwich, and that is not a refinement.** The points are
heading-augmented, so an object's two rows share their entire centre error --
correlation 0.84 at this project's fitted constants. A first implementation used
`sigma^2 A^-1` and, simulated on the deployed geometry at a nominal 5%, fired on
**63.7%** of two-object pairs against 26.6% of twenty-four-object ones. That is
not a mislabelled constant: it over-fires worst where the evidence is thinnest,
which inverts the mechanism the whole idea rests on. With the model-based
sandwich, a computed rather than counted residual divisor, and an `F(p, D)`
reference at `D = 3 n_eff_objects - 3`, the realized rate is 0.075 at two matched
objects falling to 0.045 at twenty-four. The defect was caught by review and the
sweep in flight was killed **unread**.

Three rules were tried over the statistic and no fourth: a hard threshold at a
named level, the James-Stein positive-part factor with the pair's own statistic
in place of the global one, and both.

**Only the continuous rule gets both ends of the sweep.** A hard threshold is a
step function: loosening it lets low-sigma corrections through and sigma = 0
regresses; tightening it declines pairs that needed correcting and AP@0.5 pays.
The `per_pair` arm, against the deployed IRLS arm, five seeds, both splits:

| AP@0.7, per_pair - IRLS | validation | test |
|---|---:|---:|
| grid mean (8 cells) | +0.00427 ±0.00014 | **+0.00909 ±0.00008** |
| drawn mean (7 cells) | +0.00447 ±0.00016 | +0.00829 ±0.00010 |
| sigma = 0 | +0.0029 (det) | **+0.0147** (det) |
| sigma = 0.2 | +0.0130 ±0.0005 | **+0.0273** ±0.0007 |
| sigma = 2.0 | -0.0006 ±0.0003 | +0.0002 ±0.0003 |
| coverage, grid mean | 0.861 -> 0.917 | 0.864 -> **0.903** |

**Coverage rose.** The expectation was that a gate answers less often and
flatters itself on what remains; the opposite happened, because the deployed
global tau zeroes a large fraction of low-sigma pairs outright while the
per-pair rule scales each by its own evidence. At sigma = 0 on validation it
corrects three times as many pairs, each by half as much (0.074 m against
0.146 m over answered pairs). What fell is the size of the corrections, not
their count, and `emitted_translation_m` is reported beside coverage so the two
cannot be conflated.

**Against FreeAlign at AP@0.7, report both framings or neither.** The grid mean
weights sigma = 0 -- no localization error at all -- equally with the seven
sigmas that have some:

| | grid mean (8) | drawn mean (7) |
|---|---:|---:|
| deployed IRLS, validation | -0.00354 ±0.00031 | -0.00617 ±0.00036 |
| deployed IRLS, test | -0.00041 ±0.00017 | -0.00119 ±0.00019 |
| per_pair, validation | +0.00073 ±0.00021 | **-0.00170 ±0.00024** |
| per_pair, test | +0.00869 ±0.00021 | +0.00710 ±0.00024 |

On validation the sign change is carried entirely by sigma = 0; excluding it,
FreeAlign still edges the noisy sweep at |t| = 7. **The supportable statement is
that the gate fixes the clean case and closes most of the AP@0.7 deficit --
-0.0062 to -0.0017 on validation, 3.6x -- and turns it into a lead on test.**
AP@0.3 and AP@0.5 are ours under every framing on both splits. FreeAlign remains
scored without EdgeGAT, so every margin in our favour is optimistic and every
margin against us conservative.

The sparse slice is where the structural difference lives, and it moves most.
Test, 1-2 shared objects, AP@0.7 against FreeAlign at sigma = 0: **-0.2042 for
the deployed arm, -0.0556 for per_pair** -- the worst number in this project,
reduced by 73%.

**One caveat on the split disagreement, because it is easy to misread.**
FreeAlign degrades far more between splits than we do: its answered translation
MAE is 0.321 m on validation against **1.676 m on test (5.2x)**, with coverage
0.999 falling to 0.908, on the same detector and the same fusion. Ours moves
0.103 -> 0.224 m (2.2x). So a large part of the AP@0.7 gap closing between
validation and test is the competitor's instability rather than our improvement.
Our gain over the deployed arm is measured against ourselves and is unaffected;
the head-to-head sign flip is not a clean measurement of our improvement and is
not quoted as one.

**Clause 4, the test-side guard, passes at +0.0330 ±0.0007** on the 1-2 shared
slice, positive at 7 of 8 sigmas, so all four pre-registered clauses pass and the
arm ships.

That clause's granularity is not stated in the brief, and the first reading of
this experiment supplied "at every sigma", under which the slice fails at
sigma = 2.0 (-0.0020 ±0.0015, |t| = 1.35) and the verdict is NULL. The sweep-mean
reading was adopted for three reasons, none of which depends on the direction of
the result: "beyond one paired SE" is attached to a **sweep mean** in both other
clauses that use the phrase and clause 4 names no cell; task 23's brief already
ruled that test's sparse slice "is only ever read as a *guard against
regression*, never to select a parameter"; and a per-cell bar of one paired SE at
five seeds fires on 18.7% of cells under the null, i.e. vetoes roughly four arms
in five that have no true effect at all. The task-24 report prints both readings,
both numbers and the full argument, and records that the resolution came after
the numbers were known.

**sigma = 2.0 is the one cell the rule costs**, consistently: -0.0020 ±0.0015 on
the test sparse slice, +0.0002 ±0.0003 (a draw) on the full test split,
-0.0006 ±0.0003 on validation. The top of the sweep is where the correction is
unambiguously needed and shrinking it at all can only hurt; any rule that shrinks
as a function of the statistic pays something there. Making the correction *not*
need shrinking at the top of the sweep is a training-objective question, not a
post-hoc one.

```bash
# Task 24 in full. Roughly 3 h validation + 3 h test on an RTX 4090. The
# ordering is the point: every constant is chosen on validation, and the test
# sweep carries ONLY the arm the rule already selected.
#
# 1. Validation, five seeds, all seven candidate arms beside the two deployed
#    ones and FreeAlign, in ONE invocation so the pairing is structural.
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --metric noisy_ap \
  --config configs/alignformer_detector_r140.yaml \
  --alignformer-config configs/alignformer_r140.yaml \
  --split /media/chenyi/basement2/cache/opv2v_splits/val \
  --checkpoint outputs/alignformer/r140/stage2_B_ivw_scalar/best.pth \
  --shrinkage outputs/alignformer/r140/shrinkage_ivw_scalar_calibration_result.json \
  --robust-solve huber --robust-iterations 2 --robust-min-evidence 3.0 \
  --abstain-arm abstain:0.5 --abstain-arm abstain:0.2 \
  --abstain-arm abstain:0.05 --abstain-arm abstain:0.01 \
  --abstain-arm per_pair --abstain-arm both:0.2 --abstain-arm both:0.05 \
  --freealign --ap-seeds 5 \
  --output outputs/alignformer/r140/abstain_val_result.json

# 2. Freeze on validation clauses 1-3 BEFORE reading test.
python scripts/summarize_abstention.py \
  --validation outputs/alignformer/r140/abstain_val_result.json \
  --baseline-validation outputs/alignformer/r140/seeds_val_result.json \
  --output outputs/alignformer/r140/abstention_result.json

# 3. Test, read once, carrying only the chosen arm (`per_pair`).
#    Same command as step 1 with --split .../OPV2V/test and --abstain-arm per_pair.

# 4. Clause 4, the veto, and every table above.
python scripts/summarize_abstention.py \
  --validation outputs/alignformer/r140/abstain_val_result.json \
  --baseline-validation outputs/alignformer/r140/seeds_val_result.json \
  --test outputs/alignformer/r140/abstain_test_result.json \
  --baseline-test outputs/alignformer/r140/seeds_test_result.json \
  --output outputs/alignformer/r140/abstention_result.json
```

The summarizer refuses a single-seed decision, refuses to report one split as
two, prints and **enforces** the bit-identity gate against the published runs
(990 cells per split, max |difference| = 0 here), and keeps the verdict
`PROVISIONAL_SHIP_*` until clause 4 has actually been evaluated.

## What this means for P3-P5

The plan's "After P2" branch asks which of three outcomes obtains.

**The yaw gate passes, and the closed-form premise is supported.** The plan's
feared failure mode was "flat at the predict-zero value" -- the CoLoca-QuA
outcome, a head that never leaves the conditional mean. That is exactly what
**head A** does here, and post-fix it does it more clearly than before: its
translation MAE is within 1-3% of predict-zero at every sigma and its **yaw MAE
is worse than predict-zero** (1.6428 against 1.5995 deg at sigma = 2). It is a
clean reproduction of the CoLoca-QuA finding on AlignFormer's own trunk, on a
retrained model. **Head B does not do that at all**: 8.0x better than
predict-zero on yaw and 17x on translation at sigma = 2 m, with an error
essentially independent of the input noise.

All three of the defects that produced the original failure were in the *data
the model was shown*, not in the model: a heading channel carrying a direction
the detector never estimates, a pair index built at a communication range the
evaluation protocol does not use, and a detector configured for half the range
the evaluation scores. None was visible in any loss curve. That is the
transferable lesson from P2.

**The embedding does not carry the method, and the matched rerun closed the
question.** On association it is worth nothing -- boxes-only is marginally
ahead (0.9976 against 0.9961). On fused AP, once both arms are matched on epoch
and stage-1 lineage at +/-140.8 m, it is worth **+0.0007 with the sign flipping
three times**; the +0.011 measured at `r70` was the mismatch, not the
embedding. The plan's second branch therefore applies in its strong form: the
contribution rests on the closed-form solver plus the efficiency argument. The
`match_weight 0` control continues to show that the *matching supervision* is
worth more than the embedding -- 24% of the corner loss against the embedding's
17%.

The **camera-augmentation contingency (spec 8) is not live on OPV2V.** It
addresses two of the three reasons the embedding fails -- no appearance signal,
and BEV resolution -- but not the binding one: no ego object in the validation
split has a competitor within 2 m, so nearest-centre association already scores
Top-1 1.0000 and there is nothing for a better descriptor to disambiguate. The
contingency stays open for **V2X-Real**, where real traffic supplies the
competitors OPV2V lacks. Whether any matching improvement can move the headline
metric at all is bounded by the oracle-correspondence measurement recorded in
the open items below.

Five things follow, in priority order:

1. **Close the clean case.** -0.055 AP@0.7 at sigma = 0 is the one stated target
   still unmet. Widening the detector shrank it from -0.070 but did not close
   it. Gating the correction on an estimate of the localization error,
   or training with a loss that pins the identity at sigma = 0, would turn "a
   large win everywhere above 0.2 m and a small loss at 0" into a method that is
   never worse than doing nothing. Both are tuning and neither is done here.
2. **Weight the Procrustes fit by object range, or by detection confidence.**
   This is now measured rather than suspected: correspondences beyond
   |x| = 70.4 m disagree across agents by 0.230 m against the near field's
   0.151 m, and 6.65 deg against 3.69 deg, yet they enter the closed-form fit
   with equal weight. That is why every test-split pose number got worse at the
   wider range even as every AP number got better. **Done in task 19**: the
   inverse-variance weight is now fitted and deployed, and it cuts translation
   MAE 13-16% and yaw MAE 12-14% at every sigma. See [The correspondence
   variance model](#the-correspondence-variance-model). The variable that
   drives the variance turned out to be **detection score, not range** --
   the hypothesis that dispatched the task was wrong by a factor of ~7 in
   deviance improvement.
3. **Attack the detector-limited residual.** What caps AlignFormer at ~0.77
   AP@0.7 against an oracle 0.8964 is not the injected noise -- the residual is
   nearly sigma-independent. It is the cross-agent disagreement in the
   detections themselves: 0.162 m per correspondence per axis on average, which
   an averaging over ~13 objects can only reduce so far. The levers are the
   detector and the loss, not more pose training.
4. **Repeat the embedding ablation over seeds, and at the wider range.** The
   ablation tables in this document are all `r70`; only the deployed
   configuration was retrained at +/-140.8 m.
5. **Raise `MIN_MATCH_MASS` to its documented intent** and recount the
   unalignable subset on the 70 m index, which admits more low-overlap pairs
   than the counts in this document were taken on.

## Open items carried forward

- The unalignable-pair counts in "Findings that are not the gate" are from the
  40 m index and need recounting at 70 m.
- `tau` is calibrated on a validation slice measurably easier than the test
  split (per-scenario translation MAE median 0.118 m across the train split
  against 0.153 m across the test split, Mann-Whitney one-sided p = 0.014), so
  it is a little small where it is applied. Diagnosed in
  [alignformer_pose_floor.md](alignformer_pose_floor.md) section 6; not
  corrected, because correcting it would mean calibrating on test.
- **Resolved in task 23.** The AP sweep now runs **five** independent noise
  seeds per sigma on both splits, paired across conditions; see [Error bars:
  what the AP head-to-head actually supports](#error-bars-what-the-ap-head-to-head-actually-supports).
  What it changed: the AP@0.7 head-to-head against FreeAlign is **theirs** on
  both splits, not ours, and FreeAlign vs the deployed arm at AP@0.5 on test
  is a **draw**. Both were quoted as orderings above on one seed. The pose
  sweep is still three seeds.
- The `match_weight 0` ablation, the shrinkage on/off table and the per-epoch
  training curve are all `r70`. **Head A was also run at +/-140.8 m** (task 19,
  `p2_r140_ablations_result.json`) and the conclusion strengthens there: its yaw
  MAE is worse than predict-zero at *every* sigma, not only at the high-noise
  end. The **boxes-only ablation was**
  retrained at +/-140.8 m in task 19 with both arms matched on epoch and
  lineage; see [the matched rerun](#the-matched-r140-rerun-removes-both-confounds-and-the-null-holds).
  The pose-level embedding gap (12-15% at `r70`) was never re-tested under those
  matched conditions and should not be quoted without that caveat.
- **Resolved in task 19.** Pose estimation was measurably worse at the wider
  range on the test split (translation MAE 0.2509 -> 0.3043 m at sigma = 0.2),
  and the P2 gate's translation criterion failed at sigma = 0.2 on test where it
  used to pass. Inverse-variance weighting brings it to **0.2599 m**, below
  predict-zero's 0.2797 m, so that criterion **passes again**. The yaw criterion
  still fails on test at sigma = 0.2 and 0.4 and remains open. The recommended
  fix at the time -- weighting by object *range* -- was measured and is roughly
  7x weaker than weighting by detection score; score is what was deployed.
- **Done, and it changed the framing.** V2X-ViT and six other
  intermediate-fusion baselines were re-run locally under this sweep and this
  evaluator; see [Head-to-head against intermediate
  fusion](#head-to-head-against-intermediate-fusion). No published number is
  quoted anywhere. What is left open from it: CoAlign is represented by its
  fusion module only, because this OpenCOOD fork does not ship its pose graph
  (caveat 3); and FreeAlign, the closest competitor in kind, cannot be run as
  published on this machine -- porting its training-free matcher onto our own
  detections is the recommended P3 item. (Caveat 1, the detector range, is
  resolved as of task 18.)
- **Measured in task 21, verdict `ambiguous` by 0.0001.** See [the
  oracle-correspondence ceiling](#the-oracle-correspondence-ceiling-how-much-is-matching-worth-at-all).
  The open part is the **split disagreement**: validation says matching is
  closed, test says there is headroom, and the difference is carried by test's
  6x larger sparse-pair tail. Whether to chase that tail (harder training pairs,
  refusing to correct at low match mass) is a decision the branch does not make.
  One noise seed per AP cell, so the 0.0001 margin is not meaningful on its own.
- **The sigma = 0 regression is a property, not a bug** (ruling R44). Five
  attempted fixes failed; it is the unavoidable cost of moving boxes by an
  imperfect estimate when the pose was already correct. The deployment answer is
  to gate the correction on a pose-uncertainty signal rather than to remove the
  regression. Whether that framing is acceptable for publication is a judgement
  call, not a measurement, and is recorded here as open.
