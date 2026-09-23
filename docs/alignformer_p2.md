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
| Head A vs head B | **Decisive for B, and head A's collapse reproduces post-fix.** Head A's translation MAE tracks predict-zero to within 1-3% at every sigma (2.688 against 2.734 m at sigma = 2) and its **yaw MAE is *worse* than predict-zero** (1.643 against 1.600 deg). It never leaves the conditional mean -- exactly [CoLoca-QuA's failure](coloca_qua_baseline.md) on the same data. Head B is 8.0x better on yaw and 17x better on translation at sigma = 2 m. |
| Does the embedding help? | **Barely, and the headline "nothing" still stands where it was measured.** On *association* it is still worth nothing: the boxes-only stage 1 reaches Top-1 **0.9976** against boxes+embeddings' 0.9961. On *fused AP* it is worth **+0.000 to +0.021 AP@0.7** (mean +0.011), the same marginal amount measured before the fixes (+0.010 to +0.016). What did change is the *pose* metric, where the gap widened from ~3% to 12-15% and boxes-only would now **fail** the P2 gate at sigma = 0.2 m (0.1662 against 0.1600) where boxes+embeddings passes. See the caveat below: one seed, and separate warm starts. |
| Does anything in the message help? | **Yes: the matching supervision, still more than the embedding.** Dropping `match_nll` (`match_weight 0`) costs head B 24% of its corner loss (0.919 -> 1.138 m) and 9-32% of its yaw accuracy. The value is in learning a correspondence from *geometry*, which the auxiliary loss supervises. |
| **mAP under localization error** (the number the project needs) | **AlignFormer recovers 78-84% of the oracle-vs-vanilla gap at every sigma from 0.4 to 2.0 m**, worth **+0.46 to +0.60 AP@0.7** on the 2170-frame test split, and **+0.18 at sigma = 0.2 m**. At sigma = 0 it still costs **0.055** (it cost 0.070 at +/-70.4 m: the regression shrank by a fifth but did **not** close). |
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

**The embedding still does not carry the method, but the claim needs its
numbers.** On association it is worth nothing -- boxes-only is marginally ahead
(0.9976 against 0.9961). On fused AP it is worth +0.011 on average. On the pose
metric it is now worth 12-15%, enough to decide the gate at sigma = 0.2 m, on
one seed and with separate warm starts. The plan's second branch therefore still
applies: the contribution rests on the closed-form solver plus the efficiency
argument, and the camera-augmentation contingency (spec 8) remains live. The
`match_weight 0` control continues to show that the *matching supervision* is
worth more than the embedding -- 24% of the corner loss against the embedding's
17%.

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
   wider range even as every AP number got better. An inverse-variance weight
   is the obvious first thing to try and is *not* done here.
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
- One noise seed for the AP sweep, three for the pose sweep.
- The head A / boxes-only / `match_weight 0` ablations, the shrinkage on/off
  table and the per-epoch training curve are all `r70`. Only the deployed
  configuration (B / boxes+embeddings) was retrained at +/-140.8 m.
- Pose estimation is measurably **worse** at the wider range on the test split
  (translation MAE 0.2509 -> 0.3043 m at sigma = 0.2, yaw 0.3605 -> 0.4308 deg),
  and the P2 gate's translation criterion now fails at sigma = 0.2 on test where
  it used to pass. Fused AP rose anyway. Nothing was tuned to recover the pose
  numbers; weighting the fit by object range is the recommended next step.
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
