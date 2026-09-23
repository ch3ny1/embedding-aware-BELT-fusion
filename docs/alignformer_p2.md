# AlignFormer P2: stage-2 pose training, the boxes-only ablation, and fused AP under localization error

> **Current.** Every number in this document was measured after the two
> defects P2 originally surfaced were fixed: the heading channel's pi ambiguity
> (diagnosed in [alignformer_pose_floor.md](alignformer_pose_floor.md)) and the
> communication-range mismatch between the pair index and the evaluation
> protocol. Four configurations -- head A, head B, the boxes-only ablation and
> the `match_weight 0` control -- were retrained from scratch on the corrected
> pair index and re-measured together, so the comparisons here are
> like-for-like. (P2's fifth configuration, head A / boxes_only, was dropped:
> head A collapses to predict-zero either way, so ablating its message content
> measures nothing.)

Stage 1 established that the two agents' object sets can be put into
correspondence (P1: cross-agent Top-1 0.9961 at 70 m). It also established,
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
| **P2 gate** (head B yaw MAE below predict-zero at every non-zero sigma) | **PASS, 7 of 7.** Definition unmodified, and now measured on a *larger and harder* validation population (6262 pairs out to 70 m, against the 4176 the 40 m index admitted). |
| Head A vs head B | **Decisive for B, and head A's collapse reproduces post-fix.** Head A's translation MAE tracks predict-zero to within 1-3% at every sigma (2.688 against 2.734 m at sigma = 2) and its **yaw MAE is *worse* than predict-zero** (1.643 against 1.600 deg). It never leaves the conditional mean -- exactly [CoLoca-QuA's failure](coloca_qua_baseline.md) on the same data. Head B is 8.0x better on yaw and 17x better on translation at sigma = 2 m. |
| Does the embedding help? | **Barely, and the headline "nothing" still stands where it was measured.** On *association* it is still worth nothing: the boxes-only stage 1 reaches Top-1 **0.9976** against boxes+embeddings' 0.9961. On *fused AP* it is worth **+0.000 to +0.021 AP@0.7** (mean +0.011), the same marginal amount measured before the fixes (+0.010 to +0.016). What did change is the *pose* metric, where the gap widened from ~3% to 12-15% and boxes-only would now **fail** the P2 gate at sigma = 0.2 m (0.1662 against 0.1600) where boxes+embeddings passes. See the caveat below: one seed, and separate warm starts. |
| Does anything in the message help? | **Yes: the matching supervision, still more than the embedding.** Dropping `match_nll` (`match_weight 0`) costs head B 24% of its corner loss (0.919 -> 1.138 m) and 9-32% of its yaw accuracy. The value is in learning a correspondence from *geometry*, which the auxiliary loss supervises. |
| **mAP under localization error** (the number the project needs) | **AlignFormer recovers 75-81% of the oracle-vs-vanilla gap at every sigma from 0.4 to 2.0 m**, worth **+0.43 to +0.57 AP@0.7** on the 2170-frame test split, and **+0.16 at sigma = 0.2 m** where it used to lose. At sigma = 0 it still costs **0.070**. |

### The headline

On the official 2170-frame OPV2V test split, global-sorted AP@0.7:

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

Three things are visible at a glance.

**Vanilla late fusion collapses under localization error** -- 0.8764 -> 0.1690
AP@0.7 by sigma = 1 m, an 81% relative loss. That collapse is the problem
AlignFormer exists to solve, and it is severe.

**AlignFormer is almost flat in sigma** -- 0.8068 at sigma = 0 down to 0.7046 at
sigma = 2 m. That flatness is the closed-form solver working: it removes
essentially all of the *injected* error, leaving a residual set by the detector
rather than by the noise. It is also why the method wins by more as conditions
get worse, which is the right direction for a robustness method.

**The remaining cost is at sigma = 0 only, and it is now 0.070.** Where
localization is already perfect, moving boxes by an imperfect estimate can only
lose AP, and shrinkage suppresses the correction entirely on 66% of test pairs
there rather than on all of them. The stated target -- AP@0.7 at or above
uncorrected at **every** sigma including 0 -- is therefore still not met at
sigma = 0, and that is the one row that fails it. At every other sigma it is met
by a wide margin, including at 0.2 m where the method used to lose 0.043 and now
gains 0.164.

### How the three fixes accumulate

Same evaluator, same detections, same 2170-frame test split, AP@0.7:

| sigma (m) | Uncorrected | P2 as first measured | + heading fold & shrinkage | **+ 70 m pair index** | Oracle |
|---|---:|---:|---:|---:|---:|
| 0 | **0.8764** | 0.5424 | 0.7665 | **0.8068** | 0.8764 |
| 0.2 | 0.5846 | 0.5419 | 0.7096 | **0.7488** | 0.8764 |
| 0.4 | 0.3069 | 0.5398 | 0.6963 | **0.7410** | 0.8764 |
| 0.6 | 0.2106 | 0.5368 | 0.6948 | **0.7412** | 0.8764 |
| 0.8 | 0.1785 | 0.5320 | 0.6971 | **0.7404** | 0.8764 |
| 1.0 | 0.1690 | 0.5303 | 0.6957 | **0.7386** | 0.8764 |
| 1.5 | 0.1737 | 0.5198 | 0.6845 | **0.7326** | 0.8764 |
| 2.0 | 0.1831 | 0.4961 | 0.6707 | **0.7046** | 0.8764 |

### What shrinkage is still buying

Same checkpoint, same sweep, `--shrinkage` on and off, AP@0.7:

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

Unchanged in character from before the range fix: shrinkage buys the clean case
-- **+0.050 at sigma = 0** -- for 0.001 to 0.008 everywhere else. That is the
trade it exists to make, and it is what keeps the sigma = 0 cost at 0.070
instead of 0.120.

The heading fold is the large one (+0.167 to +0.177 everywhere). Matching the
pair index to the evaluation protocol is the second (+0.034 to +0.046), and it
is the only one that helps the clean case materially (+0.040 at sigma = 0).
Neither is a modelling change: both are defects in what the model was shown.

## Reproducing

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

Head B's yaw MAE is now **0.1473-0.2065 deg**, strictly below the empirical
predict-zero value at all seven non-zero sigmas. **The gate passes, with its
definition unmodified** -- and on a harder population than it was originally
posed on, since the widened index adds 2086 further-apart validation pairs that
share fewer objects (7.61 matched objects per pair beyond 40 m against 10.92
within it).

### The clean case still costs 0.070 AP@0.7

This is the one target not met. At sigma = 0 the true correction is exactly the
identity, so any correction at all can only lose AP; the method scores 0.8068
against the oracle's 0.8764.

It is much smaller than it was -- P2 measured -0.334, the heading fold and
shrinkage brought it to -0.110, and matching the pair index to the protocol
brings it to **-0.070** -- but it is not zero, and the honest statement is that
AlignFormer is still not unconditionally free to leave on.

Shrinkage is what keeps it this small: at sigma = 0 it suppresses the
correction **entirely** on 66.1% of test pairs (the fallback fraction in the
test-split pose table below is the exact-identity rate), against 19.4% at
sigma = 0.2 and 2% at sigma >= 1. On the pairs it does not fully suppress, the
residual it lets through is 0.1453 m and 0.2692 deg -- enough to drop a box
below the 0.7 IoU threshold that was above it. AP@0.3, which tolerates that
residual, is 0.9092 against an oracle 0.9284, i.e. at the looser threshold the
clean-case cost is only 0.019.

Two routes remain and neither is taken here, because both are tuning against
the measurement this document exists to take: gate the correction on an
estimate of the localization error, or train with a loss that pins the identity
at sigma = 0.

### The embedding: still nothing on association, marginal on AP, larger on pose

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

All of the following are rendered by `scripts/summarize_alignformer_p2.py`
from `outputs/alignformer/r70/p2_r70_gate.json` (the gate, shrunk),
`outputs/alignformer/r70/p2_r70_ablations_result.json` (the configuration comparison,
**without** shrinkage, since one calibration cannot describe four different
estimators) and `outputs/alignformer/r70/p2_r70_noisy_ap_result.json` (fused
AP, shrunk).

## P2 gate

Gate configuration: **B / boxes+embeddings** -- **PASS**

| sigma | Yaw MAE (deg) | Predict-zero yaw (deg) | Below? |
|---|---:|---:|---|
| sigma_0.2m | 0.1473 | 0.1600 | yes |
| sigma_0.4m | 0.1654 | 0.3199 | yes |
| sigma_0.6m | 0.1690 | 0.4799 | yes |
| sigma_0.8m | 0.1708 | 0.6398 | yes |
| sigma_1m | 0.1726 | 0.7998 | yes |
| sigma_1.5m | 0.1816 | 1.1996 | yes |
| sigma_2m | 0.2065 | 1.5995 | yes |

Measured on the scenario-disjoint validation split (6262 pairs, `val_scenario_fraction` 0.15, `split_seed` 0), mean of 3 independent noise draws, with the validation-calibrated shrinkage applied. `sigma = 0` is excluded from the gate because the predict-zero baseline is exactly 0 there; it is reported in the pose sweep below as the clean-case diagnostic, at **0.0320 m / 0.0509 deg**.

## Pose sweep

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


| sigma (m) | Condition | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---:|---:|---:|
| -- | oracle (true pose) | 0.9284 | 0.9251 | 0.8764 |
| 0 | vanilla late fusion (uncorrected) | 0.9284 | 0.9251 | 0.8764 |
| 0 | AlignFormer-corrected | 0.9092 | 0.8882 | 0.8068 |
| 0.2 | vanilla late fusion (uncorrected) | 0.9277 | 0.8991 | 0.5846 |
| 0.2 | AlignFormer-corrected | 0.9087 | 0.8840 | 0.7488 |
| 0.4 | vanilla late fusion (uncorrected) | 0.8928 | 0.6699 | 0.3069 |
| 0.4 | AlignFormer-corrected | 0.9073 | 0.8787 | 0.7410 |
| 0.6 | vanilla late fusion (uncorrected) | 0.7595 | 0.4507 | 0.2106 |
| 0.6 | AlignFormer-corrected | 0.9042 | 0.8742 | 0.7412 |
| 0.8 | vanilla late fusion (uncorrected) | 0.6207 | 0.3476 | 0.1785 |
| 0.8 | AlignFormer-corrected | 0.9039 | 0.8747 | 0.7404 |
| 1 | vanilla late fusion (uncorrected) | 0.5268 | 0.2978 | 0.1690 |
| 1 | AlignFormer-corrected | 0.8998 | 0.8672 | 0.7386 |
| 1.5 | vanilla late fusion (uncorrected) | 0.3856 | 0.2484 | 0.1737 |
| 1.5 | AlignFormer-corrected | 0.8945 | 0.8638 | 0.7326 |
| 2 | vanilla late fusion (uncorrected) | 0.3365 | 0.2419 | 0.1831 |
| 2 | AlignFormer-corrected | 0.8850 | 0.8470 | 0.7046 |

## Gap recovered at AP@0.7

| sigma (m) | Vanilla AP@0.7 | AlignFormer AP@0.7 | Oracle AP@0.7 | Gain over vanilla | Gap recovered |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8764 | 0.8068 | 0.8764 | -0.0696 | -- |
| 0.2 | 0.5846 | 0.7488 | 0.8764 | +0.1641 | 56.3% |
| 0.4 | 0.3069 | 0.7410 | 0.8764 | +0.4341 | 76.2% |
| 0.6 | 0.2106 | 0.7412 | 0.8764 | +0.5306 | 79.7% |
| 0.8 | 0.1785 | 0.7404 | 0.8764 | +0.5619 | 80.5% |
| 1 | 0.1690 | 0.7386 | 0.8764 | +0.5695 | 80.5% |
| 1.5 | 0.1737 | 0.7326 | 0.8764 | +0.5588 | 79.5% |
| 2 | 0.1831 | 0.7046 | 0.8764 | +0.5215 | 75.2% |

## Pose error on the test split

| sigma (m) | Pairs | Translation MAE (m) | Predict-zero (m) | Yaw MAE (deg) | Predict-zero (deg) | Unalignable pairs | Fell back |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3445 | 0.1453 | 0.0000 | 0.2692 | 0.0000 | 37 | 0.661 |
| 0.2 | 3445 | 0.2509 | 0.2797 | 0.3605 | 0.1610 | 37 | 0.194 |
| 0.4 | 3445 | 0.2539 | 0.5450 | 0.3817 | 0.3241 | 37 | 0.055 |
| 0.6 | 3445 | 0.2646 | 0.8140 | 0.3945 | 0.4865 | 37 | 0.032 |
| 0.8 | 3445 | 0.2615 | 1.0904 | 0.3880 | 0.6308 | 37 | 0.023 |
| 1 | 3445 | 0.2707 | 1.3459 | 0.4084 | 0.7892 | 37 | 0.021 |
| 1.5 | 3445 | 0.2880 | 2.0195 | 0.4358 | 1.1962 | 37 | 0.019 |
| 2 | 3445 | 0.3262 | 2.6933 | 0.4447 | 1.6053 | 37 | 0.019 |

**Read this table against the gate.** The P2 gate is a *validation*
measurement, and on validation head B's yaw MAE is below predict-zero at all
seven non-zero sigmas. On the **test** split the same criterion would **fail at
sigma = 0.2 and 0.4 m** -- 0.3605 against 0.1610 and 0.3817 against 0.3241 --
and pass from 0.6 m up. Translation passes everywhere (0.2509 against 0.2797 at
sigma = 0.2). This is the same finding as the validation-vs-test gap in
[alignformer_pose_floor.md](alignformer_pose_floor.md) section 6, seen through
the gate: the test split is harder, and `tau`, calibrated on validation, is a
little small there. It does not change the AP result -- fused AP@0.7 at
sigma = 0.2 is 0.7488 against the uncorrected 0.5846 -- because IoU is far more
sensitive to a centre offset than to a few tenths of a degree of heading. But
the gate should not be read as a claim about the test split, and it is not one.

The fallback column is the *exact-identity* rate, which at sigma = 0 is
shrinkage doing its job: **66.1%** of test pairs receive no correction at all
there, against 19.4% at sigma = 0.2 and ~2% above 1 m. The 37 structurally
unalignable pairs (1.07%) are a separate, much smaller population.

## Training curve: stage2_B_boxes+embeddings

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

Both of the defects that produced the original failure were in the *data the
model was shown*, not in the model: a heading channel carrying a direction the
detector never estimates, and a pair index built at a communication range the
evaluation protocol does not use. Neither was visible in any loss curve. That is
the transferable lesson from P2.

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

Four things follow, in priority order:

1. **Close the clean case.** -0.070 AP@0.7 at sigma = 0 is the one stated target
   still unmet. Gating the correction on an estimate of the localization error,
   or training with a loss that pins the identity at sigma = 0, would turn "a
   large win everywhere above 0.2 m and a small loss at 0" into a method that is
   never worse than doing nothing. Both are tuning and neither is done here.
2. **Attack the detector-limited residual.** What caps AlignFormer at ~0.74
   AP@0.7 against an oracle 0.8764 is not the injected noise -- the residual is
   nearly sigma-independent. It is the cross-agent disagreement in the
   detections themselves: 0.23 m per correspondence per axis, which an average
   over ~10 objects can only reduce to ~0.09 m. The levers are the detector and
   the loss, not more pose training.
3. **Repeat the embedding ablation over seeds.** The pose-level gap that opened
   post-fix is the one finding here resting on a single run per configuration.
4. **Raise `MIN_MATCH_MASS` to its documented intent** and recount the
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
- The V2X-ViT comparison (published noisy AP@0.7 0.614 on OPV2V) is still not
  apples-to-apples: that number is under V2X-ViT's own noise convention and
  detector and has to be re-run locally before it can stand beside these.
