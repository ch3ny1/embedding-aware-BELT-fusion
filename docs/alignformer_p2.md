# AlignFormer P2: stage-2 pose training, the boxes-only ablation, and fused AP under localization error

> **Superseded for head B.** The sigma-independent error floor this document
> reports was diagnosed and fixed in
> [alignformer_pose_floor.md](alignformer_pose_floor.md): it was noise, not
> bias, and most of it came from the heading virtual point taking a direction
> the detector never estimates (20.3% of cross-agent detections of the same
> object disagree by ~180 deg). With that folded and the remaining noise
> shrunk, the **P2 gate passes at all seven non-zero sigmas** and AP@0.7 at
> sigma = 0.2 m is 0.7096 rather than the 0.5419 below. Every head-B number
> here is the pre-fix value. The head A comparison, the boxes-only ablation and
> the `match_weight 0` control have not been re-measured and remain as written.

Stage 1 established that the two agents' object sets can be put into
correspondence (P1: cross-agent Top-1 0.9965). It also established, through the
association diagnostic, that the *embedding* contributes nothing to that
correspondence on OPV2V. P2 asks the question the method actually exists for:

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
| **P2 gate** (head B yaw MAE below predict-zero at every non-zero sigma) | **FAIL -- at sigma = 0.2 m only.** Passes at the other seven levels (0.4 to 2.0 m). |
| Why it fails | Head B has a **sigma-independent error floor**: 0.2995-0.3010 deg yaw and 0.236-0.239 m translation, flat from sigma = 0 to 1.0 m. Below sigma = 0.38 m the pose error is smaller than that floor, so predicting zero wins. Design spec 8, Risk 4 anticipated this exactly. |
| Head A vs head B | **Decisive for B.** Head A's yaw MAE tracks predict-zero to within 2-3% at every sigma -- it never leaves the conditional mean, reproducing the CoLoca-QuA failure on the same data. Head B is 4.7x better at sigma = 2 m. |
| Does the embedding help? | **No, to a first approximation.** boxes+embeddings vs boxes_only differ by 0.008-0.015 deg of yaw (2.6-4.5% relative, consistently in the embedding's favour) and translation is a wash, with boxes_only marginally *ahead* at five of eight sigmas. Consistent with the association diagnostic's ruling. |
| Does anything in the message help? | **Yes: the matching supervision, not the embedding.** Dropping `match_nll` (`match_weight 0`) costs head B 22% of its corner loss (1.811 -> 2.314 m) and 23-41% of its yaw accuracy. The value is in learning a correspondence from *geometry*, which the auxiliary loss supervises. |
| **mAP under localization error** (the number the project needs) | **AlignFormer recovers 41-51% of the oracle-vs-vanilla gap at every sigma from 0.4 to 2.0 m**, worth **+0.23 to +0.36 AP@0.7** on the 2170-frame test split. Below ~0.3 m it *costs* AP, for the same floor that fails the gate. |


### The headline

On the official 2170-frame OPV2V test split, global-sorted AP@0.7:

| sigma (m) | Vanilla late fusion | AlignFormer | Oracle (true pose) | Gain | Gap recovered |
|---|---:|---:|---:|---:|---:|
| 0 | **0.8764** | 0.5424 | 0.8764 | -0.3340 | -- |
| 0.2 | **0.5846** | 0.5419 | 0.8764 | -0.0427 | -14.7% |
| 0.4 | 0.3069 | **0.5398** | 0.8764 | **+0.2329** | 40.9% |
| 0.6 | 0.2106 | **0.5368** | 0.8764 | **+0.3262** | 49.0% |
| 0.8 | 0.1785 | **0.5320** | 0.8764 | **+0.3535** | 50.7% |
| 1.0 | 0.1690 | **0.5303** | 0.8764 | **+0.3613** | 51.1% |
| 1.5 | 0.1737 | **0.5198** | 0.8764 | **+0.3461** | 49.3% |
| 2.0 | 0.1831 | **0.4961** | 0.8764 | **+0.3130** | 45.2% |

Two things are visible at a glance and both matter.

**Vanilla late fusion collapses under localization error** -- 0.8764 -> 0.1690
AP@0.7 by sigma = 1 m, an 81% relative loss. That collapse is the problem
AlignFormer exists to solve, and it is severe.

**AlignFormer is almost flat in sigma** -- 0.5424 at sigma = 0 down to 0.4961 at
sigma = 2 m. That flatness is the closed-form solver working: it removes
essentially all of the *injected* error, leaving a residual set by the detector
rather than by the noise. It is also why the method wins by more as conditions
get worse, which is the right direction for a robustness method.

**But the flat line starts below the clean baseline.** AlignFormer's ceiling is
~0.54 AP@0.7 where the oracle is 0.8764, so it recovers about half the gap and
never approaches the clean number. And at sigma <= 0.2 m, where vanilla is still
healthy, applying the correction *loses* AP. Deployed as an always-on
correction, AlignFormer is a large win above ~0.3 m of localization error and a
loss below it.

## Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD

# Five stage-2 configurations (~14 min each on an RTX 4090)
for head in B A; do
  for content in "boxes+embeddings" "boxes_only"; do
    python -m embedding_aware_belt_fusion.alignformer.train \
      --config configs/alignformer.yaml --stage 2 \
      --head "$head" --message-content "$content" \
      2>&1 | tee "outputs/alignformer/stage2_${head}_${content}.log"
  done
done
# Fairness control: head B's architecture without head B's extra supervision.
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 2 \
  --head B --message-content boxes+embeddings --match-weight 0 \
  2>&1 | tee outputs/alignformer/stage2_B_nomatch.log

# The P2 gate: pose MAE vs predict-zero, all five configurations, 3 noise draws
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer.yaml --metric pose \
  --checkpoint outputs/alignformer/stage2_B_boxes+embeddings/best.pth \
               outputs/alignformer/stage2_B_boxes_only/best.pth \
               outputs/alignformer/stage2_A_boxes+embeddings/best.pth \
               outputs/alignformer/stage2_A_boxes_only/best.pth \
               outputs/alignformer/stage2_B_boxes+embeddings_nomatch/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --seeds 3 \
  --output outputs/alignformer/p2_gate.json

# Fused AP under localization error, on the official 2170-frame test split
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer_detector.yaml \
  --split /media/chenyi/Elements1/Dataset/OPV2V/test \
  --metric noisy_ap --alignformer-config configs/alignformer.yaml \
  --checkpoint outputs/alignformer/stage2_B_boxes+embeddings/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
  --output outputs/alignformer/p2_noisy_ap_result.json

# Every table below is rendered from those two JSONs
python scripts/summarize_alignformer_p2.py \
  --pose outputs/alignformer/p2_gate.json \
  --noisy-ap outputs/alignformer/p2_noisy_ap_result.json \
  --history outputs/alignformer/stage2_B_boxes+embeddings/history.json
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

**Result: no non-finite gradient occurred in any of the five runs.** `tau`
stayed near 0.037 throughout stage 2 rather than collapsing further. The spec's
reachability analysis stands: the singularity remains unreached in practice.

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
each of which changes what that risk means:

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

### The P2 gate fails at sigma = 0.2 m, for a reason the spec predicted

Head B's yaw MAE is **0.2995-0.3010 deg, flat from sigma = 0 to 1.0 m**, rising
only to 0.3387 at sigma = 2 m. The predict-zero baseline is
`sigma_yaw * sqrt(2/pi)`, which is 0.1595 deg at sigma = 0.2. So the model is
*worse than doing nothing* there -- not because it fails to estimate the pose,
but because its own floor is larger than the error it is asked to remove.

Design spec section 8, Risk 4 called this in advance: "At sigma_yaw = 0.2 deg
the pose yaw error may fall below the detector's own box-heading noise.
Reported honestly rather than hidden by the sweep average."

The crossovers follow directly from the floor, on the validation split:

| Quantity | Head B floor | Predict-zero | Crossover |
|---|---:|---|---:|
| Yaw MAE | 0.2997 deg | `0.7979 * sigma` deg | sigma = **0.38 m** |
| Translation MAE | 0.2361 m | `1.3090 * sigma` m | sigma = **0.18 m** |

Translation crosses over much earlier than yaw, which is why fused AP turns
positive at sigma = 0.4 (between the two crossovers) rather than waiting for the
yaw crossover: at these box sizes, IoU is far more sensitive to a 1 m centre
offset than to a 0.4 deg heading error.

**The gate as written is therefore FAIL.** It is a real failure at one sigma and
a decisive pass at the other seven, and the mechanism is fully diagnosed. The
gate is also, on this evidence, the wrong shape: it asks whether the correction
beats the identity at *every* noise level, which no estimator with a non-zero
floor can satisfy down to sigma = 0. A gate of the form "beats predict-zero
above the noise level the deployment actually faces" would be answerable and
would pass here from 0.4 m upward.

### AlignFormer has a cost when localization is already good

The same floor has a direct fused-AP consequence, and it is the single largest
caveat on this work. At sigma = 0 the model does not return the identity: it
moves already-correct boxes by ~0.47 m and ~0.67 deg (test split), and fused
AP@0.7 falls from **0.8764 to 0.5424**. At sigma = 0.2 it still loses 0.043.

AlignFormer as trained here should be **gated on an estimate of the localization
error** -- applied above roughly 0.3 m and skipped below -- or trained with a
loss that pins the identity at sigma = 0. Neither is done here, because doing
either would be tuning against the measurement this task exists to take. The
always-on numbers above are what the method does as specified.

Note also that the gap AlignFormer does not close is *not* pose error. Its
residual is nearly sigma-independent, so the remaining 0.54 -> 0.8764 is the
cost of correcting with an imperfect, detector-limited estimate at all: every
CAV box gets moved by ~0.47 m of residual, which is enough to drop a fused box
below the 0.7 IoU threshold even when the gross misalignment is gone. AP@0.3,
which tolerates that residual, stays at 0.85-0.87 against an oracle 0.9284 --
i.e. at the looser threshold AlignFormer recovers most of the gap.


## Tables

All of the following are rendered by `scripts/summarize_alignformer_p2.py` from
`outputs/alignformer/p2_gate.json` and `outputs/alignformer/p2_noisy_ap_result.json`.

## P2 gate

Gate configuration: **B / boxes+embeddings** -- **FAIL**

| sigma | Yaw MAE (deg) | Predict-zero yaw (deg) | Below? |
|---|---:|---:|---|
| sigma_0.2m | 0.2997 | 0.1595 | NO |
| sigma_0.4m | 0.2995 | 0.3189 | yes |
| sigma_0.6m | 0.2999 | 0.4784 | yes |
| sigma_0.8m | 0.3000 | 0.6379 | yes |
| sigma_1m | 0.3010 | 0.7974 | yes |
| sigma_1.5m | 0.3101 | 1.1961 | yes |
| sigma_2m | 0.3387 | 1.5947 | yes |

## Pose sweep

| sigma (m) | Configuration | Translation MAE (m) | Predict-zero translation (m) | Yaw MAE (deg) | Predict-zero yaw (deg) | Analytic predict-zero yaw (deg) |
|---|---|---:|---:|---:|---:|---:|
| 0 | B / boxes+embeddings | 0.2362 | 0.0000 | 0.3007 | 0.0000 | 0.0000 |
| 0 | B / boxes_only | 0.2366 | 0.0000 | 0.3081 | 0.0000 | 0.0000 |
| 0 | A / boxes+embeddings (match_weight 0) | 0.1332 | 0.0000 | 0.0711 | 0.0000 | 0.0000 |
| 0 | A / boxes_only (match_weight 0) | 0.1823 | 0.0000 | 0.0938 | 0.0000 | 0.0000 |
| 0 | B / boxes+embeddings (match_weight 0) | 0.2500 | 0.0000 | 0.3690 | 0.0000 | 0.0000 |
| 0.2 | B / boxes+embeddings | 0.2361 | 0.2618 | 0.2997 | 0.1595 | 0.1596 |
| 0.2 | B / boxes_only | 0.2361 | 0.2618 | 0.3078 | 0.1595 | 0.1596 |
| 0.2 | A / boxes+embeddings (match_weight 0) | 0.2491 | 0.2618 | 0.1745 | 0.1595 | 0.1596 |
| 0.2 | A / boxes_only (match_weight 0) | 0.2759 | 0.2618 | 0.1859 | 0.1595 | 0.1596 |
| 0.2 | B / boxes+embeddings (match_weight 0) | 0.2496 | 0.2618 | 0.3693 | 0.1595 | 0.1596 |
| 0.4 | B / boxes+embeddings | 0.2364 | 0.5236 | 0.2995 | 0.3189 | 0.3192 |
| 0.4 | B / boxes_only | 0.2361 | 0.5236 | 0.3080 | 0.3189 | 0.3192 |
| 0.4 | A / boxes+embeddings (match_weight 0) | 0.4023 | 0.5236 | 0.3203 | 0.3189 | 0.3192 |
| 0.4 | A / boxes_only (match_weight 0) | 0.4181 | 0.5236 | 0.3308 | 0.3189 | 0.3192 |
| 0.4 | B / boxes+embeddings (match_weight 0) | 0.2502 | 0.5236 | 0.3708 | 0.3189 | 0.3192 |
| 0.6 | B / boxes+embeddings | 0.2370 | 0.7854 | 0.2999 | 0.4784 | 0.4787 |
| 0.6 | B / boxes_only | 0.2360 | 0.7854 | 0.3080 | 0.4784 | 0.4787 |
| 0.6 | A / boxes+embeddings (match_weight 0) | 0.5620 | 0.7854 | 0.4697 | 0.4784 | 0.4787 |
| 0.6 | A / boxes_only (match_weight 0) | 0.5731 | 0.7854 | 0.4805 | 0.4784 | 0.4787 |
| 0.6 | B / boxes+embeddings (match_weight 0) | 0.2516 | 0.7854 | 0.3740 | 0.4784 | 0.4787 |
| 0.8 | B / boxes+embeddings | 0.2377 | 1.0472 | 0.3000 | 0.6379 | 0.6383 |
| 0.8 | B / boxes_only | 0.2365 | 1.0472 | 0.3095 | 0.6379 | 0.6383 |
| 0.8 | A / boxes+embeddings (match_weight 0) | 0.7266 | 1.0472 | 0.6214 | 0.6379 | 0.6383 |
| 0.8 | A / boxes_only (match_weight 0) | 0.7368 | 1.0472 | 0.6325 | 0.6379 | 0.6383 |
| 0.8 | B / boxes+embeddings (match_weight 0) | 0.2534 | 1.0472 | 0.3775 | 0.6379 | 0.6383 |
| 1 | B / boxes+embeddings | 0.2386 | 1.3090 | 0.3010 | 0.7974 | 0.7979 |
| 1 | B / boxes_only | 0.2369 | 1.3090 | 0.3105 | 0.7974 | 0.7979 |
| 1 | A / boxes+embeddings (match_weight 0) | 0.9004 | 1.3090 | 0.7756 | 0.7974 | 0.7979 |
| 1 | A / boxes_only (match_weight 0) | 0.9099 | 1.3090 | 0.7855 | 0.7974 | 0.7979 |
| 1 | B / boxes+embeddings (match_weight 0) | 0.2562 | 1.3090 | 0.3825 | 0.7974 | 0.7979 |
| 1.5 | B / boxes+embeddings | 0.2465 | 1.9635 | 0.3101 | 1.1961 | 1.1968 |
| 1.5 | B / boxes_only | 0.2449 | 1.9635 | 0.3208 | 1.1961 | 1.1968 |
| 1.5 | A / boxes+embeddings (match_weight 0) | 1.3852 | 1.9635 | 1.1699 | 1.1961 | 1.1968 |
| 1.5 | A / boxes_only (match_weight 0) | 1.3931 | 1.9635 | 1.1765 | 1.1961 | 1.1968 |
| 1.5 | B / boxes+embeddings (match_weight 0) | 0.2741 | 1.9635 | 0.4079 | 1.1961 | 1.1968 |
| 2 | B / boxes+embeddings | 0.2854 | 2.6180 | 0.3387 | 1.5947 | 1.5958 |
| 2 | B / boxes_only | 0.2825 | 2.6180 | 0.3540 | 1.5947 | 1.5958 |
| 2 | A / boxes+embeddings (match_weight 0) | 1.9311 | 2.6180 | 1.5682 | 1.5947 | 1.5958 |
| 2 | A / boxes_only (match_weight 0) | 1.9361 | 2.6180 | 1.5721 | 1.5947 | 1.5958 |
| 2 | B / boxes+embeddings (match_weight 0) | 0.3408 | 2.6180 | 0.4770 | 1.5947 | 1.5958 |

## Fused AP under localization error

| sigma (m) | Condition | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---:|---:|---:|
| -- | oracle (true pose) | 0.9284 | 0.9251 | 0.8764 |
| 0 | vanilla late fusion (uncorrected) | 0.9284 | 0.9251 | 0.8764 |
| 0 | AlignFormer-corrected | 0.8726 | 0.7803 | 0.5424 |
| 0.2 | vanilla late fusion (uncorrected) | 0.9277 | 0.8991 | 0.5846 |
| 0.2 | AlignFormer-corrected | 0.8731 | 0.7800 | 0.5419 |
| 0.4 | vanilla late fusion (uncorrected) | 0.8928 | 0.6699 | 0.3069 |
| 0.4 | AlignFormer-corrected | 0.8731 | 0.7788 | 0.5398 |
| 0.6 | vanilla late fusion (uncorrected) | 0.7595 | 0.4507 | 0.2106 |
| 0.6 | AlignFormer-corrected | 0.8711 | 0.7734 | 0.5368 |
| 0.8 | vanilla late fusion (uncorrected) | 0.6207 | 0.3476 | 0.1785 |
| 0.8 | AlignFormer-corrected | 0.8711 | 0.7698 | 0.5320 |
| 1 | vanilla late fusion (uncorrected) | 0.5268 | 0.2978 | 0.1690 |
| 1 | AlignFormer-corrected | 0.8664 | 0.7672 | 0.5303 |
| 1.5 | vanilla late fusion (uncorrected) | 0.3856 | 0.2484 | 0.1737 |
| 1.5 | AlignFormer-corrected | 0.8604 | 0.7543 | 0.5198 |
| 2 | vanilla late fusion (uncorrected) | 0.3365 | 0.2419 | 0.1831 |
| 2 | AlignFormer-corrected | 0.8459 | 0.7348 | 0.4961 |

## Gap recovered at AP@0.7

| sigma (m) | Vanilla AP@0.7 | AlignFormer AP@0.7 | Oracle AP@0.7 | Gain over vanilla | Gap recovered |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8764 | 0.5424 | 0.8764 | -0.3340 | -- |
| 0.2 | 0.5846 | 0.5419 | 0.8764 | -0.0427 | -14.7% |
| 0.4 | 0.3069 | 0.5398 | 0.8764 | +0.2329 | 40.9% |
| 0.6 | 0.2106 | 0.5368 | 0.8764 | +0.3262 | 49.0% |
| 0.8 | 0.1785 | 0.5320 | 0.8764 | +0.3535 | 50.7% |
| 1 | 0.1690 | 0.5303 | 0.8764 | +0.3613 | 51.1% |
| 1.5 | 0.1737 | 0.5198 | 0.8764 | +0.3461 | 49.3% |
| 2 | 0.1831 | 0.4961 | 0.8764 | +0.3130 | 45.2% |

## Pose error on the test split

| sigma (m) | Pairs | Translation MAE (m) | Predict-zero (m) | Yaw MAE (deg) | Predict-zero (deg) | Unalignable pairs | Fell back |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3445 | 0.4687 | 0.0000 | 0.6677 | 0.0000 | 37 | 0.012 |
| 0.2 | 3445 | 0.4732 | 0.2797 | 0.6694 | 0.1610 | 37 | 0.012 |
| 0.4 | 3445 | 0.4789 | 0.5450 | 0.6739 | 0.3241 | 37 | 0.012 |
| 0.6 | 3445 | 0.4820 | 0.8140 | 0.6774 | 0.4865 | 37 | 0.012 |
| 0.8 | 3445 | 0.4872 | 1.0904 | 0.6786 | 0.6308 | 37 | 0.012 |
| 1 | 3445 | 0.4901 | 1.3459 | 0.6799 | 0.7892 | 37 | 0.012 |
| 1.5 | 3445 | 0.5206 | 2.0195 | 0.7119 | 1.1962 | 37 | 0.012 |
| 2 | 3445 | 0.5649 | 2.6933 | 0.7433 | 1.6053 | 37 | 0.015 |

## Training curve: stage2_B_boxes+embeddings

| Epoch | sigma (m) | Train loss | Train corner (m) | Val corner (m) | Val translation MAE (m) | Val yaw MAE (deg) | Match temperature |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 4.2792 | 4.1866 | 6.0483 | 1.0680 | 1.0013 | 0.03785 |
| 2 | 0.069 | 5.0672 | 4.8599 | 6.1841 | 1.0424 | 0.9005 | 0.03603 |
| 3 | 0.138 | 5.0386 | 4.8218 | 6.6001 | 1.0883 | 1.1075 | 0.03457 |
| 4 | 0.207 | 4.2561 | 4.0504 | 9.4231 | 1.7160 | 1.5193 | 0.03419 |
| 5 | 0.276 | 4.1957 | 3.9919 | 5.7931 | 1.0072 | 0.9660 | 0.03377 |
| 6 | 0.345 | 3.9081 | 3.7180 | 5.7488 | 0.9687 | 1.0504 | 0.03419 |
| 7 | 0.414 | 3.5724 | 3.3843 | 6.5400 | 1.0511 | 1.3697 | 0.03396 |
| 8 | 0.483 | 3.5697 | 3.3793 | 5.0648 | 0.8733 | 0.8484 | 0.03397 |
| 9 | 0.552 | 3.4203 | 3.2429 | 4.6953 | 0.8012 | 0.8564 | 0.03434 |
| 10 | 0.621 | 3.2310 | 3.0630 | 4.0849 | 0.6853 | 0.7325 | 0.03477 |
| 11 | 0.690 | 2.8919 | 2.7211 | 4.4900 | 0.7420 | 0.7725 | 0.03515 |
| 12 | 0.759 | 3.1307 | 2.9658 | 3.8997 | 0.6657 | 0.7476 | 0.03518 |
| 13 | 0.828 | 2.7926 | 2.6318 | 3.7794 | 0.6230 | 0.6876 | 0.03559 |
| 14 | 0.897 | 2.7730 | 2.6157 | 3.6064 | 0.5416 | 0.7929 | 0.03579 |
| 15 | 0.966 | 2.4973 | 2.3456 | 2.9677 | 0.4931 | 0.5502 | 0.03603 |
| 16 | 1.034 | 2.3605 | 2.2111 | 2.9421 | 0.4605 | 0.5689 | 0.03610 |
| 17 | 1.103 | 2.2933 | 2.1463 | 2.5665 | 0.4204 | 0.4805 | 0.03608 |
| 18 | 1.172 | 2.0487 | 1.9063 | 2.5285 | 0.4112 | 0.4659 | 0.03614 |
| 19 | 1.241 | 2.2241 | 2.0858 | 2.3761 | 0.3884 | 0.4239 | 0.03624 |
| 20 | 1.310 | 2.0342 | 1.8986 | 2.2668 | 0.3588 | 0.4095 | 0.03634 |
| 21 | 1.379 | 1.8512 | 1.7197 | 2.2173 | 0.3466 | 0.4159 | 0.03645 |
| 22 | 1.448 | 1.7484 | 1.6186 | 2.0588 | 0.3270 | 0.3872 | 0.03654 |
| 23 | 1.517 | 1.7335 | 1.6056 | 1.9840 | 0.3116 | 0.3633 | 0.03655 |
| 24 | 1.586 | 1.5789 | 1.4513 | 1.9561 | 0.3118 | 0.3607 | 0.03654 |
| 25 | 1.655 | 1.5816 | 1.4552 | 1.8971 | 0.2967 | 0.3586 | 0.03652 |
| 26 | 1.724 | 1.5776 | 1.4526 | 1.8790 | 0.2955 | 0.3461 | 0.03652 |
| 27 | 1.793 | 1.5847 | 1.4593 | 1.8537 | 0.2924 | 0.3458 | 0.03651 |
| 28 | 1.862 | 1.5657 | 1.4409 | 1.8222 | 0.2867 | 0.3410 | 0.03651 |
| 29 | 1.931 | 1.6028 | 1.4751 | 1.8108 | 0.2842 | 0.3378 | 0.03651 |
| 30 | 2.000 | 1.6706 | 1.5414 | 1.8168 | 0.2850 | 0.3387 | 0.03651 |


## What this means for P3-P5

The plan's "After P2" branch asks which of three outcomes obtains. The answer is
a fourth one, so it is worth stating precisely.

**The yaw gate failed, but not in the way the plan feared.** The plan's failure
mode was "flat at the predict-zero value" -- the CoLoca-QuA outcome, a head that
never leaves the conditional mean. That is exactly what **head A** does here
(yaw MAE within 2-3% of predict-zero at every sigma), and it is a clean
reproduction of the CoLoca-QuA finding on AlignFormer's own trunk. **Head B does
not do that at all**: it is 4.7x better than predict-zero at sigma = 2 m and its
error is independent of the input noise. The closed-form premise is *supported*,
not refuted. What failed is the gate's universal quantifier, at the one sigma
where the model's floor exceeds the error being corrected.

So the plan's prescribed response to a failed gate -- "stop, and diagnose with
oracle correspondences vs Sinkhorn to separate a matching failure from a solver
failure" -- is already answered by the evidence in hand: it is neither. Matching
is at Top-1 0.9965 and the solver removes essentially all of the injected error.
The residual is the **detector's** box noise propagating through an otherwise
working estimator, which is a third failure mode the diagnostic tree did not
have a branch for.

**The embedding does not help, confirming ruling R39 at the pose level too.**
The plan's second branch then applies: the contribution shifts to the
closed-form solver plus the efficiency argument, and the camera-augmentation
contingency (spec 8) is live. One qualification worth carrying forward: the
`match_weight 0` control shows the *matching supervision* is worth 22% of the
corner loss even though the *embedding* is worth nothing. The value is in
learning a correspondence from geometry, not in the appearance descriptor.

Three things follow, in priority order:

1. **Fix the floor, not the gate.** The residual is ~0.47 m and ~0.67 deg on the
   test split and is what caps fused AP@0.7 at ~0.54 against an oracle 0.8764.
   It is detector-limited, so the levers are the detector and the loss, not more
   pose training. This is the highest-value next experiment and it is not in the
   current plan.
2. **Gate the correction on estimated noise.** Always-on costs 0.334 AP@0.7 in
   the clean case. A confidence- or noise-conditioned switch turns the method
   from "a large win above 0.3 m and a loss below it" into "a large win above
   0.3 m and a no-op below it", which is strictly better and cheap.
3. **Raise `MIN_MATCH_MASS` to its documented intent** and re-measure the
   unalignable subset.
