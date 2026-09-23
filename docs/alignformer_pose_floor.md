# AlignFormer's pose error floor: diagnosis and fix

[P2](alignformer_p2.md) measured a **constant pose error floor** that did not
depend on the injected localization noise. It capped corrected fusion at ~61%
of oracle at every sigma and, because it applies at sigma = 0 too, made
AlignFormer *cost* 0.334 AP@0.7 when localization was already perfect. This
document is the diagnosis of that floor and the fix. **Every P2 number for
head B in [alignformer_p2.md](alignformer_p2.md) is superseded by the tables
here.**

## Results in one place

| Question | Answer |
|---|---|
| Is the floor bias or noise? | **Noise.** Signed mean of the residual at sigma = 0 on 4176 validation pairs: dx +0.0072 m, dy -0.0038 m, dpsi +0.0078 deg -- a 0.0081 m mean vector against a 0.119 m translation MAE, and a yaw mean whose 95% CI covers zero. |
| Is the solver at fault? | **No.** Given oracle correspondences and noiseless boxes it returns the ground-truth SE(2) to 1.3e-6 m and 1.8e-6 deg over 200 validation geometries. The solver, the ego/cav argument order and the SE(2) conventions are all correct. |
| What produced most of the noise? | **A real defect: the heading channel took a direction the detector never estimates.** 20.3% of cross-agent detections of the same physical object disagree by ~180 deg, and each one displaced a heading virtual point by `2 * heading_lambda`. |
| **P2 gate** after the fix | **PASS, 7 of 7 non-zero sigmas**, definition unmodified. |
| **AP@0.7 at sigma = 0.2 m** | **0.7096**, against 0.5846 uncorrected and 0.5419 before the fix. |
| **Gap recovered** | **68-75% of the oracle-vs-uncorrected gap** from 0.4 to 2.0 m, up from 41-51%. |
| What still costs AP | At sigma = 0, AlignFormer scores 0.7665 against the oracle's 0.8764. **33.8% of test pairs are outside the 40 m communication range the model was trained on** and their residual is 2.9x worse; that, not the estimator, is what is left. |

## 1. Bias or noise

Three measurements, and the order matters: a bias would be a bug worth finding,
and shrinking it would have hidden it.

**The signed mean of the residual**, at sigma = 0 on the scenario-disjoint
validation split, where the true correction is exactly the identity and the
emitted correction therefore *is* the residual:

| Component | Signed mean | 95% CI |
|---|---:|---|
| dx | +0.00716 m | [+0.00340, +0.01092] |
| dy | -0.00376 m | [-0.00710, -0.00043] |
| dpsi | +0.00778 deg | [-0.00092, +0.01654] |

Measured on the same pairs, the translation MAE is 0.1192 m and the yaw MAE
0.1648 deg. The mean *vector* is 0.0081 m long -- 6.8% of the translation
error in magnitude and 0.2% of its mean squared error -- and the yaw mean is
not significant at all. The estimator is unbiased and imprecise.

**The solver control.** Fed oracle one-to-one correspondences and noiseless
boxes (the ego's own boxes, displaced by a known SE(2)), `weighted_se2_kabsch`
plus `augment_with_heading` recovers that SE(2) to a maximum error of
**1.31e-6 m** and **1.81e-6 deg** across 200 validation geometries. Nothing in
the solver path is wrong.

**The decomposition.** At sigma = 0 the fitted translation is essentially the
mean disagreement over matched objects, so averaging-limited noise is
predictable:

| Quantity | Value |
|---|---:|
| Per-axis centre disagreement between agents, per correspondence | 0.2198 m (mean +0.008, +0.007 -- unbiased) |
| Matched objects per pair | 10.92 mean, 11 median |
| Prediction from averaging alone (`sigma / sqrt(n)`, Rayleigh mean) | 0.083 m |
| Centroid difference on oracle correspondences, rotation forced to zero | 0.089 m |
| **Full SE(2) solve on oracle correspondences** | **0.434 m** |

The last row is 5x the prediction. The yaw lever arm cannot explain it:
`mean(|q_bar| * |psi_error|)` is only 0.066 m with a 14.4 m mean lever. The
excess enters through the heading virtual points themselves.

## 2. The defect: a heading the detector does not estimate

`configs/alignformer_detector.yaml` declares **no `dir_args` head** and its
anchors are `r: [0, 90]`. The detector therefore fixes a box's *axis*, not
which way along it the vehicle points -- a vehicle box is symmetric under a
180 degree rotation and nothing in the network breaks that symmetry.

Measured on the 45,622 cross-agent correspondences of the validation split:

| Heading disagreement between two agents' detections of the same object | Share |
|---|---:|
| within 30 deg (core std 4.88 deg) | 79.2% |
| beyond 150 deg (a flip) | **20.3%** |

`augment_with_heading` placed a virtual point at
`centre + lambda * (cos yaw, sin yaw)`, so each flipped detection put its
virtual point `2 * lambda = 4 m` away from its counterpart. The floor scales
linearly with lambda, which is the signature:

| Heading treatment (oracle correspondences, sigma = 0) | Translation MAE | Yaw MAE |
|---|---:|---:|
| as-is, lambda = 0 (centres only) | 0.0956 m | 0.114 deg |
| as-is, lambda = 1 | 0.2544 m | 0.166 deg |
| **as-is, lambda = 2 (shipped)** | **0.4336 m** | **0.243 deg** |
| as-is, lambda = 4 | 0.7996 m | 0.417 deg |
| **pi ambiguity folded, lambda = 2** | **0.1044 m** | **0.138 deg** |

`procrustes.heading_orientation` is the fold: for each (ego object, CAV object)
it returns +1 or -1 according to whether the CAV heading points within 90
degrees of the ego heading it is being matched to, and
`AlignFormerB._soft_correspondence` applies that sign to each CAV heading
vector **before** the soft-match average. It must be per pair, not per CAV
object, because one ego row can soft-match CAV boxes that are flipped
differently from one another. Folding against the ego's own heading is safe
here: the corrections being estimated are a couple of degrees, nowhere near the
90 degree decision boundary, and if the ego's own heading is itself flipped
relative to the world, both sides move together and the fit is unaffected.

Retraining head B on the folded channel (same config, same warm start, same 30
epochs), validation at the curriculum maximum sigma = 2 m:

| | Corner loss | Translation MAE | Yaw MAE |
|---|---:|---:|---:|
| before | 1.8108 m | 0.2842 m | 0.3378 deg |
| **after** | **0.9676 m** | **0.1480 m** | **0.1997 deg** |

## 3. Shrinkage for what is left

What remains is genuine estimator noise, and an unbiased-but-imprecise
correction does harm exactly when there is little error to correct. The
treatment is shrinkage, not a noise-conditioned switch. With
`x = e + n`, `e ~ N(0, sigma^2)` the true pose error and `n ~ N(0, tau^2)` the
estimator's own noise, the posterior mean of `e` is
`sigma^2 / (sigma^2 + tau^2) * x`. `sigma` is unknown at inference, so it is
estimated from the observation itself: standardize each component by its
calibrated noise scale, pool them,

```
z^2 = 2 |t|^2 / tau_t^2  +  psi^2 / tau_psi^2          (p = 3 dimensions)
k   = max(0, 1 - p / z^2)
```

and apply the one factor `k` to the whole SE(2). This is the positive-part
empirical-Bayes rule; James-Stein's dominance result applies because the pooled
problem is three-dimensional where a translation alone would be two.

**Pooling is load-bearing, not tidiness.** Estimating `sigma` from the yaw
alone is a single observation of a heavy-tailed quantity (its RMS is 1.75x its
MAE) and over-shrinks badly: measured, a per-component rule pushed yaw MAE at
sigma = 0.2 m to 0.1637 deg against a 0.1595 predict-zero -- i.e. back onto the
baseline, failing the gate -- while the pooled rule with the *same* tau gives
0.1511 and passes. The translation carries most of the evidence about how large
the localization error is, and the yaw is entitled to use it.

`tau` is calibrated **once**, on the validation split only, at sigma = 0:

| | |
|---|---:|
| `tau_translation_m` (RMS of the residual's norm) | 0.16569 |
| `tau_yaw_deg` (RMS of the yaw residual) | 0.28912 |
| pairs | 4176 |

It is the same at every sigma and on the test split. There is no per-sigma
factor anywhere; `outputs/alignformer/shrinkage_calibration_result.json` is the
record.

## 4. The P2 gate

Head B (boxes+embeddings) yaw MAE strictly below the **empirical** predict-zero
value at every non-zero sigma. Definition unchanged from P2, mean of 3
independent noise draws on the scenario-disjoint validation split.

**Verdict: PASS (7 of 7).**

| sigma (m) | Yaw MAE (deg) | Predict-zero (deg) | Below? | P2's value |
|---|---:|---:|---|---:|
| 0.2 | **0.1511** | 0.1595 | yes | 0.2997 (no) |
| 0.4 | **0.1674** | 0.3189 | yes | 0.2995 |
| 0.6 | **0.1691** | 0.4784 | yes | 0.2999 |
| 0.8 | **0.1695** | 0.6379 | yes | 0.3000 |
| 1.0 | **0.1705** | 0.7974 | yes | 0.3010 |
| 1.5 | **0.1750** | 1.1961 | yes | 0.3101 |
| 2.0 | **0.1922** | 1.5947 | yes | 0.3387 |

And the clean case the gate excludes, which is what the fix exists for:

| sigma = 0, validation | Translation MAE | Yaw MAE |
|---|---:|---:|
| P2 (before) | 0.2361 m | 0.2997 deg |
| heading fold only | 0.1192 m | 0.1648 deg |
| **heading fold + shrinkage** | **0.0264 m** | **0.0463 deg** |

## 5. Fused AP under localization error

Official 2170-frame OPV2V test split, global-sorted AP, one set of detections
per frame shared by all three conditions. AP@0.7:

| sigma (m) | Uncorrected | AlignFormer (P2) | **AlignFormer (fixed)** | Oracle | Gain | Gap recovered |
|---|---:|---:|---:|---:|---:|---:|
| 0 | **0.8764** | 0.5424 | 0.7665 | 0.8764 | -0.1099 | -- |
| 0.2 | 0.5846 | 0.5419 | **0.7096** | 0.8764 | **+0.1250** | 42.8% |
| 0.4 | 0.3069 | 0.5398 | **0.6963** | 0.8764 | **+0.3894** | 68.4% |
| 0.6 | 0.2106 | 0.5368 | **0.6948** | 0.8764 | **+0.4842** | 72.7% |
| 0.8 | 0.1785 | 0.5320 | **0.6971** | 0.8764 | **+0.5186** | 74.3% |
| 1.0 | 0.1690 | 0.5303 | **0.6957** | 0.8764 | **+0.5267** | 74.5% |
| 1.5 | 0.1737 | 0.5198 | **0.6845** | 0.8764 | **+0.5107** | 72.7% |
| 2.0 | 0.1831 | 0.4961 | **0.6707** | 0.8764 | **+0.4876** | 70.3% |

### Which half of the fix did what

Same checkpoint, same sweep, shrinkage on and off (AP@0.7):

| sigma (m) | Uncorrected | P2 (no fold, no shrinkage) | Fold only | **Fold + shrinkage** |
|---|---:|---:|---:|---:|
| 0 | **0.8764** | 0.5424 | 0.7118 | **0.7665** |
| 0.2 | 0.5846 | 0.5419 | 0.7092 | **0.7096** |
| 0.4 | 0.3069 | 0.5398 | **0.7057** | 0.6963 |
| 0.6 | 0.2106 | 0.5368 | **0.7030** | 0.6948 |
| 0.8 | 0.1785 | 0.5320 | **0.7016** | 0.6971 |
| 1.0 | 0.1690 | 0.5303 | **0.6988** | 0.6957 |
| 1.5 | 0.1737 | 0.5198 | **0.6897** | 0.6845 |
| 2.0 | 0.1831 | 0.4961 | **0.6715** | 0.6707 |

The heading fold is where the gain is: +0.167 to +0.177 AP@0.7 at every sigma.
Shrinkage buys the clean case -- **+0.055 at sigma = 0**, break-even at 0.2 m --
for **0.001 to 0.009** at higher sigma, and it is what turns the P2 gate from
fail to pass. That is the trade it exists to make.

All three thresholds:

| Condition | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| oracle | 0.9284 | 0.9251 | 0.8764 |
| uncorrected, sigma 0 | 0.9284 | 0.9251 | 0.8764 |
| alignformer, sigma 0 | 0.8957 | 0.8677 | 0.7665 |
| uncorrected, sigma 0.2 | 0.9277 | 0.8991 | 0.5846 |
| alignformer, sigma 0.2 | 0.8959 | 0.8640 | 0.7096 |
| uncorrected, sigma 0.4 | 0.8928 | 0.6699 | 0.3069 |
| alignformer, sigma 0.4 | 0.8956 | 0.8591 | 0.6963 |
| uncorrected, sigma 0.6 | 0.7595 | 0.4507 | 0.2106 |
| alignformer, sigma 0.6 | 0.8942 | 0.8562 | 0.6948 |
| uncorrected, sigma 0.8 | 0.6207 | 0.3476 | 0.1785 |
| alignformer, sigma 0.8 | 0.8922 | 0.8514 | 0.6971 |
| uncorrected, sigma 1.0 | 0.5268 | 0.2978 | 0.1690 |
| alignformer, sigma 1.0 | 0.8901 | 0.8468 | 0.6957 |
| uncorrected, sigma 1.5 | 0.3856 | 0.2484 | 0.1737 |
| alignformer, sigma 1.5 | 0.8857 | 0.8413 | 0.6845 |
| uncorrected, sigma 2.0 | 0.3365 | 0.2419 | 0.1831 |
| alignformer, sigma 2.0 | 0.8760 | 0.8247 | 0.6707 |

The oracle condition still reproduces `p0_gate.json` to four decimals, and
`uncorrected` at sigma = 0 still equals it, so the pipeline is the same one P0
and P2 measured.

V2X-ViT's published noisy AP@0.7 on OPV2V is 0.614. AlignFormer's 0.7096 at
sigma = 0.2 m is above it, but **that comparison is not yet valid**: 0.614 was
measured under V2X-ViT's own noise convention and its own detector, and it has
to be re-run locally before it can stand beside these numbers.

## 6. What is left, and where it comes from

AlignFormer still costs 0.110 AP@0.7 at sigma = 0, so it is not yet
unconditionally safe to leave on. The sweep now splits its pose error by
whether the pair is inside the communication range the model was trained on,
and that settles where the remainder lives.

**`configs/alignformer.yaml` builds its pair index at `comm_range_m: 40.0`.
OpenCOOD's `LateFusionDataset` admits every CAV within
`opencood.data_utils.datasets.COM_RANGE = 70` m.** So a third of the test
population is out of distribution by construction, and the same filter hid
35.8% of the training pairs from the model during training.

| | Within 40 m | Beyond 40 m |
|---|---:|---:|
| Test pairs (per sigma) | 2282 (66.2%) | 1163 (**33.8%**) |
| Translation MAE at sigma = 0 | 0.1244 m | **0.3555 m** |
| Yaw MAE at sigma = 0 | 0.2227 deg | **0.5906 deg** |
| Fully suppressed by shrinkage at sigma = 0 | 68.8% | 56.2% |

The out-of-range third is 2.9x worse on translation and 2.7x worse on yaw, and
shrinkage suppresses *less* of it, because a calibration fitted on in-range
data reads its larger corrections as real signal.

Comparing like with like -- both before shrinkage, so validation and test are
measured the same way -- the range split accounts for roughly half of the
validation-to-test gap and no more:

| Translation MAE at sigma = 0, no shrinkage | Value |
|---|---:|
| Validation (all pairs, all within 40 m by construction) | 0.1192 m |
| Test, within 40 m | 0.2208 m |
| Test, all pairs | 0.2976 m |
| Test, beyond 40 m | 0.4483 m |

So the out-of-range third roughly doubles the aggregate, but the *in-range*
test residual is still 1.85x the validation one. That remainder is not
explained here: it is either genuine scenario difficulty on the held-out split
or a further difference between the cached training path and the live test
path, and separating those two is the second thing to do.

**This is the next thing to fix, and it is not a modelling problem.** Rebuild
the pair index at 70 m to match the evaluation protocol, extend the detection
cache to the agent-frames that brings in, and retrain. Until then a deployment
would be entitled to leave the correction off for pairs beyond 40 m, but that
is a workaround for a mismatch that should not exist.

## Reproducing

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD

# Retrain head B on the folded heading channel (~14 min on an RTX 4090)
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 2 --head B \
  --message-content boxes+embeddings \
  --output-dir outputs/alignformer/stage2_B_heading_fold

# Calibrate shrinkage: validation only, sigma = 0 only
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer.yaml --metric shrinkage \
  --checkpoint outputs/alignformer/stage2_B_heading_fold/best.pth \
  --output outputs/alignformer/shrinkage_calibration_result.json

# The P2 gate, definition unmodified
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer.yaml --metric pose \
  --checkpoint outputs/alignformer/stage2_B_heading_fold/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
  --shrinkage outputs/alignformer/shrinkage_calibration_result.json \
  --output outputs/alignformer/p2_fixed_gate.json

# Fused AP under localization error, official 2170-frame test split (~40 min)
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer_detector.yaml \
  --split /media/chenyi/Elements1/Dataset/OPV2V/test \
  --metric noisy_ap --alignformer-config configs/alignformer.yaml \
  --checkpoint outputs/alignformer/stage2_B_heading_fold/best.pth \
  --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 \
  --shrinkage outputs/alignformer/shrinkage_calibration_result.json \
  --output outputs/alignformer/p2_fixed_noisy_ap_result.json
```

Omitting `--shrinkage` from the last two reproduces the heading-fold-only
condition (`p2_heading_fold_gate.json`,
`p2_heading_fold_noisy_ap_result.json`).

## Caveats

1. **Only head B / boxes+embeddings was retrained on the fold.** The head A
   comparison, the boxes-only ablation and the `match_weight 0` control in
   [alignformer_p2.md](alignformer_p2.md) are all pre-fix numbers. The
   embedding-is-null conclusion is unlikely to move -- the fold is geometry,
   not appearance -- but it has not been re-measured.
2. **The clean case is still a cost** (-0.110 AP@0.7). Section 6 says where it
   comes from; it is not resolved.
3. **`tau` is a single global constant.** Scaling it by the match evidence was
   tried and is not supported: across confidence octiles the residual's
   `RMS * sqrt(confidence)` ranges 0.43-0.61 rather than staying flat, so the
   `1/sqrt(n)` model the averaging argument suggests does not hold cleanly
   enough to justify the extra parameter.
4. **The yaw residual is heavy-tailed** (RMS 1.75x MAE), so an RMS-based `tau`
   shrinks the typical pair harder than a Gaussian assumption would. Pooling
   over three dimensions absorbs most of that, but a robust scale estimate is
   untested.
5. **One noise seed for the AP sweep**, three for the pose sweep, as in P2.
6. **`MIN_MATCH_MASS` is still half what it documents** (P2 finding 6c),
   untouched here.
