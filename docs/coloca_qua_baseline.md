# CoLoca-QuA baseline on OPV2V

Reproduction of Gao, Xiang, Tan, Meng, Xia, Xiong and Ma, *"CoLoca-QuA:
Cooperative Relative Pose Estimation With Query-Based Attention Using Neural
Intermediate Features in V2X Network"*, IEEE Transactions on Vehicular
Technology 75(7), July 2026, pp. 12309-12321.

The paper is a **cooperative localization** method, not a detector: it regresses
the residual relative pose error between an ego vehicle and a CAV whose
self-reported pose is corrupted by localization noise. We reproduce it here as a
baseline for the embedding-aware BELT-fusion work.

## Method

1. Ego and CAV each run a shared PointPillars backbone on their own LiDAR. The
   CAV cloud is first projected into the ego frame using its **noisy** pose
   (Eq. 2), so the two clouds are deliberately misaligned.
2. The two BEV feature maps are concatenated along channels (Eq. 9).
3. A single strided convolution tokenizes the map into `N = HW/P^2` patches
   (Eq. 10-11); a learnable **Localization Token** is prepended and learnable
   positional embeddings are added (Eq. 12).
4. A 6-layer pre-LN ViT encoder lets the token query every patch; its final
   state goes through an MLP to `[dx, dy, dpsi]` (Eq. 13).
5. Loss is a per-component weighted MSE with `(alpha_x, alpha_y, alpha_psi) =
   (2, 2, 1)` (Eq. 14, Section IV-F setting S3).

The regressed correction `C` is defined by Eq. (5)-(7) so that
`C @ T_noisy == T_true`, where `T` maps the CAV frame into the ego frame.

## Deviations from the paper, and why

| Aspect | Paper | Here | Reason |
|---|---|---|---|
| Dataset | V2XSet, V2X-Real | OPV2V | What is available locally. OPV2V is the V2V-only sibling of V2XSet from the same CARLA/OpenCDA generator, so the comparison target is the paper's **V2V** columns of Table II. OPV2V has no infrastructure agents, so the V2I columns are out of scope. |
| Stage 1 backbone | F-Cooper trained by the authors | OpenCOOD's released F-Cooper OPV2V checkpoint | Supplied pretrained; identical architecture. |
| LiDAR | V2XSet 32-line | OPV2V 64-line | Dataset difference. OPV2V clouds are denser, which should if anything help. |
| Voxel size | stated 0.2 m | 0.4 m | See "Resolving the voxel-size inconsistency" below. |
| Validation split | not stated | 15% of train scenarios | OPV2V's `validate` split is a broken symlink on this machine. Split is by **scenario**, never by frame, since consecutive OPV2V frames are near-duplicates. |
| Agent pairing | "two-agent samples" | every ordered (ego, CAV) pair within 40 m | The correction is expressed in the ego frame, so the problem is not symmetric; using each agent as ego in turn is the natural reading and doubles the sample count. |

### Resolving the voxel-size inconsistency

Section IV-B states a voxel size of 0.2 x 0.2 m for V2XSet, and an evaluation
range of `x in [-102.4, 102.4]`, `y in [-38.4, 38.4]`. Section IV-E states the
localization module's input is `512 x 96 x 256` with `C_in = 512`, and Table I /
Fig. 4 give `C = 256, H = 96, W = 256, N = 96`.

Those two statements are inconsistent. At 0.2 m the pillar grid is 1024 x 384,
and OpenCOOD's `BaseBEVBackbone` emits a stride-2 map, giving 512 x 192 - not
256 x 96. At **0.4 m** the grid is 512 x 192 and the stride-2 map is exactly
**96 x 256**, and F-Cooper's shrink header compresses 384 -> **256** channels,
so the concatenated pair is exactly **512 x 96 x 256** with `N = 6 * 16 = 96`
tokens.

We follow the tensor dimensions, which are stated three times and are
self-consistent, over the single voxel-size sentence. The resulting
configuration reproduces every reported shape exactly.

As an independent check, the paper reports 3.70 GMACs for the localization head.
Our implementation computes ~3.4 GMACs (3.22 G in the patch embedding plus
0.17 G in the encoder), which corroborates the patch count and embedding
dimension. The parameter count does not match as cleanly: we get 38.35 M against
the paper's reported 48.74 M, an unexplained 10.4 M gap. Every hyperparameter in
Table I is implemented as stated, so this is most likely a reporting difference
in the paper's final MLP rather than an architectural divergence.

### Noise model

Section IV-B: "pose noises are ... randomly sampled from an error dataset that
includes Gaussian noise with a mean value of `mu = 0` and standard deviation of
`sigma = 2 m` and `sigma = 1 m`", plus heading noise with `sigma = 1 deg`.
Section IV-C then evaluates the trained model at each sigma separately. We read
this as **one** model trained on a mixture of both sigmas, which is what
`noise.train_xy_std: [2.0, 1.0]` implements.

Noise is applied to the CAV pose only, per-axis on x and y, and to yaw. `z`,
roll and pitch are untouched because the method estimates only the planar pose.

### The planar assumption is weaker on OPV2V than the paper assumes

Section III-A justifies the SE(2) restriction with "the vehicle's pitch and roll
angles are typically within 5 degrees". Measured over the 22,672 OPV2V train
pairs that is true for roll (std 0.27 deg) but **not** for pitch: std 2.85 deg,
p99 12.3 deg, max 17.1 deg. OPV2V includes hilly scenes.

This matters in a specific and limited way. The correction is conjugated by the
ego pose, so a world-frame yaw error remains a pure z-rotation in the ego frame
only when the *ego* is level. `tests/test_coloca_alignment.py` measures the
consequence:

| Ego tilt (roll/pitch) | Residual after applying the correction | Raw misalignment |
|---|---:|---:|
| 0.0 / 0.0 deg (median) | 0.000 m (exact) | 2.76 m |
| 0.6 / 12.3 deg (p99) | 0.338 m | 2.26 m |
| 18.5 / 17.1 deg (max) | 0.719 m | 2.77 m |

**The SE(2) label itself is exact** - it is computed in closed form and the
round-trip `C @ T_noisy == T_true` holds to 1e-9. So the reported MAE/RMSE,
which compare the predicted against the true SE(2) pose error, are unaffected.
What the tilt limits is how completely the correction re-aligns the full 3-D
clouds for a *downstream* consumer, and it adds some input noise, since BEV
features collapse the z axis and pitch-induced misalignment partly appears as
in-plane distortion. Expect a modest disadvantage relative to the paper's
V2XSet numbers from this alone.

### Metric definitions

The paper does not define MAE precisely for a 2-D quantity. We use the mean
Euclidean norm of the translation residual, which is the reading consistent with
its own numbers: at `sigma = 2 m` per-axis, the expected norm of the raw input
error is `sigma * sqrt(pi/2) = 2.51 m`, and the paper reports PCM at 2.84 m,
i.e. slightly *worse* than doing nothing - which matches its statement that
"traditional point cloud matching methods collapse". The threshold metrics
("error < 1 m") are the fraction of samples whose residual norm is below the
threshold. Evaluation always reports a **no-correction control** alongside the
model so the improvement is unambiguous.

## Setup on this machine

- Env: conda `opencood` (Python 3.8, torch 2.3.1+cu121, spconv). Note this is
  **not** the `belt-fusion` env, which lacks spconv and a working OpenCOOD.
- R26: `pyproject.toml` declares `requires-python = ">=3.9"`, but the
  `opencood` env this project actually runs in is **3.8.19** -- 3.8 is
  authoritative for all AlignFormer code (e.g. no builtin generics like
  `list[int]`; use `typing.List`/`Dict`/`Tuple`/`Optional` instead). The
  `pyproject.toml` bound has not been reconciled with this yet.
- OpenCOOD: `external/OpenCOOD` submodule, used via `PYTHONPATH` (its
  `iou3d_nms` CUDA extension is unbuilt, but CoLoca-QuA never needs NMS - the
  detection heads are discarded).
- Backbone checkpoint: `/media/chenyi/Elements1/models/opv2v/f_cooper/latest.pth`.
- GPU: RTX 4090, 24 GB. Training uses ~3 GB.

### Point-cloud cache (required for usable throughput)

OPV2V lives on a USB **spinning** disk here and its `.pcd` files are ~3 MB
**ASCII** with ~57k points each. Decoding two per sample capped the data loader
at ~10 samples/s against a GPU that absorbs 223 samples/s - a 22x gap that made
the loader 100% of the wall clock.

`coloca/pcd_cache.py` mirrors each split into flat `.npy` arrays on the basement
NVMe (`/media/chenyi/basement2`). The arrays are byte-for-byte identical to what
`opencood.utils.pcd_utils.pcd_to_np` returns, 3.4x smaller (0.92 MB vs 3.09 MB),
and need no parsing.

```bash
python -m embedding_aware_belt_fusion.coloca.pcd_cache \
  --splits /media/chenyi/Elements1/Dataset/OPV2V/test \
           /media/chenyi/Elements1/Dataset/OPV2V/train \
  --cache-root /media/chenyi/basement2/cache/opv2v_coloca --workers 24
```

## Running

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD

# Stage 2 (frozen backbone) then stage 3 (end-to-end fine-tune)
python -m embedding_aware_belt_fusion.coloca.train --config configs/coloca_qua_opv2v.yaml

# Official OPV2V test split, at each evaluated noise level
python -m embedding_aware_belt_fusion.coloca.evaluate \
  --config configs/coloca_qua_opv2v.yaml \
  --checkpoint outputs/coloca/qua_sigma2/best.pth
```

## Dataset statistics

| Split | Scenarios | Agent pairs within 40 m |
|---|---:|---:|
| train (85% of scenarios) | 37 | 18,496 |
| val (15% of scenarios) | 7 | 4,176 |
| test (official) | 15 | 6,834 |

One of the 16 official test scenarios has a single CAV and therefore
contributes no pairs.

## The published localization head does not converge here

The headline finding of this reproduction: **CoLoca-QuA exactly as specified
never leaves the predict-zero solution on OPV2V**, while the signal it needs is
demonstrably present in its own inputs.

All rows below use the *same* frozen F-Cooper features, dataset, collate and
labels, so they are directly comparable. "Predict-zero" is the trivial baseline
of trusting the noisy pose unchanged.

| Head | Train MAE | Notes |
|---|---:|---|
| Predict-zero baseline | 1.92 m | no model |
| **CoLoca-QuA as published** (linear 16x16 patch embed) | **1.95 m** | flat over 3468 steps |
| Parameter-free cross-correlation | **0.49 m** | *no learning at all* |
| Conv head (5x conv-BN-ReLU), 2400 steps | **0.71 m** | learns within 300 steps |
| CoLoca-QuA + 3-layer conv stem | see `outputs/coloca` | the fix, below |

### Why the published design fails: two independent deficiencies

A 2x3 ablation over tokenization and readout, all other things held equal
(same frozen features, same 6-layer encoder, 1200 steps, lr 1e-3), isolates the
cause. "Stem" is `conv_stem_depth`; readout is how the pose error is pooled out
of the encoded sequence:

| Stem | Readout | Train MAE | |
|---:|---|---:|---|
| 0 | `loc_token` | 1.950 m | **the paper exactly** - fails |
| 0 | `mean` | 1.953 m | fails |
| 0 | `both` | 1.952 m | fails |
| 3 | `loc_token` | 1.889 m | fails |
| **3** | **`mean`** | **0.915 m** | **converges** |
| 3 | `both` | 1.645 m | partial |

**Both changes are required; neither alone is sufficient.** The two deficiencies:

1. **Tokenization.** The patch embedding (Eq. 10) is a single *linear*
   convolution over the channel-concatenated map, and a linear map of
   `[ego, cav]` can only produce `W_e . ego + W_c . cav` - an *additive*
   combination. Registration needs a *multiplicative* ego x CAV term. In the
   published design the first nonlinearity arrives only after a 512x
   compression (16x16x512 -> 256).

2. **Readout.** The localization token (Eq. 12-13) enters the residual stream as
   a *content-free constant*, identical for every sample. Its final state is
   that constant plus attention increments, so early in training the prediction
   is nearly input-independent - which is precisely the observed failure mode, a
   constant output at the conditional mean (zero). Patch tokens are
   input-dependent from the first layer, so mean-pooling escapes it. This also
   explains why the conv stem alone does not help: it improves the tokens, but
   the paper's readout never reads them directly.

The `both` row is instructive: mixing the constant token back in dilutes the
signal and recovers only part of the gain.

What is certain is that the failure is not a porting artifact:

- The cross-correlation control shares the dataset, collate and backbone with
  training and recovers the label at **r = +0.98 / +0.985**, so features,
  labels, label sign and batch alignment are all correct.
- The configuration matches the paper *physically*, not merely in tensor shape:
  the paper's BEV cell is also 0.8 m (204.8/256 and 76.8/96), so `P = 16` spans
  the same 12.8 m window here as there.

Ruled out by experiment, each with no effect: head zero-initialization,
dropout placement, output scaling, LR warm-up, AMP gradient overflow (amp on/off
identical; features absmax 4.6, grad norm 7.5), learning rate (1e-4 and 1e-3),
and patch size (P=4 and P=2 give only marginal gains).

### The convolutional stem option

`model.conv_stem_depth` inserts a shallow conv stack (3x3 conv - BN - ReLU,
first layer strided) before the patch embedding, following the standard result
that early convolutions help transformers ([Xiao et al., NeurIPS 2021](https://arxiv.org/abs/2106.14881)).
The stem's stride is folded into the patch size, so `N = 96` tokens, `D = 256`,
`L = 6`, 4 heads, the localization token and pre-LN are all unchanged, and the
head shrinks from 38.35 M to 7.78 M parameters.

**It is retained as an option, not as a fix** - on its own it did not make the
model converge (see above). `conv_stem_depth: 0` reproduces the paper exactly
and is the default in code.

## Results

Official OPV2V test split, 6,834 agent pairs, 15 scenarios. Trained with the
converging config (`conv_stem_depth: 3`, `readout: mean`) for 30 frozen + 15
fine-tune epochs; best checkpoint by validation MAE (fine-tune epoch 13,
val MAE 0.1234 m). "No correction" is the control of trusting the noisy pose.

| sigma | Method | MAE (m) | RMSE (m) | err < 1 m | err < 0.8 m | err < 0.5 m |
|---|---|---:|---:|---:|---:|---:|
| 2 m | **Ours (OPV2V)** | **0.144** | **0.219** | **99.63%** | **99.34%** | **97.45%** |
| 2 m | No correction | 2.534 | 2.852 | 11.49% | 7.77% | 3.04% |
| 2 m | Paper (V2XSet) | 0.230 | 0.290 | 99.43% | 98.15% | 93.61% |
| 1 m | **Ours (OPV2V)** | **0.100** | **0.128** | **99.97%** | **99.91%** | **99.46%** |
| 1 m | No correction | 1.306 | 1.472 | 36.92% | 24.93% | 10.97% |
| 1 m | Paper (V2XSet) | 0.180 | 0.220 | 99.73% | 99.48% | 97.77% |

The model removes ~94% of the input localization error at sigma = 2 m
(2.534 -> 0.144 m) and beats the paper's reported V2XSet figures on every
translation metric.

**This is not evidence that the paper's method was reproduced.** Read the
comparison with three caveats:

1. **Different dataset.** The paper reports V2XSet; these are OPV2V. OPV2V uses
   a 64-line LiDAR against V2XSet's 32-line, so its BEV maps are denser and
   registration is correspondingly easier. That alone plausibly accounts for
   beating the reported numbers, and the comparison is a reference point, not a
   like-for-like reproduction.
2. **Modified architecture.** These numbers come from the converging config, not
   the published one. The published architecture produces the "no correction"
   row, since it never leaves the predict-zero solution.
3. **Yaw is not learned at all** - see below.

### Yaw is never learned

`yaw_mae_deg` sits at 0.79-0.81 deg from the first epoch to the last, across
both stages and both noise levels. For sigma = 1 deg heading noise, predicting
zero gives `E|dpsi| = 0.798` deg. The model is emitting the conditional mean for
heading throughout; it is not partially capturing yaw, it is ignoring it.

The loss arithmetic shows the gradient signal is not the limitation. By the end
of stage 2 the weighted translation term contributes ~0.07 of the loss, so
essentially all of the remaining ~1.07 train loss *is* the yaw term
(`E[dpsi^2] x 1 = 1.0`). Yaw dominates the objective and is still not learned,
which points to the heading error being hard to extract from these BEV features
rather than merely down-weighted by the `(2, 2, 1)` weights.

**This matters for interpretation:** the paper's MAE and RMSE are translation
only, in metres, so a yaw-blind model scores well on the headline metrics. As a
baseline for downstream fusion, this model corrects position but not heading.
The paper reports that its yaw errors "remain highly competitive" under setting
S3, which we did not reproduce.
