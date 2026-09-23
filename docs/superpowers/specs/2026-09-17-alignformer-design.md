# AlignFormer: embedding-aware late fusion robust to localization error

Design document. Status: approved for planning, 2026-09-17.

## 1. Problem

Cooperative perception on OPV2V collapses under localization error. In this
repository, PointPillars naive late fusion scores **AP@0.7 = 0.856** clean
(globally confidence-sorted, official test split) and **0.294** under the BELT
reference perturbation of 0.2 m / 0.2 deg / 100 ms. BELT-style uncertainty-aware
fusion recovers part of this (0.415) but does not estimate the pose error itself.

Methods that *do* estimate the pose error well, such as the CoLoca-QuA
reproduction in `docs/coloca_qua_baseline.md`, require transmitting dense BEV
feature maps (~25.8 MiB per agent per frame). That defeats the point of late
fusion.

**Goal.** A late-fusion cooperative perception method that transmits only
bounding boxes plus a light per-object embedding (~5 KB/frame/agent), estimates
each CAV's SE(2) localization error at the ego, corrects the boxes, and fuses -
recovering AP under localization noise at a fraction of the bandwidth.

## 2. Decisions and rationale

| Decision | Choice | Rationale |
|---|---|---|
| Detector backbone | PointPillars (frozen, pretrained), behind a single interface | Already 0.856 AP@0.7 clean, above published OPV2V late fusion (~0.781). SWFormer in this repo reaches only 0.682. Backbone quality is not the bottleneck; localization error is. Swap is a later ablation and touches one file. |
| Noise regime | One model trained on a mixture, results reported as a sweep | Covers both the BELT reference regime and the CoLoca regime with a single model, and directly produces the AP-vs-noise curve. Matches how CoLoca-QuA trained (mixed sigma). |
| Delay | Phase 2, after pose correction works | Localization error is exactly a rigid SE(2) transform, so its label is exact. Delay is not: the CAV's own egomotion is absorbable by SE(2), but other vehicles move independently. Training them jointly makes failures unattributable. |
| Pose head | Build **both** A (direct regression) and B (matching + Procrustes), compare | B is expected to win, but the comparison is the evidence. Shared trunk and shared frozen embedding head mean only the head differs - the same ablation methodology that isolated CoLoca's two deficiencies. |
| Prior artifacts | Rebuild fresh | `src/embedding_aware_belt_fusion/data/` was never committed (no git history) and `outputs/` is empty, so the 0.891 Top-1 Spatial TrackFormer model has no surviving code or checkpoints. Confirmed no backup. |

### Lessons inherited from the CoLoca-QuA reproduction

That reproduction established, on this exact data, that:

1. A **linear** tokenization of channel-concatenated `[ego, cav]` features cannot
   register - only `W_e.ego + W_c.cav` is expressible, and registration needs a
   **multiplicative** ego x CAV term.
2. A **content-free query token** readout collapses to the conditional mean.
   Mean-pooling over input-dependent tokens escapes it.
3. Fixing either alone does nothing; both are required.
4. **Yaw was never learned** (flat at 0.79-0.81 deg, the predict-zero value)
   even though yaw dominated the loss.

AlignFormer's architecture is designed against all four findings. Cross-attention
between the two object sets supplies (1); mean-pooled readout supplies (2);
closed-form Procrustes makes (4) structural rather than learned.

## 3. Architecture

### 3.1 Transmit side (per CAV)

1. Frozen PointPillars produces a BEV feature map and decoded boxes with scores.
2. For each box, a **rotated ROI-align** samples the BEV map over the box
   footprint **in the box's own canonical frame**, to a fixed k x k grid.
3. A small MLP maps the ROI features to an L2-normalized `d`-dimensional
   embedding (`d` ablated over {32, 64, 128}).

Extracting in the box frame makes the embedding **pose-invariant**: it describes
the object's appearance, not where the sender believes it is. This gives a clean
separation of concerns:

> **The embedding carries appearance, for matching. The box carries geometry,
> for solving.**

### 3.2 Message

Per agent per frame: `{(box_7dof, score, embedding_d)} + noisy_pose`.

At 20 objects and `d = 128` in fp16 this is ~5 KB, or ~640 B with the product
codebook already in `communication/product_codebook.py`, against ~25.8 MiB for a
dense BEV map. Byte accounting is a first-class reported metric, not an estimate.

### 3.3 Receive side

CAV boxes are first projected into the ego frame using the CAV's **noisy** pose,
so the two object sets are deliberately misaligned.

**Tokenization (shared).** One token per object:
`MLP([x, y, z, l, w, h, cos psi, sin psi, score]) + embedding + set-membership
embedding (ego | CAV)`.

**Trunk (shared).** A small transformer over the two sets concatenated:
self-attention within each set, cross-attention across. Object counts are ~20-60,
so cost is negligible.

**Head A - direct regression.** Mean-pool over all trunk tokens, MLP to
`(dx, dy, dpsi)`. Mean-pool, never a constant query token.

**Head B - soft correspondence to closed-form pose.**

1. Project trunk outputs to match features; build a score matrix with a dustbin
   row and column for objects having no cross-agent counterpart.
2. Sinkhorn-normalize to a soft assignment `P`.
3. For each ego object `i`: weight `w_i = sum_j P_ij`, virtual correspondence
   `q_hat_i = (sum_j P_ij q_j) / w_i`.
4. **Augment each object with a second virtual point** at
   `centre + lambda * unit(heading)`, so headings enter the same least-squares
   problem as centres rather than being fused by an ad-hoc rule.
5. Weighted 2-D Kabsch in closed form over the augmented set:

   ```
   psi = atan2( sum w (dq_x dp_y - dq_y dp_x),  sum w (dq . dp) )
   t   = p_bar - R(psi) q_bar
   ```

   solving for the transform taking the noisily-projected CAV points `q` onto the
   matched ego points `p`.

Gradients flow through the match weights `w`. Yaw is recovered **analytically**,
never regressed - this is the structural answer to CoLoca's yaw failure.

A consequence worth stating: because each matched object contributes both a
centre and a heading, **a single matched object already determines full SE(2)**.
This removes the collinear-correspondence degeneracy that breaks centre-only
Procrustes and makes the method degrade gracefully on low-overlap pairs.

### 3.4 Degenerate cases

These must be explicit or training produces NaNs:

- `sum w` guarded with epsilon.
- Below a match-mass threshold (`sum w < 1.0`, i.e. less than one effective
  matched object): emit zero correction and zero confidence.
- Confidence is derived from match mass plus post-fit residual, and is exported
  so the BELT-style uncertainty fusion stage can consume it.

#### `AlignFormerB`'s per-object `atan2(0, 0)` guard: reachability analysis

`model.py::AlignFormerB._soft_correspondence` derives each ego object's
virtual CAV heading via `atan2(direction_y, direction_x)`, where `direction`
is the soft-match-weighted mean CAV heading. Any row with zero soft-match
weight (an ordinary padded ego position in a collated batch, not just a
wholly empty sample) has `direction == (0, 0)` exactly, which is a genuine
`atan2` gradient singularity, guarded by routing those rows through a
placeholder direction before the call (see the code comment there for the
mechanism -- it differs between the padded-row case, where the row's own
Kabsch weight is what makes the placeholder harmless, and the wholly-empty
-sample case, where `MIN_MATCH_MASS` zeroes the whole correction instead).

This subsection records, separately from the mechanism, whether the
singularity is reachable *in the first place* for a real, trained model --
i.e. whether `direction` can land arbitrarily close to, but not exactly at,
`(0, 0)` for a row that is NOT simply zero-weight (which is safe regardless):
`atan2`'s gradient magnitude is `1/r` where `r = sqrt(x^2+y^2)`, so it is
`~7e14` at `r=1e-15` and literally `inf` by `r=1e-25` (float32 underflow of
`x^2+y^2`). Two attempts to drive the end-to-end model into that window both
failed, and the reason is structural: `weighted_se2_kabsch`'s
`MIN_MATCH_MASS = 1.0` zeroes the gradient for any sample whose total mass is
under 1, and above that floor Sinkhorn's normalization runs through a single
shared dustbin scalar, so every row's dustbin share stays comparable to every
other row's. Separating one row's weight from another's by the ~46 nats
needed to reach `1e-20` while another stays near 1 would require a learned
match-score temperature below ~0.043 (`tau` initializes at 0.1, see 3.5).
Reachable in principle if `tau` collapses during training; not reproduced
here. **If a non-finite gradient is ever seen from `AlignFormerB` in
training, this is the first place to look** — check whether `log_temperature`
has collapsed toward that range.

### 3.5 Hyperparameter defaults

Starting values, all ablatable. Fixed here so the implementation plan is
actionable rather than exploratory.

| Symbol | Meaning | Default |
|---|---|---|
| `k` | rotated ROI-align output grid | 4 x 4 |
| `d` | embedding dimension | 128 (ablated over 32 / 64 / 128) |
| `D` | trunk model dimension | 256 |
| `L` | trunk layers (self + cross per layer) | 4 |
| heads | trunk attention heads | 4 |
| `lambda` | heading virtual-point offset | 2.0 m (about half a vehicle length, so heading and centre contribute comparably to the Kabsch fit) |
| Sinkhorn iters | normalization steps | 20 |
| `tau` | match-score temperature | learned scalar, initialized to 0.1 |
| match-mass floor | degenerate-case fallback | `sum w < 1.0` |
| max objects/agent | token budget | 64, ranked by score |

## 4. Training

| Stage | Trains | Data condition | Gate |
|---|---|---|---|
| 0 | nothing (reproduction) | clean | clean AP@0.7 = 0.856 +/- 0.01 |
| 1 | embedding head + trunk, **matching loss only** | sigma sampled in [0, 0.5] m | cross-agent Top-1 >= 0.85 |
| 2 | each pose head, embedding head frozen | noise curriculum, sigma 0 -> 2 m | see phase gates |
| 3 | optional joint fine-tune | full sweep | no regression vs stage 2 |

**Match loss.** Negative log-likelihood of the ground-truth assignment under the
Sinkhorn output, including dustbins. Ground-truth correspondences come from OPV2V
physical object IDs, used only as supervision.

**Pose loss - corner loss, deviating from CoLoca.** Apply the predicted and the
true correction to the CAV boxes and take L1 over the 4 BEV corners.

CoLoca's weighted `(2, 2, 1)` MSE on `(dx, dy, dpsi)` mixes metres and radians
with arbitrary weights, and the reproduction showed yaw dominating that loss while
still not being learned. The corner loss is in metres throughout, automatically
weights yaw error by object distance, and optimizes exactly what IoU and AP
measure. Per-component MAE is still **reported**, for comparability with the
CoLoca table.

**Fairness caveat.** Head B can consume correspondence supervision because its
architecture exposes a matching matrix; Head A cannot. That extra supervision is
genuinely part of B's contribution, but it confounds "better architecture" with
"more supervision". A **B-without-match-loss** ablation separates the two.

## 5. Data and caching

Caching is load-bearing. `docs/coloca_qua_baseline.md` records that the OPV2V
loader was 100% of wall clock until the NVMe npy mirror fixed it (0.62 -> 11.2
it/s). Running PointPillars every training step would reintroduce exactly that.

Cache per frame per agent, once, to `/media/chenyi/basement2`:

- decoded boxes, scores, and ground-truth object IDs
- **ROI features** (not final embeddings, so the embedding head stays trainable)

At k = 4, 384 channels, fp16 and ~30 objects per frame this is ~180 KB/frame,
roughly 5 GB total. Training then runs on tiny object sets at high throughput.

Splits are **scenario-disjoint**, reusing `coloca/index.py`. Consecutive OPV2V
frames are near-duplicates, so frame-level splits leak. OPV2V's `validate/` split
is a broken symlink on this machine; validation is 15% of train scenarios.

## 6. Evaluation

Official OPV2V test split, globally confidence-sorted AP.

| | Method | Role |
|---|---|---|
| 1 | Ego-only, no fusion | floor |
| 2 | Vanilla late fusion, clean | late-fusion ceiling, **measured 0.8764 global-sorted / 0.8180 frame-order** (P0 gate, commit 4180ca9; supersedes the 0.856/0.781 predicted from README.md:133, which came from a checkpoint that no longer exists) |
| 3 | Vanilla late fusion, noisy | what we beat |
| 4 | RANSAC SE(2) on box centres (`integration/localization.py`) | classical control |
| 5 | CoLoca-QuA correction + late fusion | strong, ~25 MiB/frame - the bandwidth contrast |
| 6 | **AlignFormer-A** | direct regression |
| 7 | **AlignFormer-B** | matching + Procrustes |
| 8 | Oracle correspondence + Procrustes | isolates matching error from solver error |
| 9 | V2X-ViT, CoAlign, CoBEVT, AttFuse, Where2comm | **the robustness SOTA to beat** |
| 10 | **FreeAlign** (Lei, Ni, Han, Tang, Wang, Feng, Chen, Wang, ICRA 2024, arXiv 2405.02965) | **the closest competitor** - object-level geometric alignment, boxes only |

### FreeAlign: the closest competitor, and where AlignFormer differs structurally

FreeAlign builds a **salient object graph** per agent - nodes are detected boxes,
edges are **relative distances between them**, which are invariant to the
viewer's pose - and matches common subgraphs across agents with a GNN, recovering
the relative pose with no localization prior at all. Code:
`github.com/MediaBrain-SJTU/FreeAlign`, built on CoAlign and therefore
OpenCOOD-derived, so it runs in this stack for a like-for-like local comparison
at inference cost, exactly like rows 9.

This became the most important baseline once the association diagnostic
(2026-09-22) established that AlignFormer's appearance embedding contributes
nothing to matching on OPV2V, leaving object-level **geometry** as what both
methods actually use. The comparison is therefore direct, and the burden is on
AlignFormer to show a difference that matters.

**The structural difference is the minimum number of correspondences each method
needs, and it is measurable rather than rhetorical.** FreeAlign's evidence is
pairwise *distances*, so:

- 1 shared object = a 1-node graph with **0 edges**: no constraint at all.
- 2 shared objects = **1 edge**: a single scalar distance, which fixes neither
  rotation nor the reflection ambiguity.
- 3+ shared objects are needed before a distance graph rigidly determines SE(2).

AlignFormer augments every object with **heading virtual points** (lambda = 2 m,
Section 3.3), so a *single* matched object carries both a position and an
orientation and determines the full SE(2) on its own.

Measured on this machine's ROI cache (8,936 frames, ~18,200 ego-CAV pairs):
**18.5% of pairs share no object** (neither method can help; both must fall back
to the uncorrected pose), and **~7.8% share exactly one or two** - the regime
where AlignFormer is structurally solvable and a distance-graph method is not.
That ~7.8% slice is the sharpest experiment against FreeAlign and must be
reported as its own row, not averaged into the whole split where it would be
diluted eightfold.

Required, and not yet done: run FreeAlign locally on OPV2V under this project's
own noise sweep and evaluator (`alignformer/evaluate.py`, global-sorted AP,
2170-frame test split), never quoting its published numbers beside ours - the
same rule that applies to rows 9 and for the same reason (see the P0 gate note
on the measured 0.8764 baseline).

### The SOTA claim, stated precisely

Row 9 is the real target. OpenCOOD's own benchmark uses exactly this project's BELT
reference perturbation (`xyz_std 0.2`, `rpy_std 0.2`, `async_overhead 200-300 ms`),
and publishes AP@0.7 perfect -> noisy: Late Fusion **62.0 -> 30.7**, F-Cooper
68.0 -> 46.9, AttFuse 66.4 -> 48.7, V2VNet 67.7 -> 49.3, DiscoNet 69.5 -> 54.1,
Where2Comm 65.4 -> 53.4, CoBEVT 66.0 -> 54.3, **V2X-ViT 71.2 -> 61.4** (retains 86%,
the robustness SOTA).

Trained checkpoints for V2X-ViT, CoAlign, CoBEVT, AttFuse and Where2comm are
**already on this machine** under `/media/chenyi/Elements1/models/opv2v/`, so these
are run locally on OPV2V under our own noise sweep rather than cited across
datasets. That makes the comparison like-for-like and costs inference only.

The claim is therefore **robustness per byte**, not clean AP:

> Under the same localization-noise sweep, AlignFormer's AP@0.7 exceeds every
> intermediate-fusion baseline above while transmitting ~5 KB/frame/agent against
> their dense BEV feature maps.

Late fusion does not beat intermediate fusion on *clean* AP and this work does not
claim it does. The headline pairing is the retention ratio under noise against the
bandwidth gap; clean AP is reported for completeness and to prove nothing was
sacrificed to get it.

**Metrics.** AP@0.3 / 0.5 / 0.7; pose translation MAE and RMSE; yaw MAE in
degrees; cross-agent association Top-1; bytes per frame per agent.

**Sweep.** sigma_xy in {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} m, with
`sigma_yaw (deg) = sigma_xy (m)` numerically - the convention already used by this
repository's sweeps (0.2 m / 0.2 deg, 0.4 m / 0.4 deg, ...). Noise is applied to
the CAV pose only, per-axis on x and y and on yaw; z, roll and pitch are
untouched because the method estimates only the planar pose.
**>= 3 seeds, reported as mean +/- std**. The existing README sweeps
use a single seed and flag this as a weakness; it is fixed here.

**Ablations.**

- Message content: **boxes only vs boxes + embeddings** (see risk 1 - run early)
- Embedding dimension `d` in {32, 64, 128}
- Product-codebook quantization of the embedding
- Head B without match loss (fairness caveat above)
- Heading virtual points on/off (does `lambda` augmentation fix yaw?)
- Backbone swap to SWFormer (later)

## 7. Phases and go/no-go gates

- **P0** cache, dataset, clean reproduction - *gate: clean AP@0.7 = 0.856 +/- 0.01* - **PASSED, measured 0.8764**; the band came from a lost checkpoint, so 0.8764 is the baseline to beat
- **P1** embedding head + matching - *gate: cross-agent Top-1 >= 0.85*
- **P2** Head B + the boxes-only ablation - *gate: yaw MAE strictly below the
  predict-zero conditional-mean value at each sigma*
- **P3** Head A, head-to-head comparison
- **P4** correction + fusion + AP sweep - *gate: noisy AP@0.7 beats vanilla late fusion*
- **P5** bandwidth ablations and quantization

P2's gate is deliberately the precise thing CoLoca-QuA failed. If closed-form
Procrustes cannot clear it, the central premise is wrong and we learn that early
and cheaply, before any fusion or sweep work is built on top of it.

## 8. Risks and contingencies

**Risk 1 - the embedding may add nothing over geometry.** OPV2V scenes are
sparse; box centres alone may suffice for matching, in which case RANSAC or a
geometry-only AlignFormer matches the full model and the premise is unsupported.

*Mitigation:* the boxes-only vs boxes+embeddings ablation is a first-class
experiment run in **P2, not at the end**. If geometry wins, that is a real
finding and the contribution shifts to the closed-form solver plus the efficiency
argument.

> **THIS RISK FIRED, AND IS RESOLVED AGAINST THE EMBEDDING (2026-09-22).** The
> ablation was run early as planned, and a follow-up pre-registered diagnostic
> settled it: the appearance embedding contributes **nothing** to association on
> OPV2V. Hard-subset delta (r = 5 m, n = 17,484) at sigma = 0.5 m is
> **+0.0001, 95% CI [-0.0013, +0.0014]**. Three independent causes, all
> confirmed: true-partner separability is **AUC 0.560** on raw pooled ROI
> features (chance 0.5); competing vehicles differ by a median **0.071 m** in
> width against a detector width spread of +/-0.081 m (they are the same CARLA
> asset); and the BEV map is **0.8 m/cell**, so a car spans 5.7 x 2.5 cells and
> the 4x4 ROI grid samples ~a dozen - enough for size class, not instance
> identity. "Undertrained" is **ruled out**: the embedding head's weight norm
> went 19.60 -> 38.87 and a causal shuffle control costs the model 0.0036, so
> it genuinely consumes the embedding and gets ~0.4 points, which a retrained
> boxes-only trunk recovers entirely from geometry.
>
> Per this risk's own mitigation clause, **the contribution shifts to the
> closed-form solver plus the efficiency argument**, and AlignFormer is a
> boxes-only method unless a contingency below changes that. Do not claim
> embedding-driven association gains on OPV2V anywhere. Full write-up:
> `.superpowers/sdd/2026-09-17-alignformer-p0-p2/association-diagnostic-report.md`.

*Contingency A (camera semantics) - RETAINED, but it does not rescue association
on OPV2V.* The diagnostic's binding constraint is that **no ego object in the
entire validation split has a competitor within 2 m** (the r = 2 hard subset is
empty; r = 3 holds 31 objects). Geometry never fails at stage-1 noise, so there
is nothing for a better feature to disambiguate, however good it is. Camera
augmentation is therefore **not** a fix for the OPV2V association number, and
proposing it as one would repeat the error this diagnostic corrected. It remains
live for two other purposes: (a) a dataset where vehicles genuinely cluster
(Contingency B), and (b) communication delay, where objects have moved and
geometric correspondence degrades while appearance stays time-invariant - the
one regime on OPV2V where appearance could still earn its place. The mechanism
is unchanged and bandwidth-neutral: OPV2V provides
four RGB cameras per CAV, present on disk, and
`scripts/opv2v_camera_bbox_viewer.py` already implements world-to-camera cuboid
projection with occlusion rejection. Each box is projected into the cameras,
ROI-cropped, encoded by a light 2-D encoder, and fused with the LiDAR ROI
embedding. This is **bandwidth-neutral** - the embedding dimension is unchanged,
only the encoder differs - so the efficiency claim survives intact.

*Contingency B (V2X-Real) - RETAINED, and it is the principled test.* Every
cause above is a property of OPV2V specifically: CARLA's small vehicle asset
library, and scenes sparse enough that competitors never come within 2 m.
**V2X-Real** is real traffic with real vehicle diversity and genuine density, so
both the appearance premise and the "geometry fails when objects cluster"
premise become testable rather than structurally unanswerable. It is also the
dataset CoLoca-QuA reports on (with V2XSet), which makes it the natural venue
for the comparison against that method. Treat OPV2V's negative as **specific to
OPV2V** and say so; do not generalize it to "appearance embeddings do not help
cooperative association", which the evidence does not support.

**Risk 2 - low-overlap CAV pairs.** Distant CAVs share few objects. Mitigated by
confidence gating with a no-correction fallback; AP additionally reported
conditioned on shared-object count. **Measured on this machine's cache: 18.5%
of ego-CAV pairs share no object at all and ~7.8% share exactly one or two.**
The zero-overlap 18.5% is a correctness requirement - the fusion path must fall
back to the uncorrected pose there, or roughly one pair in five is actively
corrupted. The ~7.8% one-or-two slice is the sharpest experiment against
FreeAlign, whose distance-graph evidence is structurally degenerate below three
correspondences while AlignFormer's heading virtual points solve from one.

**Risk 3 - detection errors couple into matching.** Correspondences are between
*detections*, not ground-truth objects, so false positives and misses corrupt the
assignment. The dustbin handles this by construction, but it bounds performance
by detector quality.

**Risk 4 - yaw signal at small sigma.** At sigma_yaw = 0.2 deg the pose yaw error
may fall below the detector's own box-heading noise. Reported honestly rather
than hidden by the sweep average.

## 9. Module layout

New package `src/embedding_aware_belt_fusion/alignformer/`, each file small and
single-purpose per the project coding rules:

```
boxes.py        detection extraction from the frozen backbone (the swap interface)
embedding.py    rotated ROI-align + MLP embedding head
message.py      message construction, quantization hook, byte accounting
trunk.py        shared transformer over the two object sets
head_direct.py  Head A
head_match.py   Sinkhorn soft assignment
procrustes.py   differentiable weighted SE(2) Kabsch
dataset.py      OPV2V pairwise object-set dataset with the noise curriculum
losses.py       match NLL + corner loss
fusion.py       apply correction, late-fuse, NMS
train.py
evaluate.py     AP + pose + bandwidth
```

**Reused without duplication:** `coloca/geometry.py` (exact SE(2) label
machinery and pose perturbation), `coloca/index.py` (scenario-disjoint splits),
`coloca/pcd_cache.py` (NVMe point cache), `features/opencood_proposals.py`
(detection decoding and GT-ID assignment), `integration/localization.py`
(RANSAC baseline), `integration/belt_fusion.py` (uncertainty-aware fusion).

Note: only `evaluation/belt_fusion.py`, the driver script, imports the missing
`data/` package. The fusion algorithm in `integration/belt_fusion.py` is intact,
so `alignformer/evaluate.py` can call it directly.

## 10. Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
```

The `opencood` env (py3.8, torch 2.3.1+cu121, working spconv), **not**
`belt-fusion`, which has no spconv and no OpenCOOD. Backbone checkpoint:
`/media/chenyi/Elements1/models/opv2v/f_cooper/latest.pth`. GPU: RTX 4090, 24 GB.

OPV2V pose is `[x, y, z, roll, yaw, pitch]` in degrees with **index 4 = yaw**.

## 11. Out of scope

- Communication delay and the per-object motion head (phase 2 of the project)
- SWFormer or any backbone swap (later ablation)
- Reconstructing the missing `data/` subpackage and the old Spatial TrackFormer
  pipeline
- Infrastructure (V2I) agents; OPV2V is V2V only
