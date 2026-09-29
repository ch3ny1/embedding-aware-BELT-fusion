# V2X-Real: boxes-only vs LiDAR embedding vs LiDAR+camera embedding

**Goal.** Repeat the trunk comparison that retired the embedding on OPV2V
(`docs/alignformer_p2.md`, "The message") on real traffic, where the
asset-library argument behind the OPV2V nulls does not apply, and add the
one trunk OPV2V could not justify: a per-object **camera** feature. Three
trunks, one detector, one FreeAlign, on the official V2X-Real splits:

| trunk | message per object | what the ego receives |
|---|---|---|
| boxes only | 7 floats | boxes (byte-identical to FreeAlign) |
| boxes + LiDAR embedding | 7 + 128 | boxes + ROI feature of the pillar BEV |
| boxes + LiDAR + camera embedding | 7 + 128 | boxes + embedding fed by ROI feature ‖ camera ROI feature |

Reported: AP@0.7/0.5/0.3 under localization error (sigma 0..2 m) and delay
(1/2/4 frames), against FreeAlign, plus bytes per frame per agent.
**Validation calibrates, test reports**; the official `val/` is the
calibration split and `test/` is touched once per trunk.

**Nothing on this machine is trained for V2X-Real** (user, 2026-09-29;
confirmed by a survey of `/media/chenyi/basement2/repos/HetPoison/GenComm`,
whose V2X-Real path is LiDAR-only intermediate fusion with a broken
late-fusion route and no checkpoints). Everything below trains from scratch.

## Protocol borrowed from GenComm / the V2X-Real paper, and where we differ

Taken as-is (file:line refer to GenComm's `opencood/`):

- **Class map** `data_utils/__init__.py:1-7`: vehicle = {Car, PoliceCar,
  LongVehicle}; truck = {Truck, Van, TrashCan, ConcreteTruck, Bus};
  pedestrian = the rest. **We use the `vehicle` super-class only**, because
  OpenCOOD's stock late-fusion PointPillars is single-class with one anchor
  size and a 12.7 m bus does not fit a car anchor. Trucks and buses are not
  in the ground truth and a detection on one is a false positive, for every
  arm and for FreeAlign alike.
- **Ranges** `hypes_yaml/v2xreal/**/stage1/*.yaml`: `cav_lidar_range`
  `[-102.4, -51.2, -15, 102.4, 51.2, 15]`, voxel `[0.4, 0.4, 30]`,
  `GT_RANGE` the same (`datasets/__init__.py:44`). The z extent is wide
  because infrastructure LiDARs sit ~4 m up and see the ground at z ≈ -5.
- **Ego** = vehicle-centric (`vc`): vehicles first, infrastructure after,
  the first agent is ego (`v2xreal_basedataset.py:207-218, 311-313`).
  OpenCOOD's `basedataset.py:140-141` rotates only ONE negative id, so with
  folders `-1, -2, 1, 2` its ego would be `-2`; the adapter reorders.
- **Comm range** 70 m (yamls) at training; GenComm's inference uses 180.
  We keep 70 for training and evaluate at 70 first; a wider range is a
  reported variant, not a silent default.
- **Scenarios dated 2023-04-07** have no infrastructure (V2V only); GenComm
  drops them in `vc` mode at eval. We keep them: the comparison is between
  aligners on the same pairs, and V2V pairs are pairs.
- **10 Hz**, so OpenCOOD's `time_delay // 100` frame quantization holds and
  `alignformer/delay.py` applies unchanged.

Verified on the data (memory: `v2x-real-dataset-state`): every agent in a
scenario has the same timestamp set; `.bin` is float32 (x, y, z, i) with
occasional NaN rows; `extrinsic` is camera->LiDAR; images 1920x1080; three
CRC-failed train files were deleted, one of them a LiDAR frame.

## Tasks

Each task is TDD (RED -> GREEN -> REFACTOR), reviewed, committed on its
own, never touching `external/OpenCOOD/`.

### T1. `alignformer/v2xreal.py`: the dataset adapter (no GPU)
- `VEHICLE_TYPES`, `V2XREAL_GT_RANGE`, `V2XREAL_COM_RANGE` as named constants
  with the GenComm citations above.
- `load_lidar_bin(path)`: `np.fromfile(float32).reshape(-1, 4)`, drop
  non-finite rows, refuse a file whose size is not a multiple of 16 bytes.
- `bin_pcd_to_np(pcd_path)`: OpenCOOD's `basedataset` names the LiDAR file
  `<ts>.pcd`; when that does not exist and `<ts>.bin` does, load the bin;
  otherwise defer to the real reader. Installed on
  `opencood.utils.pcd_utils.pcd_to_np`, the same slot
  `scripts/train_late_fusion.py::install_pcd_cache_shim` already patches.
- `filter_vehicles(params)`: returns a NEW params dict whose `vehicles`
  keeps only `VEHICLE_TYPES`. Never mutates the input.
- `order_agents(ids)`: vehicles ascending, then infrastructure ascending
  by absolute value; pure.
- `V2XRealLateFusionDataset(LateFusionDataset)`: after `super().__init__`,
  rebuild `scenario_database` in `order_agents` order with `ego` on the
  first vehicle; `retrieve_base_data` = `super()` then `filter_vehicles`
  on every agent's `params`.
- `build_dataset(hypes, visualize, train)`: for `core_method ==
  "V2XRealLateFusionDataset"` install the bin shim, set
  `opencood.data_utils.datasets.GT_RANGE/COM_RANGE` from the config, and
  construct the subclass; otherwise delegate to OpenCOOD's builder
  unchanged. Every call site in `alignformer/` (`evaluate.py`, `cache.py`,
  `baselines.py`) and the training script switch to this one function.
- Tests: pure functions on synthetic data; an end-to-end
  `retrieve_base_data` on a synthetic scenario tree built from
  `configs/alignformer_detector_r140.yaml` with the paths overridden.

### T2. `configs/v2xreal_detector.yaml` + detector training (GPU, ~5 h)
- Copy of the r140 detector config with the V2X-Real ranges, voxel size,
  `root_dir` = NVMe `train/`, `validate_dir` = NVMe `val/`, output
  `outputs/v2xreal/point_pillar_late_fusion`. Pinned by a test the way
  `tests/test_alignformer_range.py` pins the two OPV2V configs.
- `scripts/train_late_fusion.py --dataset v2xreal`: skips the OPV2V split
  materialization and pcd cache shim, installs the V2X-Real builder into
  `opencood.tools.train`. The OPV2V path is untouched (bit-identity).
- 15 epochs, checkpoint selection on `val/` as before.

### T3. Detection + ROI cache on train/val/test (GPU, ~1 h)
- `cache.py` through the new builder; `.bin` handled by the shim.
- Skip list: the deleted LiDAR frame (`2023-04-04-15-58-18_30_0/1/000112`)
  as a named constant in `v2xreal.py`, refused loudly if encountered.

### T4. `alignformer/camera_features.py`: per-object camera ROI features (GPU, ~1 h)
- Frozen ImageNet ResNet-18 (torchvision 0.18 is in the env), stride-16
  feature map, `roi_align` on the projected 2-D box of each **detected**
  object in each of the agent's two cameras, `K @ inv(extrinsic)`
  (verified convention), keep the camera with the larger visible area,
  zero vector + `has_camera=False` when the box projects outside both
  images or is behind the camera. 512-d, stored beside the LiDAR ROI
  feature in the cache so training is identical across trunks.
- Tests: projection against the hand-verified frame; a box behind the
  camera yields the flag; the feature is deterministic.

### T5. AlignFormer configs and training, three trunks (GPU, ~3 x 3 h)
- `configs/alignformer_v2xreal.yaml`: model unchanged; `data` points at the
  V2X-Real cache; `training_comm_range_m` 70.
- Embedding head input: `lidar_roi` (trunk 2) or `lidar_roi ‖ camera`
  (trunk 3), one flag. Boxes-only is the existing ablation.
- Stage 1 then stage 2 per trunk with the existing scripts; IVW scalar,
  IRLS Huber n=2 guard 3.0, per-pair abstention -- the shipped OPV2V arm.
- Shrinkage / tau calibrated on `val/` per trunk.

### T6. Sweeps and the write-up (GPU, ~4 h)
- `evaluate.py --metric noisy_ap` on `val/` (selection) then once on
  `test/` per trunk: sigma sweep with 5 seeds, delay 1/2/4 with 3 seeds,
  `--freealign`.
- Bandwidth rows from `bandwidth.alignment_overhead_bytes` at the measured
  objects per agent-frame.
- New section in `docs/alignformer_p2.md`: the three trunks against
  FreeAlign, the OPV2V-vs-V2X-Real contrast, and whether the camera
  feature earns its place on real traffic.

## Risks named up front
- A single-anchor detector on real traffic will be weaker than the paper's
  three-class one; the comparison is internal, so it stays fair, but the
  absolute AP is not the paper's number and must not be quoted beside it.
- Infrastructure agents have a wide z range; if pillar height 30 m hurts,
  that is a detector question shared by all arms.
- The camera feature is frozen ImageNet, not fine-tuned. If it shows
  signal, fine-tuning is the follow-up; if it shows none, that null is
  about frozen features on this data, not about cameras.
