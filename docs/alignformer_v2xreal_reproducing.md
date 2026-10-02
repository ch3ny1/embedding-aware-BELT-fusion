# Reproducing the V2X-Real comparison

> The command sequence behind [alignformer_v2xreal.md](alignformer_v2xreal.md):
> cache, stage 1, variance fit, stage 2, shrinkage, FreeAlign calibration and
> re-calibration, the sweeps on val and test, delay, the second training seed,
> and the summaries. Split out of the write-up to keep it readable.


```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
D=/media/chenyi/basement2/dataset/v2x-real; V=outputs/v2xreal
AF=configs/alignformer_v2xreal.yaml; AFC=configs/alignformer_v2xreal_camera.yaml
DET=configs/v2xreal_detector.yaml
ROBUST="--robust-solve huber --robust-iterations 2 --robust-min-evidence 3.0"

# Detector (15 epochs; checkpoint epoch chosen by val loss, named in $DET).
# Cache with camera features, all three splits (~40 min; resumes).
python -m embedding_aware_belt_fusion.alignformer.cache --config $DET \
  --splits $D/train $D/val $D/test \
  --cache-root /media/chenyi/basement2/cache/alignformer_v2xreal --camera

# Stage 1, three trunks (~17 min each on an RTX 4090).
python -m embedding_aware_belt_fusion.alignformer.train --config $AF --stage 1 --output-dir $V/stage1
python -m embedding_aware_belt_fusion.alignformer.train --config $AF --stage 1 --zero-embeddings --output-dir $V/stage1_zero_embeddings
python -m embedding_aware_belt_fusion.alignformer.train --config $AFC --stage 1 --output-dir $V/stage1_camera

# Correspondence variance on val; the four numbers are pasted into both configs.
python scripts/fit_correspondence_variance.py --config $AF --output $V/variance_fit_result.json

# Stage 2 (~28 min each) and shrinkage on val, per trunk.
for spec in "boxes+embeddings|$AF|stage1|B_boxes+embeddings" \
            "boxes_only|$AF|stage1_zero_embeddings|B_boxes_only" \
            "boxes+embeddings+camera|$AFC|stage1_camera|B_camera"; do
  IFS='|' read -r content cfg s1 tag <<<"$spec"
  python -m embedding_aware_belt_fusion.alignformer.train --config $cfg --stage 2 --head B \
    --message-content $content --stage1-checkpoint $V/$s1/best.pth \
    --variance-weighting scalar --output-dir $V/stage2_$tag
  python -m embedding_aware_belt_fusion.alignformer.evaluate --config $cfg --metric shrinkage \
    --checkpoint $V/stage2_$tag/best.pth $ROBUST --output $V/shrinkage_${tag}_calibration_result.json
done

# FreeAlign's six parameters on val.
python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF \
  --split $D/val --stride 4 --sigma 1.0 --output $V/freealign_calibration_result.json
FA="--freealign --freealign-calibration $V/freealign_calibration_result.json"

# Sweeps: val 3 seeds (pipeline check), test 5 seeds (report), delay 1/2/4 on test 3 seeds.
sweep() { local tag=$1 cfg=$2 split=$3 seeds=$4; shift 4
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $cfg --split $D/$split --checkpoint $V/stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds $seeds "$@"; }
for spec in "B_boxes_only|$AF" "B_boxes+embeddings|$AF" "B_camera|$AFC"; do
  IFS='|' read -r tag cfg <<<"$spec"
  sweep $tag $cfg val 3 --output $V/${tag}_val_result.json
  sweep $tag $cfg test 5 --output $V/${tag}_test_result.json
  for d in 1 2 4; do
    sweep $tag $cfg test 3 --sweep 0 0.4 1.0 2.0 --delay-frames $d --output $V/${tag}_delay${d}_test_result.json
  done
done

# Message bytes (structural; the camera trunk shares the LiDAR trunk's bytes).
python - <<'PY'
import json; from pathlib import Path
from embedding_aware_belt_fusion.alignformer.bandwidth import alignformer_message_bytes, alignment_overhead_bytes
m = alignformer_message_bytes(Path("/media/chenyi/basement2/cache/alignformer_v2xreal/test"), 128, 64)
n = m["objects_per_agent_mean"]
Path("outputs/v2xreal/bandwidth_result.json").write_text(json.dumps({
  "method": "v2xreal_bandwidth", "metric": "bytes_per_frame_per_agent", "precision": "float32",
  "alignformer": m, "alignment_overhead": {k: alignment_overhead_bytes(n, e) for k, e in
  [("freealign", 0), ("alignformer_boxes_only", 0), ("alignformer_boxes_embedding", 128), ("alignformer_boxes_embedding_camera", 128)]}}, indent=2))
PY

# Training-seed replicate: seed-1 copies of the configs for TRAINING only;
# the sweep runs on the shipped config (whose training.seed also seeds the
# noise draws) with the seed-1 checkpoint and shrinkage swapped in.
sed 's/^  seed: 0$/  seed: 1/' $AF > $V/alignformer_v2xreal_seed1.yaml
sed 's/^  seed: 0$/  seed: 1/' $AFC > $V/alignformer_v2xreal_camera_seed1.yaml
for spec in "boxes_only|$V/alignformer_v2xreal_seed1.yaml|$AF|--zero-embeddings|B_boxes_only" \
            "boxes+embeddings|$V/alignformer_v2xreal_seed1.yaml|$AF||B_boxes+embeddings" \
            "boxes+embeddings+camera|$V/alignformer_v2xreal_camera_seed1.yaml|$AFC||B_camera"; do
  IFS='|' read -r content tcfg ecfg zero tag <<<"$spec"
  python -m embedding_aware_belt_fusion.alignformer.train --config $tcfg --stage 1 $zero --output-dir $V/seed1_stage1_$tag
  python -m embedding_aware_belt_fusion.alignformer.train --config $tcfg --stage 2 --head B \
    --message-content $content --stage1-checkpoint $V/seed1_stage1_$tag/best.pth \
    --variance-weighting scalar --output-dir $V/seed1_stage2_$tag
  python -m embedding_aware_belt_fusion.alignformer.evaluate --config $ecfg --metric shrinkage \
    --checkpoint $V/seed1_stage2_$tag/best.pth $ROBUST --output $V/shrinkage_seed1_${tag}_calibration_result.json
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $ecfg --split $D/val --checkpoint $V/seed1_stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_seed1_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds 3 --output $V/seed1_${tag}_val_result.json
  python -m embedding_aware_belt_fusion.alignformer.evaluate --metric noisy_ap --config $DET \
    --alignformer-config $ecfg --split $D/test --checkpoint $V/seed1_stage2_$tag/best.pth \
    --shrinkage $V/shrinkage_seed1_${tag}_calibration_result.json $ROBUST --abstain-arm per_pair $FA \
    --ap-seeds 5 --output $V/seed1_${tag}_test_result.json
done

# Every table under "Results" is rendered from those JSONs.
for split in val test delay1_test delay2_test delay4_test; do
  python scripts/summarize_v2xreal_trunks.py --reference boxes_only \
    --trunk boxes_only=$V/B_boxes_only_${split}_result.json \
    --trunk lidar=$V/B_boxes+embeddings_${split}_result.json \
    --trunk lidar_camera=$V/B_camera_${split}_result.json \
    --output $V/trunk_comparison_${split}_result.json
done
for split in val test; do
  python scripts/summarize_v2xreal_trunks.py --reference boxes_only \
    --trunk boxes_only=$V/seed1_B_boxes_only_${split}_result.json \
    --trunk lidar=$V/seed1_B_boxes+embeddings_${split}_result.json \
    --trunk lidar_camera=$V/seed1_B_camera_${split}_result.json \
    --output $V/trunk_comparison_seed1_${split}_result.json
  for tag in B_boxes_only B_boxes+embeddings B_camera; do
    python scripts/summarize_v2xreal_trunks.py --reference seed0 \
      --trunk seed0=$V/${tag}_${split}_result.json --trunk seed1=$V/seed1_${tag}_${split}_result.json \
      --output $V/seed_replicate_${tag}_${split}_result.json
  done
done

# FreeAlign re-calibration. More data, same MAE grid (selection unchanged):
python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF --split $D/val --stride 1 --sigma 1.0 \
  --output $V/freealign_calibration_val_stride1_result.json
C=/media/chenyi/basement2/cache/v2xreal_calib_split/valplus  # symlinks: 6 val + the 44 train scenarios sharing no clip with test
python scripts/calibrate_freealign.py --config $DET --alignformer-config $AF --split $C --stride 2 --sigma 1.0 \
  --output $V/freealign_calibration_valplus_result.json
# The *_wide_result.json files are the same two runs with the 0.1-1.5 m grid now in the script.
# Edge threshold by val AP: one calibration file per threshold (the 0.3 m point is B_boxes_only_val_result.json),
# swept on val through the deployed pipeline; argmax of FreeAlign's AP@0.7 sweep mean. A later
# --freealign-calibration overrides the one sweep() passes.
for thr in 0.1 0.15 0.2 0.5 1 1.5; do
  sweep B_boxes_only $AF val 3 --freealign-calibration $V/freealign_calibration_thr${thr}_result.json \
    --output $V/B_boxes_only_fathr${thr}_val_result.json
done
# 1.0 and 1.5 m tie: both go to test, boxes-only trunk only (FreeAlign's row is checkpoint-independent).
for thr in 1 1.5; do
  FAAP=$V/freealign_calibration_valap_thr${thr}_result.json
  sweep B_boxes_only $AF test 5 --freealign-calibration $FAAP --output $V/B_boxes_only_faap${thr}_test_result.json
  for d in 1 2 4; do
    sweep B_boxes_only $AF test 3 --sweep 0 0.4 1.0 2.0 --delay-frames $d --freealign-calibration $FAAP \
      --output $V/B_boxes_only_faap${thr}_delay${d}_test_result.json
  done
done
# Pair the re-calibrated FreeAlign column against every trunk file (seed 0 shown; seed1_ and delay files likewise).
python scripts/summarize_v2xreal_trunks.py --reference boxes_only \
  --trunk boxes_only=$V/B_boxes_only_test_result.json --trunk lidar=$V/B_boxes+embeddings_test_result.json \
  --trunk lidar_camera=$V/B_camera_test_result.json --freealign-from $V/B_boxes_only_faap1_test_result.json \
  --output $V/trunk_comparison_seed0_faap1_test_result.json

# AlignFormer's decision rule, the same treatment: every arm on val, selected on AP@0.7 sweep mean,
# deployed + selected + two runners-up on test, deployed + selected under delay and on the LiDAR trunk.
ARMS="--abstain-arm per_pair --abstain-arm abstain:0.5 --abstain-arm abstain:0.2 --abstain-arm abstain:0.1 \
  --abstain-arm abstain:0.05 --abstain-arm abstain:0.01 --abstain-arm both:0.5 --abstain-arm both:0.2 --abstain-arm both:0.05"
FAAP=$V/freealign_calibration_valap_thr1_result.json
sweep B_boxes_only $AF val 3 --freealign-calibration $FAAP $ARMS --output $V/B_boxes_only_arms_val_result.json
python scripts/select_alignformer_arm.py --result $V/B_boxes_only_arms_val_result.json \
  --output $V/alignformer_arm_selection_val_result.json          # also writes $V/.arms_for_test, $V/.arms_for_delay
sweep B_boxes_only $AF test 5 --freealign-calibration $FAAP $(cat $V/.arms_for_test) --output $V/B_boxes_only_arms_test_result.json
for d in 1 2 4; do
  sweep B_boxes_only $AF test 3 --sweep 0 0.4 1.0 2.0 --delay-frames $d --freealign-calibration $FAAP \
    $(cat $V/.arms_for_delay) --output $V/B_boxes_only_arms_delay${d}_test_result.json
done
sweep "B_boxes+embeddings" $AF test 5 --freealign-calibration $FAAP $(cat $V/.arms_for_delay) \
  --output "$V/B_boxes+embeddings_arms_test_result.json"

# The exact re-solve (alignformer.refine) on val: three gate/evidence settings beside the deployed and
# selected decision arms and FreeAlign 1.0 m; selected on val AP@0.7 sweep mean, then test.
ARMS2="--abstain-arm per_pair --abstain-arm abstain:0.2"
sweep B_boxes_only $AF val 3 --freealign-calibration $FAAP $ARMS2 --refine icp --refine-gates 2.0 1.0 0.5 --refine-min-pairs 3 \
  --output $V/B_boxes_only_icp_default_val_result.json
sweep B_boxes_only $AF val 3 --freealign-calibration $FAAP $ARMS2 --refine icp --refine-gates 3.0 1.5 0.75 0.5 --refine-min-pairs 3 \
  --output $V/B_boxes_only_icp_wide_val_result.json
sweep B_boxes_only $AF val 3 --freealign-calibration $FAAP $ARMS2 --refine icp --refine-gates 2.0 1.0 0.5 --refine-min-pairs 2 \
  --output $V/B_boxes_only_icp_minp2_val_result.json
# Camera colour on V2X-Real val (CPU, ~7 min): pre-registered bars, split by shared-object bucket.
python scripts/analyze_v2xreal_colour_separability.py --root $D/val --pairs 400 \
  --output $V/colour_separability_val_result.json
# Chain K adds the agreement arms to every re-solve setting: append to each sweep above
#   --agree-tolerance 0.3 --agree-tolerance 0.5 --agree-tolerance 1.0
# Chain L, after K: the pose graph on top of the default-gate re-solve (fill, then joint).
for mode in fill joint; do
  sweep B_boxes_only $AF val 3 --freealign-calibration $FAAP $ARMS2 --agree-tolerance 0.5 \
    --refine icp --refine-gates 2.0 1.0 0.5 --refine-min-pairs 3 \
    --graph-arm alignformer_abstain_0.2_icp --graph-mode $mode --output $V/B_boxes_only_graph_${mode}_val_result.json
done
```
