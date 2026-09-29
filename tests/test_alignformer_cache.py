"""Storage-layer tests for the per-frame detection/ROI-feature cache.

The dataclass and read/write round-trip are the load-bearing contract here;
building the cache against a live detector is exercised by the CLI itself
(see the task report for that verification), not by these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_aware_belt_fusion.alignformer import cache as cache_module
from embedding_aware_belt_fusion.alignformer import evaluate as evaluate_module
from embedding_aware_belt_fusion.alignformer.cache import (
    FrameRecord,
    cache_path,
    frame_seed,
    read_frame,
    write_frame,
)


def _record(count=3, channels=4, grid=4):
    return FrameRecord(
        boxes=np.random.randn(count, 7).astype(np.float32),
        scores=np.random.rand(count).astype(np.float32),
        gt_ids=["12", None, "45"][:count],
        roi=np.random.randn(count, channels, grid, grid).astype(np.float16),
    )


def test_frame_round_trips_through_disk(tmp_path):
    record = _record()
    path = tmp_path / "000069.npz"

    write_frame(path, record)
    loaded = read_frame(path)

    assert np.allclose(loaded.boxes, record.boxes)
    assert np.allclose(loaded.scores, record.scores)
    assert loaded.gt_ids == record.gt_ids
    assert np.allclose(loaded.roi, record.roi)


def test_none_gt_ids_survive_the_round_trip(tmp_path):
    # None means "matched no ground-truth object" and must not become the
    # string "None", which would silently create a bogus shared identity that
    # the matching loss would then train towards.
    record = _record()
    path = tmp_path / "000069.npz"

    write_frame(path, record)

    assert read_frame(path).gt_ids[1] is None


def test_empty_frame_round_trips(tmp_path):
    record = FrameRecord(
        boxes=np.zeros((0, 7), np.float32),
        scores=np.zeros((0,), np.float32),
        gt_ids=[],
        roi=np.zeros((0, 4, 4, 4), np.float16),
    )
    path = tmp_path / "000070.npz"

    write_frame(path, record)
    loaded = read_frame(path)

    assert loaded.boxes.shape == (0, 7)
    assert loaded.gt_ids == []


def test_cache_path_is_split_scenario_cav_timestamp(tmp_path):
    path = cache_path(tmp_path, "train", "2021_08_18", "1045", "000069")

    assert path == tmp_path / "train" / "2021_08_18" / "1045" / "000069.npz"


def test_frame_record_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="length"):
        FrameRecord(
            boxes=np.zeros((3, 7), np.float32),
            scores=np.zeros((2,), np.float32),
            gt_ids=["a", "b", "c"],
            roi=np.zeros((3, 4, 4, 4), np.float16),
        )


def test_write_frame_rejects_a_real_gt_id_that_is_the_empty_string(tmp_path):
    # An empty-string gt_id would decode back as None (the "no match"
    # sentinel), fabricating a shared identity between two agents that never
    # matched anything -- this must be caught at write time, not assumed
    # impossible.
    record = FrameRecord(
        boxes=np.zeros((1, 7), np.float32),
        scores=np.zeros((1,), np.float32),
        gt_ids=[""],
        roi=np.zeros((1, 4, 4, 4), np.float16),
    )

    with pytest.raises(ValueError, match="empty string"):
        write_frame(tmp_path / "000071.npz", record)


def test_evaluate_and_cache_share_the_same_frame_seed_function():
    # R31: cache.py and evaluate.py used to each define a byte-identical
    # _frame_seed. Equal *output* would not catch the two copies drifting
    # apart again after a future edit to only one of them -- only object
    # identity proves there is exactly one implementation left to edit.
    assert evaluate_module.frame_seed is cache_module.frame_seed
    assert evaluate_module.frame_seed is frame_seed


# ----------------------------------------------------------------------------
# Camera features ride beside the LiDAR ROI feature, optionally
# ----------------------------------------------------------------------------
#
# OPV2V caches were written without them and must read back unchanged; a
# V2X-Real cache built with --camera carries three extra arrays per frame.


def _record_with_camera(n: int = 3):
    import numpy as np

    from embedding_aware_belt_fusion.alignformer.cache import FrameRecord

    return FrameRecord(
        boxes=np.zeros((n, 7), dtype=np.float32),
        scores=np.ones(n, dtype=np.float32),
        gt_ids=[None] * n,
        roi=np.zeros((n, 4, 4, 8), dtype=np.float16),
        camera=np.arange(n * 256, dtype=np.float32).reshape(n, 256),
        has_camera=np.array([True, False, True][:n]),
        camera_index=np.array([0, -1, 1][:n], dtype=np.int8),
    )


def test_camera_fields_round_trip_through_the_npz(tmp_path):
    import numpy as np

    from embedding_aware_belt_fusion.alignformer.cache import read_frame, write_frame

    record = _record_with_camera()
    write_frame(tmp_path / "f.npz", record)

    loaded = read_frame(tmp_path / "f.npz")

    np.testing.assert_allclose(loaded.camera, record.camera, rtol=1e-3)  # float16 on disk
    np.testing.assert_array_equal(loaded.has_camera, record.has_camera)
    np.testing.assert_array_equal(loaded.camera_index, record.camera_index)


def test_a_record_without_camera_fields_reads_back_with_none(tmp_path):
    import numpy as np

    from embedding_aware_belt_fusion.alignformer.cache import FrameRecord, read_frame, write_frame

    record = FrameRecord(
        boxes=np.zeros((2, 7), dtype=np.float32),
        scores=np.ones(2, dtype=np.float32),
        gt_ids=[None, "7"],
        roi=np.zeros((2, 4, 4, 8), dtype=np.float16),
    )
    write_frame(tmp_path / "f.npz", record)

    loaded = read_frame(tmp_path / "f.npz")

    assert loaded.camera is None and loaded.has_camera is None and loaded.camera_index is None
    assert loaded.gt_ids == [None, "7"]


def test_camera_fields_must_match_the_box_count():
    import numpy as np

    from embedding_aware_belt_fusion.alignformer.cache import FrameRecord

    with pytest.raises(ValueError, match="camera"):
        FrameRecord(
            boxes=np.zeros((2, 7), dtype=np.float32),
            scores=np.ones(2, dtype=np.float32),
            gt_ids=[None, None],
            roi=np.zeros((2, 4, 4, 8), dtype=np.float16),
            camera=np.zeros((3, 256), dtype=np.float32),
            has_camera=np.zeros(3, dtype=bool),
            camera_index=np.full(3, -1, dtype=np.int8),
        )


def test_the_cache_cli_has_a_camera_switch(monkeypatch):
    from embedding_aware_belt_fusion.alignformer.cache import parse_args

    monkeypatch.setattr("sys.argv", ["cache", "--config", "c.yaml", "--splits", "s", "--cache-root", "r", "--camera"])
    assert parse_args().camera is True

    monkeypatch.setattr("sys.argv", ["cache", "--config", "c.yaml", "--splits", "s", "--cache-root", "r"])
    assert parse_args().camera is False
