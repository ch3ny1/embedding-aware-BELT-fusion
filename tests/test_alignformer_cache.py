"""Storage-layer tests for the per-frame detection/ROI-feature cache.

The dataclass and read/write round-trip are the load-bearing contract here;
building the cache against a live detector is exercised by the CLI itself
(see the task report for that verification), not by these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_aware_belt_fusion.alignformer.cache import (
    FrameRecord,
    cache_path,
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
