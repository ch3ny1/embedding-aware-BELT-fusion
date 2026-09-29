"""What each alignment method actually puts on the wire.

The comparison that matters, and the trap in it
-----------------------------------------------

Both AlignFormer and FreeAlign are **late** fusion: the ego cannot fuse a CAV's
detection without the full box, so the detected boxes are on the wire either
way and neither method can be charged for them as if they were its own
overhead. What separates the two is only what each adds *on top* of that
payload for the purpose of alignment.

FreeAlign adds nothing. It reads ``*_boxes`` and ``*_mask`` and builds its
relative-distance graph from the centres and yaws it already has
(``freealign.edge_features``). AlignFormer as configured adds a 128-float
per-object embedding, which at float32 is 512 bytes against the box's 28 --
**19x the message** for something the association diagnostic measured at
``+0.0001`` AUC-equivalent and the trunk ablation at ``+0.0007`` mean AP.

So the honest accounting has three rows, not two, and the middle one is the
one a reader wants: boxes-only AlignFormer and FreeAlign are the SAME number
of bytes, and the embedding is a cost AlignFormer pays and FreeAlign does not.
Reporting only "AlignFormer 8474 vs FreeAlign 439" would make the method look
19x more expensive than the variant we would actually ship; reporting only
"both 439" would hide that the deployed checkpoint is not that variant.

These tests pin the arithmetic and, more importantly, the framing: that the
shared box payload is attributed to neither method.
"""

import pytest

from embedding_aware_belt_fusion.alignformer.bandwidth import (
    BYTES_PER_FLOAT32,
    alignment_overhead_bytes,
)

# The measured mean from outputs/alignformer/r140/bandwidth_r140_result.json,
# so these numbers line up with the ones the report quotes.
OBJECTS_PER_AGENT = 15.692281991313063
BOX_FLOATS = 7


def test_freealign_adds_nothing_to_the_message():
    # It reads the boxes late fusion already sends. A method that needs no
    # extra bytes should be charged no extra bytes.
    overhead = alignment_overhead_bytes(
        OBJECTS_PER_AGENT, embed_dim=0, box_floats=BOX_FLOATS
    )

    assert overhead["alignment_overhead_bytes_per_frame_per_agent"] == 0.0


def test_the_shared_box_payload_is_attributed_to_neither_method():
    # Late fusion sends the boxes regardless of how it aligns them, so the
    # box bytes are the paradigm's cost and not the aligner's. If this ever
    # folds the boxes into the overhead, every ratio in the report changes.
    overhead = alignment_overhead_bytes(
        OBJECTS_PER_AGENT, embed_dim=128, box_floats=BOX_FLOATS
    )

    expected_shared = OBJECTS_PER_AGENT * BOX_FLOATS * BYTES_PER_FLOAT32
    assert overhead["shared_box_bytes_per_frame_per_agent"] == pytest.approx(
        expected_shared
    )
    assert overhead["alignment_overhead_bytes_per_frame_per_agent"] == pytest.approx(
        OBJECTS_PER_AGENT * 128 * BYTES_PER_FLOAT32
    )


def test_the_embedding_is_the_whole_difference_and_it_is_large():
    embedded = alignment_overhead_bytes(OBJECTS_PER_AGENT, 128, BOX_FLOATS)
    boxes_only = alignment_overhead_bytes(OBJECTS_PER_AGENT, 0, BOX_FLOATS)

    ratio = (
        embedded["total_bytes_per_frame_per_agent"]
        / boxes_only["total_bytes_per_frame_per_agent"]
    )
    # 128 embedding floats against 7 box floats: (128 + 7) / 7.
    assert ratio == pytest.approx((128 + BOX_FLOATS) / BOX_FLOATS, rel=1e-6)
    assert ratio > 19.0


def test_boxes_only_alignformer_and_freealign_cost_exactly_the_same():
    # The claim the report rests on, as an assertion rather than a sentence.
    ours = alignment_overhead_bytes(OBJECTS_PER_AGENT, 0, BOX_FLOATS)
    theirs = alignment_overhead_bytes(OBJECTS_PER_AGENT, 0, BOX_FLOATS)

    assert (
        ours["total_bytes_per_frame_per_agent"]
        == theirs["total_bytes_per_frame_per_agent"]
    )


def test_a_negative_embedding_dimension_is_refused():
    with pytest.raises(ValueError, match="negative|non-negative"):
        alignment_overhead_bytes(OBJECTS_PER_AGENT, -1, BOX_FLOATS)


def test_a_box_must_have_at_least_the_three_floats_freealign_reads():
    # freealign.edge_features reads centres (x, y) and yaw. Fewer than three
    # floats per box is not a message either method could align from, and a
    # bandwidth table built on one would be comparing impossibilities.
    with pytest.raises(ValueError, match="box_floats"):
        alignment_overhead_bytes(OBJECTS_PER_AGENT, 0, box_floats=2)
