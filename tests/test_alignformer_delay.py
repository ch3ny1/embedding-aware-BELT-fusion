"""Transmission delay as a sweep axis, alongside localization error.

Why delay is a separate axis and not more of the same
-----------------------------------------------------

Localization error moves the CAV's whole box set rigidly: there IS a single
SE(2) that undoes it, which is what head B estimates. Delay does not work like
that. The CAV transmits what it saw at ``t - d`` together with its own pose at
``t - d``, so **static** objects arrive in exactly the right world position and
need no correction at all, while **moving** objects arrive displaced by their
own velocity times the delay. No rigid transform fixes that, and a solver that
tries will be dragged off by whichever objects happened to be moving.

That is the interesting part. Under delay the matched correspondences split
into a correct majority and a velocity-corrupted minority, which is precisely
the shape robust weighting exists for -- Huber IRLS down-weights them and
per-pair abstention declines when what is left cannot carry a correction.
FreeAlign's relative-distance graph has no equivalent mechanism: a moving
object changes its distance to every other node at once.

What these tests pin
--------------------

``wild_setting`` was previously stripped wholesale before the dataset was
built (``baselines.load_baseline``) so that OpenCOOD's own constant-seed,
z-perturbing localization noise could not run beside this sweep's controlled
perturbation. That was right, and it also threw out the asynchrony model with
it. These tests pin the narrow re-entry: delay on, OpenCOOD's localization
noise still firmly off, and ``sim`` mode so the delay is a constant rather
than a draw that would fight this sweep's seeding.

The quantization is not incidental. OPV2V is 10 Hz and OpenCOOD floors the
delay to whole frames (``time_delay // 100``), so 150 ms and 100 ms are the
SAME experiment and a sweep that reported them as different points would be
reporting rounding. The axis is therefore specified in frames.
"""

import pytest

from embedding_aware_belt_fusion.alignformer.delay import (
    DELAY_FRAME_MS,
    delay_wild_setting,
    frames_for_ms,
)


def test_no_delay_asks_for_no_wild_setting_at_all():
    # Zero has to mean "the dataset is built exactly as it was before this
    # feature existed", not "wild_setting present but neutral". A neutral
    # block would still take OpenCOOD's async branch and still have to be
    # trusted to do nothing.
    assert delay_wild_setting(0) is None


def test_a_delay_never_turns_on_opencood_localization_noise():
    # The whole reason wild_setting was stripped. If this ever flips, this
    # sweep's perturbation and OpenCOOD's own would both be running and every
    # number in the result would be measuring their sum.
    setting = delay_wild_setting(2)

    assert setting["loc_err"] is False
    assert setting["xyz_std"] == 0.0
    assert setting["ryp_std"] == 0.0


def test_the_delay_is_a_constant_and_not_a_draw():
    # 'real' mode adds np.random.uniform overhead, which would consume from
    # the same global stream this sweep seeds for its own noise draws and make
    # the paired-seed comparison unpaired.
    setting = delay_wild_setting(1)

    assert setting["async"] is True
    assert setting["async_mode"] == "sim"


def test_the_requested_frame_count_survives_opencood_quantization():
    # OpenCOOD floors ms to frames with `time_delay // 100`, so the overhead
    # must be expressed in whole frames or the sweep silently reports a
    # different delay than it asked for.
    for frames in (1, 2, 3, 5):
        setting = delay_wild_setting(frames)

        assert setting["async_overhead"] // DELAY_FRAME_MS == frames


def test_milliseconds_are_floored_to_frames_the_way_opencood_floors_them():
    # 150 ms and 100 ms are the same experiment at 10 Hz. A caller that thinks
    # in milliseconds has to be told that, not quietly given one of them.
    assert frames_for_ms(0) == 0
    assert frames_for_ms(99) == 0
    assert frames_for_ms(100) == 1
    assert frames_for_ms(150) == 1
    assert frames_for_ms(200) == 2


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError, match="negative|non-negative"):
        delay_wild_setting(-1)


def test_the_setting_carries_every_key_opencood_reads_unconditionally():
    # basedataset reads seed/async/async_overhead/loc_err/xyz_std/ryp_std
    # without a default the moment `wild_setting` is present, so a missing one
    # is a KeyError halfway through dataset construction rather than here.
    setting = delay_wild_setting(1)

    for key in (
        "seed", "async", "async_mode", "async_overhead",
        "loc_err", "xyz_std", "ryp_std",
    ):
        assert key in setting


def test_the_setting_is_a_fresh_object_each_time():
    # It is handed to OpenCOOD, which is free to keep a reference; a shared
    # dict would let one sweep's dataset mutate another's.
    first, second = delay_wild_setting(1), delay_wild_setting(1)

    assert first == second
    assert first is not second
