"""Which split directories an experiment may read, enforced rather than asked.

Two constraints carry this project's headline claim, and both are the kind a
future run forgets:

1. **``OPV2V/validate/`` is not a validation split here.** On this machine it
   is a *symlink to* ``OPV2V/test`` -- not merely overlapping it by scenario
   name, but the same directory -- so a constant chosen on it has been chosen
   on test outright. The project's validation set is a scenario-disjoint 15%
   slice of ``train/`` (``split_seed 0``), cached as ``opv2v_splits/val``.
2. **Calibration and model selection never read ``test/``.** Reporting does,
   exactly once, after the constants are frozen.

Both were previously enforced only by a sentence in an argparse ``help=``
string. This module makes them refusals. ``allow_test`` is keyword-only and
has no default on purpose: a caller that has not said whether it is selecting
or reporting is precisely the caller that gets this wrong, so omitting it is a
``TypeError`` rather than a permissive default. The same reasoning as
:func:`noisy_fusion.unshrunk_conditions`' required argument.

The match is on path *components*, never substrings, so an ordinary directory
called ``validate_v2`` or ``test_mini`` passes; a guard that refuses innocent
paths is a guard that callers learn to route around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

# The OPV2V directory that looks like a validation split and is not one.
LEAKY_SPLIT_NAME = "validate"

# The split that may be read for reporting and never for selection.
TEST_SPLIT_NAME = "test"

_USE_INSTEAD = (
    "use the scenario-disjoint 15% slice of train/ (split_seed 0), cached as "
    "opv2v_splits/val"
)


def resolve_split(path: Union[str, Path], *, allow_test: bool) -> Path:
    """Return ``path`` resolved, or raise if reading it would invalidate a result.

    Parameters
    ----------
    path: the split directory a ``--split`` flag named.
    allow_test: ``True`` only for a caller that is *reporting* a frozen
        configuration. Calibration, fitting and any model selection pass
        ``False``; the refusal is what keeps "never select on test" true of the
        code rather than of the documentation.

    Raises
    ------
    ValueError
        If ``path`` names the overlapping ``validate/`` directory under any
        parent, or names ``test/`` while ``allow_test`` is ``False``.
    """
    given = Path(path)
    resolved = given.resolve()
    # Both forms. The resolved path alone misses a `validate` SYMLINK (on this
    # machine it points straight at test/, so the name disappears on resolve);
    # the given path alone misses a link or relative path that lands inside a
    # leaky directory under another name. Either spelling is a refusal.
    components = set(given.parts) | set(resolved.parts)

    if LEAKY_SPLIT_NAME in components:
        # Report the path AS GIVEN: this one usually resolves to test/, and
        # printing the resolved form would show "test" while complaining
        # about "validate".
        raise ValueError(
            f"{given} names OPV2V's validate/ directory, which overlaps the "
            f"test split -- on this machine it is a symlink to {resolved}, so "
            f"selecting anything on it is selecting on test. "
            f"Instead, {_USE_INSTEAD}."
        )

    if not allow_test and TEST_SPLIT_NAME in components:
        raise ValueError(
            f"{resolved} names the test split, and this caller selects rather "
            f"than reports: a constant fitted here is never a held-out number. "
            f"Instead, {_USE_INSTEAD}."
        )

    return resolved


def validation_split_description(config: dict) -> str:
    """The one line a result file records for the validation split it used.

    With ``data.val_root`` set, validation is that official split and the
    carved-slice parameters in the config are inert, so naming them would
    misdescribe the measurement.
    """
    data = config["data"]
    if data.get("val_root"):
        return f"{data['val_root']} :: the official validation split (data.val_root)"
    return (
        f"{data['train_root']} :: validation scenarios (scenario-disjoint, "
        f"val_scenario_fraction={data['val_scenario_fraction']}, split_seed={data['split_seed']})"
    )
