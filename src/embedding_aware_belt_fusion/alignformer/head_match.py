"""Log-domain Sinkhorn with dustbins, for soft cross-agent correspondence.

Objects seen by only one agent have no counterpart, so the score matrix carries
a dustbin row and column that absorb their mass. Running Sinkhorn in the log
domain keeps it stable for the peaked score matrices a trained model produces.

Follows Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural
Networks", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_SINKHORN_ITERATIONS = 20


def log_sinkhorn(scores: Tensor, alpha: Tensor, iterations: int) -> Tensor:
    """Normalize ``scores`` into a log-domain soft assignment with dustbins.

    Parameters
    ----------
    scores: ``(B, M, N)`` match scores between ego and CAV objects.
    alpha: scalar tensor, the learnable dustbin score.
    iterations: number of Sinkhorn normalization steps.

    Returns
    -------
    Tensor
        ``(B, M + 1, N + 1)`` log-assignment. The last row and column are the
        dustbins. Rows sum to 1 and columns sum to 1 after ``.exp()`` — including
        in the degenerate ``rows == 0`` or ``columns == 0`` case, where every
        real object on the other side is assigned to its dustbin with certainty
        (``log(1) == 0``, hence an all-zeros return).
    """
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, M, N), got {tuple(scores.shape)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    batch, rows, columns = scores.shape
    bin_score = alpha.to(scores)

    couplings = torch.cat(
        [
            torch.cat([scores, bin_score.expand(batch, rows, 1)], dim=2),
            torch.cat(
                [bin_score.expand(batch, 1, columns), bin_score.expand(batch, 1, 1)],
                dim=2,
            ),
        ],
        dim=1,
    )

    # An agent can legitimately detect nothing. With rows or columns at zero the
    # marginals would contain log(0) = -inf, which turns into NaN gradients.
    # The semantically correct assignment there is that every real row (or
    # column) on the other side matches its dustbin with certainty — the single
    # remaining column (or row) *is* the dustbin — so each carries log(1) = 0.
    # ``couplings * 0`` satisfies the row/column-sums-to-1 contract exactly and
    # keeps the backward pass finite: unlike ``torch.zeros_like``, multiplying
    # by zero stays attached to the autograd graph, so scores/alpha still get a
    # (zero-valued) gradient instead of an error from a disconnected constant.
    # There is no matching information to learn from an empty detection set,
    # so a zero gradient here is correct.
    if rows == 0 or columns == 0:
        return couplings * 0

    row_count = scores.new_tensor(float(rows))
    column_count = scores.new_tensor(float(columns))

    normalizer = -(row_count + column_count).log()
    log_mu = torch.cat(
        [normalizer.expand(rows), column_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)
    log_nu = torch.cat(
        [normalizer.expand(columns), row_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)

    return couplings + u.unsqueeze(2) + v.unsqueeze(1) - normalizer
