import torch

from embedding_aware_belt_fusion.alignformer.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
    log_sinkhorn,
)


def test_assignment_rows_and_columns_are_normalized():
    scores = torch.randn(2, 4, 6)
    alpha = torch.tensor(0.5)

    assignment = log_sinkhorn(scores, alpha, DEFAULT_SINKHORN_ITERATIONS).exp()

    # Each real row distributes unit mass across real columns plus its dustbin.
    assert torch.allclose(assignment[:, :-1, :].sum(dim=2), torch.ones(2, 4), atol=1e-4)
    assert torch.allclose(assignment[:, :, :-1].sum(dim=1), torch.ones(2, 6), atol=1e-4)


def test_shape_includes_dustbin_row_and_column():
    assignment = log_sinkhorn(torch.randn(1, 3, 5), torch.tensor(0.0), 10)

    assert assignment.shape == (1, 4, 6)


def test_a_dominant_score_wins_its_row():
    scores = torch.full((1, 2, 2), -5.0)
    scores[0, 0, 1] = 10.0

    assignment = log_sinkhorn(scores, torch.tensor(0.0), 50).exp()

    assert assignment[0, 0, 1] > 0.9


def test_handles_asymmetric_and_single_object_sets():
    assignment = log_sinkhorn(torch.randn(1, 1, 7), torch.tensor(0.0), 20)

    assert assignment.shape == (1, 2, 8)
    assert torch.isfinite(assignment).all()


def test_an_empty_object_set_does_not_produce_inf_or_nan():
    # An agent can legitimately detect nothing. log(0) in the marginals would
    # put -inf into the iteration and NaN into the gradients.
    assignment = log_sinkhorn(torch.randn(1, 0, 4), torch.tensor(0.0), 20)

    assert assignment.shape == (1, 1, 5)
    assert torch.isfinite(assignment).all()


def test_is_differentiable_with_finite_gradients():
    scores = torch.randn(1, 3, 3, requires_grad=True)
    alpha = torch.tensor(0.3, requires_grad=True)

    log_sinkhorn(scores, alpha, 20).exp().sum().backward()

    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(alpha.grad).all()
