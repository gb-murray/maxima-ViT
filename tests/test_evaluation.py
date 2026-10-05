import torch

from xrd_geometry.evaluation import gaussian_metrics, point_metrics


def test_point_metrics_are_parameterwise():
    mean = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    target = torch.tensor([[0.0, 2.0], [5.0, 2.0]])
    metrics = point_metrics(mean, target)
    torch.testing.assert_close(metrics.mae, torch.tensor([1.5, 2.0]))
    torch.testing.assert_close(metrics.rmse, torch.tensor([2.5**0.5, 8.0**0.5]))


def test_gaussian_metrics_report_marginal_and_joint_coverage():
    mean = torch.zeros(4, 2)
    target = torch.tensor([[0.0, 0.0], [0.25, -0.25], [3.0, 0.0], [0.0, -3.0]])
    covariance = torch.eye(2).expand(4, 2, 2).clone()
    metrics = gaussian_metrics(mean, covariance, target, levels=(0.5, 0.95))
    assert metrics.nll > 0
    torch.testing.assert_close(metrics.marginal_coverage[0.5], torch.tensor([0.75, 0.75]))
    torch.testing.assert_close(metrics.marginal_coverage[0.95], torch.tensor([0.75, 0.75]))
    assert metrics.joint_coverage[0.5] == 0.5
    assert metrics.joint_coverage[0.95] == 0.5
    assert torch.all(metrics.marginal_width[0.95] > metrics.marginal_width[0.5])


def test_gaussian_metrics_validate_shapes_and_levels():
    mean = torch.zeros(2, 2)
    covariance = torch.eye(2).expand(2, 2, 2)
    target = torch.zeros(2, 2)
    try:
        gaussian_metrics(mean, covariance, target, levels=(1.0,))
    except ValueError as error:
        assert "Coverage levels" in str(error)
    else:
        raise AssertionError("Expected invalid coverage level to fail")
