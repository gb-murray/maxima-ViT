import torch

from xrd_geometry.contracts import ParameterSpace
from xrd_geometry.diagnostics import parameter_jacobians, uncertainty_sensitivity
from xrd_geometry.losses import MultivariateGaussianNLL
from xrd_geometry.models import FullCovarianceHead, ImageRegressor
from xrd_geometry.prediction import predict_with_uncertainty

from .helpers import CountingBackbone


def _parameter_space():
    return ParameterSpace.xrd_five_parameter([0.1, 0.02, 0.03, 0.0, 0.0], [0.01, 0.01, 0.01, 0.02, 0.02])


def _model():
    return ImageRegressor(CountingBackbone(), FullCovarianceHead(6, 5, hidden_dim=10, dropout=0.4))


def test_full_covariance_is_positive_definite_and_nll_backpropagates():
    head = FullCovarianceHead(6, 5, hidden_dim=10)
    features = torch.randn(4, 6)
    mean, factor = head(features)
    covariance = factor @ factor.transpose(-1, -2)
    assert mean.shape == (4, 5)
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0)
    loss = MultivariateGaussianNLL()((mean, factor), torch.randn(4, 5))
    loss.backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_head_only_mc_dropout_caches_backbone_features():
    model = _model()
    inputs = torch.randn(2, 3, 8, 8)
    result = predict_with_uncertainty(model, inputs, _parameter_space(), samples=5)
    assert model.feature_extractor.calls == 1
    assert result.mean.shape == (2, 5)
    assert result.total_covariance.shape == (2, 5, 5)
    torch.testing.assert_close(
        result.total_covariance,
        result.conditional_covariance + result.model_covariance,
    )


def test_pixel_space_diagnostics_have_detector_tensor_shape():
    model = _model()
    inputs = torch.randn(2, 3, 8, 8)
    jacobians = parameter_jacobians(model, inputs)
    sensitivity = uncertainty_sensitivity(model, inputs, samples=3, seed=11)
    assert jacobians.shape == (2, 5, 3, 8, 8)
    assert sensitivity.shape == inputs.shape
    assert torch.isfinite(jacobians).all()
    assert torch.isfinite(sensitivity).all()


def test_uncertainty_sensitivity_reuses_seeded_dropout_draws():
    model = _model()
    inputs = torch.randn(1, 3, 8, 8)
    first = uncertainty_sensitivity(model, inputs, samples=4, seed=7)
    second = uncertainty_sensitivity(model, inputs, samples=4, seed=7)
    torch.testing.assert_close(first, second)
