import torch

from xrd_geometry.contracts import ParameterSpace
from xrd_geometry.diagnostics import parameter_jacobians, uncertainty_sensitivity
from xrd_geometry.losses import MultivariateGaussianNLL
from xrd_geometry.models import FullCovarianceHead, ImageRegressor, enable_stochastic_depth
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


def test_quality_adapter_is_identity_initialized_and_changes_covariance():
    head = FullCovarianceHead(6, 5, hidden_dim=10, quality_dim=3, quality_hidden_dim=4)
    head.eval()
    features = torch.randn(2, 6)
    quality = torch.randn(2, 3)
    _, adapted = head(features, quality)
    with torch.no_grad():
        assert head.quality_adapter is not None
        head.quality_adapter[-1].bias.fill_(0.25)
    _, inflated = head(features, quality)
    torch.testing.assert_close(inflated, adapted * torch.exp(torch.tensor(0.25)))


def test_stochastic_depth_activation_is_scoped():
    class SwinDropPath(torch.nn.Module):
        def forward(self, inputs):
            return inputs

    module = torch.nn.Sequential(torch.nn.BatchNorm1d(2), SwinDropPath())
    module.eval()
    enable_stochastic_depth(module)
    assert module[0].training is False
    assert module[1].training is True


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


def test_quality_conditioned_diagnostics_hold_summaries_fixed():
    torch.manual_seed(11)
    model = ImageRegressor(
        CountingBackbone(),
        FullCovarianceHead(6, 5, hidden_dim=10, dropout=0.0, quality_dim=3, quality_hidden_dim=4),
    )
    with torch.no_grad():
        model.head.quality_adapter[0].weight.fill_(0.1)
        model.head.quality_adapter[0].bias.zero_()
        model.head.quality_adapter[-1].weight.fill_(0.2)
        model.head.quality_adapter[-1].bias.zero_()
    inputs = torch.randn(1, 3, 8, 8)
    low_quality = torch.zeros(1, 3, requires_grad=True)
    high_quality = torch.ones(1, 3, requires_grad=True)

    low_jacobians = parameter_jacobians(model, inputs, low_quality)
    high_jacobians = parameter_jacobians(model, inputs, high_quality)
    torch.testing.assert_close(low_jacobians, high_jacobians)
    assert low_jacobians.shape == (1, 5, 3, 8, 8)

    low_sensitivity = uncertainty_sensitivity(model, inputs, quality=low_quality, samples=3)
    high_sensitivity = uncertainty_sensitivity(model, inputs, quality=high_quality, samples=3)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    _, factor = model(reference_inputs, low_quality.detach())
    covariance_trace = factor.square().sum()
    reference_gradient = torch.autograd.grad(covariance_trace, reference_inputs)[0]
    torch.testing.assert_close(low_sensitivity, reference_gradient)
    assert not torch.allclose(low_sensitivity, high_sensitivity)
    assert torch.isfinite(high_sensitivity).all()
    assert low_quality.grad is None
    assert high_quality.grad is None
