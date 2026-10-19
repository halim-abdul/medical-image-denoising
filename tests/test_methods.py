import numpy as np
import pytest

from mri_denoise.methods import denoise, na_wtv
from mri_denoise.metrics import compute_metrics
from mri_denoise.noise import add_rician_noise, estimate_sigma_mad


def test_rician_noise_reproducible_and_nonnegative():
    x = np.full((32, 32), 0.5, dtype=np.float32)
    a, b = add_rician_noise(x, 0.03, 4), add_rician_noise(x, 0.03, 4)
    assert np.array_equal(a, b)
    assert np.all(a >= 0)


def test_methods_shape_and_finite():
    x = np.zeros((32, 32), dtype=np.float32)
    x[8:24, 8:24] = 0.8
    noisy = add_rician_noise(x, 0.02, 1)
    for name in ("gaussian", "median", "nlm", "wavelet", "tv", "na_wtv"):
        y = denoise(noisy, name, sigma=0.02)
        assert y.shape == x.shape and np.isfinite(y).all()


def test_na_wtv_reduces_mse_on_simple_phantom():
    x = np.zeros((48, 48), dtype=np.float32)
    x[12:36, 12:36] = 0.7
    noisy = add_rician_noise(x, 0.02, 12)
    assert np.mean((na_wtv(noisy, sigma=0.02) - x) ** 2) < np.mean((noisy - x) ** 2)


def test_metrics_identity_and_validation():
    x = np.ones((8, 8))
    m = compute_metrics(x, x)
    assert np.isinf(m["psnr"])
    assert m["ssim"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        denoise(np.zeros((8, 8, 1)), "tv")


def test_sigma_estimate_nonnegative():
    assert estimate_sigma_mad(np.zeros((8, 8))) == 0.0
