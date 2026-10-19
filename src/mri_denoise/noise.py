"""Noise utilities; simulated noise is not a replacement for measured acquisition noise."""
import numpy as np
from scipy.ndimage import convolve


def add_rician_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Add magnitude-domain Rician noise via independent Gaussian quadrature channels."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    x = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(seed)
    real = x + rng.normal(0.0, sigma, size=x.shape)
    imag = rng.normal(0.0, sigma, size=x.shape)
    return np.hypot(real, imag).astype(np.float32)


def estimate_sigma_mad(image: np.ndarray) -> float:
    """Estimate scale from robust MAD of diagonal high-pass coefficients."""
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("image must be 2-D and at least 2x2")
    detail = convolve(x, np.array([[1, -1], [-1, 1]]) / 2.0, mode="reflect")
    med = np.median(detail)
    return float(np.median(np.abs(detail - med)) / (0.67448975 * np.sqrt(2.0)))
