"""Classical and hybrid 2-D denoising methods."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_wavelet

from .noise import estimate_sigma_mad

METHODS = ("gaussian", "median", "nlm", "wavelet", "tv", "na_wtv", "bm3d")


def available_methods():
    return METHODS


def _image(image):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("image must be finite, 2-D, and at least 2x2")
    return x


def na_wtv(image, sigma=None, data_range=1.0, wavelet="db2", level=2):
    """Noise-adaptive wavelet-TV blend; deterministic baseline, not learned residual learning."""
    x = _image(image)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    sig = max(float(estimate_sigma_mad(x) if sigma is None else sigma), 0.0) / data_range
    wav = denoise_wavelet(x, sigma=sig, wavelet=wavelet, wavelet_levels=level,
                          mode="soft", rescale_sigma=False, channel_axis=None)
    tv = denoise_tv_chambolle(x, weight=max(sig, 1e-5), channel_axis=None)
    alpha = float(np.clip(sig / (sig + 0.04), 0.15, 0.85))
    return ((1 - alpha) * wav + alpha * tv).astype(np.float32)


def denoise(image, method, sigma=None, data_range=1.0):
    """Denoise one 2-D image; preserve scale and never normalize implicitly."""
    x = _image(image)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    sig = max(float(estimate_sigma_mad(x) if sigma is None else sigma), 0.0)
    s = sig / data_range
    if method == "gaussian":
        return gaussian_filter(x, sigma=max(0.2, s * 1.5)).astype(np.float32)
    if method == "median":
        return median_filter(x, size=3).astype(np.float32)
    if method == "nlm":
        return denoise_nl_means(x, h=max(s, 1e-4), fast_mode=True, patch_size=5,
                                patch_distance=6, preserve_range=True,
                                channel_axis=None).astype(np.float32)
    if method == "wavelet":
        return denoise_wavelet(x, sigma=s, mode="soft", rescale_sigma=False,
                               channel_axis=None).astype(np.float32)
    if method == "tv":
        return denoise_tv_chambolle(x, weight=max(s, 1e-5), channel_axis=None).astype(np.float32)
    if method == "na_wtv":
        return na_wtv(x, sigma=sig, data_range=data_range)
    if method == "bm3d":
        try:
            import bm3d
        except ImportError as exc:
            raise ImportError("Install BM3D with pip install -e '.[bm3d]'") from exc
        return np.asarray(bm3d.bm3d(x, s), dtype=np.float32)
    raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")
