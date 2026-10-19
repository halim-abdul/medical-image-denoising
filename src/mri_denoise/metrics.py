"""Image quality metrics with explicit intensity range."""
import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(reference: np.ndarray, estimate: np.ndarray,
                    data_range: float = 1.0) -> dict[str, float]:
    ref, pred = np.asarray(reference, float), np.asarray(estimate, float)
    if ref.ndim != 2 or ref.shape != pred.shape:
        raise ValueError("reference and estimate must be same-shaped 2-D images")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    err = pred - ref
    denom = max(float(np.mean(ref ** 2)), np.finfo(float).eps)
    hfen_ref = gaussian_laplace(ref, sigma=1.5)
    hfen_err = gaussian_laplace(err, sigma=1.5)
    return {
        "psnr": float(peak_signal_noise_ratio(ref, pred, data_range=data_range)),
        "ssim": float(structural_similarity(ref, pred, data_range=data_range)),
        "nmse": float(np.mean(err ** 2) / denom),
        "hfen": float(np.linalg.norm(hfen_err) / max(np.linalg.norm(hfen_ref), np.finfo(float).eps)),
    }
