"""Metric evaluation and CSV output."""
import csv
from pathlib import Path

from .methods import METHODS, denoise
from .metrics import compute_metrics


def evaluate_methods(clean, noisy, sigma=None, data_range=1.0, methods=METHODS):
    rows = []
    for name in methods:
        try:
            estimate = denoise(noisy, name, sigma, data_range)
        except ImportError:
            continue
        rows.append({"method": name, **compute_metrics(clean, estimate, data_range)})
    return rows


def save_csv(rows, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no metric rows to save")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
