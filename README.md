# MRI Denoising Bench

Research scaffold for reproducible denoising experiments on 2-D magnitude MRI. Includes Gaussian and median filters, non-local means, wavelet shrinkage, total variation, optional BM3D, a residual CNN architecture, and a noise-adaptive Wavelet–TV hybrid.

> Research status: this is not a validated clinical tool. No method is claimed to preserve diagnostic information. Synthetic results do not establish clinical utility.

## Included
- Reproducible Rician magnitude-noise simulator and robust high-frequency MAD sigma estimate.
- Classical baselines: Gaussian, median, NLM, wavelet, TV, optional BM3D.
- NA-WTV: transparent adaptive blend of wavelet and TV outputs (a hand-designed baseline, not learned residual learning).
- Optional PyTorch residual CNN architecture; no pretrained weights.
- PSNR, SSIM, NMSE and HFEN metrics; CLI, tests, CI, and a synthetic notebook.

## Install and run

~~~bash
python -m venv .venv
python -m pip install -e ".[dev]"
mri-denoise demo --output outputs/demo
~~~

Optional dependencies: install with pip install -e ".[bm3d]" for BM3D, or pip install -e ".[deep]" for PyTorch.

Denoise one NumPy image (finite 2-D array, intensities in [0,1]):

~~~bash
mri-denoise run --input image.npy --method na_wtv --output outputs/denoised.npy --sigma 0.04 --data-range 1
mri-denoise evaluate --clean clean.npy --noisy noisy.npy --output outputs/metrics.csv --sigma 0.04
~~~

Input is not silently normalized. Keep image, sigma and data_range on the same scale. NIfTI/DICOM volume loading and orientation management are intentionally left to a dedicated adapter (e.g. NiBabel or SimpleITK); this CLI selects explicit 2-D arrays only.

## NA-WTV hypothesis

NA-WTV estimates noise from robust high-frequency MAD (unless sigma is supplied), computes wavelet and TV estimates, then blends them with a bounded noise-dependent weight. It is included as a falsifiable baseline, not as a validated or learned method. Tune parameters on validation participants only. The optional CNN predicts the residual y-x; it requires subject-level training and is distributed without weights.

## Reproducible evaluation
Split by participant, use the same noisy input and fixed intensity scaling for each method, tune only on validation subjects, and reserve test subjects for final reporting. Report per-subject metrics, mean and standard deviation, runtime, acquisition/noise assumptions, software versions, and representative structure/edge inspection. PSNR/SSIM do not establish anatomical fidelity. Never commit patient-identifiable data or derived images.

See docs/METHODS.md, docs/EXPERIMENTS.md, and docs/DATA.md. Start with notebooks/01_synthetic_demo.ipynb.

## Citation and license
No paper is associated with this scaffold yet. Cite the repository and commit hash if reused; do not present the hybrid as a published result. MIT License; see LICENSE.
