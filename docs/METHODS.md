# Methods and assumptions

All methods accept finite 2-D arrays without implicit normalization. Intensities, sigma, and data_range must use the same scale. The included simulator produces magnitude values from two independent Gaussian channels; real MRI noise depends on acquisition, coil combination, reconstruction, and preprocessing.

## Baselines
- Gaussian and median filtering: local smoothing references.
- NLM: patch-similarity filtering.
- Wavelet: soft coefficient shrinkage.
- TV: Chambolle total-variation denoising.
- BM3D: optional dependency adapter; verify its version and scaling before comparison.
- Residual CNN: predicts an image-domain noise residual and subtracts it. Architecture only; no weights or results are supplied.

## Proposed noise-adaptive Wavelet–TV
The deterministic NA-WTV baseline estimates noise using robust MAD of high-frequency coefficients unless sigma is given, computes wavelet and TV estimates, and blends them using a bounded rule whose TV share increases with estimated noise. It is a testable hand-designed hypothesis, not learned residual learning and not evidence of superiority. Compare against each component on subject-level splits and multiple noise levels; inspect structures and edges as well as scalar metrics.
