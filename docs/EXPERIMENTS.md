# Experiment protocol

1. Split by participant, then scan, then slice; never place slices from one participant in multiple splits.
2. Keep orientation, preprocessing, intensity scaling and crop policy fixed across methods.
3. Evaluate multiple noise levels and noise types. Magnitude MRI may be Rician/noncentral-chi, depending on acquisition and reconstruction; additive Gaussian noise is not universally appropriate.
4. Tune on validation participants only and freeze parameters before test evaluation.
5. Report per-subject metrics and mean ± standard deviation, runtime, hardware, software versions, assumptions, and failures.
6. Use PSNR, SSIM, NMSE and HFEN as complementary measures. Scalar scores do not prove anatomical fidelity; use documented visual or task-based assessment where possible.

The CLI evaluates one 2-D pair. The synthetic Shepp–Logan demo is for smoke testing and teaching, not a public MRI benchmark or clinical validation.
