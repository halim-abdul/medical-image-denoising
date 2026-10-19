"""CLI for explicit 2-D NumPy images and a synthetic smoke experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from .evaluate import evaluate_methods, save_csv
from .methods import denoise
from .noise import add_rician_noise


def _load(path):
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Input NumPy file must contain a finite 2-D image")
    return x.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mri-denoise")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--method", default="na_wtv")
    run.add_argument("--output", required=True)
    run.add_argument("--sigma", type=float)
    run.add_argument("--data-range", type=float, default=1.0)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--clean", required=True)
    ev.add_argument("--noisy", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--sigma", type=float)
    ev.add_argument("--data-range", type=float, default=1.0)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--sigma", type=float, default=0.05)
    demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.command == "run":
        out = denoise(_load(args.input), args.method, args.sigma, args.data_range)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, out)
    elif args.command == "evaluate":
        rows = evaluate_methods(_load(args.clean), _load(args.noisy), args.sigma, args.data_range)
        save_csv(rows, args.output)
        print(json.dumps(rows, indent=2))
    else:
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        clean = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True).astype(np.float32)
        noisy = add_rician_noise(clean, args.sigma, args.seed)
        np.save(folder / "clean.npy", clean)
        np.save(folder / "noisy.npy", noisy)
        rows = evaluate_methods(clean, noisy, args.sigma)
        save_csv(rows, folder / "metrics.csv")
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
