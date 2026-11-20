#!/usr/bin/env python3
"""Check of the flow convention of a compiled package: is `p + flow[p]` the
position in image 2 of pixel `p` of image 1, in pixels of the input size?

    python scripts/flow_convention_check.py \
        --package outputs/aoti/fp16_448x336_t3/waft_flow.pt2

Image 2 is sampled at `p + s * flow[p]` and compared with image 1, for the stated
convention (s = 1) and for the mistakes a consumer could make: no flow (s = 0), wrong
sign (s = -1), wrong scale (s = 0.5, 2) and swapped axes. The mean absolute intensity
difference must be smallest for s = 1. Only pixels whose target lies inside the image
for every variant are counted. The pairs are the frames flow_sweep.py extracted.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {"as stated (s = 1)": (1.0, False), "no flow (s = 0)": (0.0, False),
            "wrong sign (s = -1)": (-1.0, False), "half scale (s = 0.5)": (0.5, False),
            "double scale (s = 2)": (2.0, False), "axes swapped": (1.0, True)}


def load(path: Path, width: int, height: int, device: str) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].contiguous().to(device)


def sample(image2: torch.Tensor, flow: torch.Tensor, scale: float, swap: bool):
    """image2 at p + scale * flow[p], and whether that target is inside the image."""
    _, _, height, width = image2.shape
    ys, xs = torch.meshgrid(torch.arange(height, device=flow.device, dtype=torch.float32),
                            torch.arange(width, device=flow.device, dtype=torch.float32), indexing="ij")
    dx, dy = (flow[0, 1], flow[0, 0]) if swap else (flow[0, 0], flow[0, 1])
    x, y = xs + scale * dx, ys + scale * dy
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack([2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1], dim=-1)[None]
    return F.grid_sample(image2, grid, mode="bilinear", align_corners=True), inside


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "outputs" / "flow_sweep" / "frames")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = json.loads((args.package.parent / "build_info.json").read_text())
    _, _, height, width = manifest["input_shape"]
    pairs = json.loads((args.frames / "pairs.json").read_text())
    runner = torch._inductor.aoti_load_package(str(args.package))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    totals = {name: [] for name in VARIANTS}
    best_count = 0
    for pair in pairs:
        image1 = load(args.frames / f"{pair['first']:06d}.png", width, height, args.device)
        image2 = load(args.frames / f"{pair['second']:06d}.png", width, height, args.device)
        with torch.no_grad():
            flow = runner(image1, image2)[0]
        sampled = {name: sample(image2, flow, *variant) for name, variant in VARIANTS.items()}
        common = torch.stack([inside for _, inside in sampled.values()]).all(dim=0)
        errors = {name: (warped - image1).abs().mean(dim=1)[0][common].mean().item()
                  for name, (warped, _) in sampled.items()}
        for name, value in errors.items():
            totals[name].append(value)
        best_count += min(errors, key=errors.get) == "as stated (s = 1)"

    print(f"{args.package}: {len(pairs)} pairs at {width}x{height}")
    print("mean absolute intensity difference to image 1 (0..255), mean over the pairs:")
    for name, values in totals.items():
        print(f"  {name:<22} {np.mean(values):6.2f}")
    print(f"the stated convention has the smallest difference on {best_count} of {len(pairs)} pairs")
    return 0 if best_count == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
