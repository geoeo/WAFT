#!/usr/bin/env python3
"""Follow a grid of pixels through N consecutive images of an OpenREALM rosbag2 recording
and draw the path each of them takes.

    python scripts/flow_paths.py --skip 3000
    python scripts/flow_paths.py --skip 3000 --num_images 8 --arrow_step 0.1 \
        --bag /bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight

The flow is computed between each image and the next, as dataloader/flow/openrealm.py
does, which gives N - 1 flows for N images. The pixels that are followed are the ones
that script draws its arrows from: a regular grid over image 0 (--arrow_step).

A pixel starts at its grid position in image 0. Flow 0, read at that position, moves it
to where the same piece of the scene is in image 1; flow 1, read at the new position,
moves it on to image 2; and so on. The flow is sampled bilinearly, because after the
first step a pixel is no longer on integer coordinates.

One image is written per flow. The image of flow k shows, for every followed pixel, the
whole path from image 0 to image k + 1 as a chain of k + 1 arrows: one arrow in the
image of flow 0, two in that of flow 1, ... Each arrow is one step, coloured by the flow
it comes from (legend in the top left corner), so a step keeps its colour from image to
image. The dot marks where the pixel started in image 0.

A pixel that leaves the image is not followed further: its last arrow points out of the
image and its path stays as it is in the later images.

Outputs go to outputs/<bag name>_flow_paths_skip<skip> (--output_path), so runs that
start at different places in the bag do not overwrite each other. The folder is emptied
first unless --no_clean is given.

--precision selects the arithmetic mode the network runs in (default: torch's own flags;
fp32, tf32 or fp16 for float16 autocast, see utils/precision.py).
"""

import argparse
import os.path as osp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataloader.flow.openrealm import (  # noqa: E402
    DEFAULT_ARROW_STEP, DEFAULT_BAG, DEFAULT_TOPIC, OpenRealmBag, arrow_grid, clean_output,
    parse_resolution,
)

BACKGROUNDS = ("start", "end", "first")


def sample_flow(flow, positions):
    """Bilinearly interpolated flow (H x W x 2) at float (x, y) `positions`, P x 2."""
    map_x = positions[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = positions[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(flow, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)


def inside(positions, height, width):
    """Which (x, y) positions lie on the image."""
    x, y = positions[:, 0], positions[:, 1]
    return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)


def step_colors(steps):
    """One BGR colour per step, in step order, from the turbo colour map. Its dark ends are
    left out: every arrow has a black outline, and they would vanish into it."""
    values = np.linspace(64, 230, steps).round().astype(np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return [tuple(int(c) for c in color) for color in colors]


def draw_paths(name, image, paths, lengths, colors, thickness):
    """Draw every pixel's path on top of `image` (RGB) and write it to `name`.

    `paths` is (steps + 1) x P x 2: row j holds the (x, y) of the P followed pixels in
    image j. `lengths` says how many steps of each pixel's path exist; one that left the
    image has fewer than the others. Step j is drawn in colors[j], as an arrow from row j
    to row j + 1, at true scale."""
    canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    points = np.rint(paths).astype(np.int64)
    # dark outline first, then the bright arrows, so they stay visible on any background
    for outline in (True, False):
        extra = 2 * thickness if outline else 0
        for index in range(points.shape[1]):
            start = tuple(int(v) for v in points[0, index])
            cv2.circle(canvas, start, thickness + extra // 2,
                       (0, 0, 0) if outline else colors[0], -1, cv2.LINE_AA)
            for step in range(int(lengths[index])):
                a = tuple(int(v) for v in points[step, index])
                b = tuple(int(v) for v in points[step + 1, index])
                if a == b:
                    continue
                tip = min(4.0 * thickness / np.hypot(b[0] - a[0], b[1] - a[1]), 1.0)
                cv2.arrowedLine(canvas, a, b, (0, 0, 0) if outline else colors[step],
                                thickness + extra, cv2.LINE_AA, tipLength=tip)
    draw_legend(canvas, colors[:paths.shape[0] - 1], thickness)
    cv2.imwrite(name, canvas)


def draw_legend(canvas, colors, thickness):
    """Which colour is which step: one line per flow drawn so far, on a dark box top left."""
    factor = max(canvas.shape[1] / 640, 1.0)
    scale, line, margin = 0.45 * factor, int(round(18 * factor)), int(round(6 * factor))
    texts = [f"flow {step}: image {step} -> {step + 1}" for step in range(len(colors))]
    text_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for text in texts)
    box = canvas[:line * len(texts) + margin, :text_width + 2 * margin]
    box[:] = (box * 0.25).astype(np.uint8)
    for step, (text, color) in enumerate(zip(texts, colors)):
        cv2.putText(canvas, text, (margin, line * (step + 1) - margin // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)


@torch.no_grad()
def follow(args, model, dataset, output_path, device):
    """Run the model over the image pairs of `dataset`, advance the followed pixels with
    every flow and write one path image per flow. Returns the paths, (flows + 1) x P x 2,
    and which of the pixels are still in the image at the end."""
    images, paths, lengths, alive, colors, thickness = [], None, None, None, None, None
    for k, (image1, image2, (ts1, ts2)) in enumerate(dataset):
        output = model.calc_flow(image1[None].to(device), image2[None].to(device))
        flow = output['flow'][-1][0].permute(1, 2, 0).cpu().numpy()
        height, width = flow.shape[:2]
        if paths is None:
            grid, thickness = arrow_grid(height, width, args.arrow_step)
            paths = [np.array(grid, dtype=np.float64)]
            lengths = np.zeros(len(grid), dtype=np.int64)
            alive = np.ones(len(grid), dtype=bool)
            colors = step_colors(len(dataset))
            images.append(image1.permute(1, 2, 0).numpy())
        images.append(image2.permute(1, 2, 0).numpy())

        # a pixel that has left the image stays where it went out; the others move on
        position = paths[-1].copy()
        if alive.any():
            position[alive] += sample_flow(flow, position[alive])
            lengths[alive] += 1
            alive &= inside(position, height, width)
        paths.append(position)

        background = {'start': images[k], 'end': images[k + 1], 'first': images[0]}[args.background]
        name = osp.join(output_path, f"paths_{k:02d}_{ts1}_{ts2}.jpg")
        draw_paths(name, background, np.stack(paths), lengths, colors, thickness)
        moved = np.linalg.norm(paths[-1] - paths[-2], axis=1)[lengths == k + 1]
        print(f"flow {k} (image {k} -> {k + 1}): {k + 1} arrow(s) per followed pixel, {int(alive.sum())} of "
              f"{len(alive)} pixels still in the image, mean step "
              f"{moved.mean() if len(moved) else 0.0:.1f} px -> {name}")
    if paths is None:
        raise RuntimeError(f"No image pairs read from {dataset.topic} in {dataset.bag}")
    return np.stack(paths), alive


def main():
    from config.parser import json_to_args
    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.precision import MODES, describe_precision, is_half, set_precision
    from utils.utils import load_ckpt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', help='experiment configure file name', default=str(REPO_ROOT / 'config/a1/tar.json'), type=str)
    parser.add_argument('--ckpt', help='checkpoint path', default=str(REPO_ROOT / 'weights/tar-c-t.pth'), type=str)
    parser.add_argument('--bag', help='rosbag2 directory (the one holding metadata.yaml)', default=DEFAULT_BAG, type=str)
    parser.add_argument('--topic', help='sensor_msgs/CompressedImage topic to read', default=DEFAULT_TOPIC, type=str)
    parser.add_argument('--num_images', help='number of consecutive images N to follow the pixels through; '
                        'gives N - 1 flows and as many output images (default: %(default)s)', default=5, type=int)
    parser.add_argument('--skip', help='drop this many images at the start of the bag (10 images are about 1 s)', default=0, type=int)
    parser.add_argument('--stride', help='keep every N-th image; the flow is computed between consecutive kept images', default=1, type=int)
    parser.add_argument('--resolution', help='resize every image to WIDTHxHEIGHT before the flow is computed; paths and '
                        'outputs are then in pixels of that size (default: 640x480)', default=(640, 480), type=parse_resolution)
    parser.add_argument('--no_undistort', help='use the images as recorded instead of removing the lens distortion with the hardcoded intrinsics', action='store_true')
    parser.add_argument('--arrow_step', help='spacing of the followed pixels in image 0: a fraction of the image width if below 1, '
                        'otherwise pixels (default: %(default)s, i.e. 1/20 of the image width)', default=DEFAULT_ARROW_STEP, type=float)
    parser.add_argument('--background', help='image the paths of flow k are drawn on. start (default): image k, the first of the '
                        'pair, as the arrow image of dataloader/flow/openrealm.py; the newest arrow then starts on the piece of '
                        'the scene it follows. end: image k + 1, the newest arrow then ends on it. first: always image 0.',
                        choices=BACKGROUNDS, default='start')
    parser.add_argument('--output_path', help='where to write results (default outputs/<bag name>_flow_paths_skip<skip>)', default=None, type=str)
    parser.add_argument('--no_clean', help='keep what is already in the output folder instead of deleting it before the run', action='store_true')
    parser.add_argument('--device', help='inference device', default='cuda', type=str)
    parser.add_argument('--precision', help="arithmetic mode (utils/precision.py). default: torch's own flags, as before. fp32: full fp32. tf32: TF32 everywhere. fp16: float16 autocast, about twice as fast as fp32 and about 0.01 px from it on average; needs a CUDA device", choices=MODES, default='default', type=str)
    args = parser.parse_args()
    if args.num_images < 2:
        parser.error(f"--num_images must be at least 2, got {args.num_images}")
    if is_half(args.precision) and not args.device.startswith('cuda'):
        parser.error('--precision fp16 needs --device cuda')

    dataset = OpenRealmBag(args.bag, topic=args.topic, skip=args.skip, stride=args.stride, max_pairs=args.num_images - 1,
                           resolution=args.resolution, undistort=not args.no_undistort)
    if len(dataset) < args.num_images - 1:
        print(f"WARNING: only {len(dataset) + 1} images are left in the bag after --skip/--stride, not {args.num_images}")
    print(f"following pixels through {len(dataset) + 1} images ({len(dataset)} flows) of {args.topic} at "
          "{}x{}{}".format(*args.resolution, '' if args.no_undistort else ', undistorted'))

    model_args = json_to_args(args.cfg)
    device = torch.device(args.device)
    set_precision(args.precision)
    print(f"precision: {args.precision} ({describe_precision()})")
    model = fetch_model(model_args)
    load_ckpt(model, args.ckpt)
    model = model.to(device)
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=0, train_size=model_args.image_size, pad_to_train_size=False, tiling=False,
                                     half=is_half(args.precision))

    output_path = args.output_path or osp.join('outputs', osp.basename(osp.normpath(args.bag)) + f'_flow_paths_skip{args.skip}')
    if not args.no_clean:
        clean_output(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    paths, alive = follow(args, wrapped_model, dataset, output_path, device)
    travelled = np.linalg.norm(paths[-1] - paths[0], axis=1)[alive]
    print(f"Followed {paths.shape[1]} pixels through {paths.shape[0]} images: {int(alive.sum())} are still in the image, "
          f"their mean displacement from image 0 to the last image is {travelled.mean() if alive.any() else 0.0:.1f} px. "
          f"Outputs written to {output_path}")


if __name__ == '__main__':
    main()
