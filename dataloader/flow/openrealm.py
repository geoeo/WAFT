"""Offline optical flow on the camera stream of an OpenREALM rosbag2 recording.

The bag is read straight from its files (metadata.yaml + *.mcap), so no ROS installation or
running ROS graph is needed. Run from the repository root:

    python dataloader/flow/openrealm.py --resolution 640x480 \
        --bag /bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight

--cfg, --ckpt and --bag default to config/a1/tar.json, weights/tar-c-t.pth and the bag above.

The images are undistorted with the camera intrinsics hardcoded below (--no_undistort turns
that off). Per image pair three images are written to outputs/<bag name>: the colour-coded
flow, the flow of a grid of pixels drawn as arrows, and the uncertainty heatmap, together with
a stats.csv. Pass --save_flow to also keep the raw flow.
Everything already in that output folder is deleted before the run (--no_clean keeps it).
"""
import argparse
import csv
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

DEFAULT_TOPIC = '/AIT_Fighter6/down/image/compressed'
DEFAULT_BAG = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight'
DEFAULT_ARROW_STEP = 1 / 20

# hardcoded for F6

WIDTH = 2048
HEIGHT = 1536

CX = 1032.7638177042552
CY = 753.7287922955178
# Focal length
FX = 1331.1713955614885
FY = 1329.7824642924002
# Distortion
DIST_MODEL = 'equi'
C1 = 0.014127946548806695
C2 = -0.007081005355739692
C3 = 0.02477688100909874
C4 = -0.012940251836647083


def camera_matrix(resolution=None):
    """3x3 pinhole matrix of the hardcoded camera, for images resized to `resolution`
    (width, height) if given. It describes the undistorted images, see `undistort_maps`."""
    sx, sy = (1.0, 1.0) if resolution is None else (resolution[0] / WIDTH, resolution[1] / HEIGHT)
    # resizing moves pixel centres as u' = (u + 0.5) * s - 0.5, hence the half-pixel terms
    return np.array([[FX * sx, 0.0, (CX + 0.5) * sx - 0.5],
                     [0.0, FY * sy, (CY + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_maps():
    """cv2.remap lookup tables that turn a raw camera image into a pinhole image with the same
    camera matrix. 'equi' is the equidistant (fisheye) model, cv2.fisheye in OpenCV."""
    if DIST_MODEL != 'equi':
        raise NotImplementedError(f"Distortion model '{DIST_MODEL}' is not supported, only 'equi'")
    K = camera_matrix()
    D = np.array([C1, C2, C3, C4], dtype=np.float64)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (WIDTH, HEIGHT), cv2.CV_32F)


class OpenRealmBag(data.IterableDataset):
    """Consecutive image pairs from a sensor_msgs/CompressedImage topic of a rosbag2 recording.

    `bag` is the bag directory (the one holding metadata.yaml); all of its storage files are
    read in recording order. The first `skip` images are dropped, then only every `stride`-th
    image is kept and each kept image is paired with the next kept one. With `undistort` the
    lens distortion of the hardcoded camera is removed first. With `resolution` (width, height)
    every image is then resized to that size, otherwise it keeps the camera resolution. `self.K`
    is the camera matrix of the images as they are yielded. The bag is a stream, so this is an iterable dataset: it yields
    (image1, image2, [timestamp1, timestamp2]) like the test split of the other flow datasets,
    with images as float RGB tensors (3, H, W) in [0, 255] and timestamps as the nanosecond
    header stamps of the two images.
    """
    def __init__(self, bag, topic=DEFAULT_TOPIC, skip=0, stride=1, max_pairs=None, resolution=None, undistort=True):
        super(OpenRealmBag, self).__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.bag = Path(bag)
        self.topic = topic
        self.skip = skip
        self.stride = stride
        self.max_pairs = max_pairs
        self.resolution = resolution
        self.undistort = undistort
        self.K = camera_matrix(resolution)
        self._maps = undistort_maps() if undistort else None
        with self._open() as reader:
            self.num_images = sum(c.msgcount for c in self._connections(reader))

    def _open(self):
        from rosbags.highlevel import AnyReader
        if not (self.bag / 'metadata.yaml').is_file():
            raise FileNotFoundError(f"{self.bag} is not a rosbag2 directory (no metadata.yaml)")
        return AnyReader([self.bag])

    def _connections(self, reader):
        connections = [c for c in reader.connections if c.topic == self.topic]
        if not connections:
            image_topics = sorted({c.topic for c in reader.connections if c.msgtype.endswith('Image')})
            raise KeyError(f"Topic {self.topic} not found in {self.bag}. Image topics in this bag: {image_topics}")
        return connections

    def __len__(self):
        pairs = max((self.num_images - self.skip - 1) // self.stride, 0)
        return pairs if self.max_pairs is None else min(pairs, self.max_pairs)

    def frames(self):
        """Yield (timestamp_ns, RGB uint8 image) for every `stride`-th image after the first `skip`."""
        with self._open() as reader:
            for index, (connection, log_time, rawdata) in enumerate(reader.messages(connections=self._connections(reader))):
                if index < self.skip or (index - self.skip) % self.stride != 0:
                    continue
                msg = reader.deserialize(rawdata, connection.msgtype)
                image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode image {index} of {self.topic} (format '{msg.format}')")
                if self.undistort:
                    if (image.shape[1], image.shape[0]) != (WIDTH, HEIGHT):
                        raise ValueError(f"Image {index} is {image.shape[1]}x{image.shape[0]}, but the hardcoded "
                                         f"intrinsics are for {WIDTH}x{HEIGHT}; pass --no_undistort to use it as is")
                    image = cv2.remap(image, *self._maps, interpolation=cv2.INTER_LINEAR)
                if self.resolution is not None and self.resolution != (image.shape[1], image.shape[0]):
                    image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_AREA)
                stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                # fall back to the time the message was recorded if the driver left the stamp empty
                yield stamp or log_time, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        previous = None
        pairs = 0
        for stamp, image in self.frames():
            if self.max_pairs is not None and pairs >= self.max_pairs:
                return
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            if previous is not None:
                yield previous[1], image, [previous[0], stamp]
                pairs += 1
            previous = (stamp, image)


def clean_output(output_path):
    """Delete everything inside `output_path`, so results of earlier runs do not mix with the
    new ones. Refuses a folder that holds this repository or the working directory."""
    path = Path(output_path).resolve()
    if not path.is_dir():
        return
    for protected in (Path(__file__).resolve(), Path.cwd().resolve()):
        if path == protected or path in protected.parents:
            raise ValueError(f"Refusing to clean {path}: it contains {protected}. Choose another --output_path or pass --no_clean")
    entries = list(path.iterdir())
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f"Cleaned {output_path}: deleted {len(entries)} files/folders from earlier runs")


def arrow_grid(height, width, step):
    """The pixels whose flow is drawn as arrows: a regular grid of (x, y), row by row. `step`
    is the grid spacing: a fraction of the image width if below 1, otherwise pixels. Also
    returns the line thickness that goes with an image of this width."""
    step = max(int(step * width if step < 1 else step), 1)
    points = [(x, y) for y in range(step // 2, height, step) for x in range(step // 2, width, step)]
    return points, max(round(width / 640), 1)


def vis_arrows(name, image, flow, step):
    """Draw the flow of a regular grid of pixels as arrows on top of `image` (RGB, the first
    image of the pair). Each arrow starts at a pixel of the first image and ends where that
    piece of the scene appears in the second image, at true scale. This is the apparent motion
    of the image content; the camera moves the opposite way over a static scene. `step` is the
    grid spacing: a fraction of the image width if below 1, otherwise pixels."""
    canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    points, thickness = arrow_grid(*flow.shape[:2], step)
    arrows = [((x, y), (round(x + float(flow[y, x, 0])), round(y + float(flow[y, x, 1]))))
              for x, y in points]
    # dark outline first, then the bright arrow, so it stays visible on any background
    for color, extra in (((0, 0, 0), 2 * thickness), ((0, 255, 255), 0)):
        for start, end in arrows:
            cv2.circle(canvas, start, thickness + extra // 2, color, -1, cv2.LINE_AA)
            if start != end:
                tip = min(4.0 * thickness / np.hypot(end[0] - start[0], end[1] - start[1]), 1.0)
                cv2.arrowedLine(canvas, start, end, color, thickness + extra, cv2.LINE_AA, tipLength=tip)
    cv2.imwrite(name, canvas)


@torch.no_grad()
def process_bag(args, model, dataset, output_path, device):
    """ Run inference on every image pair of `dataset` and write the results to `output_path`.
        The bag has no optical-flow ground truth, so no error metrics are computed. stats.csv
        gets one row per pair as soon as it is processed, so an interrupted run keeps its rows. """
    from custom import get_heatmap, vis_heatmap
    from utils.flow_viz import flow_to_image
    from tqdm import tqdm

    os.makedirs(output_path, exist_ok=True)
    synchronize = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)

    rows = []
    with open(osp.join(output_path, 'stats.csv'), 'w', newline='') as f:
        writer = None
        for image1, image2, (ts1, ts2) in tqdm(dataset, total=len(dataset)):
            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            if not rows:
                # warm-up so the first timed pair does not include CUDA / cuDNN initialisation
                model.calc_flow(image1, image2)
            synchronize()
            start = time.perf_counter()
            output = model.calc_flow(image1, image2)
            synchronize()
            inference_time_ms = (time.perf_counter() - start) * 1000.0

            flow = output['flow'][-1][0].permute(1, 2, 0).cpu().numpy()
            name = f"{ts1}_{ts2}"
            if args.save_flow:
                np.save(osp.join(output_path, f"flow_{name}.npy"), flow)
            if not args.no_vis:
                cv2.imwrite(osp.join(output_path, f"flow_{name}.jpg"), flow_to_image(flow, convert_to_bgr=True))
                vis_arrows(osp.join(output_path, f"arrows_{name}.jpg"), image1[0].permute(1, 2, 0).cpu().numpy(),
                           flow, step=args.arrow_step)

            magnitude = np.linalg.norm(flow, axis=-1)
            row = {'frame1': ts1, 'frame2': ts2, 'dt_ms': (ts2 - ts1) / 1e6, 'inference_time_ms': inference_time_ms,
                   'flow_mean_px': magnitude.mean(), 'flow_max_px': magnitude.max()}
            if 'info' in output:
                heatmap = get_heatmap(output['info'][-1], args)
                row['uncertainty_mean'] = heatmap.mean().item()
                if not args.no_vis:
                    vis_heatmap(osp.join(output_path, f"heatmap_{name}.jpg"),
                                image1[0].permute(1, 2, 0).cpu().numpy(),
                                heatmap[0].permute(1, 2, 0).cpu().numpy())
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No image pairs read from {dataset.topic} in {dataset.bag}")
    summary = (f"Processed {len(rows)} image pairs from {dataset.bag}: "
               f"mean inference time {np.mean([r['inference_time_ms'] for r in rows]):.1f} ms, "
               f"mean flow magnitude {np.mean([r['flow_mean_px'] for r in rows]):.2f} px")
    if 'uncertainty_mean' in rows[0]:
        summary += f", mean uncertainty {np.mean([r['uncertainty_mean'] for r in rows]):.3f}"
    print(summary + f". Outputs written to {output_path}")


def parse_resolution(text):
    """Parse 'WIDTHxHEIGHT' (e.g. 640x480) into (width, height)."""
    try:
        width, height = (int(v) for v in text.lower().split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 640x480, got '{text}'")
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError(f"width and height must be positive, got '{text}'")
    return width, height


def main():
    from config.parser import parse_args
    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.utils import load_ckpt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', help='experiment configure file name', default='config/a1/tar.json', type=str)
    parser.add_argument('--ckpt', help='checkpoint path', default='weights/tar-c-t.pth', type=str)
    parser.add_argument('--bag', help='rosbag2 directory (the one holding metadata.yaml)', default=DEFAULT_BAG, type=str)
    parser.add_argument('--topic', help='sensor_msgs/CompressedImage topic to read', default=DEFAULT_TOPIC, type=str)
    parser.add_argument('--skip', help='drop this many images at the start of the bag (10 images are about 1 s)', default=0, type=int)
    parser.add_argument('--stride', help='keep every N-th image; flow is computed between consecutive kept images', default=1, type=int)
    parser.add_argument('--max_pairs', help='stop after this many image pairs (default: whole bag)', default=None, type=int)
    parser.add_argument('--resolution', help='resize every image to WIDTHxHEIGHT (e.g. 640x480) before the flow is computed; '
                        'flow and outputs are then in pixels of that size (default: camera resolution)', default=None, type=parse_resolution)
    parser.add_argument('--no_undistort', help='use the images as recorded instead of removing the lens distortion with the hardcoded intrinsics', action='store_true')
    parser.add_argument('--scale', help='the network runs on the (resized) images scaled by 2**scale; the flow is returned at their resolution', default=0.0, type=float)
    parser.add_argument('--output_path', help='where to write results (default outputs/<bag name>)', default=None, type=str)
    parser.add_argument('--no_clean', help='keep what is already in the output folder instead of deleting it before the run', action='store_true')
    parser.add_argument('--save_flow', help='also write the raw flow of every pair as .npy (H x W x 2 float32, large)', action='store_true')
    parser.add_argument('--no_vis', help='skip the flow / arrow / heatmap images and only write stats.csv', action='store_true')
    parser.add_argument('--arrow_step', help='spacing of the arrows in the arrow image: a fraction of the image width if below 1, '
                        'otherwise pixels (default: %(default)s, i.e. 1/20 of the image width)', default=DEFAULT_ARROW_STEP, type=float)
    parser.add_argument('--device', help='inference device', default='cuda', type=str)
    args = parse_args(parser)

    dataset = OpenRealmBag(args.bag, topic=args.topic, skip=args.skip, stride=args.stride, max_pairs=getattr(args, 'max_pairs', None),
                           resolution=getattr(args, 'resolution', None), undistort=not args.no_undistort)
    size = 'camera resolution' if dataset.resolution is None else '{}x{}'.format(*dataset.resolution)
    size += ', undistorted' if dataset.undistort else ', not undistorted'
    print(f"load data success: {dataset.num_images} images on {args.topic}, {len(dataset)} pairs to process at {size}")
    print("camera matrix of these images: fx {:.3f} fy {:.3f} cx {:.3f} cy {:.3f}".format(
        dataset.K[0, 0], dataset.K[1, 1], dataset.K[0, 2], dataset.K[1, 2]))

    args.gpus = [0]
    device = torch.device(args.device)
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.to(device)
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)

    # parse_args drops options that were not given, so the optional ones are read with getattr
    output_path = getattr(args, 'output_path', None) or osp.join('outputs', osp.basename(osp.normpath(args.bag)))
    if not args.no_clean:
        clean_output(output_path)
    process_bag(args, wrapped_model, dataset, output_path, device)


if __name__ == '__main__':
    if not __package__:
        # started as `python dataloader/flow/openrealm.py`: Python then puts this file's folder
        # on the import path instead of the repository root, where config/, model/, ... live
        sys.path[0] = str(Path(__file__).resolve().parents[2])
    main()
