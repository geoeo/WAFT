import numpy as np

from glob import glob
import os.path as osp

from dataloader.template import FlowDataset


def read_intrinsics(path):
    """Parse an OpenREALM intrinsics file (`fx fy cx cy width height`, '#' comments allowed).
    Returns the 3x3 camera matrix and the (height, width) of the images."""
    with open(path, 'r') as f:
        rows = [line.split() for line in f if line.strip() and not line.startswith('#')]
    fx, fy, cx, cy, width, height = [float(v) for v in rows[0][:6]]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, (int(height), int(width))


def read_trajectory(path):
    """Parse an OpenREALM keyframe trajectory: one line per keyframe with
    `timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz` (row-major 3x4 camera-to-world).
    Returns the ordered list of timestamps and a dict timestamp -> 4x4 pose."""
    timestamps, poses = [], {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            tokens = line.split()
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(tokens[1:13], dtype=np.float64).reshape(3, 4)
            timestamps.append(tokens[0])
            poses[tokens[0]] = T
    return timestamps, poses


class Realm(FlowDataset):
    """Keyframe sequence exported by the OpenREALM pipeline.

    Expected layout:
        <root>/imgs/<timestamp>.png
        <root>/intrinsics.txt   # fx fy cx cy width height
        <root>/kf_traj.txt      # timestamp + row-major 3x4 camera-to-world pose per keyframe

    The sequence ships without optical-flow ground truth, so the dataset behaves like a
    test split: __getitem__ returns (image1, image2, [timestamp1, timestamp2]) for keyframes
    that are `stride` apart in trajectory order. Camera intrinsics and poses are exposed as
    `self.K` and `self.poses` for downstream processing.
    """
    def __init__(self, aug_params=None, root='custom/realm_1_100m_w_tandem', stride=1):
        super(Realm, self).__init__(aug_params)
        self.is_test = True
        self.root = root
        self.K, self.image_shape = read_intrinsics(osp.join(root, 'intrinsics.txt'))
        self.timestamps, self.poses = read_trajectory(osp.join(root, 'kf_traj.txt'))

        images = {osp.splitext(osp.basename(p))[0]: p for p in glob(osp.join(root, 'imgs', '*.png'))}
        frames = [ts for ts in self.timestamps if ts in images]
        for ts1, ts2 in zip(frames[:-stride], frames[stride:]):
            self.image_list += [[images[ts1], images[ts2]]]
            self.extra_info += [[ts1, ts2]]

        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No keyframe pairs found under {root} (images: {len(images)}, trajectory entries: {len(self.timestamps)})")
