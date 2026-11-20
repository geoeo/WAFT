#!/usr/bin/env python3
"""How flow and confidence change with input size, refinement iterations, float precision
and batch size, measured against 640x480, T = 5 inference on a recorded flight.

    python scripts/flow_sweep.py                      # all phases
    python scripts/flow_sweep.py --phase sweep         # one phase, others cached

Phases, each cached under --output_dir (default outputs/flow_sweep):

  extract    Undistorted full-resolution frames of --bag: `--anchors` anchor images
             spread over the flight after `--skip` images, each with the images
             `--gaps` frames later. One pair per anchor and gap.
  reference  --reference_resolution (default 640x480), T = 5, fp32 inference per pair.
             Flow, alpha1 and log b0 are reduced to every sweep size up to that size by
             area averaging (flow scaled to that size's pixels) and stored per pair.
             Larger sweep sizes are context: they get the reference-free figures only.
  sweep      Every size x iterations x precision configuration on every pair, compared
             with the reduced reference.
  timing     Latency and peak GPU memory per configuration at batch 1 and 2, and the
             difference of a batch-2 result from the batch-1 result.

This runs the ordinary pipeline (xformers attention, cuDNN grid sampler), not the
export-safe one, so that --reference_resolution 2048x1536 stays possible: full resolution
does not fit in memory with plain attention. Under fp32 the two differ by 2.6e-05 px on
average (scripts/aot_compile_flow.py).

Full resolution was the first reference and is not a usable one: on the 40 Calvarina
pairs the network's mean alpha1 at 2048x1536 is below 0.5 on 28 and about zero on 15
(outputs/flow_sweep/reference_2048x1536_t5/summary.json). The same motion is 4.6 times as many
pixels there as at 448x336.

Agreement with the reference is not accuracy: nothing here has flow ground truth. Each
result therefore also carries a measure
that needs no reference, the epipolar residual: an essential matrix is fitted to a grid
of the flow's correspondences with the camera intrinsics, and the Sampson distance of
those correspondences to it is reported. It measures the flow component across the
epipolar lines only, and it includes whatever the static pinhole model misses (rolling
shutter, residual distortion, moving objects), so it has a floor above zero.

All pixel figures in the summary are in pixels of a 448-wide image ("448-equivalent"),
so sizes can be compared: a figure measured at width W is multiplied by 448 / W.

Latency is that of the development GPU. It says nothing about the Jetson Orin NX.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.parser import json_to_args  # noqa: E402
from dataloader.flow.openrealm import DEFAULT_TOPIC, OpenRealmBag, camera_matrix  # noqa: E402
from flow_padding import padded_size, set_padding  # noqa: E402
from model import fetch_model  # noqa: E402
from utils.precision import describe_precision, set_precision  # noqa: E402

DEFAULT_BAG = "/bags/2026_03_23-Calvarina-rec/seq0/bags/all/rosbag2_2026_03_23-09_01_52"
REFERENCE_WIDTH = 448

# (width, height, padding): 448x336, one size below and several above. "none" runs a multiple of 112 unpadded.
SIZES = [
    (336, 252, "reference"),
    (448, 336, "reference"),
    (448, 336, "none"),
    (544, 408, "reference"),
    (640, 480, "reference"),
    (896, 672, "reference"),
]
ITERATIONS = [5, 3, 2]
# "fp16" is automatic mixed precision (torch.autocast, float16) on top of TF32.
PRECISIONS = ["fp32", "tf32", "fp16"]


def size_key(width: int, height: int, padding: str) -> str:
    return f"{width}x{height}" + ("" if padding == "reference" else f"-pad-{padding}")


# ------------------------------------------------------------------------------ frames


def extract(args, frames_dir: Path) -> list[dict]:
    """Write the undistorted full-resolution frames and return the pair list."""
    index_path = frames_dir / "pairs.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())

    frames_dir.mkdir(parents=True, exist_ok=True)
    dataset = OpenRealmBag(args.bag, topic=args.topic, undistort=True)
    last_gap = max(args.gaps)
    anchors = np.linspace(args.skip, dataset.num_images - 1 - last_gap, args.anchors).round().astype(int)
    wanted = sorted({int(a + g) for a in anchors for g in (0, *args.gaps)})
    print(f"extracting {len(wanted)} of {dataset.num_images} images from {args.bag}")

    stamps = {}
    remaining = set(wanted)
    with dataset._open() as reader:
        messages = reader.messages(connections=dataset._connections(reader))
        for index, (connection, log_time, rawdata) in enumerate(messages):
            if index not in remaining:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode image {index}")
            image = cv2.remap(image, *dataset._maps, interpolation=cv2.INTER_LINEAR)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec or log_time
            cv2.imwrite(str(frames_dir / f"{index:06d}.png"), image)  # BGR on disk
            stamps[index] = int(stamp)
            remaining.discard(index)
            if not remaining:
                break
    if remaining:
        raise SystemExit(f"the bag ended before images {sorted(remaining)}")

    pairs = []
    for anchor in anchors:
        for gap in args.gaps:
            first, second = int(anchor), int(anchor + gap)
            pairs.append({"name": f"{first:06d}_{second:06d}", "first": first, "second": second,
                          "gap_frames": gap, "stamp1": stamps[first], "stamp2": stamps[second],
                          "gap_seconds": (stamps[second] - stamps[first]) * 1e-9})
    index_path.write_text(json.dumps(pairs, indent=2) + "\n")
    return pairs


def load_frame(frames_dir: Path, index: int) -> np.ndarray:
    """Undistorted full-resolution frame as RGB uint8."""
    image = cv2.imread(str(frames_dir / f"{index:06d}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"frame {index} is missing from {frames_dir}; run --phase extract")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray, width: int, height: int, device: str) -> torch.Tensor:
    """[1, 3, H, W] float32 RGB 0..255, contiguous, resized as OpenRealmBag does."""
    if (rgb.shape[1], rgb.shape[0]) != (width, height):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].contiguous().to(device)


# ------------------------------------------------------------------------------- model


def build_model(args):
    model_args = json_to_args(str(args.cfg))
    model = fetch_model(model_args)
    state = torch.load(str(args.ckpt), map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise SystemExit(f"{len(result.missing_keys)} model weights are missing from {args.ckpt}")
    return model.to(args.device).eval(), model_args


@torch.no_grad()
def infer(model, model_args, image1, image2, iters: int, precision: str):
    """(flow, alpha1, log_b0) of the last iteration, float32. The precision flags of the
    process must already be set (see apply_mode)."""
    if precision == "fp16":
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(image1, image2, iters=iters)
    else:
        out = model(image1, image2, iters=iters)
    flow = out["flow"][-1].float()
    info = out["info"][-1].float()
    alpha1 = torch.softmax(info[:, 0:2], dim=1)[:, 1:2]
    log_b0 = torch.clamp(info[:, 2:3], min=0.0, max=float(model_args.var_max))
    return flow, alpha1, log_b0


def apply_mode(precision: str) -> None:
    set_precision("fp32" if precision == "fp32" else "tf32")


# ----------------------------------------------------------------------------- metrics


def epipolar(flow: torch.Tensor, width: int, height: int) -> dict:
    """Sampson distance of a 56x42 grid of the flow's correspondences to an essential
    matrix fitted to them, in 448-equivalent pixels."""
    scale = REFERENCE_WIDTH / width
    K = camera_matrix((width, height))
    xs = np.clip(np.round((np.arange(56) + 0.5) * width / 56 - 0.5), 0, width - 1).astype(int)
    ys = np.clip(np.round((np.arange(42) + 0.5) * height / 42 - 0.5), 0, height - 1).astype(int)
    gx, gy = np.meshgrid(xs, ys)
    field = flow[0].cpu().numpy()
    p1 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    p2 = p1 + np.stack([field[0, gy.ravel(), gx.ravel()], field[1, gy.ravel(), gx.ravel()]], axis=1)
    keep = np.isfinite(p2).all(axis=1) & (p2[:, 0] >= 0) & (p2[:, 0] <= width - 1) \
        & (p2[:, 1] >= 0) & (p2[:, 1] <= height - 1)
    p1, p2 = p1[keep], p2[keep]
    if len(p1) < 50:
        return {"epi_median": float("nan"), "epi_p90": float("nan"), "epi_points": int(len(p1))}
    cv2.setRNGSeed(0)
    E, _ = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=1.0 / scale)
    if E is None or E.shape[0] < 3:
        return {"epi_median": float("nan"), "epi_p90": float("nan"), "epi_points": int(len(p1))}
    K_inv = np.linalg.inv(K)
    fundamental = K_inv.T @ E[:3] @ K_inv
    h1 = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
    h2 = np.concatenate([p2, np.ones((len(p2), 1))], axis=1)
    line2 = h1 @ fundamental.T
    line1 = h2 @ fundamental
    numerator = np.einsum("ij,ij->i", h2, line2) ** 2
    denominator = line2[:, 0] ** 2 + line2[:, 1] ** 2 + line1[:, 0] ** 2 + line1[:, 1] ** 2
    distance = np.sqrt(numerator / np.maximum(denominator, 1e-30)) * scale
    return {"epi_median": float(np.median(distance)), "epi_p90": float(np.percentile(distance, 90)),
            "epi_points": int(len(p1))}


def describe(flow, alpha1, log_b0, width: int, height: int) -> dict:
    """Reference-free figures of one result."""
    scale = REFERENCE_WIDTH / width
    finite = bool(torch.isfinite(flow).all() and torch.isfinite(alpha1).all() and torch.isfinite(log_b0).all())
    row = {
        "finite": finite,
        "flow_mean": flow.norm(dim=1).mean().item() * scale,
        "alpha1_mean": alpha1.mean().item(),
        "alpha1_ge_0.5": (alpha1 >= 0.5).float().mean().item(),
        "alpha1_ge_0.9": (alpha1 >= 0.9).float().mean().item(),
        "log_b0_mean": log_b0.mean().item(),
    }
    row.update(epipolar(flow, width, height) if finite else
               {"epi_median": float("nan"), "epi_p90": float("nan"), "epi_points": 0})
    return row


def against_reference(flow, alpha1, reference: dict, width: int) -> dict:
    """Difference to the reference reduced to this size, flow in 448-equivalent pixels."""
    scale = REFERENCE_WIDTH / width
    reference_flow = reference["flow"].to(flow.device)
    reference_alpha1 = reference["alpha1"].to(flow.device)
    epe = (flow - reference_flow).norm(dim=1) * scale
    confident = alpha1[:, 0] >= 0.5
    flat = epe.flatten()
    return {
        "epe_mean": flat.mean().item(),
        "epe_median": flat.median().item(),
        "epe_p90": torch.quantile(flat, 0.9).item(),
        "epe_gt_1px": (flat > 1.0).float().mean().item(),
        "epe_mean_confident": epe[confident].mean().item() if confident.any() else float("nan"),
        "alpha1_abs_difference": (alpha1 - reference_alpha1).abs().mean().item(),
        "alpha1_shift": (alpha1 - reference_alpha1).mean().item(),
    }


# --------------------------------------------------------------------------- reference


def reduce(tensor: torch.Tensor, width: int, height: int, is_flow: bool) -> torch.Tensor:
    reduced = F.interpolate(tensor, size=(height, width), mode="area")
    if is_flow:
        factor = torch.tensor([width / tensor.shape[-1], height / tensor.shape[-2]],
                              device=tensor.device).view(1, 2, 1, 1)
        reduced = reduced * factor
    return reduced


def reference_phase(args, pairs, frames_dir: Path, reference_dir: Path, model, model_args) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    width, height = args.reference_resolution
    apply_mode("fp32")
    set_padding("reference")
    print(f"reference: {width}x{height}, T = {args.reference_iters}, fp32 ({describe_precision()}), "
          f"network runs at {padded_size(height, width, 'reference')[::-1]}")
    summary = {}
    summary_path = reference_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    for pair in pairs:
        target = reference_dir / f"{pair['name']}.pt"
        if target.is_file() and pair["name"] in summary:
            continue
        image1 = to_tensor(load_frame(frames_dir, pair["first"]), width, height, args.device)
        image2 = to_tensor(load_frame(frames_dir, pair["second"]), width, height, args.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        flow, alpha1, log_b0 = infer(model, model_args, image1, image2, args.reference_iters, "fp32")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        reduced = {}
        for w, h in sorted({(w, h) for w, h, _ in SIZES if w <= width and h <= height}):
            reduced[f"{w}x{h}"] = {"flow": reduce(flow, w, h, True).cpu(),
                                   "alpha1": reduce(alpha1, w, h, False).cpu(),
                                   "log_b0": reduce(log_b0, w, h, False).cpu()}
        torch.save(reduced, str(target))
        row = describe(flow, alpha1, log_b0, width, height)
        row["seconds"] = elapsed
        summary[pair["name"]] = row
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  {pair['name']} ({pair['gap_seconds']:.2f} s): flow {row['flow_mean']:.1f} px, "
              f"alpha1 {row['alpha1_mean']:.3f}, epipolar median {row['epi_median']:.3f} px "
              f"(448-equivalent), {elapsed:.1f} s")
        del flow, alpha1, log_b0, image1, image2
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------- sweep


def sweep_phase(args, pairs, frames_dir: Path, reference_dir: Path, model, model_args, output: Path) -> None:
    rows = []
    for width, height, padding in SIZES:
        set_padding(padding)
        key = size_key(width, height, padding)
        inputs = {}
        references = {}
        for pair in pairs:
            inputs[pair["name"]] = (to_tensor(load_frame(frames_dir, pair["first"]), width, height, args.device),
                                    to_tensor(load_frame(frames_dir, pair["second"]), width, height, args.device))
            references[pair["name"]] = torch.load(str(reference_dir / f"{pair['name']}.pt"),
                                                  map_location="cpu").get(f"{width}x{height}")
        for precision in PRECISIONS:
            apply_mode(precision)
            for iters in ITERATIONS:
                failed = 0
                for pair in pairs:
                    image1, image2 = inputs[pair["name"]]
                    flow, alpha1, log_b0 = infer(model, model_args, image1, image2, iters, precision)
                    row = {"size": key, "width": width, "height": height, "padding": padding,
                           "precision": precision, "iters": iters, "pair": pair["name"],
                           "gap_frames": pair["gap_frames"]}
                    row.update(describe(flow, alpha1, log_b0, width, height))
                    if not row["finite"]:
                        failed += 1
                    elif references[pair["name"]] is not None:
                        row.update(against_reference(flow, alpha1, references[pair["name"]], width))
                    rows.append(row)
                recent = [r for r in rows[-len(pairs):] if r["finite"]]
                mean = lambda name: (float(np.nanmean([r[name] for r in recent]))
                                     if recent and name in recent[0] else float("nan"))
                print(f"  {key:>18} {precision} T={iters}: EPE to reference {mean('epe_mean'):.3f} px, "
                      f"epipolar {mean('epi_median'):.3f} px, alpha1 {mean('alpha1_mean'):.3f}"
                      + (f", {failed} non-finite" if failed else ""), flush=True)
        del inputs
        torch.cuda.empty_cache()
        output.write_text(json.dumps(rows) + "\n")
    set_padding("reference")


# ------------------------------------------------------------------------------ timing


def timing_phase(args, pairs, frames_dir: Path, model, model_args, output: Path) -> None:
    rows = []
    first, second = pairs[0], pairs[1]
    for width, height, padding in SIZES:
        set_padding(padding)
        key = size_key(width, height, padding)
        a1 = to_tensor(load_frame(frames_dir, first["first"]), width, height, args.device)
        a2 = to_tensor(load_frame(frames_dir, first["second"]), width, height, args.device)
        b1 = to_tensor(load_frame(frames_dir, second["first"]), width, height, args.device)
        b2 = to_tensor(load_frame(frames_dir, second["second"]), width, height, args.device)
        batches = {1: (a1, a2), 2: (torch.cat([a1, b1]).contiguous(), torch.cat([a2, b2]).contiguous())}
        for precision in PRECISIONS:
            apply_mode(precision)
            for iters in ITERATIONS:
                single = None
                for batch, (image1, image2) in batches.items():
                    try:
                        for _ in range(3):
                            result = infer(model, model_args, image1, image2, iters, precision)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        times = []
                        for _ in range(args.timing_calls):
                            started = time.perf_counter()
                            result = infer(model, model_args, image1, image2, iters, precision)
                            torch.cuda.synchronize()
                            times.append((time.perf_counter() - started) * 1e3)
                        peak = torch.cuda.max_memory_allocated() / 2**20
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rows.append({"size": key, "precision": precision, "iters": iters, "batch": batch,
                                     "out_of_memory": True})
                        continue
                    row = {"size": key, "padding": padding, "precision": precision, "iters": iters,
                           "batch": batch, "ms_per_call_median": float(np.median(times)),
                           "ms_per_call_min": float(np.min(times)),
                           "ms_per_pair_median": float(np.median(times)) / batch,
                           "peak_memory_mib": peak}
                    if batch == 1:
                        single = result
                    elif single is not None:
                        row["batch_vs_single_epe_mean"] = (result[0][:1] - single[0]).norm(dim=1).mean().item()
                        row["batch_vs_single_epe_max"] = (result[0][:1] - single[0]).norm(dim=1).max().item()
                        row["batch_vs_single_alpha1_max"] = (result[1][:1] - single[1]).abs().max().item()
                    rows.append(row)
                    print(f"  {key:>18} {precision} T={iters} batch {batch}: "
                          f"{row['ms_per_pair_median']:.0f} ms/pair, {peak:.0f} MiB"
                          + (f", batch-1 difference {row['batch_vs_single_epe_mean']:.1e} px mean"
                             if "batch_vs_single_epe_mean" in row else ""), flush=True)
        output.write_text(json.dumps(rows) + "\n")
    set_padding("reference")


# -------------------------------------------------------------------------------- main


def parse_resolution(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["all", "extract", "reference", "sweep", "timing"], default="all")
    parser.add_argument("--bag", type=Path, default=Path(DEFAULT_BAG))
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--skip", type=int, default=2400,
                        help="images dropped at the start (default 2400: the flight is replayed from 240 s)")
    parser.add_argument("--anchors", type=int, default=20, help="anchor images spread over the rest")
    parser.add_argument("--gaps", type=int, nargs="+", default=[6, 12],
                        help="frames between the two images of a pair (default 6 12: 0.6 s and 1.2 s at 10 Hz)")
    parser.add_argument("--cfg", type=Path, default=REPO_ROOT / "config" / "a1" / "tar.json")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "weights" / "tar-c-t.pth")
    parser.add_argument("--reference_resolution", type=parse_resolution, default=(640, 480),
                        help="size of the reference inference (default 640x480)")
    parser.add_argument("--reference_iters", type=int, default=5)
    parser.add_argument("--timing_calls", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "flow_sweep")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames_dir = args.output_dir / "frames"
    tag = f"{args.reference_resolution[0]}x{args.reference_resolution[1]}_t{args.reference_iters}"
    reference_dir = args.output_dir / f"reference_{tag}"
    pairs = extract(args, frames_dir)
    print(f"{len(pairs)} pairs")
    if args.phase == "extract":
        return 0

    model, model_args = build_model(args)
    meta = {"bag": str(args.bag), "skip": args.skip, "anchors": args.anchors, "gaps": args.gaps,
            "cfg": str(args.cfg), "ckpt": str(args.ckpt),
            "reference": {"resolution": list(args.reference_resolution), "iters": args.reference_iters,
                          "precision": "fp32"},
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "created": datetime.now().isoformat(timespec="seconds")}
    (args.output_dir / f"meta_ref-{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")

    if args.phase in ("all", "reference"):
        reference_phase(args, pairs, frames_dir, reference_dir, model, model_args)
    if args.phase in ("all", "sweep"):
        sweep_phase(args, pairs, frames_dir, reference_dir, model, model_args, args.output_dir / f"sweep_ref-{tag}.json")
    if args.phase in ("all", "timing"):
        timing_phase(args, pairs, frames_dir, model, model_args, args.output_dir / "timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
