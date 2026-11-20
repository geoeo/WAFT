"""Shared pieces for the flow-model export pipeline.

IMPORTANT: importing this module sets `XFORMERS_DISABLED` and `AOTI_SAFE_GRID_SAMPLE`
before anything under `model`, `utils` or `thirdparty` is imported. Both are read at
*import* time:

  * xformers' `memory_efficient_attention` is an opaque custom op with no ATen lowering,
    so `torch.export` cannot capture it. The fallback is the plain matmul/softmax
    attention in thirdparty/DepthAnythingV2/.../dinov2_layers/attention.py, which
    exports fine. Without this the export fails outright.
  * `F.grid_sample` is recorded as `aten.cudnn_grid_sampler`, which has no c-shim, so
    AOTInductor can only reach it through the proxy executor. The switch makes the warp
    use `aten.grid_sampler_2d`, which inductor compiles; see utils/export_compat.py.

So `import flow_export_common` must come before any `model` import.

A third substitution is applied to the model instance, in `FlowExport`:
`torchvision.transforms.Normalize` raises on a zero std through
`if (std == 0).any():`, a data-dependent branch that `torch.export` rejects in both
strict and non-strict mode. `normalize_image()` below is the same arithmetic without the
check. It is bit-exact with the original, so unlike the other two it does not move the
result.

The first two do move it, by 2.6e-05 px on average and 4.4e-02 px at worst: the eager
model these scripts build is therefore the export-safe one, and "package vs eager"
measures the compile and nothing else. scripts/aot_compile_flow.py quotes how far that
eager model is from what evaluate.py computes.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["AOTI_SAFE_GRID_SAMPLE"] = "1"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config.parser import json_to_args  # noqa: E402
from dataloader.flow.openrealm import DEFAULT_TOPIC, parse_resolution  # noqa: E402
from model import fetch_model  # noqa: E402
from utils.precision import MODES, describe_precision, is_half, set_precision  # noqa: E402

DEFAULT_CFG = REPO_ROOT / "config" / "a1" / "tar.json"
DEFAULT_CKPT = REPO_ROOT / "weights" / "tar-c-t.pth"
DEFAULT_DATA_ROOT = REPO_ROOT / "custom" / "realm_1_100m_w_tandem"

# Recorded in build_info.json; see FlowExport. Packages built before this was pinned
# down expect the loaders' channels-last strides instead and carry no such record.
INPUT_LAYOUT = "contiguous NCHW"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """Export-safe stand-in for the models' `normalize_image` ((B,C,H,W), RGB, 0-255).

    Same operations, in the same order, as torchvision's Normalize, minus its
    `if (std == 0).any()` guard.
    """
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img / 255.0).sub(mean).div(std).contiguous()


class FlowExport(torch.nn.Module):
    """The deployment signature: two images in positionally, flow and confidence out.

    A flat positional tensor signature is what the C++ loader wants, and it keeps the
    exported graph's input spec trivial. Everything non-tensor (the number of refinement
    iterations) is baked in as a constant, and only the last iteration's prediction is
    returned.

        flow, alpha1, log_b0 = model(image1, image2)

        image1  [B, 3, H, W]     float32, RGB, 0..255 (not normalised), contiguous
        image2  [B, 3, H, W]     same
        -> flow   [B, 2, H, W]   pixels, (x, y) displacement from image1 to image2
        -> alpha1 [B, 1, H, W]   softmax(info[:, 0:2])[:, 1]: weight of the tight
                                 mixture component, whose scale is 1 px
        -> log_b0 [B, 1, H, W]   clamp(info[:, 2], 0, var_max): log-scale in pixels of
                                 the wide component

    `info` is the raw uncertainty head. Its softmax and clamp are applied here, as
    custom.get_heatmap() applies them, so that a consumer cannot leave them out; the
    heatmap is (1 - alpha1) * log_b0. The fourth channel is dropped: clamped to
    [var_min, 0] it is always 0 for var_min = 0, which build_model() requires.

    "Contiguous" is part of the signature of a compiled package, not a nicety. The
    generated code reads its inputs with the strides of the tracing example and does not
    check them, so a tensor of the right shape in another memory layout is silently read
    as a scrambled image. That is easy to hit: the loaders of this repository build
    their images as `torch.from_numpy(hwc).permute(2, 0, 1).float()`, which has the
    shape [3, H, W] but channels-last strides. A package traced from such a tensor
    computes flow that is ~11 px off (on 13 px of motion) when it is then given an
    ordinary [1, 3, H, W] tensor, as a C++ caller would build one. iter_pairs() therefore
    hands out plain contiguous NCHW tensors, for tracing and for running alike, and
    build_info.json records it (INPUT_LAYOUT).

    This is the network alone. The Python pipeline calls it through InferenceWrapper,
    which at scale 0 without tiling adds nothing but `x * w / w` with a Gaussian `w`;
    scripts/aot_compile_flow.py quotes what that costs in agreement.

    With `half` the network runs under float16 autocast (utils/precision.py); inputs
    and outputs stay float32, and an exported package computes in float16.
    """

    def __init__(self, core: torch.nn.Module, iters: int, var_max: float, half: bool = False):
        super().__init__()
        if not hasattr(core, "normalize_image"):
            raise TypeError(f"{type(core).__name__} has no normalize_image to substitute")
        core.normalize_image = normalize_image
        self.core = core
        self.iters = iters
        self.var_max = float(var_max)
        self.half = half

    def forward(self, image1, image2):
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.core(image1, image2, iters=self.iters)
        else:  # no autocast region in a float32 graph
            out = self.core(image1, image2, iters=self.iters)
        flow, info = out["flow"][-1].float(), out["info"][-1].float()
        alpha1 = torch.softmax(info[:, 0:2], dim=1)[:, 1:2]
        log_b0 = torch.clamp(info[:, 2:3], min=0.0, max=self.var_max)
        return flow, alpha1, log_b0


# --------------------------------------------------------------------------- arguments


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Model architecture and trained-checkpoint selection."""
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG,
                        help="experiment configure file name (default: config/a1/tar.json)")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="checkpoint path (default: weights/tar-c-t.pth)")
    parser.add_argument("--iters", type=int, default=None,
                        help="refinement iterations baked into the graph (default: the "
                             "cfg's `iters`)")
    parser.add_argument("--allow_missing_keys", action="store_true",
                        help="continue when the checkpoint lacks weights the model has; "
                             "those stay randomly initialised, so without this it is an error")


def add_pair_args(parser: argparse.ArgumentParser) -> None:
    """Image-pair selection. Three sources, in order of precedence: a snapshot written by
    export_dataset.py, a rosbag2 recording, an OpenREALM keyframe sequence."""
    parser.add_argument("--dataset_pt2", type=Path, default=None,
                        help="snapshot written by export_dataset.py; if set it takes "
                             "precedence over every other option in this group")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="root of an OpenREALM keyframe sequence, as for evaluate.py "
                             "--dataset realm (default: custom/realm_1_100m_w_tandem)")
    parser.add_argument("--bag", type=Path, default=None,
                        help="read the pairs from this rosbag2 directory instead of "
                             "--data_root, as dataloader/flow/openrealm.py does")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help="sensor_msgs/CompressedImage topic to read (--bag only)")
    parser.add_argument("--no_undistort", action="store_true",
                        help="use the bag images as recorded (--bag only)")
    parser.add_argument("--offset", type=int, default=0,
                        help="index of the first pair; with --bag, the number of images "
                             "dropped at the start of the bag")
    parser.add_argument("--count", type=int, default=None,
                        help="number of pairs (default: every pair from --offset on)")
    parser.add_argument("--stride", type=int, default=1,
                        help="--data_root: pair each keyframe with the one this many "
                             "later. --bag: keep every N-th image and pair consecutive "
                             "kept ones")
    parser.add_argument("--resolution", type=parse_resolution, default=None,
                        help="resize every image to WIDTHxHEIGHT first (default: as "
                             "stored). H and W are baked into the graph, so this is what "
                             "decides the size a package accepts.")
    parser.add_argument("--device", default="cuda")


def add_precision_arg(parser: argparse.ArgumentParser) -> None:
    """`--precision fp32|tf32|default|fp16`, the arithmetic mode of *this process*.

    The three float32 modes are not a property of a package: the same package runs
    under whichever of them the loading process is in (see utils/precision.py). fp32 is
    the default because it is the mode in which a build can be validated -- there the
    package agrees with eager to rounding noise, while under the other two the same
    sound package is ~1e-2 px from eager on average and whole pixels in places. See
    scripts/aot_compile_flow.py for the measured table.

    fp16 is compiled into a package that is exported under it (check_precision()).
    """
    parser.add_argument("--precision", choices=MODES, default="fp32",
                        help="fp32 (default): full fp32. tf32: allow TF32 everywhere, "
                             "~20%% faster. default: leave torch's flags alone (TF32 "
                             "convolutions, fp32 matmul), which is what evaluate.py "
                             "runs under. Between these the flow moves by ~0.01 px on "
                             "average and by whole pixels in a few spots. fp16: float16 "
                             "autocast, about twice as fast; compiled into the package.")


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line and, when the script takes --cfg, attach the model
    configuration as `args.model_args`.

    Unlike config.parser.parse_args the two are kept apart rather than merged into one
    namespace, so an option that was not given stays an explicit None instead of
    disappearing. `--iters` is the one command-line value written into the model config.
    """
    args = parser.parse_args()
    if hasattr(args, "cfg"):
        if not args.cfg.is_file():
            raise SystemExit(f"no such config: {args.cfg}")
        args.model_args = json_to_args(str(args.cfg))
        if args.iters is not None:
            if args.iters < 1:
                raise SystemExit(f"--iters must be at least 1, got {args.iters}")
            args.model_args.iters = args.iters
    if getattr(args, "count", None) is not None and args.count < 1:
        raise SystemExit(f"--count must be at least 1, got {args.count}")
    if getattr(args, "offset", 0) < 0:
        raise SystemExit(f"--offset must not be negative, got {args.offset}")
    return args


def apply_precision(args) -> str:
    """Apply --precision to this process and report. Returns the mode."""
    set_precision(args.precision)
    print(f"precision: {args.precision} ({describe_precision()})")
    return args.precision


def _repo_relative(path: Path) -> str:
    """A path as it should be recorded: relative to the repository when inside it, so the
    record reads the same from the host and from the devcontainer."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def model_info(args) -> dict:
    """What identifies the model a result came from. Plain str/int/float only: this goes
    into snapshots, which have to stay readable from C++ (see SNAPSHOT_SAFE_GLOBALS)."""
    model_args = args.model_args
    return {
        "cfg": _repo_relative(args.cfg),
        "ckpt": _repo_relative(args.ckpt),
        "algorithm": str(model_args.algorithm),
        "iters": int(model_args.iters),
        "var_min": float(model_args.var_min),
        "var_max": float(model_args.var_max),
    }


# ------------------------------------------------------------------------------- model


def build_model(args) -> FlowExport:
    """Build the network, load the trained weights, wrap it in the export signature."""
    model_args = args.model_args
    core = fetch_model(model_args)
    if not args.ckpt.is_file():
        raise SystemExit(f"no such checkpoint: {args.ckpt}")
    state = torch.load(str(args.ckpt), map_location="cpu")
    # strict=False as utils.load_ckpt does, but not silently: a key the checkpoint lacks
    # leaves that weight at its random initialisation, and the package would bake it in.
    result = core.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        print(f"  WARNING: {len(result.unexpected_keys)} checkpoint keys the model does "
              f"not use, e.g. {result.unexpected_keys[:3]}")
    if result.missing_keys:
        message = (f"{len(result.missing_keys)} model weights are missing from "
                   f"{args.ckpt}, e.g. {result.missing_keys[:3]}")
        if not args.allow_missing_keys:
            raise SystemExit(f"{message}. They would be exported randomly initialised; "
                             f"pass --allow_missing_keys if that is intended.")
        print(f"  WARNING: {message}")
    half = is_half(getattr(args, "precision", None))
    if half and not str(args.device).startswith("cuda"):
        raise SystemExit("--precision fp16 needs a CUDA device")
    print(f"  {model_args.algorithm} from {_repo_relative(args.cfg)}, weights "
          f"{_repo_relative(args.ckpt)}, {model_args.iters} refinement iterations"
          + (", float16 autocast" if half else ""))
    if float(model_args.var_min) != 0.0:
        raise SystemExit(f"var_min is {model_args.var_min}; the exported outputs assume 0 (see FlowExport)")
    return FlowExport(core, int(model_args.iters), float(model_args.var_max), half=half).to(args.device).eval()


def as_outputs(result) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(flow, alpha1, log_b0) out of whatever a model, exported module or package returns."""
    if not isinstance(result, torch.Tensor) and len(result) == 2:
        raise SystemExit("two outputs (flow, info): this package or snapshot predates the "
                         "(flow, alpha1, log_b0) signature; rebuild it")
    if isinstance(result, torch.Tensor) or len(result) != 3:
        raise TypeError("expected three outputs (flow, alpha1, log_b0)")
    flow, alpha1, log_b0 = result
    return flow, alpha1, log_b0


# ------------------------------------------------------------------------------- pairs


def _resize(image: torch.Tensor, resolution) -> torch.Tensor:
    """Resize a [3, H, W] 0..255 image to (width, height) the way OpenRealmBag does."""
    if resolution is None or tuple(resolution) == (image.shape[2], image.shape[1]):
        return image
    array = image.permute(1, 2, 0).numpy().astype(np.uint8)
    array = cv2.resize(array, tuple(resolution), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(array).permute(2, 0, 1).float()


def iter_pairs(args, device=None):
    """Yield (name, (image1, image2)) for every selected pair, lazily.

    The images are [1, 3, H, W] float32 RGB in 0..255 on `device`, contiguous -- exactly
    the tensors the model is called with. `name` is "<timestamp1>_<timestamp2>".
    """
    for name, (image1, image2) in _iter_pairs_as_loaded(args, device or args.device):
        # The loaders return permuted HWC arrays, i.e. channels-last strides, and older
        # snapshots stored them that way. See FlowExport for why this must not leak.
        yield name, (image1.contiguous(), image2.contiguous())


def _iter_pairs_as_loaded(args, device):
    if args.dataset_pt2 is not None:
        print(f"  using dataset snapshot {args.dataset_pt2}; ignoring the other pair options")
        for row in load_snapshot(args.dataset_pt2)["pairs"]:
            yield row["name"], (row["image1"].to(device), row["image2"].to(device))
        return

    if args.bag is not None:
        from dataloader.flow.openrealm import OpenRealmBag

        dataset = OpenRealmBag(args.bag, topic=args.topic, skip=args.offset, stride=args.stride,
                               max_pairs=args.count, resolution=args.resolution,
                               undistort=not args.no_undistort)
        for image1, image2, (ts1, ts2) in dataset:
            yield f"{ts1}_{ts2}", (image1[None].to(device), image2[None].to(device))
        return

    from dataloader.flow.realm import Realm

    dataset = Realm(root=str(args.data_root), stride=args.stride)
    if args.offset >= len(dataset):
        raise SystemExit(f"--offset {args.offset} is past the {len(dataset)} pairs of {args.data_root}")
    stop = len(dataset) if args.count is None else min(args.offset + args.count, len(dataset))
    for index in range(args.offset, stop):
        # fetch(), not dataset[index]: __getitem__ answers a failed read with a random
        # other pair, which is right for training and wrong for a reference.
        image1, image2, (ts1, ts2) = dataset.fetch(index)
        yield f"{ts1}_{ts2}", (_resize(image1, args.resolution)[None].to(device),
                               _resize(image2, args.resolution)[None].to(device))


def first_inputs(args):
    """The first selected pair, used as the tracing example."""
    try:
        name, inputs = next(iter_pairs(args))
    except StopIteration:
        raise SystemExit("the pair selection is empty") from None
    print(f"  tracing example: pair {name}")
    print("  input shapes: " + ", ".join(str(tuple(t.shape)) for t in inputs))
    return inputs


def build_model_and_inputs(args):
    """Build the model and return it with one example input tuple."""
    model = build_model(args)
    return model, first_inputs(args)


# -------------------------------------------------------------------------- comparison


def diff_sums(result, reference) -> dict[str, float]:
    """Difference between two (flow, alpha1, log_b0) results, as running totals; the
    two confidence outputs are reported together as "info".

    The flow difference is measured as endpoint error: the length of the difference of
    the two flow vectors at each pixel, in pixels. Both results come from the same model
    on the same inputs, so there is no validity mask and every pixel counts.
    """
    flow, alpha1, log_b0 = as_outputs(result)
    flow_ref, alpha1_ref, log_b0_ref = as_outputs(reference)
    info = torch.cat([alpha1, log_b0], dim=1)
    info_ref = torch.cat([alpha1_ref, log_b0_ref], dim=1)
    for name, got, want in (("flow", flow, flow_ref), ("alpha1/log_b0", info, info_ref)):
        if got.shape != want.shape:
            raise ValueError(f"{name} shape {tuple(got.shape)} does not match the "
                             f"reference's {tuple(want.shape)}")
        if not torch.isfinite(got).all():
            raise ValueError(f"{name} contains non-finite values")
        if not torch.isfinite(want).all():
            raise ValueError(f"reference {name} contains non-finite values")

    epe = (flow.double() - flow_ref.double()).norm(dim=-3)
    info_error = (info.double() - info_ref.double()).abs()
    return {
        "pixels": float(epe.numel()),
        "epe": epe.sum().item(),
        "epe_sq": epe.square().sum().item(),
        "epe_max": epe.max().item(),
        "info_values": float(info_error.numel()),
        "info": info_error.sum().item(),
        "info_max": info_error.max().item(),
        "bitexact": float(torch.equal(flow, flow_ref) and torch.equal(info, info_ref)),
    }


def merge_sums(total: dict[str, float] | None, sums: dict[str, float]) -> dict[str, float]:
    """Fold one pair's diff_sums() into a running total (None to start one)."""
    if total is None:
        return dict(sums)
    merged = {key: total[key] + sums[key] for key in ("pixels", "epe", "epe_sq", "info_values", "info")}
    merged["epe_max"] = max(total["epe_max"], sums["epe_max"])
    merged["info_max"] = max(total["info_max"], sums["info_max"])
    merged["bitexact"] = min(total["bitexact"], sums["bitexact"])
    return merged


def format_diff(sums: dict[str, float]) -> str:
    return (
        f"flow EPE mean={sums['epe'] / sums['pixels']:.3e} px "
        f"max={sums['epe_max']:.3e} px rmse={(sums['epe_sq'] / sums['pixels']) ** 0.5:.3e} px | "
        f"alpha1/log_b0 mean|d|={sums['info'] / sums['info_values']:.3e} max|d|={sums['info_max']:.3e} | "
        f"bitexact={bool(sums['bitexact'])}"
    )


# ---------------------------------------------------------------------------- manifest


def compute_dtype(precision: str | None) -> str:
    """What a package exported under `precision` computes in (build_info.json)."""
    return "float16" if is_half(precision) else "float32"


def check_precision(manifest: dict, precision: str) -> None:
    """Report when --precision and the package's compiled-in dtype do not go together:
    float16 is part of a build, not a run-time switch (utils/precision.py)."""
    built = manifest.get("compute_dtype", "float32") if manifest else "float32"
    wanted = compute_dtype(precision)
    if built != wanted:
        print(f"  *** MISMATCH: the package computes in {built} (compiled in), but "
              f"--precision {precision} asks for {wanted}. ***")


def manifest_path(package_path: Path) -> Path:
    return Path(package_path).parent / "build_info.json"


def write_manifest(build_dir: Path, package_name: str, args, inputs, check: dict,
                   layout_optimization: bool) -> None:
    """Record how a build was compiled, next to the .pt2 package.

    None of this can be read back out of the package, and every item decides what a
    package computes, how fast, or whether it runs at all. The float32 arithmetic mode
    is deliberately not among them: it belongs to the process that runs the package
    (utils/precision.py). `compute_dtype` is: float16 is compiled in when the package
    was exported under --precision fp16. `check` is the build's own comparison against
    eager, kept for the record.
    """
    (build_dir / "build_info.json").write_text(json.dumps({
        "package": package_name,
        "torch": torch.__version__,
        "device": str(inputs[0].device),
        "input_shape": list(inputs[0].shape),
        "input_layout": INPUT_LAYOUT,
        "compute_dtype": compute_dtype(getattr(args, "precision", None)),
        "outputs": ["flow", "alpha1", "log_b0"],
        "layout_optimization": layout_optimization,
        "model": model_info(args),
        "check": check,
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")


def check_manifest(package_path: Path) -> dict:
    """Report what the package was built for and warn if the runtime's torch differs;
    return the manifest (empty when there is none).

    An AOTInductor build is locked to the torch that produced it, both its C++ ABI and
    its runtime API, so a package compiled under a different version either fails to
    load or is subtly wrong. Worth catching before inference rather than after.
    """
    path = manifest_path(package_path)
    if not path.is_file():
        print("  no build_info.json beside the package; cannot verify the torch "
              "version or the image size it was built for")
        return {}
    info = json.loads(path.read_text())
    if info.get("package") not in (None, Path(package_path).name):
        print(f"  note: build_info.json describes {info['package']}, not "
              f"{Path(package_path).name}; its contents may not apply")
    model = info.get("model", {})
    shape = info.get("input_shape") or [None] * 4
    layout = {True: "on", False: "off"}.get(info.get("layout_optimization"), "not recorded")
    print(f"  build_info.json: {model.get('algorithm')} / {model.get('ckpt')}, "
          f"{model.get('iters')} iterations, images {shape[3]}x{shape[2]}, "
          f"{info.get('compute_dtype', 'float32')}, layout optimization {layout}")

    built_torch = info.get("torch")
    if built_torch and built_torch != torch.__version__:
        print(f"  *** MISMATCH: package was compiled with torch {built_torch}, "
              f"running torch {torch.__version__}. An AOTInductor build is not "
              f"portable across torch versions -- recompile it with "
              f"aot_compile_flow.py. ***")
    return info


def check_model_matches(what: str, theirs: dict | None, ours: dict | None) -> None:
    """Warn when two records of "which model" disagree (see model_info()).

    A package compared against a reference from other weights or another number of
    refinement iterations is a comparison between two models (3 iterations against 5:
    9e-02 px mean, 8.5 px max), and that reads as a broken build.
    """
    if not theirs or not ours:
        return
    differing = [key for key in ("algorithm", "ckpt", "iters") if theirs.get(key) != ours.get(key)]
    if differing:
        detail = ", ".join(f"{key}: {theirs.get(key)} vs {ours.get(key)}" for key in differing)
        print(f"  *** MISMATCH: {what} ({detail}). A difference between the two is then "
              f"a different model, not a compile error. ***")


def check_input_shape(manifest: dict, inputs, name: str) -> None:
    """Stop on an image size or memory layout the package was not compiled for. H and W
    are static in the graph and the runtime's own complaint about them is not readable;
    a wrong layout draws no complaint at all, only wrong flow (see FlowExport)."""
    if manifest and manifest.get("input_layout") != INPUT_LAYOUT:
        raise SystemExit(
            f"the package's build_info.json does not say input_layout={INPUT_LAYOUT!r}: "
            f"it was built before the input layout was pinned down and expects "
            f"channels-last strides, so it would silently compute wrong flow from the "
            f"contiguous tensors these scripts now pass. Recompile it with "
            f"aot_compile_flow.py."
        )
    if not all(t.is_contiguous() for t in inputs):
        raise SystemExit(f"pair {name} is not contiguous; the package would misread it")
    built = manifest.get("input_shape")
    if built and list(inputs[0].shape) != list(built):
        raise SystemExit(
            f"pair {name} is {tuple(inputs[0].shape)} but the package was compiled for "
            f"{tuple(built)}; the image size is baked in. Pass --resolution "
            f"{built[3]}x{built[2]}, or recompile with aot_compile_flow.py."
        )


# ---------------------------------------------------------------------------- snapshot


def export_dataset(args, model: FlowExport, output_path: str | Path, precision: str) -> Path:
    """Serialize every selected pair's input tensors, and the flow, alpha1 and log_b0 the
    eager Python model produces for them, to a .pt2 snapshot.

    Embedding the tensors makes the snapshot self-contained: nothing is re-read from
    --data_root or a bag afterwards, and a C++ input pipeline can be checked against the
    exact tensors the model was fed.

    `flow_eager` / `alpha1_eager` / `log_b0_eager` are the reference a compiled package is scored against:
    the eager model's own output, since none of these sequences has flow ground truth.
    They depend on the arithmetic mode the model ran under, so that is recorded
    alongside and checked by check_eager_meta() before any comparison.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    shape = None
    for name, (image1, image2) in iter_pairs(args):
        if shape is None:
            shape = image1.shape
        if image1.shape != shape or image2.shape != shape:
            raise SystemExit(f"pair {name} is {tuple(image1.shape)}/{tuple(image2.shape)}, "
                             f"earlier pairs are {tuple(shape)}; a snapshot holds one image "
                             f"size (pass --resolution)")
        with torch.no_grad():
            flow, alpha1, log_b0 = model(image1, image2)
        pairs.append({
            "name": name,
            "image1": image1.cpu(),
            "image2": image2.cpu(),
            "flow_eager": flow.cpu(),
            "alpha1_eager": alpha1.cpu(),
            "log_b0_eager": log_b0.cpu(),
        })
        magnitude = flow.norm(dim=1)
        print(f"  {name}: embedding eager flow {tuple(flow.shape)} "
              f"mean={magnitude.mean():.2f} px max={magnitude.max():.2f} px")
    if not pairs:
        raise SystemExit("the pair selection is empty")
    torch.save({
        "pairs": pairs,
        "model": model_info(args),
        "eager": {
            "precision": precision,
            # str(): torch.__version__ is a TorchVersion, see SNAPSHOT_SAFE_GLOBALS.
            "torch": str(torch.__version__),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    }, str(path))
    return path


SNAPSHOT_SAFE_GLOBALS: tuple[type, ...] = ()
"""Types a snapshot payload may hold beyond tensors and plain containers.

Deliberately empty, and it should stay that way.

A snapshot is meant to remain readable by libtorch's IValue unpickler
(`torch::pickle_load`), so that a C++ input pipeline can use it as the reference for
building correct input tensors. That unpickler resolves only a fixed set of pickle
GLOBALs -- `torch._utils._rebuild_tensor_v2`, the storage types,
`collections.OrderedDict`. Anything else makes the *whole* file unreadable from C++, not
just the offending key, because the payload deserializes into one IValue.

`torch.__version__` is exactly that trap: it is a `TorchVersion`, not a `str`, and
storing it directly emits a GLOBAL that libtorch cannot construct. Hence the `str()` at
the write site in export_dataset(), and the plain types in model_info().

Python alone would accept such a type, via `torch.serialization.safe_globals(...)`,
which is what this tuple feeds. Adding to it trades away C++ readability.
"""


def load_snapshot(path: str | Path) -> dict:
    """Read a .pt2 snapshot written by export_dataset().

    The single entry point for reading these files. torch's `weights_only=True` default
    is left in place: against an empty allowlist it fails on precisely the types
    libtorch cannot read, so every Python-side load doubles as a check that a C++ reader
    could still open the file.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no such dataset snapshot: {path}")
    if SNAPSHOT_SAFE_GLOBALS:
        with torch.serialization.safe_globals(list(SNAPSHOT_SAFE_GLOBALS)):
            payload = torch.load(str(path), map_location="cpu")
    else:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise SystemExit(f"{path} is not a snapshot written by export_dataset.py")
    return payload


def check_eager_meta(payload: dict, precision: str) -> None:
    """Warn when the snapshot's eager reference was produced under another arithmetic
    mode than this process runs in.

    Nothing in the stored tensor says which mode produced it, and a mismatch reads as a
    slightly broken compile when it is only a rounding-mode difference.
    """
    meta = payload.get("eager", {})
    if meta.get("precision") == precision:
        print(f"  snapshot eager reference: precision {precision} -- matches")
    else:
        print(f"  *** MISMATCH: snapshot's eager reference was produced under precision "
              f"{meta.get('precision')} but this process runs under {precision}. The "
              f"diff will be precision noise, not a compile error. ***")
    made_with = meta.get("torch")
    if made_with and made_with != torch.__version__:
        print(f"  note: eager reference came from torch {made_with}, "
              f"running torch {torch.__version__}")
