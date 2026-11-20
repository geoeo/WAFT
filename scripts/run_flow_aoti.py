#!/usr/bin/env python3
"""Run a compiled AOTInductor flow package over image pairs.

Takes the same pair-selection arguments as the other export scripts:

    python scripts/run_flow_aoti.py \
        --package outputs/aoti/fp32_640x480/waft_flow.pt2 \
        --data_root custom/realm_1_100m_w_tandem

    python scripts/run_flow_aoti.py --package ... \
        --dataset_pt2 outputs/dataset_flow.pt2 \
        --compare_eager --benchmark --output_dir outputs/aoti_flow

The build directory's build_info.json is checked at startup, to catch a torch version
that no longer matches the one that compiled the package, and an image size the package
was not compiled for (--resolution resizes the pairs to the size it wants).

--precision sets the arithmetic mode of this process, which is what the package then
computes in: the mode is not compiled in (see utils/precision.py). It defaults to fp32,
the mode in which --compare_eager is a verdict on the build.

One caveat on --compare_eager: importing flow_export_common sets XFORMERS_DISABLED and
AOTI_SAFE_GRID_SAMPLE, so the eager model built here for comparison is the export-safe
one the package was compiled from (see utils/export_compat.py). The reported diff is
therefore package-vs-inductor, which is what you want when checking that a build is
sound. It is *not* the gap against the flow evaluate.py produces; that gap is quoted in
aot_compile_flow.py.

Step 2 of 2. This is the Python stand-in for the C++ loader: it drives the same .pt2
through torch._inductor.aoti_load_package(), the counterpart of C++'s
torch::inductor::AOTIModelPackageLoader.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `model` import: sets XFORMERS_DISABLED / AOTI_SAFE_GRID_SAMPLE.
import flow_export_common as common  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from custom import get_heatmap, vis_heatmap  # noqa: E402
from dataloader.flow.openrealm import clean_output  # noqa: E402
from utils.flow_viz import flow_to_image  # noqa: E402


def time_calls(fn, inputs, iters: int) -> float:
    """Mean milliseconds per call, after one untimed warm-up call."""
    with torch.no_grad():
        fn(*inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            fn(*inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_pair_args(parser)
    common.add_model_args(parser)
    parser.add_argument("--package", type=Path, required=True,
                        help="compiled .pt2 package to run (from aot_compile_flow.py)")
    common.add_precision_arg(parser)
    parser.add_argument("--compare_eager", action="store_true",
                        help="also build the eager model and diff against it. NOTE: "
                             "that model is built in this process, so it is the "
                             "export-safe one -- the diff measures the package against "
                             "the same arithmetic it was compiled from, not against "
                             "what evaluate.py computes. See utils/export_compat.py.")
    parser.add_argument("--benchmark", action="store_true",
                        help="time the package (and the eager model, with --compare_eager)")
    parser.add_argument("--bench_iters", type=int, default=8, help="benchmark iterations")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="write flow_<pair>.npy (H x W x 2), its colour coding and "
                             "the uncertainty heatmap here, named as evaluate.py "
                             "--dataset realm names them")
    parser.add_argument("--clean", action="store_true",
                        help="empty --output_dir first, so results of earlier runs do "
                             "not mix with the new ones")
    args = common.parse_args(parser)

    if not args.package.is_file():
        raise SystemExit(
            f"no such package: {args.package} (build it with aot_compile_flow.py)"
        )

    precision = common.apply_precision(args)
    manifest = common.check_manifest(args.package)
    if args.compare_eager:
        common.check_model_matches("the package was built from another model than "
                                   "--cfg/--ckpt/--iters select for the eager comparison",
                                   manifest.get("model"), common.model_info(args))

    print(f"loading {args.package} ...")
    # No device argument: the package is compiled for one device and carries it.
    # --device still decides where the input tensors are placed, and must agree.
    runner = torch._inductor.aoti_load_package(str(args.package))

    model = None
    if args.compare_eager:
        print("building eager model for comparison ...")
        model = common.build_model(args)

    if args.output_dir:
        if args.clean:
            clean_output(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # The heatmap's clamping range belongs to the model the package was built from.
    heatmap_args = argparse.Namespace(**vars(args.model_args))
    for key in ("var_min", "var_max"):
        if key in manifest.get("model", {}):
            setattr(heatmap_args, key, manifest["model"][key])

    total = None
    count = 0
    for name, inputs in common.iter_pairs(args):
        common.check_input_shape(manifest, inputs, name)
        count += 1
        with torch.no_grad():
            flow, info = common.as_flow_info(runner(*inputs))
        magnitude = flow.norm(dim=1)
        heatmap = get_heatmap(info, heatmap_args)
        print(f"\n[{name}]")
        print(f"  flow {tuple(flow.shape)} on {flow.device} mean={magnitude.mean():.2f} px "
              f"max={magnitude.max():.2f} px, mean uncertainty {heatmap.mean():.3f}")

        if model is not None:
            with torch.no_grad():
                reference = model(*inputs)
            sums = common.diff_sums((flow, info), reference)
            total = common.merge_sums(total, sums)
            print(f"  vs eager: {common.format_diff(sums)}")

        if args.benchmark:
            per_call = time_calls(runner, inputs, args.bench_iters)
            line = (f"  {per_call:.1f} ms/call over {args.bench_iters} iters "
                    f"({1000 / per_call:.2f} calls/s)")
            if model is not None:
                line += f"; eager {time_calls(model, inputs, args.bench_iters):.1f} ms/call"
            print(line)

        if args.output_dir:
            flow_hwc = flow[0].permute(1, 2, 0).cpu().numpy()
            np.save(args.output_dir / f"flow_{name}.npy", flow_hwc)
            cv2.imwrite(str(args.output_dir / f"flow_{name}.png"),
                        flow_to_image(flow_hwc, convert_to_bgr=True))
            vis_heatmap(str(args.output_dir / f"heatmap_{name}.png"),
                        inputs[0][0].permute(1, 2, 0).cpu().numpy(),
                        heatmap[0].permute(1, 2, 0).cpu().numpy())
            if model is not None:
                np.save(args.output_dir / f"flow_{name}_eager.npy",
                        reference[0][0].permute(1, 2, 0).cpu().numpy())
            print(f"  wrote flow_{name}.npy/.png and heatmap_{name}.png to {args.output_dir}")

    if count == 0:
        raise SystemExit("the pair selection is empty")
    print(f"\nran {count} pair(s)")
    if total is not None:
        print(f"package vs eager over all pairs: {common.format_diff(total)}")
        if precision != "fp32":
            print(f"NOTE: under --precision {precision} a sound package is ~1e-2 px mean "
                  f"from eager, pixels in places; run with --precision fp32 to judge "
                  f"the build (see aot_compile_flow.py)")
        elif total["epe"] / total["pixels"] > 1e-3:
            print("NOTE: a mean endpoint error above 1e-3 px against eager under fp32 is "
                  "not rounding noise; the build is suspect, or was made from another "
                  "model (see aot_compile_flow.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
