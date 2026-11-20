#!/usr/bin/env python3
"""Compile the flow network to a self-contained AOTInductor .pt2 package.

    python scripts/aot_compile_flow.py --data_root custom/realm_1_100m_w_tandem
    python scripts/aot_compile_flow.py --dataset_pt2 outputs/dataset_flow.pt2 --clean
    python scripts/aot_compile_flow.py --resolution 1024x768 --output_dir /opt/models/waft

The first selected pair becomes the tracing example. Its size is baked into the graph,
together with the batch size of 1 and the number of refinement iterations, so a package
accepts exactly one image size: compile one per resolution, with --resolution. The
default build directory is named after it, outputs/aoti/<width>x<height>/.

This uses the two-step export workflow:

    ep      = torch.export.export(model, inputs)
    package = torch._inductor.aoti_compile_and_package(ep, package_path=...)

The compiled .so and every generated kernel live inside one zip archive, which the
loader unpacks to a temporary directory. The package is a single self-contained file --
copy it anywhere.

    Python: torch._inductor.aoti_load_package(path)
    C++:    torch::inductor::AOTIModelPackageLoader loader(path);
            loader.run(inputs);

A package is still locked to the torch that built it, so it must be recompiled after a
torch upgrade. build_info.json records the version and run_flow_aoti.py reports a
mismatch at startup.

Precision is not part of the build. The package reaches cuDNN and cuBLAS through the
same runtime as eager does, and they read the TF32 flags at call time, so the arithmetic
mode is whatever the process that *runs* the package is in (utils/precision.py).
Verified: a build made under --precision fp32 and one made under tf32 hold a
byte-identical .so and kernels and return bit-identical flow in either runtime mode.
--precision here therefore only sets the mode of the build's own check against eager,
and that is why it defaults to fp32:

    mode      package   eager    package vs eager, same mode (endpoint error)
    fp32      348 ms    340 ms   2.4e-05 px mean, 1.5e-02 px max
    default   296 ms    325 ms   1.1e-02 px mean, 3.3 px max
    tf32      269 ms    293 ms   1.6e-02 px mean, 10 px max

Only the fp32 row is agreement, and it is the only mode in which the comparison says
anything about the build. The other two are not a broken graph: the model itself is
that sensitive to TF32 rounding. The unmodified pipeline, compared with itself, moves
by 1.0e-02 px mean / 3.0 px max between torch's defaults and fp32, and by 1.1e-02 px /
5.7 px between tf32 and fp32. The refinement iterations amplify the rounding in a few
spots of the large-motion pairs, and under TF32 the package and eager do not round
alike. Judge by the mean: a sound build is around 1e-05 px under fp32, any TF32 in play
puts it around 1e-02 px, and a package compared with the wrong model is further out
still (3 refinement iterations against 5: 9e-02 px mean, 8.5 px max).

"eager" above is the export-safe model these scripts build. Against what evaluate.py
actually writes (torch's defaults, xformers, cuDNN grid sampler, 322 ms), the package
lands at 1.0e-02 px mean / 3.0 px max when run under fp32 (1.3% of the pixels differ by
more than 0.1 px, 0.02% by more than 1 px) and at 9.5e-03 px / 2.4 px under default.
That is the mode sensitivity just described and nothing else: with both sides in fp32,
the two substitutions (xformers attention, cuDNN grid sampler) cost 2.6e-05 px mean /
4.4e-02 px max, and InferenceWrapper's `x * w / w` another 7.6e-06 px max (one pair).

So the package is not faster than eager under fp32 on this GPU -- the time goes into
cuDNN convolutions either way -- and about 9% faster where TF32 is allowed. What it
buys is deployment without the Python model code.

Measured 2026-11-20 on torch 2.15.0+cu134 / RTX 3060, over the twelve 640x480 keyframe
pairs of custom/realm_1_100m_w_tandem (52 to 91 px of mean flow), a1 model with
weights/tar-c-t.pth, 5 iterations. At 1024x768 (two rosbag pairs) the fp32 package is at
9.1e-06 px mean / 1.5e-04 px max from eager, 1150 ms against 1130 ms.

Step 1 of 2. Takes a couple of minutes.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `model` import: sets XFORMERS_DISABLED / AOTI_SAFE_GRID_SAMPLE.
import flow_export_common as common  # noqa: E402

import torch  # noqa: E402

from dataloader.flow.openrealm import clean_output  # noqa: E402

# Under fp32 a sound package has a mean endpoint error around 1e-5 px against the eager
# model it was compiled from, and TF32 or another model put it at 1e-2 px and beyond
# (see the table above). The mean, not the max: single pixels reach 1e-2 px even then.
FP32_EPE_MEAN_LIMIT_PX = 1e-3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_pair_args(parser)
    common.add_model_args(parser)
    common.add_precision_arg(parser)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="build directory, created if absent (default: "
                             "outputs/aoti/<width>x<height>/). Holds the waft_flow.pt2 "
                             "package and its build_info.json; the package alone is "
                             "enough to run inference and can be copied anywhere.")
    parser.add_argument("--clean", action="store_true",
                        help="empty the build directory first")
    parser.add_argument("--skip_check", action="store_true",
                        help="skip loading the package and comparing against eager")
    args = common.parse_args(parser)

    precision = common.apply_precision(args)

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    height, width = inputs[0].shape[-2:]
    build_dir = (args.output_dir
                 or common.REPO_ROOT / "outputs" / "aoti" / f"{width}x{height}").resolve()
    if args.clean:
        clean_output(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / "waft_flow.pt2"

    with torch.no_grad():
        reference = model(*inputs)
    magnitude = reference[0].norm(dim=1)
    print(f"  eager flow {tuple(reference[0].shape)} "
          f"mean={magnitude.mean():.2f} px max={magnitude.max():.2f} px")

    print("exporting the graph ...")
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
    targets = [str(n.target) for n in exported.graph.nodes]
    print(f"  {len(targets)} graph nodes")
    if any("cudnn" in t for t in targets):
        # Not fatal: the proxy executor does run it. But it means the substitution in
        # utils/export_compat.py is not in effect, which nothing here should allow.
        print("  WARNING: a cudnn op is in the graph; it will go through the proxy "
              "executor (see utils/export_compat.py)")

    print(f"compiling -> {output} (a couple of minutes) ...")
    started = time.time()
    with torch.no_grad():
        package = Path(torch._inductor.aoti_compile_and_package(
            exported, package_path=str(output),
        ))
    print(f"  compiled in {time.time() - started:.0f}s "
          f"({package.stat().st_size / 2**20:.0f} MiB)")

    check = {"precision": precision, "ran": False}
    if not args.skip_check:
        runner = torch._inductor.aoti_load_package(str(package))
        with torch.no_grad():
            got = runner(*inputs)
        sums = common.diff_sums(got, reference)
        print(f"  package vs eager: {common.format_diff(sums)}")
        check.update(ran=True, epe_mean_px=sums["epe"] / sums["pixels"],
                     epe_max_px=sums["epe_max"])
        if precision != "fp32":
            print(f"  (under --precision {precision} a sound build is ~1e-2 px mean from "
                  f"eager, pixels in places; check with --precision fp32 to judge the build)")
        elif check["epe_mean_px"] > FP32_EPE_MEAN_LIMIT_PX:
            print(f"  WARNING: mean endpoint error above {FP32_EPE_MEAN_LIMIT_PX} px "
                  f"against eager under fp32 -- that is not rounding noise, the build "
                  f"is suspect")

    common.write_manifest(build_dir, output.name, args, inputs, check)
    audit(package)

    print(f"\nbuild directory: {build_dir}")
    print(f"run it with: python scripts/run_flow_aoti.py --package {package}")
    return 0


def audit(package: Path) -> None:
    """Report what the .pt2 package contains, by file type.

    Everything the runtime needs is inside the archive -- the compiled .so and the
    generated kernels -- and nothing is referenced by an absolute path, so the package
    is relocatable. This just makes the contents visible.
    """
    with zipfile.ZipFile(package) as archive:
        entries = [e for e in archive.infolist() if not e.is_dir()]

    mib = lambda items: sum(i.file_size for i in items) / 2**20
    by_ext: dict[str, list[zipfile.ZipInfo]] = {}
    for entry in entries:
        by_ext.setdefault(Path(entry.filename).suffix or "(none)", []).append(entry)

    print(f"  {len(entries)} entries, {mib(entries):.0f} MiB uncompressed:")
    for ext, items in sorted(by_ext.items(), key=lambda kv: -mib(kv[1])):
        print(f"    {len(items):>4} {ext:<8} {mib(items):>8.1f} MiB")
    if not by_ext.get(".so"):
        print("  WARNING: no .so inside the package")


if __name__ == "__main__":
    raise SystemExit(main())
