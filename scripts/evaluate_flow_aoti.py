#!/usr/bin/env python3
"""Diff a compiled flow package against the eager reference stored in a snapshot.

Example:
    python scripts/evaluate_flow_aoti.py \
        --package outputs/aoti/640x480/waft_flow.pt2 \
        --dataset_pt2 outputs/dataset_flow.pt2

The reference is `flow_eager` / `info_eager`: what the eager Python model produced for
the very same input tensors, recorded by scripts/export_dataset.py. So the reported
numbers are a plain per-pixel difference between two results. For the flow that is the
endpoint error in pixels -- mean, max and RMSE -- and for the uncertainty head the mean
and max absolute difference. There is no validity mask to apply: both come from the
same model on the same inputs, so every pixel is comparable.

Unlike run_flow_aoti.py --compare_eager this needs neither the checkpoint nor the model
code at run time, only the package and the snapshot.

--precision must match the mode the snapshot's eager reference was produced under;
that is checked at startup, because a mismatch shows up as a difference and reads as a
broken build. How the package was compiled does not enter into it: the arithmetic mode
belongs to the process that runs a package, see utils/precision.py. Only under fp32 is
the result a verdict on the build -- there a sound package has a mean endpoint error
around 1e-5 px, while under tf32 or default the same package is around 1e-2 px from
eager, with single pixels off by whole pixels (table in aot_compile_flow.py).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede torch/model imports; this also matches the compiled runner's setup.
import flow_export_common as common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True,
                        help="compiled AOTInductor .pt2 package")
    parser.add_argument("--dataset_pt2", type=Path, required=True,
                        help="dataset snapshot containing model inputs and the eager flow")
    parser.add_argument("--device", default="cuda",
                        help="device for dataset tensors; must match the compiled package")
    common.add_precision_arg(parser)
    args = common.parse_args(parser)

    if not args.package.is_file():
        raise SystemExit(f"no such compiled package: {args.package}")
    payload = common.load_snapshot(args.dataset_pt2)

    precision = common.apply_precision(args)
    manifest = common.check_manifest(args.package)
    common.check_eager_meta(payload, precision)
    common.check_model_matches("the snapshot's eager reference and the package come "
                               "from different models",
                               payload.get("model"), manifest.get("model"))
    print(f"loading compiled package {args.package} ...")
    runner = torch._inductor.aoti_load_package(str(args.package))

    total = None
    evaluated = 0
    for row in payload["pairs"]:
        name = row["name"]
        # contiguous(): snapshots written before the input layout was pinned down hold
        # channels-last strided images; see flow_export_common.FlowExport.
        inputs = (row["image1"].to(args.device).contiguous(),
                  row["image2"].to(args.device).contiguous())
        reference = (row["flow_eager"].to(args.device), row["info_eager"].to(args.device))
        common.check_input_shape(manifest, inputs, name)

        with torch.no_grad():
            prediction = common.as_flow_info(runner(*inputs))
        try:
            sums = common.diff_sums(prediction, reference)
        except ValueError as error:
            raise SystemExit(f"{name}: {error}") from error
        total = common.merge_sums(total, sums)
        evaluated += 1
        print(f"[{name}] {common.format_diff(sums)}")

    if evaluated == 0:
        raise SystemExit("dataset snapshot contains no pairs")

    print(f"\noverall ({evaluated} pair(s), {int(total['pixels'])} pixels)")
    print(common.format_diff(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
