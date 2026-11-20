#!/usr/bin/env python3
"""Export the flow network to a torch.export ExportedProgram.

    python scripts/export_flow_model.py --data_root custom/realm_1_100m_w_tandem \
        --output outputs/waft_flow.pt2

The first selected pair becomes the tracing example.

The result is a serialised graph, not a state dict -- load it with
`torch.export.load`, never `torch.load`. It is device- and shape-specialised: B, H, W
and the number of refinement iterations are baked in.

Steps:
    export_dataset.py     -> .pt2  (pair tensors + eager flow) [Optional]
    export_flow_model.py  -> .pt2  (serialised graph) [Optional]
    aot_compile_flow.py   -> .pt2  (AOTInductor package: compiled .so + kernels)
    run_flow_aoti.py      -> runs the package

Three different things wear the .pt2 extension here, and they are not
interchangeable:

    torch.save()                        -> read with torch.load()
      (export_dataset.py's snapshot: a pickled dict of tensors)
    torch.export.save()                 -> read with torch.export.load()
      (this script: an ExportedProgram, still interpreted at runtime)
    aoti_compile_and_package()          -> read with aoti_load_package()
      (aot_compile_flow.py: native code, no Python interpreter in the loop)

This script is optional -- aot_compile_flow.py re-exports the model itself rather than
consuming the graph written here. Its value is inspection: the node count, the
banned-op scan below, and a check that the graph still computes what eager does.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `model` import: sets XFORMERS_DISABLED.
import flow_export_common as common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_pair_args(parser)
    common.add_model_args(parser)
    parser.add_argument("--output", type=Path,
                        default=common.REPO_ROOT / "outputs" / "waft_flow.pt2",
                        help="destination .pt2 (torch.export convention; a .pth "
                             "state dict is a different thing)")
    parser.add_argument("--strict", choices=["true", "false"], default="false",
                        help="dynamo-based tracing (true) or non-strict (false, torch's "
                             "own default and what aot_compile_flow.py uses). Both "
                             "work; strict is the stronger check.")
    common.add_precision_arg(parser)
    parser.add_argument("--skip_check", action="store_true",
                        help="skip the eager-vs-exported comparison")
    args = common.parse_args(parser)

    common.apply_precision(args)

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    with torch.no_grad():
        reference = model(*inputs)
    magnitude = reference[0].norm(dim=1)
    print(f"  eager flow {tuple(reference[0].shape)} "
          f"mean={magnitude.mean():.2f} px max={magnitude.max():.2f} px")

    strict = args.strict == "true"
    print(f"exporting (strict={strict}) ...")
    with torch.no_grad():
        exported = torch.export.export(model, inputs, strict=strict)
    print(f"  {len(list(exported.graph.nodes))} graph nodes")

    targets = [str(n.target) for n in exported.graph.nodes]
    for banned in ("cudnn", "flags", "xformers", "memory_efficient"):
        if any(banned in t for t in targets):
            print(f"  WARNING: '{banned}' present in the graph")

    if not args.skip_check:
        with torch.no_grad():
            got = exported.module()(*inputs)
        print(f"  exported vs eager: {common.format_diff(common.diff_sums(got, reference))}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(args.output))
    size_mb = args.output.stat().st_size / 2**20
    print(f"saved {args.output} ({size_mb:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
