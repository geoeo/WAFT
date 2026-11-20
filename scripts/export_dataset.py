#!/usr/bin/env python3
"""Export a set of image pairs, as the raw tensors the model is fed, together with the
flow the eager Python model produces for them, to a .pt2 snapshot.

    python scripts/export_dataset.py --data_root custom/realm_1_100m_w_tandem \
        --count 3 --output outputs/dataset_flow.pt2

    python scripts/export_dataset.py --resolution 640x480 --offset 3000 --count 3 \
        --bag /bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight

The resulting .pt2 is self-contained -- load it with
flow_export_common.load_snapshot(), and pass it to aot_compile_flow.py,
run_flow_aoti.py and evaluate_flow_aoti.py via --dataset_pt2.

Each pair also carries `flow_eager` and `info_eager`: what the eager Python model
produces for that pair. That is the reference evaluate_flow_aoti.py scores a compiled
package against; these sequences have no flow ground truth, and ground truth would
measure the model rather than the compile anyway.

Because the model is run here, this script needs the checkpoint and a GPU. The eager
flow depends on the arithmetic mode it was produced under (--precision, see
utils/precision.py), so that is recorded in the snapshot, and a comparison against it
has to run under the same mode; evaluate_flow_aoti.py checks that.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `model` import: sets XFORMERS_DISABLED.
import flow_export_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_pair_args(parser)
    common.add_model_args(parser)
    common.add_precision_arg(parser)
    parser.add_argument("--output", type=Path,
                        default=common.REPO_ROOT / "outputs" / "dataset_flow.pt2",
                        help="destination .pt2 holding the pairs' tensors and eager flow")
    args = common.parse_args(parser)
    if args.dataset_pt2 is not None:
        raise SystemExit("--dataset_pt2 is what this script writes; select the pairs "
                         "with --data_root or --bag")
    if args.bag is not None and args.count is None:
        raise SystemExit("a bag is a long stream and every pair is stored uncompressed; "
                         "pass --count")

    # The eager reference is only valid under the arithmetic mode it was produced in,
    # and that is what gets recorded.
    precision = common.apply_precision(args)

    print("building model ...")
    model = common.build_model(args)
    saved = common.export_dataset(args, model, args.output, precision)
    size_mb = saved.stat().st_size / 2**20
    print(f"saved pair tensors + eager reference to {saved} ({size_mb:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
