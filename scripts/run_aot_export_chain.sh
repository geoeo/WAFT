#!/usr/bin/env bash
# Run the dataset export, AOT compile, package inference and package evaluation in
# sequence.
#
#   bash scripts/run_aot_export_chain.sh [--precision fp32|tf32|default]
#
# Meant to run inside the devcontainer. From the host:
#   devcontainer exec --workspace-folder . bash scripts/run_aot_export_chain.sh
#
# --precision is the float32 arithmetic mode all four steps run under (see
# utils/precision.py); it is passed to every one of them, so the eager reference, the
# build's own check and the package runs agree. The default, fp32, is the only mode in
# which the comparisons against eager are a verdict on the build: under tf32 or
# default a sound package is ~1e-2 px from eager on average. The mode is not compiled
# into the package, so the package built here is the same whichever one is given.
#
# The image size the package is built for defaults to 640x480; set RESOLUTION
# (WIDTHxHEIGHT) for another one. The pairs are resized to it, and the build goes to
# outputs/aoti/<precision>_<RESOLUTION>/, by default outputs/aoti/fp32_640x480/.
set -euo pipefail

precision="fp32"
while [ $# -gt 0 ]; do
    case "$1" in
        --precision)
            [ $# -ge 2 ] || { echo "--precision needs a value: fp32, tf32 or default" >&2; exit 2; }
            precision="$2"
            shift 2
            ;;
        --precision=*)
            precision="${1#--precision=}"
            shift
            ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1 (usage: $0 [--precision fp32|tf32|default])" >&2
            exit 2
            ;;
    esac
done
case "$precision" in
    fp32|tf32|default) ;;
    *) echo "--precision must be fp32, tf32 or default, got '$precision'" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="${PYTHON:-/opt/conda/envs/waft/bin/python}"
# The snapshot holds the eager model's flow, which depends on the precision, so each
# mode gets its own file and the modes do not overwrite each other's reference.
dataset_pt2="$repo_root/outputs/dataset_flow_realm_1_100m_${precision}.pt2"
resolution="${RESOLUTION:-640x480}"
build_dir="$repo_root/outputs/aoti/${precision}_$resolution"
export CUDA_VISIBLE_DEVICES=0

echo "[1/4] Export Dataset (.pt2) at $resolution, precision $precision"
"$python" "$repo_root/scripts/export_dataset.py" \
    --data_root "$repo_root/custom/realm_1_100m_w_tandem" \
    --resolution "$resolution" \
    --precision "$precision" \
    --offset 0 \
    --count 3 \
    --output "$dataset_pt2"

echo "[2/4] AOT Compile Flow"
"$python" "$repo_root/scripts/aot_compile_flow.py" \
    --dataset_pt2 "$dataset_pt2" \
    --output_dir "$build_dir" \
    --precision "$precision" \
    --clean

echo "[3/4] Run Flow Package"
"$python" "$repo_root/scripts/run_flow_aoti.py" \
    --package "$build_dir/waft_flow.pt2" \
    --dataset_pt2 "$dataset_pt2" \
    --precision "$precision" \
    --benchmark --bench_iters 8 \
    --compare_eager \
    --output_dir "$repo_root/outputs/aoti_flow" \
    --clean

echo "[4/4] Evaluate Flow Package against the snapshot"
"$python" "$repo_root/scripts/evaluate_flow_aoti.py" \
    --package "$build_dir/waft_flow.pt2" \
    --dataset_pt2 "$dataset_pt2" \
    --precision "$precision"
