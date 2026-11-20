#!/usr/bin/env bash
# Run the dataset export, AOT compile, package inference and package evaluation in
# sequence, all under full fp32 (the scripts' default --precision).
#
# Meant to run inside the devcontainer. From the host:
#   devcontainer exec --workspace-folder . bash scripts/run_aot_export_chain.sh
#
# The image size the package is built for defaults to 640x480; set RESOLUTION
# (WIDTHxHEIGHT) for another one. The pairs are resized to it, and the build goes to
# outputs/aoti/<RESOLUTION>/.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="${PYTHON:-/opt/conda/envs/waft/bin/python}"
dataset_pt2="$repo_root/outputs/dataset_flow_realm_1_100m.pt2"
resolution="${RESOLUTION:-640x480}"
build_dir="$repo_root/outputs/aoti/$resolution"
export CUDA_VISIBLE_DEVICES=0

echo "[1/4] Export Dataset (.pt2) at $resolution"
"$python" "$repo_root/scripts/export_dataset.py" \
    --data_root "$repo_root/custom/realm_1_100m_w_tandem" \
    --resolution "$resolution" \
    --offset 0 \
    --count 3 \
    --output "$dataset_pt2"

echo "[2/4] AOT Compile Flow"
"$python" "$repo_root/scripts/aot_compile_flow.py" \
    --dataset_pt2 "$dataset_pt2" \
    --output_dir "$build_dir" \
    --clean

echo "[3/4] Run Flow Package"
"$python" "$repo_root/scripts/run_flow_aoti.py" \
    --package "$build_dir/waft_flow.pt2" \
    --dataset_pt2 "$dataset_pt2" \
    --benchmark --bench_iters 8 \
    --compare_eager \
    --output_dir "$repo_root/outputs/aoti_flow" \
    --clean

echo "[4/4] Evaluate Flow Package against the snapshot"
"$python" "$repo_root/scripts/evaluate_flow_aoti.py" \
    --package "$build_dir/waft_flow.pt2" \
    --dataset_pt2 "$dataset_pt2"
