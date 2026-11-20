[New!] Please also check [WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo), our new efficient state-of-the-art method for stereo matching.

# WAFT

[[Paper](https://arxiv.org/abs/2506.21526v2)]

We introduce Warping-Alone Field Transforms (WAFT), a simple and effective
method for optical flow. WAFT is similar to RAFT but replaces cost volume with
high-resolution warping, achieving better accuracy with lower memory cost. This
design challenges the conventional wisdom that constructing cost volumes is necessary for strong performance. WAFT is a simple and flexible meta-architecture
with minimal inductive biases and reliance on custom designs. Compared with
existing methods, WAFT ranks 1st on Spring, Sintel, and KITTI benchmarks,
achieves the best zero-shot generalization on KITTI, while being up to 4.1× faster
than methods with similar performance.

<img src="assets/Vis.png" width='1000'>

If you find WAFT useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2506.21526v2">
        WAFT: Warping-Alone Field Transforms for Optical Flow
    </a>
</h3>
<p align="center">
    <a href="https://memoryslices.github.io/">Yihan Wang</a>,
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{wang2025waft,
  title={WAFT: Warping-Alone Field Transforms for Optical Flow},
  author={Wang, Yihan and Deng, Jia},
  journal={arXiv preprint arXiv:2506.21526},
  year={2025}
}
```

## News
- [2025/09] WAFT now achieves sota performance on all benchmarks and supports multiple backbones (Twins/DAv2/DINOv3).

## Requirements
Our code is developed with pytorch 2.7.0, CUDA 12.8 and python 3.12. 
```Shell
conda create --name waft python=3.12
conda activate waft
pip install -r requirements.txt
```

Please also install [xformers](https://github.com/facebookresearch/xformers) following instructions.

## Model Zoo

Google Drive: [link](https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_?usp=sharing). For downstream applications, we recommend using a1 adaptation with this [checkpoint](https://drive.google.com/file/d/1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC/view?usp=drive_link).

## Datasets
To evaluate/train WAFT, you will need to download the required datasets: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/). Please also check [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) for more details.

## Training

You may need to prepare pre-trained [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) or [DINOv3](https://ai.meta.com/dinov3/) before you start training corresponding models.

```Shell
python train.py --cfg config/a2/twins/chairs.json
python train.py --cfg config/a2/twins/chairs-things.json --restore_ckpt ckpts/twins/chairs.pth
```

## Evaluation & Submission

```Shell
python evaluate --cfg config/a2/twins/chairs-things.json --ckpt ckpts/twins/zero-shot.pth --dataset sintel
python submission --cfg config/a2/twins/tar-c-t-kitti.json --ckpt ckpts/twins/kitti.pth --dataset kitti
```

To run WAFT over an OpenREALM keyframe sequence (images, `intrinsics.txt` and `kf_traj.txt`; no flow ground truth, so raw flow, visualisations, uncertainty heatmaps and per-pair statistics are written to `outputs/<sequence name>`):

```Shell
python evaluate.py --cfg config/a1/tar.json --ckpt weights/tar-c-t.pth --dataset realm --data_root custom/realm_1_100m_w_tandem
```

To run WAFT offline over the camera images of an OpenREALM rosbag2 recording (read straight from the bag files, no ROS needed; colour-coded flow, flow arrows, uncertainty heatmaps and per-pair statistics are written to `outputs/<bag name>`, raw flow with `--save_flow`; images are undistorted with the camera intrinsics hardcoded in the script, `--no_undistort` turns that off; `--resolution WIDTHxHEIGHT` then resizes them, flow is in pixels of that size; the output folder is emptied before each run unless `--no_clean` is given):

```Shell
python dataloader/flow/openrealm.py --cfg config/a1/tar.json --ckpt weights/tar-c-t.pth --resolution 640x480 --bag /bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight
```

To follow a grid of pixels through several consecutive images of such a recording and draw the path each one takes (N images, default 5, give N - 1 flows; one image per flow is written to `outputs/<bag name>_flow_paths_skip<skip>`, the image of flow k showing every pixel's path from image 0 as a chain of k + 1 arrows, one colour per step; the followed pixels are the grid the arrow image above uses, `--arrow_step`; images are resized to 640x480 unless `--resolution` says otherwise):

```Shell
python scripts/flow_paths.py --skip 3000 --num_images 5
```

## AOT export

The flow network can be compiled ahead of time with `torch.export` + AOTInductor into a single self-contained `.pt2` package that runs without the Python model code, from Python (`torch._inductor.aoti_load_package`) or C++ (`torch::inductor::AOTIModelPackageLoader`). The scripts live in [scripts/](scripts) and run inside the devcontainer; from the host, the whole chain is

```Shell
devcontainer exec --workspace-folder . bash scripts/run_aot_export_chain.sh
```

(inside the container: `bash scripts/run_aot_export_chain.sh`). The [script](scripts/run_aot_export_chain.sh) uses GPU 0 and `/opt/conda/envs/waft/bin/python`; set `PYTHON` to use another interpreter. It builds for 640x480 images; set `RESOLUTION` (e.g. `RESOLUTION=1024x768`) for another size, which resizes the pairs and moves the build to `outputs/aoti/fp32_<RESOLUTION>/`. All four steps run under `--precision fp32`; pass `--precision tf32` or `--precision default` to the script to run them under another float32 mode. The build directory carries that mode in its name, `outputs/aoti/<precision>_<RESOLUTION>/`, although the package itself does not depend on it (see below). It works from any working directory, stops if a step fails, and replaces what earlier runs left at its output locations. Its four steps are also available as VS Code launch configurations:

| Step | Script | Writes |
|---|---|---|
| Export Dataset (.pt2) | `scripts/export_dataset.py` | `outputs/dataset_flow_realm_1_100m_fp32.pt2` (the name ends in the `--precision` the chain ran under, since the stored eager flow depends on it): the first 3 keyframe pairs of `custom/realm_1_100m_w_tandem`, resized to 640x480, as input tensors, plus the eager model's flow for them |
| AOT Compile Flow | `scripts/aot_compile_flow.py` | `outputs/aoti/fp32_640x480/waft_flow.pt2` and a `build_info.json` beside it (image size, model, torch version) |
| Run Flow Package | `scripts/run_flow_aoti.py` | `outputs/aoti_flow/`: raw flow, flow visualisation and uncertainty heatmap per pair; prints timing and the difference to the eager model |
| Evaluate Flow Package | `scripts/evaluate_flow_aoti.py` | prints the package's difference to the eager flow stored in the snapshot |

`scripts/export_flow_model.py` additionally saves the intermediate `torch.export` graph for inspection; nothing consumes it.

The package signature is `flow, info = model(image1, image2)`: two `[1, 3, H, W]` float32 RGB images in 0..255, giving the flow `[1, 2, H, W]` in pixels and the raw uncertainty head `[1, 4, H, W]` (`custom.get_heatmap` turns it into the heatmap). Things to know:

- **One package per image size.** H, W, the batch size of 1 and the number of refinement iterations are baked in. Compile for another size with `--resolution WIDTHxHEIGHT`; pairs can come from a keyframe sequence (`--data_root`) or a rosbag (`--bag`), as for the scripts above. Feeding a package another size is not a clean error, it fails inside cuDNN, so `run_flow_aoti.py` checks the size against `build_info.json` first.
- **Inputs must be contiguous.** The package reads its inputs with the memory layout of the tracing example, which these scripts fix to plain contiguous `[1, 3, H, W]`, and it does not check. A tensor of the same shape with other strides gives wrong flow without any error. Beware that `torch.from_numpy(hwc_image).permute(2, 0, 1).float()`, as the dataloaders here build their images, is such a tensor (channels-last strides); call `.contiguous()` on it. `run_flow_aoti.py` and `evaluate_flow_aoti.py` do, and refuse packages built before this was fixed.
- **One package per torch version.** A package is locked to the torch that compiled it; recompile after an upgrade.
- **Precision is chosen at run time, not at compile time.** The same package computes in whichever float32 mode the loading process is in: `--precision fp32` (default of these scripts), `tf32`, or `default` (torch's own defaults: TF32 convolutions, fp32 matmul, which is what `evaluate.py` runs under). Between modes the flow moves by about 0.01 px on average and by whole pixels in a few spots of large-motion pairs (this is the model's own sensitivity to TF32 rounding and applies to the eager pipeline just the same), so compare like with like; a C++ consumer sets the mode through `at::globalContext()`, see [utils/precision.py](utils/precision.py).
- **Accuracy.** Under fp32 the package reproduces the eager model it was compiled from to 2.5e-05 px mean / 1.4e-02 px max endpoint error (twelve 640x480 pairs, RTX 3060, torch 2.15.0). The full table, including the distance to what `evaluate.py` produces, is in the docstring of [scripts/aot_compile_flow.py](scripts/aot_compile_flow.py).
- **Speed.** The package is 10 to 12% faster than the eager model in every precision mode: 298 ms per pair against 336 ms under fp32. That relies on `--layout_optimization off`, the default of `aot_compile_flow.py`: inductor's own default of rewriting convolutions to channels-last costs this model 47 ms per pair under fp32 (345 ms, slower than eager) and gains nothing in the other modes. `--layout_optimization on` restores it; `build_info.json` records which one a package was built with.
- **Export-safe substitutions.** While exporting, xformers attention is replaced by plain attention and the cuDNN grid sampler by `aten.grid_sampler_2d` (see [scripts/flow_export_common.py](scripts/flow_export_common.py) and [utils/export_compat.py](utils/export_compat.py)). Both are switched on by environment variables that only the export scripts set; the normal pipeline is unchanged.

Only the a1 model (`config/a1/tar.json` with `weights/tar-c-t.pth`) has been compiled and checked. The a2 models with the DAv2 and Twins backbones get through `torch.export` with untrained weights, but have not been compiled or compared against a checkpoint; the DINOv3 backbone has not been tried.

## Acknowledgements

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [DPT](https://github.com/isl-org/DPT), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), and [DINOv3](https://ai.meta.com/dinov3/). We thank the original authors for their excellent work.
