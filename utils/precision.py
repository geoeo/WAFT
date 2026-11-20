"""Single definition of the arithmetic mode used by the scripts of this repository.

On an Ampere-or-newer GPU there are three float32 modes, and they give measurably
different flow -- about 0.01 px on average and whole pixels in a few spots, because the
refinement iterations amplify the rounding:

  * "fp32":    full fp32 everywhere.
  * "tf32":    TF32 in cuDNN convolutions and in cuBLAS matmul. Fastest.
  * "default": what torch does when nothing is set -- TF32 in cuDNN convolutions, fp32
               matmul. This is the mode evaluate.py, custom.py and
               dataloader/flow/openrealm.py run under, and the mode of a libtorch
               process that sets nothing.

The mode is a property of the *process that runs the model*, not of a compiled
AOTInductor package: the package calls cuDNN and cuBLAS through the same runtime as
eager does, and they read these flags at call time. Verified on torch 2.15.0+cu134 --
a package compiled under fp32 and one compiled under tf32 hold a byte-identical .so and
kernels, and give bit-identical flow under either runtime mode. So a comparison is only
meaningful when both sides ran under the same mode, and a C++ consumer has to pick its
mode itself:

    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setFloat32MatmulPrecision("highest");

`set_float32_matmul_precision` alone does not cover cuDNN, whose `allow_tf32` defaults
to True, so every flag is set explicitly here and neither explicit mode depends on a
torch default. scripts/aot_compile_flow.py carries the measured differences.

A fourth mode, "fp16", runs the network under `torch.autocast(dtype=float16)` with TF32
allowed for what autocast keeps in float32; inputs and outputs stay float32. For an
eager model the call has to be wrapped in autocast(mode). For a compiled package it is
part of the build: a package exported under fp16 computes in float16 whatever the
loading process sets, and only the TF32 flags still apply at run time.
"""

import torch

MODES = ("fp32", "tf32", "default", "fp16")


def set_precision(mode: str) -> None:
    """Put this process into one of MODES. "default" leaves torch's flags untouched."""
    if mode == "fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode in ("tf32", "fp16"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif mode != "default":
        raise ValueError(f"unknown precision mode {mode!r}, expected one of {MODES}")


def is_half(mode: str | None) -> bool:
    """Whether `mode` computes the network in float16."""
    return mode == "fp16"


def autocast(mode: str | None, device_type: str = "cuda"):
    """Context manager for one eager model call under `mode`: float16 autocast for
    "fp16", nothing otherwise. The caller casts the outputs back with `.float()`."""
    if is_half(mode) and device_type != "cuda":
        raise ValueError("fp16 needs a CUDA device")
    return torch.autocast(device_type, dtype=torch.float16, enabled=is_half(mode))


def describe_precision() -> str:
    return (
        f"matmul={torch.get_float32_matmul_precision()} "
        f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
    )
