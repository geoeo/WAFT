"""Op substitutions that keep the flow network exportable.

An operation that is correct in eager but does not survive `torch.export()` +
AOTInductor cleanly is swapped for an equivalent that does. The switch is read from the
environment at import time, so `scripts/flow_export_common.py` sets it before anything
under `model` is imported, and the normal Python pipeline is untouched.

`AOTI_SAFE_GRID_SAMPLE` exists because of the warp. `F.grid_sample` on a 4-D CUDA
tensor, bilinear with zero padding, dispatches to cuDNN, and export records that choice:
the graph holds `aten.cudnn_grid_sampler`. That op has no c-shim, so AOTInductor cannot
call it from generated code and routes it through the proxy executor instead:

    aten.cudnn_grid_sampler.default is missing a c-shim implementation,
    using proxy executor as fallback

once per refinement iteration. On torch 2.15.0+cu134 that fallback does compute the
right thing when the package is run from Python. It is still the one op in the graph
that is not compiled: it goes back through the dispatcher by name with arguments
deserialized from JSON at load time, and so depends on runtime machinery that the rest
of the package does not need.

`aten.grid_sampler_2d` is the kernel dispatch picks once cuDNN is out of the way --
calling it directly is the same as wrapping `F.grid_sample` in
`torch.backends.cudnn.flags(enabled=False)`, without a context manager in the traced
region. Inductor has a decomposition for it, so the warp becomes generated code like
everything around it and the package has no proxy-executor call left.

It is the same bilinear interpolation, not an approximation, but it is a different
kernel from cuDNN's, so the two agree to rounding only and the refinement iterations
amplify that: 7e-06 px mean / 2.5e-04 px max on the flow of three ordinary pairs, more
on large-motion ones. `scripts/aot_compile_flow.py` quotes the gap of both substitutions
together over a whole sequence.

Recheck on the next torch upgrade: if `cudnn_grid_sampler` gains a c-shim (the list is
`torchgen/aoti/fallback_ops.py`), the compile stops printing the warning above with the
switch unset, and the substitution can be dropped.
"""

import os

import torch
import torch.nn.functional as F

# Presence of the variable enables the substitution; unset means F.grid_sample (the
# default, for the eager pipeline).
SAFE_GRID_SAMPLE = os.environ.get("AOTI_SAFE_GRID_SAMPLE") is not None

_BILINEAR = 0  # aten interpolation_mode
_ZEROS = 0     # aten padding_mode


def grid_sample(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """`F.grid_sample(img, grid, align_corners=True)` -- bilinear, zero padding -- or its
    cuDNN-free equivalent when AOTI_SAFE_GRID_SAMPLE is set.

    `img` is [N, C, H, W], `grid` is [N, H_out, W_out, 2] in normalised coordinates.
    """
    if not SAFE_GRID_SAMPLE:
        return F.grid_sample(img, grid, align_corners=True)
    if img.dtype != grid.dtype:
        # Under float16 autocast the feature map is float16 and the sampling grid
        # float32. F.grid_sample is on autocast's float32 list and casts both; a direct
        # aten.grid_sampler_2d call is not and raises "expected scalar type Half but
        # found Float". Do here what autocast does. With equal dtypes, i.e. in every
        # float32 mode, nothing is added to the graph.
        img, grid = img.float(), grid.float()
    return torch.ops.aten.grid_sampler_2d(img, grid, _BILINEAR, _ZEROS, True)
