"""Wrap a cost-volume CNN for the generic fp16 path: prepare_generic(model), fp32 islands for the probability /
regression math (softmax upcast through a proxy of the model modules' `F`), fp16 channels-last inputs, fp32 outputs."""
import types

import torch
import torch.nn.functional as _F

from .generic import prepare_generic


class _FProxy(types.ModuleType):
    """torch.nn.functional with softmax / log_softmax computed in fp32 (disparity probabilities + regression: fp16
    would quantize disparities to 0.06-0.125 px at 64-192 px)."""
    def __init__(self):
        super().__init__("F_fp32_softmax")

    def __getattr__(self, n):
        return getattr(_F, n)

    @staticmethod
    def softmax(x, dim=None, *a, **k):
        return _F.softmax(x.float(), dim, *a, **k)

    @staticmethod
    def grid_sample(input, grid, *a, **k):  # sampling in fp32 (fp16 grids are too coarse; mixed dtypes are an error)
        if input.dtype != grid.dtype or input.dtype == torch.float16:
            return _F.grid_sample(input.float(), grid.float(), *a, **k).to(input.dtype)
        return _F.grid_sample(input, grid, *a, **k)

    @staticmethod
    def log_softmax(x, dim=None, *a, **k):
        return _F.log_softmax(x.float(), dim, *a, **k)


FP = _FProxy()


def _cast_in(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dim() == 4:
        return x.half().contiguous(memory_format=torch.channels_last)
    return x


def _cast_out(y):
    if isinstance(y, torch.Tensor):
        return y.float()
    if isinstance(y, (list, tuple)):
        return type(y)(_cast_out(v) for v in y)
    if isinstance(y, dict):
        return {k: _cast_out(v) for k, v in y.items()}
    return y


def _softmax_module_forward(self, x):
    return _F.softmax(x.float(), self.dim)


def prepare_wrapped(model, repo_root=None, py_modules=()):
    """All python modules loaded from repo_root (and py_modules) that hold `F` = torch.nn.functional get the fp32-softmax
    proxy; nn.Softmax modules compute in fp32."""
    import os
    import sys
    mods = list(py_modules)
    if repo_root:
        root = os.path.abspath(repo_root) + os.sep
        mods += [m for m in list(sys.modules.values()) if (getattr(m, "__file__", None) or "").startswith(root)]
    for pm in mods:
        if getattr(pm, "F", None) is _F:
            pm.F = FP
    for m in model.modules():
        if isinstance(m, torch.nn.Softmax):
            m.forward = _softmax_module_forward.__get__(m)
    n = prepare_generic(model)
    fwd = model.forward

    def forward(*args, **kw):
        return _cast_out(fwd(*[_cast_in(a) for a in args], **{k: _cast_in(v) for k, v in kw.items()}))
    model.forward = forward
    return n
