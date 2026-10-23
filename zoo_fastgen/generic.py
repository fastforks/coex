"""Generic inference preparation for CNN stereo models (zoo round 2, any input size, no per-size compile / tune):

prepare_generic(model, half_acc=True, fused=True):
  - nn.Sequential runs conv -> BatchNorm [-> ReLU / LeakyReLU(0.01) / ReLU6 / SiLU / Sigmoid / Tanh] become one conv with a
    fused epilogue (BN kept as an fp32 per-channel affine, not folded into fp16 weights); other BatchNorms run as an
    in-place affine kernel on NHWC fp16; every Conv/ConvTranspose 2d/3d runs through conv_ep (fp16-acc cuDNN precompiled
    engines or CUTLASS fused-epilogue kernels for >= 128 outputs); weights fp16 channels-last.
  - modules with attributes (conv, bn) and a known activation can be registered with fuse_conv_bn_module().
Model code must feed fp16 channels-last inputs and keep disparity regression / softmax in fp32 itself.
"""
import torch
import torch.nn as nn

from .inference import conv_ep, epilogue, _bn_st, _bn_module_forward, _conv_forward, _in_module_forward, FUSED, ACT

_ACT = {nn.ReLU: "relu", nn.ReLU6: "relu6", nn.SiLU: "silu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}
CONVS = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
BNS = (nn.BatchNorm2d, nn.BatchNorm3d)


def act_code(m):
    if m is None:
        return 0
    if isinstance(m, nn.LeakyReLU):
        return ACT["leaky"] if abs(m.negative_slope - 0.01) < 1e-12 else None
    for t, n in _ACT.items():
        if type(m) is t:
            return ACT.get(n)
    return None


class FusedConvBN(nn.Module):
    """conv -> BN -> act as conv_ep(conv, x, s, t, act) (same math; BN affine in fp32 in the epilogue)."""
    def __init__(self, conv, bn, act):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.act = act
        self.s, self.t = (None, None) if bn is None else _bn_st(bn, conv)

    def forward(self, x):
        return conv_ep(self.conv, x, self.s, self.t, self.act)


def _fuse_sequentials(model):
    n = 0
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        ch = list(seq._modules.items())
        i = 0
        while i < len(ch):
            name, m = ch[i]
            if isinstance(m, CONVS) and m.padding_mode == "zeros" and i + 1 < len(ch) and type(ch[i + 1][1]) in BNS:  # plain BN only (BatchNormAct has its own act)
                act, j = 0, i + 2
                if j < len(ch):
                    a = act_code(ch[j][1])
                    if a is not None and a != 0 and a <= 6:
                        act, j = a, j + 1
                seq._modules[name] = FusedConvBN(m, ch[i + 1][1], act)
                for k in range(i + 1, j):
                    seq._modules[ch[k][0]] = nn.Identity()
                n += 1
                i = j
            else:
                i += 1
    return n


def _bn_act_forward(self, x):
    y = _bn_module_forward(self, x)
    post = getattr(self, "_post", None)
    return post(y) if post is not None else y


def prepare_generic(model, half_acc=True, fused=True):
    model.eval()
    nf = _fuse_sequentials(model)
    for m in model.modules():
        if isinstance(m, BNS) and not hasattr(m, "_s"):
            m._s, m._t = _bn_st(m, None)
            a = getattr(m, "act", None)  # timm BatchNormAct2d: activation inside the BN module
            code = 0 if a is None or isinstance(a, nn.Identity) else act_code(a)
            if code is None or code > 5:  # not in the epilogue kernel (e.g. SiLU): affine kernel + torch activation
                m._post = a
                code = 0
            m._act = code
            m.forward = _bn_act_forward.__get__(m)
        elif type(m) is nn.InstanceNorm2d and not m.affine and not m.track_running_stats:
            m.forward = _in_module_forward.__get__(m)  # layout-preserving (torch's returns NCHW for NHWC input)
    model.half()
    for m in model.modules():
        if isinstance(m, CONVS):
            m._bias32 = None if m.bias is None else m.bias.float().contiguous()
            mf = torch.channels_last if m.weight.dim() == 4 else torch.channels_last_3d
            m.weight.data = m.weight.data.contiguous(memory_format=mf)
    if half_acc:
        from .cudnn_conv import patch_convs
        patch_convs(model, mark_only=True)
    for m in model.modules():
        if type(m) in CONVS and m.padding_mode == "zeros":  # conv_ep ignores padding_mode
            m.forward = _conv_forward.__get__(m)
            m._fuse = fused and FUSED and type(m) is nn.Conv2d and m.groups == 1 and tuple(m.dilation) == (1, 1) and \
                m.stride[0] == m.stride[1] and m.weight.shape[0] >= 128 and m.weight.shape[0] % 8 == 0 and \
                m.weight.shape[1] % 8 == 0 and isinstance(m.padding, tuple) and m.padding_mode == "zeros"
    for m in model.modules():  # BN modules inside FusedConvBN keep fp32 affine computed before the cast
        if isinstance(m, FusedConvBN) and m.s is not None:
            m.s, m.t = m.s.cuda(), m.t.cuda()
    model.inf_dtype, model.inf_cl, model.inf_kernels = torch.float16, True, True
    return nf
