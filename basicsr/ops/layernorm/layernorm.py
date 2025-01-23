import os

import torch
from torch import nn
from torch.autograd import Function

BASICSR_JIT = os.getenv("BASICSR_JIT")
if BASICSR_JIT == "True":
    from torch.utils.cpp_extension import load

    module_path = os.path.dirname(__file__)
    layernorm_ext = load(
        "layernorm",
        extra_cflags=["-O3"],
        sources=[
            os.path.join(module_path, "src", "layernorm_kernel.cpp"),
            # os.path.join(module_path, "src", "layernorm_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
    )
else:
    try:
        from . import layernorm_ext
    except ImportError:
        pass
        # avoid annoying print output
        # print(f'Cannot import deform_conv_ext. Error: {error}. You may need to: \n '
        #       '1. compile with BASICSR_EXT=True. or\n '
        #       '2. set BASICSR_JIT=True during running')


class LayerNormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        ctx.eps = eps
        output, y, var_sqrt = layernorm_ext.forward(input, weight, bias, eps)
        ctx.save_for_backward(weight, y, var_sqrt)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight_, y, var_sqrt = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = layernorm_ext.backward(
            grad_output.contiguous(), y, var_sqrt, weight_
        )
        return (
            grad_input,
            grad_weight.sum(dim=[3, 2, 0], keepdim=True),
            grad_bias.sum(dim=[3, 2, 0], keepdim=True),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(1, channels, 1, 1)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(1, channels, 1, 1)))
        self.eps = eps

    def forward(self, input):
        return layernorm(input, self.weight, self.bias, self.eps)


def layernorm(input, weight, bias, eps):
    return LayerNormFunction.apply(input, weight, bias, eps)
