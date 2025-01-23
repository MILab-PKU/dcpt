# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

import torch
from torch.autograd import Function
from torch.nn import functional as F

BASICSR_JIT = os.getenv("BASICSR_JIT")
if BASICSR_JIT == "True":
    from torch.utils.cpp_extension import load

    module_path = os.path.dirname(__file__)
    window_process = load(
        "window_process",
        sources=[
            os.path.join(module_path, "src", "window_process.cpp"),
            os.path.join(module_path, "src", "window_process_kernel.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
        ],
    )
else:
    try:
        from . import window_process
    except ImportError:
        pass
        # avoid annoying print output
        # print(f'Cannot import deform_conv_ext. Error: {error}. You may need to: \n '
        #       '1. compile with BASICSR_EXT=True. or\n '
        #       '2. set BASICSR_JIT=True during running')


class WindowProcess(Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = window_process.roll_and_window_partition_forward(
            input, B, H, W, C, shift_size, window_size
        )

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output.contiguous().requires_grad_(input.requires_grad)

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = window_process.roll_and_window_partition_backward(
            grad_in.contiguous(), B, H, W, C, shift_size, window_size
        )
        return grad_out.contiguous(), None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = window_process.window_merge_and_roll_forward(
            input, B, H, W, C, shift_size, window_size
        )

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output.contiguous().requires_grad_(input.requires_grad)

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        # grad_out = ctx.saved_tensors[0]
        # grad_out = torch.zeros((B, H, W, C), dtype=dtype).cuda()
        grad_out = window_process.window_merge_and_roll_backward(
            grad_in.contiguous(), B, H, W, C, shift_size, window_size
        )
        return grad_out.contiguous(), None, None, None, None, None, None, None
