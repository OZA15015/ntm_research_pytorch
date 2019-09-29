import torch

class Mysoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.exp() - x.max()
        result = y/y.sum()
        ctx.save_for_backward(result)
        return result
