"""Sparse logical embedding gradients using the vocabulary head's transpose."""
import weakref

import torch
from torch import nn
from torch.autograd import Function


class _TiedEmbedding(Function):
    @staticmethod
    def forward(ctx, context, token_ids, tile, chunk_size):
        ids = token_ids.reshape(-1)
        outputs = []
        for chunk in ids.split(chunk_size):
            one_hot = torch.zeros(chunk.numel(), tile.out_size,
                                  device=ids.device, dtype=context.dtype)
            one_hot.scatter_(1, chunk[:, None], 1)
            outputs.append(tile.backward(one_hot, ctx=None))
        ctx.context = context
        ctx.vocab_size = tile.out_size
        ctx.width = tile.in_size
        ctx.save_for_backward(ids)
        return torch.cat(outputs).reshape(*token_ids.shape, tile.in_size)

    @staticmethod
    def backward(ctx, gradients):
        ids, = ctx.saved_tensors
        gradient = gradients.new_zeros(ctx.vocab_size, ctx.width)
        gradient.index_add_(0, ids, gradients.reshape(-1, ctx.width))
        if ctx.context.grad is None:
            ctx.context.grad = gradient
        else:
            ctx.context.grad.add_(gradient)
        return None, None, None, None


class TiedAnalogEmbedding(nn.Module):
    def __init__(self, head, read_chunk_size=128):
        super().__init__()
        if read_chunk_size <= 0:
            raise ValueError('read_chunk_size must be positive.')
        self._head = weakref.ref(head)
        self.read_chunk_size = read_chunk_size
        self.num_embeddings = head.out_features
        self.embedding_dim = head.in_features

    @property
    def analog_linear(self):
        head = self._head()
        if head is None:
            raise RuntimeError('The tied vocabulary head no longer exists.')
        return head.analog_linear

    def forward(self, token_ids):
        tile = self.analog_linear.analog_module
        return _TiedEmbedding.apply(tile.analog_ctx, token_ids, tile, self.read_chunk_size)
