import pytorch_lightning as pl
from torch import nn

from .attention import Attention
from .helpers import FeedForward, PreNorm


class Transformer(pl.LightningModule):
    def __init__(self, dim, depth, heads, expansion, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout)),
                PreNorm(dim, FeedForward(dim, expansion, dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
