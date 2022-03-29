import pytorch_lightning as pl
from torch import nn


class PreNorm(pl.LightningModule):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim, 4, 4])
        self.fn = fn

    def forward(self, x, **kwargs):
        print(x.shape)
        x = self.norm(x)
        print(x.shape)
        x = self.fn(x, **kwargs)
        print(x.shape)
        return x
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, expansion, dropout=0.):
        super().__init__(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
