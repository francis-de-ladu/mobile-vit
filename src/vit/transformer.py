import pytorch_lightning as pl
from einops import rearrange
from torch import einsum, nn


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


class PreNorm(pl.LightningModule):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class Attention(pl.LightningModule):
    def __init__(self, d_model, n_heads, dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scaling = (d_model // n_heads)**(-0.5)

        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # print("x", x.shape)
        qkv = self.to_qkv(x)
        # print("qkv", qkv.shape)
        attn = self.attention(qkv)
        # print("attn", attn.shape)
        out = self.to_out(attn)
        # print("out", out.shape)
        return out

    def attention(self, qkv):
        q, k, v = qkv.chunk(3, dim=-1)
        scores = einsum('bhqd, bhkd -> bhqk', q, k) * self.scaling
        attn = einsum('bhad, bhdv -> bhav', self.softmax(scores), v)
        return attn
        return rearrange(attn, 'b h n d -> b n (h d)')


class FeedForward(nn.Sequential):
    def __init__(self, dim, expansion, dropout=0.):
        super().__init__(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
