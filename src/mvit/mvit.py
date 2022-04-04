import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchmetrics.functional import accuracy

from ..vit import Transformer
from .helpers import conv_1x1_bn, conv_nxn_bn


class MobileViT(pl.LightningModule):
    def __init__(self, image_size, num_classes, chs, dims, depths,
                 expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        self.conv1 = conv_nxn_bn(3, chs[0], kernel_size, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(chs[0], chs[1], 1, expansion))
        self.mv2.append(MV2Block(chs[1], chs[2], 2, expansion))
        self.mv2.append(MV2Block(chs[2], chs[3], 1, expansion))
        self.mv2.append(MV2Block(chs[3], chs[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(chs[3], chs[4], 2, expansion))
        self.mv2.append(MV2Block(chs[4], chs[5], 2, expansion))
        self.mv2.append(MV2Block(chs[5], chs[6], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MViTBlock(
            dims[0], depths[0], chs[4], kernel_size, patch_size, expansion=2))
        self.mvit.append(MViTBlock(
            dims[1], depths[1], chs[5], kernel_size, patch_size, expansion=4))
        self.mvit.append(MViTBlock(
            dims[2], depths[2], chs[6], kernel_size, patch_size, expansion=4))

        self.conv2 = conv_1x1_bn(chs[-2], chs[-1])

        self.pool = nn.AvgPool2d(image_size[0] // 32, stride=1)
        self.fc = nn.Linear(chs[-1], num_classes, bias=False)

        self.save_hyperparameters()
        self.hparams.lr = 2e-3

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[-3](x)
        x = self.mvit[0](x)

        x = self.mv2[-2](x)
        x = self.mvit[1](x)

        x = self.mv2[-1](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        acc = accuracy(logits.argmax(dim=1), y)
        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        acc = accuracy(logits.argmax(dim=1), y)
        self.log('val_loss', loss, on_epoch=True, reduce_fx=torch.mean)
        self.log('val_acc', acc, on_epoch=True, reduce_fx=torch.mean)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @property
    def lr(self):
        return self.hparams.lr

    @lr.setter
    def lr(self, lr):
        self.hparams.lr = lr


# class MV2MViTBlock(nn.Module):
#     def __init__(self, dim, depth, c_in, c_out, kernel_size, patch_size,
#                  expansion):
#         super().__init__()
#         self.mv2 = MV2Block(c_in, c_out, stride=2, expansion=expansion[0])
#         self.mvit = MViTBlock(
#             dim, depth, c_out, kernel_size, patch_size, expansion[1])
#
#     def forward(self, x):
#         x = self.mv2(x)
#         x = self.mvit(x)
#         return x


class MV2Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1, expansion=4):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride

        h_dim = int(c_in * expansion)
        self.use_res_connect = stride == 1 and c_in == c_out

        self.conv = nn.Sequential(
            conv_1x1_bn(c_in, h_dim),
            conv_nxn_bn(h_dim, h_dim, 3, stride=stride, groups=h_dim),
            conv_1x1_bn(h_dim, c_out, activation=False),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MViTBlock(nn.Module):
    def __init__(self, dim, depth, channels, kernel_size, patch_size,
                 expansion=4, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channels, channels, kernel_size)
        self.conv2 = conv_1x1_bn(channels, dim)

        self.transformer = Transformer(dim, depth, 4, 8, expansion, dropout)

        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size)

    def forward(self, x):
        # clone input tensor for concatenation in fusion step
        x_hat = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        ph, pw = min(h, self.ph), min(w, self.pw)
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=ph, pw=pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      ph=ph, pw=pw, h=(h // ph), w=(w // pw))

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, x_hat), axis=1)
        x = self.conv4(x)

        return x
