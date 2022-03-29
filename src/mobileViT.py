import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .helpers import conv_1x1_bn, conv_nxn_bn
from .vit import Transformer


class MobileViT(pl.LightningModule):
    def __init__(self, image_size, dims, depths, channels, num_classes,
                 expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        self.conv1 = conv_nxn_bn(1, 16, 3, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(16, 16, stride=1))
        self.mv2.append(MV2Block(16, 24, stride=2))
        self.mv2.append(MV2Block(24, 24, stride=1))
        self.mv2.append(MV2Block(24, 24, stride=1))  # Repeat

        self.mv2mvit = nn.ModuleList([])
        self.mv2mvit.append(MV2MViTBlock(24, 48, kernel_size, dim=64, depth=2,
                                         patch_size=patch_size, expansion=expansion))
        self.mv2mvit.append(MV2MViTBlock(48, 64, kernel_size, dim=80, depth=4,
                                         patch_size=patch_size, expansion=expansion))
        self.mv2mvit.append(MV2MViTBlock(64, 80, kernel_size, dim=96, depth=3,
                                         patch_size=patch_size, expansion=expansion))

        self.conv2 = conv_1x1_bn(80, 320)

        self.pool = nn.AvgPool2d(image_size[0] // 32, stride=1)
        self.fc = nn.Linear(320, num_classes, bias=False)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)

        for mv2_block in self.mv2:
            x = mv2_block(x)

        for mv2mvit_block in self.mv2mvit:
            x = mv2mvit_block(x)

        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.shape[0], -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3)
        return optimizer


class MV2MViTBlock(pl.LightningModule):
    def __init__(self, c_in, c_out, kernel_size, dim, depth, patch_size, expansion):
        super().__init__()
        self.mv2 = MV2Block(c_in, c_out, stride=2)
        self.mvit = MobileViTBlock(
            dim, depth, c_out, kernel_size, patch_size, expansion)

    def forward(self, x):
        x = self.mv2(x)
        x = self.mvit(x)
        return x


class MV2Block(pl.LightningModule):
    def __init__(self, c_in, c_out, *, stride=1, expansion=4):
        super().__init__()

        assert stride in (1, 2)
        self.stride = stride

        h_dim = int(c_in * expansion)

        self.conv = nn.Sequential(
            conv_1x1_bn(c_in, h_dim),
            conv_nxn_bn(h_dim, h_dim, 3, stride=stride, groups=h_dim),
            conv_1x1_bn(h_dim, c_out),
        )

    def forward(self, x):
        if self.stride == 1:
            return self.conv(x)
        else:
            return x + self.conv(x)


class MobileViTBlock(pl.LightningModule):
    def __init__(self, dim, depth, channels, kernel_size, patch_size, expansion,
                 dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channels, channels, kernel_size)
        self.conv2 = conv_1x1_bn(channels, dim)

        self.transformer = Transformer(dim, depth, 4, expansion, dropout)

        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size)

    def forward(self, x):
        # clone input tensor for concatenation in the fusion step
        x_hat = x.clone()

        # Local representation
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representation
        x = self.transformer(x)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, x_hat), axis=1)
        x = self.conv4(x)

        return x
