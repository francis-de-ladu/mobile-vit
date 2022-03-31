import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from src import MobileViT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = MobileViT(
    image_size=(32, 32),
    dims=[],
    depths=[2, 4, 3],
    channels=[],
    num_classes=10,
    # expansion=4,
    # kernel_size=3,
    # patch_size=(2, 2),
)

cifar10 = CIFAR10DataModule(
    data_dir='../data',
    batch_size=2048,
    num_workers=12,
)


print('n_params:', count_parameters(model))
# print(model)

trainer = pl.Trainer(
    gpus=-1,
    # auto_scale_batch_size=True,
    # auto_lr_find=True,
    max_steps=1000,
    weights_summary='top',
)

trainer.tune(model, cifar10)

trainer.fit(model, datamodule=cifar10)
