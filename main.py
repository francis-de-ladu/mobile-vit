import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from src.mobilevit import mobilevit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


cifar10 = CIFAR10DataModule(
    data_dir='../data',
    batch_size=2048,
    num_workers=12,
)

# model = mobilevit(image_size=(32, 32), num_classes=10, kind='xxs')
model = mobilevit(image_size=(256, 256), num_classes=1000, kind='s')

logger = TensorBoardLogger('tb_logs')

print('n_params:', count_parameters(model))
# print(model)


trainer = pl.Trainer(
    gpus=-1,
    log_every_n_steps=10,
    # logger=logger,
    max_steps=10000,
    # overfit_batches=1,
    precision=16,
    weights_summary='top',
)

trainer.fit(model, datamodule=cifar10)
