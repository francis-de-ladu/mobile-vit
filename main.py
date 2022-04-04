import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.mobilevit import mobilevit
from torchvision import transforms as T


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


datamodule = CIFAR10DataModule(
    data_dir='../data',
    batch_size=2048,
    num_workers=12,
)

datamodule.train_transforms = T.Compose([
    T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
])


for kind in ('xxs', 'xs', 's'):
    model = mobilevit(
        image_size=datamodule.dims,
        num_classes=datamodule.num_classes,
        kind=kind,
    )

    logger = TensorBoardLogger('tb_logs')

    metric = 'Loss/Valid'
    callbacks = [
        EarlyStopping(metric, patience=50),
        ModelCheckpoint(
            filename=f'epoch={{epoch:03d}}-val_loss={{{metric}:.2f}}',
            monitor=metric,
            save_top_k=5,
            auto_insert_metric_name=False,
        ),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        gpus=-1,
        log_every_n_steps=10,
        # logger=logger,
        max_steps=10000,
        # overfit_batches=1,
        precision=16,
        weights_summary='top',
    )

    trainer.fit(model, datamodule=datamodule)
