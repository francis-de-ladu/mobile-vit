import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
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


model = mobilevit(
    image_size=datamodule.dims,
    num_classes=datamodule.num_classes,
    kind='s',
)


ckpt_fn_info = [
    'epoch={epoch:03d}',
    'val_acc={Accuracy/Valid:.3f}',
    'val_loss={Loss/Valid:.3f}',
]

metric = 'Accuracy/Valid'
mode = 'max'

callbacks = [
    EarlyStopping(metric, mode=mode, patience=100),
    ModelCheckpoint(
        filename='-'.join(ckpt_fn_info),
        auto_insert_metric_name=False,
        monitor=metric,
        mode=mode,
        save_last=True,
        save_top_k=5,
    ),
    LearningRateMonitor(logging_interval='step'),
]

trainer = pl.Trainer(
    callbacks=callbacks,
    gpus=-1,
    log_every_n_steps=10,
    max_epochs=-1,
    precision=16,
    weights_summary='top',
)

trainer.fit(model, datamodule=datamodule)
