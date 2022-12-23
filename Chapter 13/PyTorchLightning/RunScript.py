import torch
from MLP import MultiLayerPerceptron
from MNISTLoader import MnistDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


def main():
    torch.manual_seed(1)
    mnist_dm = MnistDataModule()
    mnistclassifier = MultiLayerPerceptron()

    # save top 1 model
    callbacks = [ModelCheckpoint(
        save_top_k=1, mode='max', monitor="valid_acc")]

    if torch.cuda.is_available():  # if you have GPUs
        trainer = pl.Trainer(
            max_epochs=10, callbacks=callbacks, accelerator='gpu')
    else:
        trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)

    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)
