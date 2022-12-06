import argparse
import logging
from typing import List, Any # noqa
from runpy import run_path

import timm
import torch
from src.base_config import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID # noqa
from src.dataset import get_class_names, get_loaders

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, RichProgressBar # noqa
from pytorch_lightning import loggers # noqa
from pytorch_lightning.utilities.model_summary import ModelSummary # noqa
from pytorch_lightning.loggers import CSVLogger # noqa
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger # noqa
from pytorch_lightning.callbacks import ModelPruning # noqa

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from torchmetrics import MetricCollection # noqa

from clearml import Task, Logger # noqa

from src.utils import set_global_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config):

    pl.seed_everything(config.seed)

    task = Task.init(project_name=config.project_name, task_name=config.experiment_name)
    task.connect(config.to_dict())
    clearml_logger = task.get_logger()

    loaders = get_loaders(config)
    class_names = get_class_names(config)

    model = timm.create_model(num_classes=len(class_names), **config.model_kwargs)
    if config.checkpoint_name is not None:
        model.load_state_dict(torch.load(config.checkpoint_name))

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    if config.scheduler is not None:
        scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    else:
        scheduler = None

    callbacks = [
        ModelCheckpoint(dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}',
                        monitor=config.valid_metric, verbose=False, save_last=None,
                        save_top_k=10, save_weights_only=False, mode='min' if config.minimize_metric else 'max'),
        # EarlyStopping(monitor='val_loss', patience=10),
        # TQDMProgressBar(refresh_rate=1),
        RichProgressBar(leave=False),
        # ModelPruning("l1_unstructured", amount=0.5),
    ]
    callbacks.extend(config.callbacks)

    class DataModule(pl.LightningDataModule):
        def __init__(self, loaders):
            super(DataModule, self).__init__()
            self.loaders = loaders

        def train_dataloader(self):
            return self.loaders['train']

        def val_dataloader(self):
            return self.loaders['valid']

        def test_dataloader(self):
            return self.loaders['infer']

    class Model(pl.LightningModule):
        def __init__(self, model, loss, class_names):
            super(Model, self).__init__()

            self.save_hyperparameters()
            self.model = model
            self.class_names = class_names
            self.loss = loss
            self.train_auroc = MultilabelAUROC(num_labels=len(class_names))
            self.val_auroc = MultilabelAUROC(num_labels=len(class_names))

            self.train_f1 = MultilabelF1Score(num_labels=len(class_names))
            self.val_f1 = MultilabelF1Score(num_labels=len(class_names))

        def training_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()

            loss = self.loss(output, labels)
            self.log("train_loss", loss, prog_bar=True, logger=True)
            clearml_logger.report_scalar("train", "loss", iteration=batch_idx, value=loss)

            self.train_auroc(output, labels)
            self.log('train_auroc', self.train_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            self.train_f1(output, labels)
            self.log('train_f1', self.train_f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            # clearml_logger.report_scalar("train", "auroc", iteration=batch_idx, value=self.train_acc)

            return {"loss": loss, "predictions": output, "labels": labels}

        def validation_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()

            loss = self.loss(output, labels)
            self.log("val_loss", loss, prog_bar=True, logger=True)
            clearml_logger.report_scalar("val", "loss", iteration=batch_idx, value=loss)

            self.val_auroc(output, labels)
            self.log('val_auroc', self.val_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            self.val_f1(output, labels)
            self.log('val_f1', self.val_f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)
#             clearml_logger.report_scalar("val", "auroc", iteration=batch_idx, value=self.val_acc)

            return loss

        def test_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()

            loss = self.loss(output, labels)
            self.log("test_loss", loss, prog_bar=True, logger=True)
#             clearml_logger.report_scalar("test", "loss", iteration=batch_idx, value=loss)

            return loss

        # def training_epoch_end(self, outputs):
        #     labels = []
        #     predictions = []

        #     for output in outputs:
        #         for out_labels in output["labels"].detach().cpu():
        #             labels.append(out_labels)

        #     for output in outputs:
        #         for out_preds in output["predictions"].detach().cpu():
        #             predictions.append(out_preds)

        #     labels = torch.stack(labels)
        #     predictions = torch.stack(predictions)

        #     for i, name in enumerate(LABEL_COLUMNS):
        #         roc_score = auroc(predictions[:, i], labels[:, i])
        #         self.logger.experiment.add_scalar(
        #             f"{name}_roc_auc/Train", roc_score, self.current_epoch
        #         )

        def configure_optimizers(self):

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }

    data = DataModule(loaders)
    model_pl = Model(model, config.loss, class_names)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=callbacks,
        # logger=csv_logger,
    )

    # train
    trainer.fit(model_pl, data)

    # evaluate
    trainer.test(model_pl, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)
