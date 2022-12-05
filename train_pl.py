import argparse
import logging
from typing import List, Any
from runpy import run_path

import timm
import torch
# from catalyst import dl
# from catalyst.callbacks import Callback
from src.base_config import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID
from src.dataset import get_class_names, get_loaders
# from src.loggers import ClearMLLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, RichProgressBar
from pytorch_lightning import loggers
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelPruning

from clearml import Task, Logger

from src.utils import set_global_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


# def get_base_callbacks(config: Config, class_names: List[str]) -> List[Callback]:
#     return [
#         dl.BatchTransformCallback(
#             transform=torch.sigmoid,
#             scope="on_batch_end",
#             input_key=LOGITS,
#             output_key=SCORES,
#         ),
#         dl.BatchTransformCallback(
#             transform=lambda x: x > config.binary_thresh,
#             scope="on_batch_end",
#             input_key=SCORES,
#             output_key=PREDICTS,
#         ),
#         dl.AUCCallback(input_key=SCORES, target_key=TARGETS, compute_per_class_metrics=True),
#         dl.MultilabelPrecisionRecallF1SupportCallback(
#             input_key=PREDICTS,
#             target_key=TARGETS,
#             num_classes=len(class_names),
#             log_on_batch=False,
#             compute_per_class_metrics=True,
#         ),
#     ]


# def get_train_callbacks(config: Config, class_names: List[str]) -> List[Callback]:
#     callbacks = get_base_callbacks(config, class_names)
#     callbacks.extend(
#         [
#             dl.CheckpointCallback(
#                 logdir=config.checkpoints_dir,
#                 loader_key=VALID,
#                 topk=10,
#                 metric_key=config.valid_metric,
#                 minimize=config.minimize_metric,
#             ),
#         ]
#     )
#     if config.scheduler is not None:
#         callbacks.append(
#             dl.SchedulerCallback(
#                 mode='batch',
#                 loader_key=VALID,
#                 metric_key=config.valid_metric,
#             ),
#         )
#     return callbacks


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

    # clearml_logger = ClearMLLogger(config, class_names)

    # callbacks = get_train_callbacks(config, class_names)
    # callbacks.extend(config.callbacks)
    callbacks = [
        ModelCheckpoint(dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}', 
                    monitor='val_loss', verbose=False, save_last=None, 
                    save_top_k=10, save_weights_only=False, mode='min' if config.minimize_metric else 'max'),
        # EarlyStopping(monitor='val_loss', patience=10),
    #     TQDMProgressBar(refresh_rate=1),
        RichProgressBar(leave=False),
    #     ModelPruning("l1_unstructured", amount=0.5),
    ]

    class DataModule(pl.LightningDataModule):
        def __init__(self, loaders):
            super(DataModule, self).__init__()
            self.loaders = loaders

        def train_dataloader(self):
            return self.loaders['train']

        def val_dataloader(self):
            return self.loaders['valid']

        def test_dataloader(self):
            return self.loaders["infer"]

    class Model(pl.LightningModule):
        def __init__(self, model, loss):
            super(Model, self).__init__()
            
            self.save_hyperparameters()
            self.model = model
            self.loss = loss

        def training_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()
            loss = self.loss(output, labels)
            self.log("train_loss", loss, prog_bar=True, logger=True)
            clearml_logger.report_scalar("train", "loss", iteration=batch_idx, value=loss)
            return {"loss": loss, "predictions": output, "labels": labels}

        def validation_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()
            loss = self.loss(output, labels)
            self.log("val_loss", loss, prog_bar=True, logger=True)
            clearml_logger.report_scalar("val", "loss", iteration=batch_idx, value=loss)
            return loss

        def test_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]

            output = self.model(images)
            output = torch.sigmoid(output)
            # output = (output > config.binary_thresh).float()
            loss = self.loss(output, labels)
            self.log("test_loss", loss, prog_bar=True, logger=True)
            clearml_logger.report_scalar("test", "loss", iteration=batch_idx, value=loss)
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
    model_pl = Model(model, config.loss)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=callbacks,
        # logger=csv_logger,
    )

    trainer.fit(model_pl, data)
    # trainer.fit(
    #     model=model_pl, 
    #     train_dataloaders=train_loader, 
    #     val_dataloaders=val_loader,
    # )

    trainer.test(model_pl, data)

    # runner.train(
    #     model=model,
    #     criterion=config.loss,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     loaders=loaders,
    #     callbacks=callbacks,
    #     loggers={
    #         "_clearml": clearml_logger,
    #         "_tensorboard": TensorboardLogger(config.checkpoints_dir, log_batch_metrics=True),
    #     },
    #     num_epochs=config.n_epochs,
    #     valid_loader=VALID,
    #     valid_metric=config.valid_metric,
    #     minimize_valid_metric=config.minimize_metric,
    #     seed=config.seed,
    #     verbose=True,
    #     load_best_on_end=True,
    # )

    # metrics = runner.evaluate_loader(
    #     model=model,
    #     loader=loaders["infer"],
    #     callbacks=get_base_callbacks(config, class_names),
    #     verbose=True,
    #     seed=config.seed,
    # )

    # clearml_logger.log_metrics(metrics, scope="loader", runner=runner, infer=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)
