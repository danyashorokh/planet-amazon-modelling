import argparse
import logging
from typing import List, Any # noqa
from runpy import run_path

import pandas as pd
import timm
import torch

from src.base_config_pl import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID, LOSS # noqa
from src.dataset import get_class_names, get_loaders

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor # noqa
from pytorch_lightning.callbacks import ModelPruning # noqa
from pytorch_lightning import loggers # noqa

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from torchmetrics import MetricCollection

from clearml import Task, Logger # noqa

from src.utils import set_global_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):

    pl.seed_everything(config.seed)

    # init logger
    task = Task.init(project_name=config.project_name, task_name=config.experiment_name)
    task.connect(config.to_dict())
    clearml_logger = task.get_logger()

    # get data and class names
    loaders = get_loaders(config)
    class_names = get_class_names(config)

    # get model
    model = timm.create_model(num_classes=len(class_names), **config.model_kwargs)
    if config.checkpoint_name is not None:
        model.load_state_dict(torch.load(config.checkpoint_name))

    # get optimizer and scheduler
    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    if config.scheduler is not None:
        scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    else:
        scheduler = None

    # init metrics
    metrics = MetricCollection(
        metrics=[
            MultilabelAUROC(num_labels=len(class_names), average=None),
            MultilabelF1Score(num_labels=len(class_names), average=None),
        ]
    ),

    # init callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}_{val_f1:.2f}',
        monitor=config.valid_metric, verbose=False, save_last=None,
        save_top_k=10, save_weights_only=True, mode='min' if config.minimize_metric else 'max')
    model_checkpoint.FILE_EXTENSION = '.pth'
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    callbacks = [
        model_checkpoint,
        early_stopping,
        lr_monitor,
        RichProgressBar(leave=False),
        # ModelPruning('l1_unstructured', amount=0.5),
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

    class TrainModule(pl.LightningModule):
        def __init__(self,
                     model, loss, class_names, optimizer, scheduler, warmup_iter, class_threshold):
            super(TrainModule, self).__init__()

            self.save_hyperparameters()
            self.model = model
            self.loss = loss
            self.class_names = class_names
            self.num_classes = len(class_names)
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.warmup_iter = warmup_iter
            self.class_threshold = class_threshold
            # for warmupup
            self.init_lr = self.optimizer.param_groups[0]['lr']
            # metrics
            self.train_auroc = MultilabelAUROC(num_labels=self.num_classes, average=None)
            self.train_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None)
            self.val_auroc = MultilabelAUROC(num_labels=self.num_classes, average=None)
            self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None)
            self.test_auroc = MultilabelAUROC(num_labels=self.num_classes, average=None)
            self.test_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None)

        def forward(self, x: torch.Tensor):
            """Get output from model.

            Args:
                x: torch.Tensor - batch of images.

            Returns:
                output: torch.Tensor - predicted logits.
            """
            return self.model(x)

        def training_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]
            logits = self(images)
            scores = torch.sigmoid(logits)

            loss = self.loss(logits, labels)
            self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True)
            clearml_logger.report_scalar('loss', 'train', iteration=batch_idx, value=loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.log('LR', current_lr, prog_bar=True, logger=True, on_step=True)

            return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

        def training_epoch_end(self, training_step_outputs):

            labels = []
            scores = []
            for output in training_step_outputs:
                for out_labels in output[TARGETS].cpu():
                    labels.append(out_labels)
                for out_predictions in output[SCORES].cpu():
                    scores.append(out_predictions)
            scores = torch.stack(scores)
            labels = torch.stack(labels)
            predictions = (scores > self.class_threshold).to(torch.int32)

            train_aurocs = self.train_auroc(scores, labels).cpu()
            train_f1 = self.train_f1(predictions, labels).cpu()

            self.log('train_auroc', float(train_aurocs.mean()), prog_bar=True, logger=True)
            clearml_logger.report_scalar('auroc_macro', 'train', iteration=self.current_epoch,
                                         value=float(train_aurocs.mean()))
            self.log('train_f1', float(train_f1.mean()), prog_bar=True, logger=True)

            for i, name in enumerate(self.class_names):
                self.log(f'train_{name}_rocauc', train_aurocs[i])
                clearml_logger.report_scalar(f'rocauc_{name}', 'train', train_aurocs[i], self.current_epoch)

                self.log(f'train_{name}_f1', train_f1[i])
                clearml_logger.report_scalar(f'f1_{name}', 'train', train_f1[i], self.current_epoch)

        def validation_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]
            logits = self(images)
            scores = torch.sigmoid(logits)

            loss = self.loss(logits, labels)
            self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True)
            clearml_logger.report_scalar('loss', 'val', iteration=batch_idx, value=loss)

            return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

        def validation_epoch_end(self, validation_step_outputs):

            labels = []
            scores = []
            for output in validation_step_outputs:
                for out_labels in output[TARGETS].detach().cpu():
                    labels.append(out_labels)
                for out_predictions in output[SCORES].detach().cpu():
                    scores.append(out_predictions)
            scores = torch.stack(scores)
            labels = torch.stack(labels)
            predictions = (scores > self.class_threshold).to(torch.int32)

            val_aurocs = self.val_auroc(scores, labels).cpu()
            val_f1 = self.val_f1(predictions, labels).cpu()

            self.log('val_auroc', float(val_aurocs.mean()), prog_bar=True, logger=True)
            clearml_logger.report_scalar('auroc_macro', 'val', iteration=self.current_epoch,
                                         value=float(val_aurocs.mean()))
            self.log('val_f1', float(val_f1.mean()), prog_bar=True, logger=True)

            for i, name in enumerate(self.class_names):
                self.log(f'val_{name}_rocauc', val_aurocs[i])
                clearml_logger.report_scalar(f'rocauc_{name}', 'val', val_aurocs[i], self.current_epoch)

                self.log(f'val_{name}_f1', val_f1[i])
                clearml_logger.report_scalar(f'f1_{name}', 'val', val_f1[i], self.current_epoch)

        def test_step(self, batch, batch_idx):

            images, labels = batch[IMAGES], batch[TARGETS]
            logits = self(images)
            scores = torch.sigmoid(logits)
            loss = self.loss(logits, labels)

            return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

        def test_epoch_end(self, test_step_outputs):

            labels = []
            scores = []
            for output in test_step_outputs:
                for out_labels in output[TARGETS].detach().cpu():
                    labels.append(out_labels)
                for out_predictions in output[SCORES].detach().cpu():
                    scores.append(out_predictions)
            scores = torch.stack(scores)
            labels = torch.stack(labels)
            predictions = (scores > self.class_threshold).to(torch.int32)

            test_aurocs = self.test_auroc(scores, labels).cpu()
            test_f1 = self.test_f1(predictions, labels).cpu()

            test_results = pd.DataFrame(index=config.log_metrics)
            for i, name in enumerate(self.class_names):
                test_results[name] = [test_aurocs[i], test_f1[i]]

            clearml_logger.report_table(title='Test Results', series='Test Results', iteration=0,
                                        table_plot=test_results)

        def configure_optimizers(self):
            """To callback for configuring optimizers.

            Returns:
                optimizer: torch.optim - optimizer for PL.
            """
            # return [self.optimizer], [self.scheduler]
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

        def optimizer_step(
                self,
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure,
                on_tpu=False,
                using_native_amp=False,
                using_lbfgs=False,
        ):
            """Set optimizer warmup.

            Args:
                epoch: int - epoch num.
                batch_idx: int - batch index.
                optimizer: torch.optim - torch optimizer.
                optimizer_idx: int - optimizer pg index.
                optimizer_closure: function - optimizer closure.
                on_tpu: bool - tpu flag.
                using_native_amp: bool - amp flag.
                using_lbfgs: bool - lbfgs flag.
            """
            optimizer.step(optimizer_closure)
            if self.trainer.global_step < self.warmup_iter:
                lr_scale = min(
                    1,
                    float(self.trainer.global_step + 1) / self.warmup_iter
                )
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.init_lr

        def on_save_checkpoint(self, checkpoint):
            """Save custom state dict.

            Function is needed, because we want only timm state_dict for scripting.

            Args:
                checkpoint: pl.checkpoint - checkpoint from PL.
            """
            checkpoint['my_state_dict'] = self.model.state_dict()

    data = DataModule(loaders)
    model = TrainModule(model, config.loss, class_names, optimizer, scheduler, config.warmup_iter, config.warmup_iter)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        **config.trainer_kwargs,
    )

    # train
    trainer.fit(model, data)

    # evaluate
    trainer.test(model, data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)
