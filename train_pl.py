import argparse
import logging
from typing import List, Any # noqa
from runpy import run_path

import timm
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor # noqa
from pytorch_lightning.callbacks import ModelPruning # noqa
from pytorch_lightning import loggers # noqa
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from torchmetrics import MetricCollection
from clearml import Task

from src.utils import set_global_seed
from src.base_config_pl import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID, LOSS # noqa
from src.dataset import get_class_names, get_loaders
from src.pl.data_module import DataModule
from src.pl.train_module import TrainModule

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for timm model download


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):

    pl.seed_everything(config.seed)

    task = Task.init(project_name=config.project_name, task_name=config.experiment_name)
    task.connect(config.to_dict())
    logger = task.get_logger()

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

    metrics = MetricCollection({
        'auroc': MultilabelAUROC(num_labels=len(class_names), average=None),
        'auroc_micro': MultilabelAUROC(num_labels=len(class_names), average='micro'),
        'auroc_macro': MultilabelAUROC(num_labels=len(class_names), average='macro'),
        'f1': MultilabelF1Score(num_labels=len(class_names), average=None),
        'f1_micro': MultilabelF1Score(num_labels=len(class_names), average='micro'),
        'f1_macro': MultilabelF1Score(num_labels=len(class_names), average='macro'),
    }).to(config.trainer_kwargs['accelerator'])

    model_checkpoint = ModelCheckpoint(
        dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}_{val_f1:.2f}',
        monitor=config.valid_metric, verbose=False, save_last=None,
        save_top_k=1, save_weights_only=True, mode='min' if config.minimize_metric else 'max')
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

    data = DataModule(loaders)
    model = TrainModule(model, config.loss, metrics, class_names, optimizer, scheduler, config.warmup_iter,
                        config.cls_thresh, logger=logger)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        **config.trainer_kwargs,
    )

    # train
    trainer.fit(model, data)

    # evaluate
    trainer.test(model, data)
    trainer.test(ckpt_path=model_checkpoint.best_model_path, datamodule=data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)
