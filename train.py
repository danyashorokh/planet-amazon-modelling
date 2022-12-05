import argparse
import logging
from typing import List, Any
from runpy import run_path

import timm
import torch
from catalyst import dl
from catalyst.callbacks import Callback
from src.base_config import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID
from src.dataset import get_class_names, get_loaders
from src.loggers import ClearMLLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from src.utils import set_global_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def get_base_callbacks(config: Config, class_names: List[str]) -> List[Callback]:
    return [
        dl.BatchTransformCallback(
            transform=torch.sigmoid,
            scope="on_batch_end",
            input_key=LOGITS,
            output_key=SCORES,
        ),
        dl.BatchTransformCallback(
            transform=lambda x: x > config.binary_thresh,
            scope="on_batch_end",
            input_key=SCORES,
            output_key=PREDICTS,
        ),
        dl.AUCCallback(input_key=SCORES, target_key=TARGETS, compute_per_class_metrics=True),
        dl.MultilabelPrecisionRecallF1SupportCallback(
            input_key=PREDICTS,
            target_key=TARGETS,
            num_classes=len(class_names),
            log_on_batch=False,
            compute_per_class_metrics=True,
        ),
    ]


def get_train_callbacks(config: Config, class_names: List[str]) -> List[Callback]:
    callbacks = get_base_callbacks(config, class_names)
    callbacks.extend(
        [
            dl.CheckpointCallback(
                logdir=config.checkpoints_dir,
                loader_key=VALID,
                topk=10,
                metric_key=config.valid_metric,
                minimize=config.minimize_metric,
            ),
        ]
    )
    if config.scheduler is not None:
        callbacks.append(
            dl.SchedulerCallback(
                mode='batch',
                loader_key=VALID,
                metric_key=config.valid_metric,
            ),
        )
    return callbacks


def train(config: Config):
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
    clearml_logger = ClearMLLogger(config, class_names)

    callbacks = get_train_callbacks(config, class_names)
    callbacks.extend(config.callbacks)

    runner = dl.SupervisedRunner(
        input_key=IMAGES,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    runner.train(
        model=model,
        criterion=config.loss,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        loggers={
            "_clearml": clearml_logger,
            "_tensorboard": TensorboardLogger(config.checkpoints_dir, log_batch_metrics=True),
        },
        num_epochs=config.n_epochs,
        valid_loader=VALID,
        valid_metric=config.valid_metric,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )

    metrics = runner.evaluate_loader(
        model=model,
        loader=loaders["infer"],
        callbacks=get_base_callbacks(config, class_names),
        verbose=True,
        seed=config.seed,
    )

    clearml_logger.log_metrics(metrics, scope="loader", runner=runner, infer=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)
