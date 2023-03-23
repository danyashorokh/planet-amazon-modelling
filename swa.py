import argparse
import json
import logging
import math
import os
from collections import OrderedDict
from runpy import run_path
from typing import Any, List

import timm
import torch
import torch.nn as nn
from catalyst import dl
from catalyst.callbacks import Callback
from src.catalyst.base_config import Config
from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS
from src.dataset import get_class_names, get_loaders


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def swa(config: Config):

    class_names = get_class_names(config)
    loaders, infer_loader = get_loaders(config)
    valid_loader = loaders['valid']

    work_dir = config.checkpoints_dir

    with open(os.path.join(work_dir, 'model.storage.json')) as f:
        history = json.load(f)

    checkpoint_names = [checkpoint_name['logpath'].split('/')[-1] for checkpoint_name in history['storage']]
    model: nn.Module = timm.create_model(num_classes=len(class_names), **config.model_kwargs)
    model.load_state_dict(torch.load(os.path.join(work_dir, checkpoint_names[0])))

    runner = dl.SupervisedRunner(
        input_key=IMAGES,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    best_metric = math.inf if config.minimize_metric else 0
    best_weights = []
    result_checkpoint_names = []

    for checkpoint_name in checkpoint_names:
        weights = torch.load(os.path.join(work_dir, checkpoint_name))
        current_weights = best_weights + [weights]

        averaged_weights = average_weights(current_weights)
        model.load_state_dict(averaged_weights)

        metrics = runner.evaluate_loader(
            model=model,
            loader=valid_loader,
            callbacks=get_base_callbacks(config),
            verbose=False,
            seed=config.seed,
        )

        logging.info(f'Best metric = {best_metric}, current metric = {metrics[config.valid_metric]}')
        if (config.minimize_metric and metrics[config.valid_metric] < best_metric) or (
            not config.minimize_metric and metrics[config.valid_metric] > best_metric
        ):
            logging.info(f'Append checkpoint `{checkpoint_name}`')
            result_checkpoint_names.append(checkpoint_name)
            best_metric = metrics[config.valid_metric]
            best_weights.append(weights)
        else:
            logging.info(f'Ignore checkpoint `{checkpoint_name}`')
        logging.info('=' * 50)
        logging.info(len(best_weights))

    logging.info(f'Result checkpoint names: {result_checkpoint_names}')
    logging.info(f'Best metric = {best_metric}')
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.swa.pth'))


def average_weights(state_dicts: List[OrderedDict]) -> OrderedDict:
    average_dict = OrderedDict()
    for key in state_dicts[0].keys():
        average_dict[key] = sum([state_dict[key] for state_dict in state_dicts]) / len(state_dicts)
    return average_dict


def get_base_callbacks(config: Config) -> List[Callback]:
    return [
        dl.BatchTransformCallback(
            transform=torch.sigmoid,
            scope="on_batch_end",
            input_key=LOGITS,
            output_key=SCORES,
        ),
        dl.BatchTransformCallback(
            transform=lambda x: x > config.cls_thresh,
            scope="on_batch_end",
            input_key=SCORES,
            output_key=PREDICTS,
        ),
        dl.AUCCallback(input_key=SCORES, target_key=TARGETS, compute_per_class_metrics=True),
    ]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    swa(config)
