import typing as tp
from collections import defaultdict
from typing import Any

import pandas as pd
from catalyst.core.logger import ILogger
from catalyst.core.runner import IRunner
from clearml import Task
from src.base_config import Config


class ClearMLLogger(ILogger):  # noqa: WPS338
    @property
    def logger(self) -> Any:
        return self.clearml_logger

    def __init__(
        self,
        config: Config,
        class_names: tp.List[str],
        log_batch_metrics: bool = False,
        log_epoch_metrics: bool = True,
    ):
        super().__init__(
            log_batch_metrics=log_batch_metrics,
            log_epoch_metrics=log_epoch_metrics,
        )
        self.class_names = class_names
        self.metrics_to_log = config.log_metrics
        self.metrics_to_log.extend(['loss', 'lr'])

        task = Task.init(
            project_name=config.project_name,
            task_name=config.experiment_name,
        )
        task.connect(config.to_dict())
        self.clearml_logger = task.get_logger()

    def log_metrics(
        self,
        metrics: tp.Dict[str, float],
        scope: str,
        runner: 'IRunner',
        infer: bool = False,
    ):
        step = runner.sample_step if self.log_batch_metrics else runner.epoch_step

        if scope == 'loader' and self.log_epoch_metrics:
            if infer:
                self._log_infer_metrics(
                    metrics=metrics,
                )
            else:
                self._log_train_metrics(
                    metrics=metrics,
                    step=step,
                    loader_key=runner.loader_key,
                )

    def _report_scalar(self, title: str, mode: str, value: float, epoch: int):
        self.logger.report_scalar(
            title=title,
            series=mode,
            value=value,
            iteration=epoch,
        )

    def _log_train_metrics(self, metrics: tp.Dict[str, float], step: int, loader_key: str):
        log_keys = [k for log_m in self.metrics_to_log for k in metrics.keys() if log_m in k]
        for k in log_keys:
            title = k
            if 'class' in k:
                title, cl = k.split('/')
                title = f"{title}_{self.class_names[int(cl.split('_')[1])]}"
            self._report_scalar(title, loader_key, metrics[k], step)

    def _log_infer_metrics(self, metrics: tp.Dict[str, float]):
        infer_metrics = defaultdict(dict)
        self.metrics_to_log.append('support')

        log_keys = [k for log_m in self.metrics_to_log for k in metrics.keys() if log_m in k]
        for k in log_keys:
            if '/' in k:
                title, cl = k.split('/')
                if 'class' in cl:
                    cl = self.class_names[int(cl.split('_')[1])]
                infer_metrics[title].update({cl: metrics[k]})
        test_results = pd.DataFrame(infer_metrics)
        test_results = test_results.rename(columns={'support': 'num'}).T
        self.logger.report_table(title='Test Results', series='Test Results', iteration=0, table_plot=test_results)

    def flush_log(self):
        self.logger.flush()
