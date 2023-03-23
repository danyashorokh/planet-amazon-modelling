from typing import Union, List
from catalyst.callbacks import Callback, CallbackOrder
from catalyst.runners.runner import IRunner
import torch.nn as nn


class FreezeModelParts(Callback):
    def __init__(
        self,
        parts: Union[List[str], str],
        on_epoch: int = 1,
    ):
        """
        :param on_epoch: Индексация начинается с 1!
        """
        super().__init__(CallbackOrder.Internal + 5)
        self.parts = parts if isinstance(parts, list) else [parts]
        self.on_epoch = on_epoch

    def on_epoch_start(self, runner: 'IRunner') -> None:
        if runner.epoch_step == self.on_epoch:
            for part in self.parts:
                for param in self.get_model_part(part, runner).parameters():
                    param.requires_grad = False

    def on_stage_end(self, runner: 'IRunner') -> None:
        for part in self.parts:
            for param in self.get_model_part(part, runner).parameters():
                param.required_grad = True

    def get_model_part(self, part: str, runner: IRunner) -> nn.Module:
        model_part = runner.model
        for sub_part in part.split('.'):
            model_part = getattr(model_part, sub_part)
        return model_part
