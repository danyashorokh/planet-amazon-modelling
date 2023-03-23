
import pandas as pd
import torch
import pytorch_lightning as pl

from src.const import IMAGES, LOGITS, PREDICTS, SCORES, TARGETS, VALID, LOSS # noqa


class TrainModule(pl.LightningModule):
    def __init__(self,
                 model, loss, metrics, class_names, optimizer, scheduler, warmup_iter, class_threshold, logger):
        super(TrainModule, self).__init__()

        self.model = model
        self.loss = loss
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_iter = warmup_iter
        self.class_threshold = class_threshold
        self.custom_logger = logger
        self.init_lr = self.optimizer.param_groups[0]['lr']
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.save_hyperparameters()

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
        self.train_metrics.update(scores, labels)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True)
        self.custom_logger.report_scalar('loss', 'train', iteration=batch_idx, value=loss)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('LR', current_lr, prog_bar=True, logger=True, on_step=True)

        return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

    def training_epoch_end(self, outputs):

        train_metrics = self.train_metrics.compute()

        # self.log('train_auroc_macro', train_metrics['train_auroc_macro'], prog_bar=True, logger=True)
        # self.log('train_f1_macro', train_metrics['train_f1_macro'], prog_bar=True, logger=True)
        self.custom_logger.report_scalar('auroc_macro', 'train', iteration=self.current_epoch,
                                         value=float(train_metrics['train_auroc_macro']))
        self.custom_logger.report_scalar('f1_macro', 'train', iteration=self.current_epoch,
                                         value=float(train_metrics['train_f1_macro']))

        for i, name in enumerate(self.class_names):
            self.custom_logger.report_scalar(f'rocauc_{name}', 'train', float(train_metrics['train_auroc'][i]),
                                             self.current_epoch)

            self.custom_logger.report_scalar(f'rocauc_{name}', 'train', float(train_metrics['train_f1'][i]),
                                             self.current_epoch)

    def validation_step(self, batch, batch_idx):

        images, labels = batch[IMAGES], batch[TARGETS]
        logits = self(images)
        scores = torch.sigmoid(logits)

        loss = self.loss(logits, labels)
        self.val_metrics.update(scores, labels)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True)
        self.custom_logger.report_scalar('loss', 'val', iteration=batch_idx, value=loss)

        return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

    def validation_epoch_end(self, outputs):

        val_metrics = self.val_metrics.compute()

        # self.log('val_auroc_macro', val_metrics['val_auroc_macro'], prog_bar=True, logger=True)
        # self.log('val_f1_macro', val_metrics['val_f1_macro'], prog_bar=True, logger=True)
        self.custom_logger.report_scalar('auroc_macro', 'val', iteration=self.current_epoch,
                                          value=float(val_metrics['val_auroc_macro']))
        self.custom_logger.report_scalar('f1_macro', 'val', iteration=self.current_epoch,
                                         value=float(val_metrics['val_f1_macro']))
        for i, name in enumerate(self.class_names):
            self.custom_logger.report_scalar(f'rocauc_{name}', 'val', float(val_metrics['val_auroc'][i]),
                                             self.current_epoch)

            self.custom_logger.report_scalar(f'rocauc_{name}', 'val', float(val_metrics['val_f1'][i]),
                                             self.current_epoch)

    def test_step(self, batch, batch_idx):

        images, labels = batch[IMAGES], batch[TARGETS]
        logits = self(images)
        scores = torch.sigmoid(logits)
        loss = self.loss(logits, labels)

        self.test_metrics.update(scores, labels)

        return {LOSS: loss, LOGITS: logits, SCORES: scores, TARGETS: labels}

    def test_epoch_end(self, outputs):

        test_metrics = self.test_metrics.compute()

        test_not_agg_metrics = [el for el in test_metrics.keys() if 'micro' not in el and 'macro' not in el]
        test_results = pd.DataFrame(index=test_not_agg_metrics)
        for i, name in enumerate(self.class_names):
            for metric_name in test_not_agg_metrics:
                test_results.loc[metric_name, name] = float(test_metrics[metric_name][i])

        self.custom_logger.report_table(title='Test Results', series='Test Results', iteration=0,
                                        table_plot=test_results)

    def configure_optimizers(self):
        """To callback for configuring optimizers.

        Returns:
            optimizer: torch.optim - optimizer for PL.
        """
        return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler}}

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
