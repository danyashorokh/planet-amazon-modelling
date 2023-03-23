
import pytorch_lightning as pl


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
