import logging
import os
import typing as tp
from collections import OrderedDict
from random import shuffle, sample
from copy import deepcopy

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from src.base_config_pl import Config
from src.const import IMAGES, TARGETS
from src.utils import worker_init_fn
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        augmentation: tp.Optional[albu.Compose] = None,
        preprocessing: tp.Optional[tp.Callable] = None,
    ):
        self.df = df
        self.config = config
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx: int) -> tp.Dict[str, np.ndarray]:
        row = self.df.iloc[idx]

        img_path = f'{os.path.join(self.config.images_dir, row[self.config.image_col_id])}.jpg'
        target = np.array(row.drop([self.config.image_col_id]), dtype='float32')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation is not None:
            image = self.augmentation(image=image)['image']
        if self.preprocessing is not None:
            image = self.preprocessing(image)

        return {IMAGES: image, TARGETS: target}

    def __len__(self) -> int:
        return len(self.df)


class CustomSampler:
    def __init__(self, df: pd.DataFrame, batch_size: int = 1, iteration_counts: int = 1):
        self.df = df
        self.batch_size = batch_size
        self.iteration_counts = iteration_counts

        self.indexes = list(df.index)
        shuffle(self.indexes)

        self.current_indexes = deepcopy(self.indexes)

    def __iter__(self):
        for _ in range(self.iteration_counts):
            indexes = []
            for _ in range(self.batch_size):
                if not self.current_indexes:
                    self.current_indexes = sample(self.indexes, len(self.indexes))
                indexes.append(self.current_indexes.pop())
            yield indexes

    def __len__(self):
        return self.iteration_counts


def _get_dataframes(config: Config) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.train_dataset_path)
    valid_df = pd.read_csv(config.valid_dataset_path)
    test_df = pd.read_csv(config.test_dataset_path)

    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    return train_df, valid_df, test_df


def get_class_names(config: Config) -> tp.List[str]:
    class_names = list(pd.read_csv(config.train_dataset_path, nrows=1).drop(config.image_col_id, axis=1))
    logging.info(f'Classes num: {len(class_names)}')
    return class_names


def get_datasets(config: Config) -> tp.Tuple[Dataset, Dataset, Dataset]:
    df_train, df_val, df_test = _get_dataframes(config)

    train_dataset = CustomDataset(
        df_train,
        config,
        augmentation=config.augmentations,
        preprocessing=config.preprocessing,
    )

    valid_dataset = CustomDataset(df_val, config, preprocessing=config.preprocessing)

    test_dataset = CustomDataset(df_test, config, preprocessing=config.preprocessing)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(config: Config) -> tp.Tuple[tp.OrderedDict[str, DataLoader], tp.Dict[str, DataLoader]]:
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    if config.num_iteration_on_epoch > 0:
        batch_sampler = CustomSampler(
            df=train_dataset.df,
            batch_size=config.batch_size,
            iteration_counts=config.num_iteration_on_epoch,
        )
    else:
        batch_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if batch_sampler else config.batch_size,
        shuffle=False if batch_sampler else True,  # noqa: WPS502
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        batch_sampler=batch_sampler,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return OrderedDict({'train': train_loader, 'valid': valid_loader, 'infer': test_loader})
