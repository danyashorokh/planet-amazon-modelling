from functools import partial
import os

import albumentations as albu
import torch
from src.catalyst.base_config import Config
from src.utils import preprocess_imagenet
from torch.nn import BCEWithLogitsLoss
from catalyst.callbacks import LRFinder

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 256
N_EPOCHS = 100
ROOT_PATH = os.path.join(os.environ.get("ROOT_PATH"))

augmentations = albu.Compose(
    [
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5,
        ),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.ShiftScaleRotate(),
        albu.GaussianBlur(),
    ]
)

callbacks = [
    LRFinder(final_lr=1.),  # ограничиваем lr максимальным значением
]

config = Config(
    num_workers=4,
    seed=SEED,
    loss=BCEWithLogitsLoss(),
    optimizer=torch.optim.SGD,
    optimizer_kwargs={
        "lr": 1e-7,  # минимальное значение lr
    },
    scheduler=None,
    scheduler_kwargs={
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
    },
    img_size=IMG_SIZE,
    augmentations=augmentations,
    callbacks=callbacks,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    num_iteration_on_epoch=100,  # увеличиваем количество итераций на эпоху
    n_epochs=N_EPOCHS,
    model_kwargs={"model_name": "resnet18", "pretrained": False},
    checkpoint_name='weights/train_only_head/model.best.pth',
    log_metrics=["auc", "f1"],
    cls_thresh=0.1,
    valid_metric="auc",
    minimize_metric=False,
    images_dir=os.path.join(ROOT_PATH, "Images"),
    train_dataset_path=os.path.join(ROOT_PATH, "train_df.csv"),
    valid_dataset_path=os.path.join(ROOT_PATH, "valid_df.csv"),
    test_dataset_path=os.path.join(ROOT_PATH, "test_df.csv"),
    project_name="[Classification]planet-amazon",
    experiment_name=f'{os.path.basename(__file__).split(".")[0]}',
)
