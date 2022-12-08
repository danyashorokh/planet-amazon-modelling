from datetime import datetime
from functools import partial
import os

import albumentations as albu
import torch
from src.base_config import Config
from src.utils import preprocess_imagenet
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim.lr_scheduler import StepLR

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
N_EPOCHS = 2
NUM_ITERATION_ON_EPOCH = 10
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


config = Config(
    num_workers=0,
    seed=SEED,
    loss=BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        "lr": 1e-3,
        "weight_decay": 5e-4,
    },
    warmup_iter=0,
    scheduler=StepLR,
    scheduler_kwargs={
        "step_size": 30 * NUM_ITERATION_ON_EPOCH,
        "gamma": 0.1,
    },
    img_size=IMG_SIZE,
    augmentations=augmentations,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    num_iteration_on_epoch=NUM_ITERATION_ON_EPOCH,
    n_epochs=N_EPOCHS,
    model_kwargs={"model_name": "resnet18", "pretrained": True},
    log_metrics=["auc", "f1"],
    binary_thresh=0.1,
    valid_metric="val_loss",
    minimize_metric=True,
    images_dir=os.path.join(ROOT_PATH, "raw", "train"),
    image_col_id='image_name',
    train_dataset_path=os.path.join(ROOT_PATH, "train_v2.csv", "train_df_1024.csv"),
    valid_dataset_path=os.path.join(ROOT_PATH, "train_v2.csv", "valid_df_256.csv"),
    test_dataset_path=os.path.join(ROOT_PATH, "train_v2.csv", "test_df_1024.csv"),
    project_name="[Classification]planet-amazon",
    experiment_name=f'{os.path.basename(__file__).split(".")[0]}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}',
)
