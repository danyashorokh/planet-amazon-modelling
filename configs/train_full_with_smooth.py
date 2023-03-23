from functools import partial
import os

import albumentations as albu
import torch
from src.catalyst.base_config import Config
from src.scheduler import CosineAnnealingWarmUpRestarts
from src.utils import preprocess_imagenet
from src.losses import BCEWithLogitsLossSmoothing

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 256
N_EPOCHS = 100
NUM_ITERATION_ON_EPOCH = 100
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
    num_workers=4,
    seed=SEED,
    loss=BCEWithLogitsLossSmoothing(label_smoothing=0.02),
    optimizer=torch.optim.SGD,
    optimizer_kwargs={
        "lr": 5e-6,  # минимальное значение lr
        "momentum": 0.9,
    },
    scheduler=CosineAnnealingWarmUpRestarts,
    scheduler_kwargs={
        'T_0': 20 * NUM_ITERATION_ON_EPOCH,  # будет 5 цикла
        'T_mult': 1,
        'eta_max': 5e-4,  # максимальный lr
        'T_up': 0,  # итераций warm up
        'gamma': 0.7,  # затухание амплитуды
    },
    img_size=IMG_SIZE,
    augmentations=augmentations,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    num_iteration_on_epoch=NUM_ITERATION_ON_EPOCH,  # увеличиваем количество итераций на эпоху
    n_epochs=N_EPOCHS,
    model_kwargs={"model_name": "resnet18", "pretrained": True},
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
