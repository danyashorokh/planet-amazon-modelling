
import os

import albumentations as albu

TESTS_DIR = os.path.dirname(__file__)


class Config:
    weights = os.path.join(TESTS_DIR, '..', 'weights', 'model.best.zip')
    device = 'cpu'
    size = 224
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
