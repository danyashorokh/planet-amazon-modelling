import os
import random

import cv2
import numpy as np
import torch

MAX_PIXEL_INTENSITY = 255


def preprocess_imagenet(im: np.ndarray, img_size: int) -> np.ndarray:
    im = im.astype(np.float32)
    im /= MAX_PIXEL_INTENSITY
    im = cv2.resize(im, (img_size, img_size))
    im = np.transpose(im, (2, 0, 1))
    im -= np.array([0.485, 0.456, 0.406])[:, None, None]
    im /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return im


def set_global_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int, initial_seed: int = 42):
    """Fixes bug with identical augmentations.
    More info: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    """
    seed = initial_seed**2 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
