
import pytest  # noqa
from typing import Tuple, Union

import numpy as np
import cv2
import torch


CLASSES = [
    'haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road',
    'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy', 'conventional_mine',
    'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down'
]
MAX_PIXEL_INTENSITY = 255


def preprocess_imagenet(im: np.ndarray, img_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    im = im.astype(np.float32)
    im /= MAX_PIXEL_INTENSITY
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    elif isinstance(img_size, tuple):
        target_size = img_size
    else:
        raise ValueError(f'bad img_size format {img_size}, one need int or tuple')
    im = cv2.resize(im, target_size)
    im = np.transpose(im, (2, 0, 1))
    im -= np.array([0.485, 0.456, 0.406])[:, None, None]
    im /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return im


def test_model_loading(model_path):
    model = torch.jit.load(model_path, map_location='cpu')
    assert model


def test_model_classes(model_path):
    model = torch.jit.load(model_path, map_location='cpu')
    assert model.classes == CLASSES


def test_model_empty_tensor(model_path):
    model = torch.jit.load(model_path, map_location='cpu')
    dummy_input = torch.rand(1, 3, *model.size)
    with torch.no_grad():
        preds = model(dummy_input).detach().cpu().numpy()[0]
    assert len(preds) == len(CLASSES)


def test_model_np_image(model_path, sample_image_np):
    model = torch.jit.load(model_path, map_location='cpu')

    batch = preprocess_imagenet(sample_image_np, model.size)
    batch = torch.from_numpy(batch)[None]

    with torch.no_grad():
        preds = model(batch.to('cpu')).detach().cpu().numpy()[0]

    assert len(preds) == len(CLASSES)
