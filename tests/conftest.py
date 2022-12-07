
import typing as tp
import os
import pytest

import numpy as np
import cv2
import torch

from config_tests import Config

TESTS_DIR = os.path.dirname(__file__)
MAX_PIXEL_INTENSITY = 255


def preprocess_imagenet(im: np.ndarray, img_size: tp.Union[int, tp.Tuple[int, int]]) -> np.ndarray:
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


@pytest.fixture(scope='session')
def dummy_input():
    return np.random.rand(Config.size, Config.size, 3)


@pytest.fixture(scope='session')
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'fixtures', 'images', 'train_0.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def augmentations():
    return Config.augmentations


class ImageClassifier:

    def __init__(self, model_path: str, device: str):
        self._model_path = model_path
        self._device = device
        self._model = torch.jit.load(self._model_path, map_location=self._device)
        self._classes = np.array(self._model.classes)
        self._size = self._model.size
        self._thresholds = np.array(self._model.thresholds)

    @property
    def classes(self) -> tp.List:
        return list(self._classes)

    @property
    def size(self) -> tp.Tuple:
        return self._size

    def predict(self, image: np.ndarray) -> tp.List[str]:
        return self._postprocess_predict(self._predict(image))

    def predict_proba(self, image: np.ndarray) -> tp.Dict[str, float]:
        return self._postprocess_predict_proba(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        batch = preprocess_imagenet(image, self._size)
        batch = torch.from_numpy(batch)[None]

        with torch.no_grad():
            model_predict = self._model(batch.to(self._device)).detach().cpu()[0]

        return model_predict.numpy()

    def _postprocess_predict(self, predict: np.ndarray) -> tp.List[str]:
        return self._classes[predict > self._thresholds].tolist()

    def _postprocess_predict_proba(self, predict: np.ndarray) -> tp.Dict[str, float]:
        return {self._classes[i]: float(predict[i]) for i in predict.argsort()[::-1]}


@pytest.fixture(scope='session')
def model_wrapper():
    return ImageClassifier(model_path=Config.weights, device=Config.device)
