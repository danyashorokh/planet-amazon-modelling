
import argparse
import typing as tp
import os

import numpy as np
import cv2
import torch
import torch.nn as nn

from src.utils import preprocess_imagenet


class ImageClassifier:

    def __init__(self, model_path: str, device: str):
        self._model_path = model_path
        self._device = device

        self._model: nn.Module = torch.jit.load(self._model_path, map_location=self._device)
        self._classes: np.ndarray = np.array(self._model.classes)
        self._size: tp.Tuple[int, int] = self._model.size
        self._thresholds: np.ndarray = np.array(self._model.thresholds)

    @property
    def classes(self) -> tp.List:
        return list(self._classes)

    @property
    def size(self) -> tp.Tuple:
        return self._size

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """Предсказание списка класслв.

        :param image: RGB изображение;
        :return: список классов.
        """
        return self._postprocess_predict(self._predict(image))

    def predict_proba(self, image: np.ndarray) -> tp.Dict[str, float]:
        """Предсказание вероятностей принадлежности к классам изображений.

        :param image: RGB изображение.
        :return: словарь вида `класс изображения`: вероятность.
        """
        return self._postprocess_predict_proba(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей.

        :param image: RGB изображение;
        :return: вероятности после прогона модели.
        """
        batch = preprocess_imagenet(image, self._size)
        batch = torch.from_numpy(batch)[None]

        with torch.no_grad():
            model_predict = self._model(batch.to(self._device)).detach().cpu()[0]

        return model_predict.numpy()

    def _postprocess_predict(self, predict: np.ndarray) -> tp.List[str]:
        """Постобработка для получения списка классов.

        :param predict: вероятности после прогона модели;
        :return: список классов.
        """
        return self._classes[predict > self._thresholds].tolist()

    def _postprocess_predict_proba(self, predict: np.ndarray) -> tp.Dict[str, float]:
        """Постобработка для получения словаря с вероятностями.

        :param predict: вероятности после прогона модели;
        :return: словарь вида `класс изображения`: вероятность.
        """
        return {self._classes[i]: float(predict[i]) for i in predict.argsort()[::-1]}


def main():
    parser = argparse.ArgumentParser(description='Demo script')
    parser.add_argument('-w', type=str, help='model weights path', dest='model_path')
    parser.add_argument('-i', type=str, help='input image path', dest='img_path')
    parser.add_argument('-d', type=str, help='device type', dest='device', default='cpu')
    parser.add_argument('--show', dest='show', action='store_true', default=False,
                        help='show image flag')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise ValueError(f'{args.model_path} doesn`t exist')

    if not os.path.exists(args.img_path):
        raise ValueError(f'{args.img_path} doesn`t exist')

    detector = ImageClassifier(model_path=args.model_path, device=args.device)

    image = cv2.imread(args.img_path)
    preds = detector.predict_proba(image)
    print(preds)

    if args.show:
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
