
import os

import pytest
import cv2

TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'fixtures', 'images', 'train_0.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture()
def model_path():
    # return './weights/model.best.zip'
    return os.path.join(TESTS_DIR, '..', 'weights', 'model.best.zip')
