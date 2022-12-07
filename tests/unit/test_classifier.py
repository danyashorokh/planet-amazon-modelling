
from copy import deepcopy

import pytest  # noqa

import numpy as np


def test_model_empty_tensor(model_wrapper, dummy_input):
    labels = model_wrapper.predict(dummy_input)
    probs = model_wrapper.predict_proba(dummy_input)
    assert labels
    assert probs


def test_model_np_image(model_wrapper, sample_image_np):
    labels = model_wrapper.predict(sample_image_np)
    probs = model_wrapper.predict_proba(sample_image_np)
    assert labels
    assert probs


def test_correct_probs(model_wrapper, dummy_input, sample_image_np):

    probs = model_wrapper.predict_proba(dummy_input)
    for prob in probs.values():
        assert prob <= 1
        assert prob >= 0

    probs = model_wrapper.predict_proba(sample_image_np)
    for prob in probs.values():
        assert prob <= 1
        assert prob >= 0


def test_predict_dont_mutate_orig_image(model_wrapper, sample_image_np):
    initial_image = deepcopy(sample_image_np)
    model_wrapper.predict(sample_image_np)
    assert np.allclose(initial_image, sample_image_np)
