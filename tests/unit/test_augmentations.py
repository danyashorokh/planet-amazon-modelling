
def test_augs_not_fail(sample_image_np, augmentations):
    augmentations(image=sample_image_np)


def test_augs_convert_to_tensor(sample_image_np, augmentations):
    mutated_im = augmentations(image=sample_image_np)['image']
    assert type(mutated_im) == type(sample_image_np)
