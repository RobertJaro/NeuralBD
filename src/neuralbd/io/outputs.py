from pathlib import Path

import numpy as np


def restore_image_order(array, indices, images_shape, kind):
    if kind in {"convolved_true", "convolved_pred"}:
        ordered = np.full(
            (images_shape[0] * images_shape[1], images_shape[2], images_shape[3]),
            np.nan,
            dtype=array.dtype,
        )
        ordered[indices] = array
        return ordered.reshape(images_shape)
    if kind == "image_pred":
        ordered = np.full((images_shape[0] * images_shape[1], images_shape[3]), np.nan, dtype=array.dtype)
        ordered[indices] = array
        return ordered.reshape(images_shape[0], images_shape[1], images_shape[3])
    return array


def save_validation_outputs(directory, outputs, images_shape=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indices = outputs.get("indices")
    if indices is not None:
        indices = indices.detach().cpu().numpy().astype("int64")
    for key, value in outputs.items():
        if key in {"indices", "images_shape"}:
            continue
        array = value.detach().cpu().numpy()
        if images_shape is not None and indices is not None:
            array = restore_image_order(array, indices, images_shape, key)
        elif images_shape is not None and key in {"convolved_true", "convolved_pred"}:
            array = array.reshape(images_shape)
        elif images_shape is not None and key == "image_pred":
            array = array.reshape(images_shape[0], images_shape[1], images_shape[-1])
        np.save(directory / f"{key}.npy", array)
