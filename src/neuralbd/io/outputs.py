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
