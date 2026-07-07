import numpy as np


def image_coordinates(shape, pixel_per_ds=1.0):
    height, width = shape[:2]
    coords = np.stack(np.mgrid[:height, :width], axis=-1).astype("float32")
    return coords / float(pixel_per_ds)
