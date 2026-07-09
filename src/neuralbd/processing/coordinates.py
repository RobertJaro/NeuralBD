import numpy as np


def image_coordinates(shape, pixel_per_ds=1.0):
    height, width = shape[:2]
    yy, xx = np.mgrid[:height, :width]
    coords = np.stack(
        [
            xx - (width - 1) / 2,
            (height - 1) / 2 - yy,
        ],
        axis=-1,
    ).astype("float32")
    return coords / float(pixel_per_ds)
