import numpy as np


def integer_shift(image, shift_x, shift_y):
    shifted = np.zeros_like(image)
    height, width = image.shape[:2]

    dst_x0, dst_x1 = max(0, shift_x), min(height, height + shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(width, width + shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(height, height - shift_x)
    src_y0, src_y1 = max(0, -shift_y), min(width, width - shift_y)
    shifted[dst_x0:dst_x1, dst_y0:dst_y1, ...] = image[src_x0:src_x1, src_y0:src_y1, ...]
    return shifted
