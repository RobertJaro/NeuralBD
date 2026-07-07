def crop_at(image, x, y, size):
    half = size // 2
    x0, x1 = x - half, x + half
    y0, y1 = y - half, y + half
    if x0 < 0 or y0 < 0 or x1 > image.shape[0] or y1 > image.shape[1]:
        raise ValueError("Crop extends outside image bounds")
    return image[x0:x1, y0:y1, ...]


def center_crop(image, size):
    return crop_at(image, image.shape[0] // 2, image.shape[1] // 2, size)


def subframe(image, x=None, y=None, size=None, x_range=None, y_range=None):
    if size is not None:
        center_x = image.shape[0] // 2 if x is None else int(x)
        center_y = image.shape[1] // 2 if y is None else int(y)
        return crop_at(image, center_x, center_y, int(size))

    if x_range is None and y_range is None:
        return image

    x0, x1 = x_range or (0, image.shape[0])
    y0, y1 = y_range or (0, image.shape[1])
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    if x0 < 0 or y0 < 0 or x1 > image.shape[0] or y1 > image.shape[1] or x0 >= x1 or y0 >= y1:
        raise ValueError("Subframe extends outside image bounds")
    return image[x0:x1, y0:y1, ...]
