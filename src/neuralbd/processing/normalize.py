import numpy as np


def normalize(image, method="minmax", eps=1e-8):
    if method is None or method == "none":
        return image
    if method == "minmax":
        min_value = np.nanmin(image)
        max_value = np.nanmax(image)
        return (image - min_value) / (max_value - min_value + eps)
    if method == "mean":
        return image / (np.nanmean(image) + eps)
    if method == "standard":
        return (image - np.nanmean(image)) / (np.nanstd(image) + eps)
    raise ValueError(f"Unknown normalization method: {method}")
