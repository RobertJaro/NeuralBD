import numpy as np


def rms_contrast(image):
    mean = np.nanmean(image)
    return float(np.sqrt(np.nanmean((image - mean) ** 2)))


def select_highest_contrast(images, n_images):
    scores = [rms_contrast(images[:, :, idx]) for idx in range(images.shape[2])]
    indices = np.argsort(scores)[::-1][:n_images]
    return images[:, :, indices], indices, [scores[idx] for idx in indices]
