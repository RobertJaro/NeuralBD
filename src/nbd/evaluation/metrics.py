import numpy as np


def mean_squared_error(prediction, target):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    return float(np.mean((prediction - target) ** 2))


def peak_signal_to_noise_ratio(prediction, target, data_range=1.0, eps=1e-12):
    mse = mean_squared_error(prediction, target)
    return float(20 * np.log10(data_range / np.sqrt(mse + eps)))
