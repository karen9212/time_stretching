import numpy as np


def hanning(window_size: int) -> float:
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_size * np.arange(window_size)))


def get_pitch(r: float) -> float:
    return float(12 * np.log2(r))
