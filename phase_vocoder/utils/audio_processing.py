import numpy as np

from typing import Tuple
from scipy.interpolate import interp1d


def make_new_sr(y: np.ndarray, old_sr: int, new_sr: int) -> Tuple[int, np.ndarray]:
    """
        Resamples an audio signal to a new sampling rate using linear interpolation.

        Args:
            y (np.ndarray): Input audio signal as a numpy array.
            old_sr (int): Original sampling rate of the input audio signal.
            new_sr (int): Desired sampling rate of the output audio signal.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the new sampling rate and the resampled audio signal.

        Raises:
            ValueError: If the new sampling rate is less than or equal to 0 or if the length of the input audio signal is 0.

        Example:
            >>> y, sr = librosa.load('audio.wav', sr=44100)
            >>> new_sr, y_resampled = make_new_sr(y, sr, 22050)
    """

    duration = y.shape[0] / old_sr

    time_old = np.linspace(0, duration, y.shape[0])
    time_new = np.linspace(0, duration, int(y.shape[0] * new_sr / old_sr))

    interpolator = interp1d(time_old, y.T)
    y = interpolator(time_new).T

    return new_sr, y
