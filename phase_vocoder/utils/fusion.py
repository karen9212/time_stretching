import numpy as np


def get_fusion_frames(frames_matrix: np.ndarray, hop_output: int) -> np.ndarray:
    """
        Fuse the frames matrix into a single audio vector.

        Args:
            frames_matrix (numpy.ndarray): A matrix of audio frames, where each row represents a frame of audio.
            hop_output (int): The number of samples to skip between each output frame.

        Returns:
            numpy.ndarray: A 1D audio vector created by fusing the frames matrix.
    """

    vector_time = np.zeros(frames_matrix.shape[0] * hop_output - hop_output + frames_matrix.shape[1])

    current_index = 0

    for index in range(frames_matrix.shape[0]):

        vector_time[current_index:current_index + frames_matrix.shape[1]] += frames_matrix[index]
        current_index = current_index + hop_output

    return vector_time
