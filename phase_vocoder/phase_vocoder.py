import sys
import logging
import argparse

import soundfile as sf

from utils.audio_processing import make_new_sr
from utils.fusion import get_fusion_frames
from utils.functions import *

from scipy.io import wavfile
from scipy.fft import fft, ifft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def phase_vocoder(input_file: str,
                  output_file: str,
                  ratio: float = 1.,
                  sr: int = 16000,
                  window_size: int = 1024,
                  hop_input_resize: float = .25) -> None:
    """
       Perform speed up/delay an audio file without changing the pitch using phase vocoder algorithm.

       Args:
            input_file (str): Path to the input audio file.
            output_file (str): Path to the output audio file.
            ratio (float, optional): Pitch shift factor. A value of 1.0 means no pitch shift. Defaults to 1.0.
            sr (int, optional): Desired sampling rate of the output audio file. Defaults to 16000.
            window_size (int, optional): Size of the analysis window in samples. Defaults to 1024.
            hop_input_resize (float, optional): Amount of overlap between consecutive analysis windows, as a fraction of the window size. Defaults to 0.25.
    """

    # Attempt to read the input file
    try:
        cur_sr, y = wavfile.read(input_file)
    except Exception as e:
        # If there's an error reading the file, log it and exit the script
        logger.critical(e)
        sys.exit(-1)

    # If the audio is stereo, convert it to mono by summing the channels
    if len(y.shape) == 2 and y.shape[1] == 2:
        y = y.sum(axis=1) / 2

    # If the audio's sampling rate is different from the desired sampling rate, resample it
    if cur_sr != sr:
        sr, y = make_new_sr(y, cur_sr, sr)

    # Calculate the frequency shift factor based on the pitch shift factor
    pitch = get_pitch(ratio)
    alpha = 2 ** (pitch / 12)

    hop_input = int(window_size * hop_input_resize)
    hop_output = round(alpha * hop_input)
    wn = hanning(window_size)

    # Create a matrix of FFTs of each input frame, normalized by the number of overlapping frames
    frames = np.array([fft(wn * y[index:index + window_size]) for index in range(0, y.shape[0] - window_size, hop_input)])
    frames /= np.sqrt(((float(window_size) / float(hop_input)) / 2))

    previous_phase, phase_cumulative = 0, 0
    output = np.array([[0] * window_size] * frames.shape[0])

    for index, frame in enumerate(frames):

        magnitude, phase = np.abs(frame), np.angle(frame)

        # Calculate the delta phase between the current and previous frames
        delta_phase = phase - previous_phase
        previous_phase = phase

        # Unwrap the delta phase
        delta_phase -= hop_input * 2 * np.pi * np.arange(frame.shape[0]) / frame.shape[0]

        # Perform modulo 2pi on the delta phase to keep it in the range [-pi, pi]
        delta_phase_mod = np.mod(delta_phase + np.pi, 2 * np.pi) - np.pi

        # Calculate the true frequency of each bin in the FFT
        true_freq = 2 * np.pi * np.arange(frame.shape[0]) / frame.shape[0] + delta_phase_mod / hop_input
        # Update the cumulative phase based on the true frequency and the desired output window size
        phase_cumulative += hop_output * true_freq

        # Reconstruct the audio frame using the modified magnitude and cumulative phase information
        frame_output = np.real(ifft(magnitude * np.exp(phase_cumulative * 1j)))
        frame_output = frame_output * wn / np.sqrt(((float(window_size) / float(hop_output)) / 2.))

        output[index] = frame_output

    # Fuse overlapping frames in the output matrix
    output = get_fusion_frames(output, hop_output)
    # Calculate the length of the output audio based on the desired pitch shift ratio
    audio_length_output = int(ratio * y.shape[0])

    # Resample the audio to the desired length
    interpolator = lambda x_new: np.interp(x_new, np.arange(0, output.shape[0]), output)
    resampled = np.linspace(0, output.shape[0] - 1, audio_length_output)
    resampled = interpolator(resampled)

    # Write the output audio to a WAV file
    sf.write(output_file, np.asarray(resampled, dtype=np.int16), sr)
    # Log a message indicating that the output file was saved successfully
    logger.info(f"File {output_file} successfully saved")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Perform speed up/delay an audio file without changing the pitch using phase vocoder algorithm.')

    parser.add_argument('--input_file', required=True, type=str, help='Path to the input audio file.')
    parser.add_argument('--output_file', required=True, type=str, help='Path to the output audio file.')
    parser.add_argument('--ratio', required=True, type=float, help='Pitch shift factor. A value of 1.0 means no pitch shift.')
    parser.add_argument('--sr', required=False, type=int, default=16000,
                        help='Desired sampling rate of the output audio file. Default is 16000.')
    parser.add_argument('--window-size', required=False, type=int, default=1024,
                        help='Size of the analysis window in samples. Default is 1024.')
    parser.add_argument('--hop-input-resize', required=False, type=float, default=0.25,
                        help='Amount of overlap between consecutive analysis windows, as a fraction of the window size. Default is 0.25.')

    args = parser.parse_args()

    phase_vocoder(args.input_file, args.output_file, float(args.ratio),
                  sr=args.sr, window_size=args.window_size, hop_input_resize=args.hop_input_resize)
