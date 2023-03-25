# Time stretching

This project is an implementation of the Phase Vocoder algorithm in Python. This script allows you to speed up/delay an audio file without changing the pitch, while maintaining its harmonic content.

## Installation

1. Clone this repository.
3. Run the `run.sh` script.

## Running the script
To run the Phase Vocoder algorithm using the run.sh script, run the following command:

```sh
./run.sh --input_file <input_file> --output_file <output_file> --ratio <ratio>
```

Example:
```sh
./run.sh --input_file examples/input_file.wav --output_file examples/output_file_05.wav --ratio 0.5
```

Replace <input_file> with the path to the input audio file, <output_file> with the path to the output audio file, 
and <ratio> with the time stretch/pitch shift ratio. 

A ratio of 1.0 means no time stretch or pitch shift, 
a ratio of 0.5 means 2x speed without changing the pitch.

## Usage
To use the Phase Vocoder algorithm in your own Python code, import the phase_vocoder function from the phase_vocoder.py module and call it with the following parameters:

- input_file (str): path to the input audio file
- output_file (str): path to the output audio file
- ratio (float): time stretch/pitch shift ratio. A ratio of 1.0 means no time stretch or pitch shift.
- sr (int, optional): sample rate of the input audio file (default 16000)
- window_size (int, optional): size of the analysis window (default 1024)
- hop_input_resize (float, optional): hop size of the analysis window (default 0.25)

## References
- Implements from this papper: [Guitar Pitch Shifter](https://www.guitarpitchshifter.com/algorithm.html)