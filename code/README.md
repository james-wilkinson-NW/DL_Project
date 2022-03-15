# DL_Project

## Basic runthroughs

The main models can be found in the following modules. Running each of these models will run the model on the dummy data included
in this repo.

1) VGGultramini.py -> our smallest CNN, which was used extensively but did not feature in the report.
2) VGGmini.py -> our staple CNN, which is outlined in architecture 1 of the report.
3) VGGpool.py -> our pooling CNN, which is outlined in architecture 2 of the report.
4) VGGnet.py -> A spin on the deep-CNN (VGG-M) model produced by Oxford's VGG group.

In order to run these models fully, data is required. For the sake of getting the models running, we have included a
small portion of data (4 speaker IDs) in ./dataset/raw/.

You can see the parameters on which spectrogram is generated come as arguments in the dataloader.

## Bigger runthroughs

Bigger runthroughs of the models can be performed (across multiple parameters, multiple runs etc) using the .ipynb files
within this repo. EG: run_VGGmini.ipynb.

You can see from the contents that we pass the hyperparameters to runner.py using a config dictionary.

### Runner

The runner.py module is a wrapper for easily running and logging models. The logging is automated, along with the naming
conventions of logs.

## Dataloaders

We take advantage of the fast processing speed of the FFT (and equivalent algorithms) to produce spectrograms on the fly
as the models train. This is demonstrated in the dataloader.py module. You can also see how this module reads data from disk
using a pre-made csv file defining the training, validation and testing data. This is crucial, given the large dataset and our
limited RAM.

## Preprocessing

concerning preprocessing.py:

### Setup before preprocessing

The initial dataset (audio files in .m4a format) is held within the directory ./dataset/raw/.

The naming convention follows the format ./dataset/raw/id/context/filename.m4a.

### 1) dataset_to_wav()

This function takes raw .m4a files from ./dataset/raw/ , and builds an identical dataset in ./dataset/processed/ but with
all .m4a files converted to .wav.

### 2) gen_phases

This function generates a file 'phase_map.csv' in .dataset/processed/ which contains a complete indexation of the contents
of ./dataset/processed, along with a variable 'phase' which identifies which contents will be used for training (phase=1),
validation (phase=2) and testing (phase=4).

### 3) check_sample_rates()

This function checks all the wav files inside .dataset/processed/ and plots a histogram of the sample rates. NB: we
require all data to be the same samplerate (16kHz)

### 4) wav_to_spectogram()

Function to convert the .wav files in ./dataset/processed/ into spectograms. Spectograms must be in pytorch binary file
format.

### 4) normalise_spectograms()

Function to perform normlaisation on the spectograms (held in ./dataset/processed/) and to perform a normlaisation on them.
The function overwrites the existing spectograms with their new, normalised version, also saving them as torch binary
files.


### Other

Additional preprocessing steps can be included in this script. Eg: additional function to convert the wav files into
a spectogram in a torch binary file format.

