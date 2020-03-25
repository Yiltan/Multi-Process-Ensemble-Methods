import preprocess
import reader
import filters
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

def plot_eeg():
    plt.figure(figsize=(8, 4))
    column_name = { 3 : "AF3",
                    4 : "F7",
                    5 : "F3",
                    6 : "FC5",
                    7 : "T7",
                    8 : "P7",
                    9 : "O1",
                   10 : "O2",
                   11 : "P8",
                   12 : "T8",
                   13 : "FC6",
                   14 : "F4",
                   15 : "F8",
                   16 : "AF4"}

    # Pre-processing changes data in place so we must do a deep copy
    raw_data = reader.get_matlab_data('eeg', 1, 12)
    data = preprocess.process_data('eeg', copy.deepcopy(raw_data))

    # EEG channels are 3-16
    #for channel in range(3, 17):
    for channel in range(3, 5):
        # We must also normalize as the processed data is
        y_raw = filters.normalize(raw_data[channel])
        y = data[channel]

        plt.magnitude_spectrum (y, Fs=128, label='Preprocessed')
        plt.magnitude_spectrum(y_raw, Fs=128, label='Raw Data')

        plt.legend()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(column_name[channel] + " " + 'Volatge ($\mu V$)')
        plt.savefig(column_name[channel] + ".png", bbox_inches='tight')
        plt.clf()

def plot_ecg():
    plt.figure(figsize=(8, 4))
    # ECG Ploting
    raw_data = reader.get_matlab_data('ecg', 1, 12)
    data = preprocess.process_data('ecg', copy.deepcopy(raw_data))

    # ECG channels are 1 & 2
    for channel in range(1, 3):
        # We must also normalize as the processed data is
        y_raw = filters.normalize(raw_data[channel])
        y = data[channel]

        plt.magnitude_spectrum(y_raw, Fs=256, label='Raw Data')
        plt.magnitude_spectrum(y, Fs=256, label='Preprocessed')

        plt.legend()
        plt.xlabel('Frequency (Hz)')

        label = "Left Arm" if channel is 1 else "Right Arm"
        plt.ylabel(label + ' Volatge ($\mu V$)')
        plt.savefig("ECG " + label + ".png", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    plot_eeg()
    plot_ecg()
