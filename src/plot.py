import preprocess
import reader
import filters
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

if os.name != 'nt':
    from matplotlib import rc
    rc('text', usetex=True)

def plot_eeg_freq():
	plt.figure(figsize=(8, 4))
	column_name = { 0 : "AF3",
					1 : "F7",
					2 : "F3",
					3 : "FC5",
					4 : "T7",
					5 : "P7",
					6 : "O1",
					7 : "O2",
					8: "P8",
					9 : "T8",
				   10 : "FC6",
				   11 : "F4",
				   12 : "F8",
				   13 : "AF4"}

	# Pre-processing changes data in place so we must do a deep copy
	raw_data = reader.get_matlab_data('eeg', 1, 12)
	data = preprocess.filter_eeg(copy.deepcopy(raw_data))

	# EEG channels are 3-16
	#for channel in range(0, 14):
	for channel in range(0, 2):
		# We must also normalize as the processed data is
		y_raw = filters.normalize(raw_data[channel])
		y = filters.normalize(data[channel])

		plt.magnitude_spectrum (y, Fs=128, label='Filtered')
		plt.magnitude_spectrum(y_raw, Fs=128, label='Raw Data')

		plt.legend()
		plt.xlabel('Frequency (Hz)')
		plt.ylabel(column_name[channel] + " " + 'Volatge ($\mu V$)')
		plt.savefig(column_name[channel] + ".png", bbox_inches='tight')
		plt.clf()

def plot_ecg_freq():
	plt.figure(figsize=(8, 4))
	# ECG Ploting
	raw_data = reader.get_matlab_data('ecg', 1, 12)

	data = [preprocess.filter_ecg(copy.deepcopy(raw_data[0])),
			preprocess.filter_ecg(copy.deepcopy(raw_data[1]))]

	# ECG channels are 1 & 2
	for channel in range(0, 2):
		# We must also normalize as the processed data is
		y_raw = filters.normalize(raw_data[channel])
		y = filters.normalize(data[channel])

		plt.magnitude_spectrum(y_raw, Fs=256, label='Raw Data')
		plt.magnitude_spectrum(y, Fs=256, label='Preprocessed')

		plt.legend()
		plt.xlabel('Frequency (s)')

		label = "Left Arm" if channel is 1 else "Right Arm"
		plt.ylabel(label + ' Volatge ($\mu V$)')
		plt.savefig("ECG " + label + ".png", bbox_inches='tight')
		plt.clf()

def plot_ecg_time():
	plt.figure(figsize=(8, 4))
	# ECG Ploting
	raw_data = reader.get_matlab_data('ecg', 1, 12)

	data = [preprocess.filter_ecg(copy.deepcopy(raw_data[0])),
			preprocess.filter_ecg(copy.deepcopy(raw_data[1]))]

	x = data[0] - data[0][0]

	# ECG channels are 1 & 2
	for channel in range(0, 2):
		# We must also normalize as the processed data is
		y_raw = filters.normalize(raw_data[channel])
		y = filters.normalize(data[channel])

		plt.plot(x, y_raw, label='Raw Data')
		plt.plot(x, y, label='Preprocessed')

		plt.legend()
		plt.xlabel('Time (s)')

		label = "Left Arm" if channel is 1 else "Right Arm"
		plt.ylabel(label + ' Volatge ($\mu V$)')
		plt.savefig("ECG " + label + " time.png", bbox_inches='tight')
		plt.clf()


if __name__ == "__main__":
	plot_eeg_freq()
	plot_ecg_freq()
	plot_ecg_time()
