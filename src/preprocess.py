import reader
import filters
import scipy.signal as signal
import numpy as np

def process_data(signal_type, raw_data):
	# Each signal is filtered differently
	# We are following the guidelines set here:
	# http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html#prep
	if signal_type == 'eeg':
		return process_eeg(raw_data)
	elif signal_type == 'ecg':
		return filter_ecg(raw_data)
	elif signal_type == 'gsr':
		return raw_data
	else:
		raise Exception("Invalid data type")
	# Add SVM etc inside here

def filter_eeg(data):
	# EEG channels are 3-16
	for channel in range(3, 17):
		data[channel] = filters.band_pass(4, 45, 128, data[channel])
	return data

def filter_ecg(data):
	# ECG channels are 1 & 2
	for channel in [1, 2]:
		data[channel] = filters.high_pass(0.05, 256, data[channel])
		data[channel] = filters.band_stop(50, 256, data[channel])
		data[channel] = filters.band_stop(64, 256, data[channel])
	return data

def get_freqs_power(data, sample_rate, nperseg, scaling):
	# Calculate power density or power spectrum density
	if scaling == "density":
		freqs, power = signal.welch(data, fs=sample_rate, nperseg=nperseg, scaling='density')
		return freqs, power
	elif scaling == "spectrum":
		freqs, power = signal.welch(data, fs=sample_rate, nperseg=nperseg, scaling='spectrum')
		return freqs, power
	else:
		return 0, 0

def get_band_power(freqs, power, lower, upper):
	# Sum band power within desired frequency range '''
	low_idx = np.array(np.where(freqs <= lower)).flatten()
	up_idx = np.array(np.where(freqs > upper)).flatten()
	band_power = np.sum(power[low_idx[-1]:up_idx[0]])

	return band_power
	
def get_five_bands_power(freqs, power):
	# Calculate 5 bands power
	theta_power = get_band_power(freqs, power, 3, 7)
	slow_alpha_power = get_band_power(freqs, power, 8, 10)
	alpha_power = get_band_power(freqs, power, 8, 13)
	beta_power = get_band_power(freqs, power, 14, 29)
	gamma_power = get_band_power(freqs, power, 30, 47)
	return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power

def process_eeg(data):

	data = filter_eeg(data)

	# PSD features
	theta_power = []
	slow_alpha_power = []
	alpha_power = []
	beta_power = []
	gamma_power = []
	psd_list = [theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power]

	# Spectral features
	theta_spec_power = []
	slow_alpha_spec_power = []
	alpha_spec_power = []
	beta_spec_power = []
	gamma_spec_power = []
	spec_power_list = [theta_spec_power, slow_alpha_spec_power,
					   alpha_spec_power, beta_spec_power, gamma_spec_power]

	# Spectral Parameter Analysis features 
	theta_spa = []
	slow_alpha_spa = []
	alpha_spa = []
	beta_spa = []
	gamma_spa = []

	# Relative PSD features
	theta_relative_power = []
	slow_alpha_relative_power = []
	alpha_relative_power = []
	beta_relative_power = []
	gamma_relative_power = []

	for channel in range(3, 17):
		freqs, power = get_freqs_power(data[channel], sample_rate=128., 
				nperseg=data[channel].size, scaling='density')

		psd = get_five_bands_power(freqs, power)
		for band, band_list in zip(psd, psd_list):
			band_list.append(band)

		freqs_, power_ = get_freqs_power(data[channel], sample_rate=128.,
				nperseg=data[channel].size, scaling='spectrum')

		spec_power = get_five_bands_power(freqs_, power_)
		for band, band_list in zip(spec_power, spec_power_list):
			band_list.append(band)

	for i in range(7):
		theta_spa.append((theta_spec_power[i] - theta_spec_power[13 - i]) /
						 (theta_spec_power[i] + theta_spec_power[13 - i]))
		slow_alpha_spa.append((slow_alpha_spec_power[i] - slow_alpha_spec_power[13 - i]) /
							  (slow_alpha_spec_power[i] + slow_alpha_spec_power[13 - i]))
		alpha_spa.append((alpha_spec_power[i] - alpha_spec_power[13 - i]) /
						 (alpha_spec_power[i] + alpha_spec_power[13 - i]))
		beta_spa.append((beta_spec_power[i] - beta_spec_power[13 - i]) /
						(beta_spec_power[i] + beta_spec_power[13 - i]))
		gamma_spa.append((gamma_spec_power[i] - gamma_spec_power[13 - i]) /
						 (gamma_spec_power[i] + gamma_spec_power[13 - i]))
	
	total_power = np.array(theta_power) + np.array(alpha_power) + \
		np.array(beta_power) + np.array(gamma_power)
	
	print(total_power.shape)
	
	for i in range(total_power.shape[0]):
		theta_relative_power.append(theta_power[i] / total_power[i])
		slow_alpha_relative_power.append(slow_alpha_power[i] / total_power[i])
		alpha_relative_power.append(alpha_power[i] / total_power[i])
		beta_relative_power.append(beta_power[i] / total_power[i])
		gamma_relative_power.append(gamma_power[i] / total_power[i])

	print(theta_power)
	print(np.max(theta_power))
	print(np.min(theta_power))
	print(np.mean(theta_power))
	print(np.std(theta_power))
	features = theta_power + slow_alpha_power + alpha_power + beta_power + \
		gamma_power + theta_spa + slow_alpha_spa + alpha_spa + beta_spa + \
		gamma_spa + theta_relative_power + slow_alpha_relative_power + \
		alpha_relative_power + beta_relative_power + gamma_relative_power + \
		np.std(theta_power) + np.std(alpha_power) + np.std(beta_power) + np.std(gamma_power) + \
		np.mean(theta_power) + np.mean(alpha_power) + np.mean(beta_power) + np.mean(gamma_power) + \
		np.min(theta_power) + np.min(alpha_power) + np.min(beta_power) + np.min(gamma_power) + \
		np.max(theta_power) + np.max(alpha_power) + np.max(beta_power) + np.max(gamma_power)

	return features

# Used for testing
if __name__ == "__main__":
	raw_data = reader.get_matlab_data('eeg', 1, 12)
	eeg_features = process_data('eeg', raw_data)
	print(eeg_features.shape)
	print(type(eeg_features))
	
	raw_data = reader.get_matlab_data('ecg', 1, 12)
	print(raw_data.shape)
	print(type(raw_data))
	#print(process_data('ecg', raw_data))
