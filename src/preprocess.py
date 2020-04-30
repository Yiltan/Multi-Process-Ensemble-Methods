import reader
import filters
import scipy.signal as signal
import scipy.stats as stats
import numpy as np
import biosppy.signals as bio
import PyEMD

def filter_ecg(data): 
	# ECG sample rate has to be 256 - After 10 hours debug!
	# Just on signal at a time
	data = filters.high_pass(1., 256., data)
	data = filters.low_pass(40., 256., data)
	return data

def filter_eeg(data):
	# EEG sample rate has to be 128 - After 10 hours debug!
	# Accepts on channel at a time
	data = filters.band_pass(4, 45, 128, data)
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
	# Sum band power within desired frequency range 
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
	
def detrend(data):
	# No idea what it's doing! Doesn't matter as long as it works.
    emd = PyEMD.EMD()
    imfs = emd(data)
    detrended = np.sum(imfs[:int(imfs.shape[0] / 2)], axis=0)
    trend = np.sum(imfs[int(imfs.shape[0] / 2):], axis=0)

    return detrended, trend

def process_eeg(data):

	# Total 14 EEG channels
	# Assume that the data contains only EEG signals
	for channel in range(len(data)):
		data[channel] = filter_eeg(data[channel])

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

	for channel in range(len(data)):
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
	
	for i in range(total_power.shape[0]):
		theta_relative_power.append(theta_power[i] / total_power[i])
		slow_alpha_relative_power.append(slow_alpha_power[i] / total_power[i])
		alpha_relative_power.append(alpha_power[i] / total_power[i])
		beta_relative_power.append(beta_power[i] / total_power[i])
		gamma_relative_power.append(gamma_power[i] / total_power[i])
		
	features = theta_power + slow_alpha_power + alpha_power + beta_power + \
		gamma_power + theta_spa + slow_alpha_spa + alpha_spa + beta_spa + \
		gamma_spa + theta_relative_power + slow_alpha_relative_power + \
		alpha_relative_power + beta_relative_power + gamma_relative_power + \
		[np.mean(theta_power), np.mean(alpha_power), np.mean(beta_power), np.mean(gamma_power), \
		np.std(theta_power), np.std(alpha_power), np.std(beta_power), np.std(gamma_power), \
		np.min(theta_power), np.min(alpha_power), np.min(beta_power), np.min(gamma_power), \
		np.max(theta_power), np.max(alpha_power), np.max(beta_power), np.max(gamma_power)]

	return features

def process_ecg(data):
	
	data = filter_ecg(data)
	# Consult with Yiltan about this filter and compare with yours
	# data = filters.butter_highpass_filter(data, 1.0, 256.0)
	
	ecg_all = bio.ecg.ecg(signal=data, sampling_rate=256., show=False)
	rpeaks = ecg_all['rpeaks']  # R-peak location indices.

	# ECG
	freqs, power = get_freqs_power(data, sample_rate=256., nperseg=data.size, scaling='spectrum')
	power_0_6 = []
	for i in range(60):
		power_0_6.append(get_band_power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))

	IBI = np.array([])
	for i in range(len(rpeaks) - 1):
		IBI = np.append(IBI, (rpeaks[i + 1] - rpeaks[i]) / 256.0)

	heart_rate = np.array([])
	for i in range(len(IBI)):
		append_value = 60.0 / IBI[i] if IBI[i] != 0 else 0
		heart_rate = np.append(heart_rate, append_value)

	mean_IBI = np.mean(IBI)
	rms_IBI = np.sqrt(np.mean(np.square(IBI)))
	std_IBI = np.std(IBI)
	skew_IBI = stats.skew(IBI)
	kurt_IBI = stats.kurtosis(IBI)
	per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
	per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)

	# IBI
	freqs_, power_ = get_freqs_power(IBI, sample_rate=1.0 / mean_IBI, 
									nperseg=IBI.size, scaling='spectrum')
	power_000_004 = get_band_power(freqs_, power_, lower=0., upper=0.04)  # VLF
	power_004_015 = get_band_power(freqs_, power_, lower=0.04, upper=0.15)  # LF
	power_015_040 = get_band_power(freqs_, power_, lower=0.15, upper=0.40)  # HF
	power_000_040 = get_band_power(freqs_, power_, lower=0., upper=0.40)  # TF
	# maybe these five features indicate same meaning
	LF_HF = power_004_015 / power_015_040
	LF_TF = power_004_015 / power_000_040
	HF_TF = power_015_040 / power_000_040
	nLF = power_004_015 / (power_000_040 - power_000_004)
	nHF = power_015_040 / (power_000_040 - power_000_004)

	mean_heart_rate = np.mean(heart_rate)
	std_heart_rate = np.std(heart_rate)
	skew_heart_rate = stats.skew(heart_rate)
	kurt_heart_rate = stats.kurtosis(heart_rate)
	per_above_heart_rate = float(heart_rate[heart_rate > mean_heart_rate + 
								std_heart_rate].size) / float(heart_rate.size)
	per_below_heart_rate = float(heart_rate[heart_rate < mean_heart_rate - 
								std_heart_rate].size) / float(heart_rate.size)
								
	features = [rms_IBI, mean_IBI] + power_0_6 + [power_000_004, power_004_015, 
				power_015_040, mean_heart_rate, std_heart_rate, skew_heart_rate,
				kurt_heart_rate, per_above_heart_rate, per_below_heart_rate, 
				std_IBI, skew_IBI, kurt_IBI, per_above_IBI, per_below_IBI, 
				LF_HF, LF_TF, HF_TF, nLF, nHF]

	return features

def process_gsr(data):

	der_data = np.gradient(data)
	con_data = 1.0 / data
	nor_con_data = (con_data - np.mean(con_data)) / np.std(con_data)

	mean = np.mean(data)
	der_mean = np.mean(der_data)
	neg_der_mean = np.mean(der_data[der_data < 0])
	neg_der_pro = float(der_data[der_data < 0].size) / float(der_data.size)

	local_min = 0
	for i in range(data.shape[0] - 1):
		if i == 0:
			continue
		if data[i - 1] > data[i] and data[i] < data[i + 1]:
			local_min += 1

	# Using SC calculates rising time
	det_nor_data, trend = detrend(nor_con_data)
	lp_det_nor_data = filters.low_pass(0.5, 128., det_nor_data)
	der_lp_det_nor_data = np.gradient(lp_det_nor_data)

	rising_time = 0
	rising_cnt = 0
	for i in range(der_lp_det_nor_data.size - 1):
		if der_lp_det_nor_data[i] > 0:
			rising_time += 1
			if der_lp_det_nor_data[i + 1] < 0:
				rising_cnt += 1

	avg_rising_time = rising_time * (1. / 128.) / rising_cnt

	freqs, power = get_freqs_power(data, sample_rate=128., nperseg=data.size, scaling='spectrum')
	power_0_24 = []
	for i in range(21):
		power_0_24.append(get_band_power(freqs, power, lower=0 +
										(i * 0.8 / 7), upper=0.1 + (i * 0.8 / 7)))

	SCSR, _ = detrend(filters.low_pass(0.2, 128., nor_con_data))
	SCVSR, _ = detrend(filters.low_pass(0.08, 128., nor_con_data))

	zero_cross_SCSR = 0
	zero_cross_SCVSR = 0
	peaks_cnt_SCSR = 0
	peaks_cnt_SCVSR = 0
	peaks_value_SCSR = 0.
	peaks_value_SCVSR = 0.

	zc_idx_SCSR = np.array([], int)  # must be int, otherwise it will be float
	zc_idx_SCVSR = np.array([], int)
	for i in range(nor_con_data.size - 1):
		if SCSR[i] * next((j for j in SCSR[i + 1:] if j != 0), 0) < 0:
			zero_cross_SCSR += 1
			zc_idx_SCSR = np.append(zc_idx_SCSR, i + 1)
		if SCVSR[i] * next((j for j in SCVSR[i + 1:] if j != 0), 0) < 0:
			zero_cross_SCVSR += 1
			zc_idx_SCVSR = np.append(zc_idx_SCVSR, i)

	for i in range(zc_idx_SCSR.size - 1):
		peaks_value_SCSR += np.absolute(SCSR[zc_idx_SCSR[i]:zc_idx_SCSR[i + 1]]).max()
		peaks_cnt_SCSR += 1
	for i in range(zc_idx_SCVSR.size - 1):
		peaks_value_SCVSR += np.absolute(SCVSR[zc_idx_SCVSR[i]:zc_idx_SCVSR[i + 1]]).max()
		peaks_cnt_SCVSR += 1

	zcr_SCSR = zero_cross_SCSR / (nor_con_data.size / 128.)
	zcr_SCVSR = zero_cross_SCVSR / (nor_con_data.size / 128.)

	mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
	mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

	features = [mean, der_mean, neg_der_mean, neg_der_pro, local_min, avg_rising_time] + \
		power_0_24 + [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR]

	return features
	
def process_amigos_data():
	
	amigos_data = np.array([])
	corrupted = np.array([])
	for person in range(1, 3): #reader._number_of_users
		if person in reader._missing_data_subject:
			continue;
		for video in range(1, 2): #reader._videos_per_user
			try:
				raw_eeg_data = reader.get_matlab_data('eeg', person, video)
				eeg_features = process_eeg(raw_eeg_data[3:17])
				
				raw_ecg_data = reader.get_matlab_data('ecg', person, video)
				ecg_features_R = process_ecg(raw_ecg_data[1])
				ecg_features_L = process_ecg(raw_ecg_data[2])	
				
				raw_gsr_data = reader.get_matlab_data('gsr', person, video)
				gsr_features = process_gsr(raw_gsr_data[1])
				
				all_features = np.array(eeg_features + ecg_features_R + ecg_features_L + gsr_features)
				amigos_data = np.vstack((amigos_data, all_features)) if len(amigos_data) else all_features
			
			except:
				print("Couldn't extract feature, person: {0}, video: {1}".format(person, video))
				#corrupted = np.vstack((corrupted, [person, video])) if len(corrupted) else [person, video]
	
	return amigos_data
	
# Used for testing
if __name__ == "__main__":
	
	raw_eeg_data = reader.get_matlab_data('eeg', 6, 12)
	eeg_features = process_eeg(raw_eeg_data[3:17])
	print(len(eeg_features))
	
	raw_ecg_data = reader.get_matlab_data('ecg', 1, 15)
	ecg_features_R = process_ecg(raw_ecg_data[1])
	ecg_features_L = process_ecg(raw_ecg_data[2])
	print(len(ecg_features_R))
	print(len(ecg_features_L))
	
	raw_gsr_data = reader.get_matlab_data('gsr', 1, 13)
	gsr_features = process_gsr(raw_gsr_data[1])
	print(len(gsr_features))
	
	# Let's store the preprocessed extracted features.
	np.savetxt('data/features.csv', process_amigos_data(), delimiter=',', fmt='%s')
	
