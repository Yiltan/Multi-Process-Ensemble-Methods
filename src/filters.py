import scipy
import scipy.signal as signal

def normalize(data):
	mean = data.mean()
	stddev = data.std()
	return (data - mean) / stddev

def high_pass(highcut, sample_rate, data, order=5):
	cutoff = highcut / (0.5 * sample_rate)
	b, a = signal.butter(order, cutoff, btype='high', analog=False)
	filtered = signal.filtfilt(b, a, data) # padlen=150
	return filtered
	
def low_pass(highcut, sample_rate, data, order=5):
	cutoff = highcut / (0.5 * sample_rate)
	b, a = signal.butter(order, cutoff, btype='low')
	filtered = signal.filtfilt(b, a, data)
	return filtered

def band_pass(lowcut, highcut, sample_rate, data, order=5):
	low = lowcut / (0.5 * sample_rate)
	high = highcut / (0.5 * sample_rate)
	b, a = signal.butter(order, [low, high], btype='band')
	filtered = signal.filtfilt(b, a, data, padlen=150)
	return filtered

def band_stop(stop, sample_rate, data, order=5):
	low  = (stop - 0.5) / (0.5 * sample_rate)
	high = (stop + 0.5) / (0.5 * sample_rate)
	b, a = signal.butter(order, [low, high], btype='bandstop')
	filtered = signal.filtfilt(b, a, data, padlen=150)
	return filtered

'''
def butter_lowpass_filter(data, cutoff, fs, order=5):
	def butter_lowpass(cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
		return b, a
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = signal.lfilter(b, a, data)
	return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
'''
	
# Used for testing
if __name__ == "__main__":
	print(scipy.__version__)
