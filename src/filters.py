import scipy.signal as signal

def normalize(data):
    mean = data.mean()
    stddev = data.std()
    return (data - mean) / stddev

def low_pass(lowcut, sample_rate, data, order=5):
    cutoff= lowcut / (0.5 * sample_rate)
    b, a = signal.butter(order, cutoff, btype='low')
    filtered = signal.filtfilt(b, a, data, padlen=150)
    return normalize(filtered)

def band_pass(lowcut, highcut, sample_rate, data, order=5):
    low = lowcut / (0.5 * sample_rate)
    high = highcut / (0.5 * sample_rate)
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, padlen=150)
    return normalize(filtered)

# Used for testing
if __name__ == "__main__":
    print(scipy.__version__)
