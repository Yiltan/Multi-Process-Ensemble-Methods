import reader
import filters

def process_data(signal_type, raw_data):
    # Each signal is filtered differently
    # We are following the guidelines set here:
    # http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html#prep
    if signal_type == 'eeg':
        return filter_eeg(raw_data)
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
        data[channel] = filters.high_pass(0.5, 256, data[channel])
        data[channel] = filters.band_stop(60, 256, data[channel])
    return data

# Used for testing
if __name__ == "__main__":
    print(get_processed_data('eeg', 1, 12))
    print(get_processed_data('ecg', 1, 12))
