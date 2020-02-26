from scipy.io import loadmat

def get_matlab_data(signal_type, participant_id, video_id):
    data = loadmat('data/Data_Original_P01.mat')

    # Sensor Data
    eeg = data['EEG_DATA']
    ecg = data['ECG_DATA']
    gsr = data['GSR_DATA']

    # Other
    video_ids = data['VideoIDs']
