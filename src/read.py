from scipy.io import loadmat

data = loadmat('data/Data_Original_P01.mat')

# Sensor Data
eeg = data['EEG_DATA']
ecg = data['ECG_DATA']
gsr = data['GSR_DATA']

# Other
video_ids = data['VideoIDs']

print("EEG")
print(eeg.shape)
print(eeg[0].shape)
print(eeg[0][0].shape)

print("ECG")
print(ecg.shape)
print(ecg[0].shape)
print(ecg[0][0].shape)

print("GSR")
print(gsr.shape)
print(gsr[0].shape)
print(gsr[0][0].shape)
