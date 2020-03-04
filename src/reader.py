import pandas as pd
from scipy.io import loadmat

_data_dir = "./data"

def get_matlab_data(signal_type, user_id, video_id):
    # Do we want to do Preprossing here or in another function?
    data = loadmat('data/Data_Original_P01.mat')

    # Sensor Data
    eeg = data['EEG_DATA']
    ecg = data['ECG_DATA']
    gsr = data['GSR_DATA']

    # Other
    video_ids = data['VideoIDs']

def get_annotations(user_id, video_number):

    valid_inputs = 0 < user_id < 41 and 0 < video_number < 21

    if valid_inputs:
        df = pd.read_excel(_data_dir + '/External_Annotations.xlsx')

        # Filter
        df = df[df['UserID'] == user_id]
        df = df[df['Video_Number'] == video_number]

        # Clean
        df = df.reset_index()
        df = df.drop(['index', 'UserID', 'Video_Number', 'VideoID'], axis=1)

        return df

    raise Exception("Invalid Inputs")

# Used for testing
if __name__ == "__main__":
    df = get_annotations(5, 18)
    print(df)
    df = get_annotations(50, 18)

