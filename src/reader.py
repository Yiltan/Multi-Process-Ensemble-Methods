import pandas as pd
from scipy.io import loadmat

_number_of_users = 10
_videos_per_user = 20

_annotations_cache = None
_matlab_data_cache = dict()

_data_dir = "./data"
_annotation_path = _data_dir + '/External_Annotations.xlsx'

def get_matlab_data(signal_type, user_id, video_number):

    # - Do we want to do Preprossing here or in another function?
    # + Maybe return the dictionary and filter its data in another function

    global _number_of_users, _videos_per_user
    valid_inputs = 0 < user_id <= _number_of_users and 0 < video_number <= _videos_per_user

    global _matlab_data_cache
    if user_id not in _matlab_data_cache:
        _matlab_data_cache[user_id] = loadmat('data/Data_Original_P{:02d}.mat'.format(user_id));

    data = _matlab_data_cache[user_id]

    # Sensor Data of user with user_id
    if signal_type == 'eeg':
        data = data['EEG_DATA']
    elif signal_type == 'ecg':
        data = data['ECG_DATA']
    elif signal_type == 'gsr':
        data = data['GSR_DATA']
    else:
        raise Exception("Invalid data type")

    # [0] is present due to parsing of .mat files.
    # Transpose so that we return an array index by [col][row]
    return data[0][video_number].transpose()


def get_annotations(user_id, video_number):

    global _number_of_users, _videos_per_user
    valid_inputs = 0 < user_id <= _number_of_users and 0 < video_number <= _videos_per_user

    assert valid_inputs, "Invalid Inputs!"

    global _annotations_cache
    if _annotations_cache is None:
        _annotations_cache = pd.read_excel(_annotation_path)
    df = _annotations_cache

    # Filter
    df = df[df['UserID'] == user_id]
    df = df[df['Video_Number'] == video_number]

    # Clean
    df = df.reset_index()
    df = df.drop(['index', 'UserID', 'Video_Number', 'VideoID'], axis=1)
    return df


# Used for testing
if __name__ == "__main__":
    # Get data of a specific user and a specific video
    df = get_annotations(5, 18)
    print(df)

    # This line should raise the Exception
    # df = get_annotations(50, 18)

    data = get_matlab_data('eeg', 5)
    data = get_matlab_data('ecg', 10)
    data = get_matlab_data('efg', 10)

