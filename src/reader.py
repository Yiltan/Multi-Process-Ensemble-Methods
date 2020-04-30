import pandas as pd
from scipy.io import loadmat

_number_of_users = 40
_videos_per_user = 20
_missing_data_subject = [9, 12, 21, 22, 23, 24, 33]
_annotations_cache = None
_matlab_data_cache = dict()

_data_dir = "./data"
_annotation_path = _data_dir + '/External_Annotations.xlsx'

def get_matlab_data(signal_type, user_id, video_number):
	
	# - Do we want to do Preprossing here or in another function?
	# + Maybe return the dictionary and filter its data in another function
	# - We kept everything untouched here then

	global _number_of_users, _videos_per_user
	valid_inputs = 0 < user_id <= _number_of_users and 0 < video_number <= _videos_per_user

	global _matlab_data_cache
	if user_id not in _matlab_data_cache:
		_matlab_data_cache[user_id] = loadmat('data/Data_Original_P{:02d}.mat'.format(user_id));

	data = _matlab_data_cache[user_id]
	
	# Sensor Data of user with user_id
	# [0] is present due to parsing of .mat files.
	# Transpose so that we return an array index by [col][row]
	if signal_type == 'eeg':
		eeg_data = data['EEG_DATA']
		eeg_data = eeg_data[0][video_number].transpose()
		# return eeg_data[4:18]
		return eeg_data
	elif signal_type == 'ecg':
		ecg_data = data['ECG_DATA']
		ecg_data = ecg_data[0][video_number].transpose()
		# return ecg_data[2:4]
		return ecg_data
	elif signal_type == 'gsr':
		gsr_data = data['GSR_DATA']
		gsr_data = gsr_data[0][video_number].transpose()
		# return gsr_data[2]
		return gsr_data
	else:
		raise Exception("Invalid data type")

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
	#print(df)
	
	# This line should raise the Exception
	# df = get_annotations(50, 18)
	data = get_matlab_data('eeg', 5, 1)
	print(data.shape)
	data = get_matlab_data('ecg', 10, 2)
	print(data.shape)
	data = get_matlab_data('gsr', 10, 3)
	print(data.shape)
