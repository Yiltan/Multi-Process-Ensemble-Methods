import pandas as pd
from scipy.io import loadmat

_number_of_users = 40
_videos_per_user = 16
# 17-20 correspond to the long videos experiment
_missing_data_subject = [9, 12, 21, 22, 23, 24, 33]

_annotations_cache = None
_assessments_cache = None
_matlab_data_cache = dict()

_data_dir = "./data"
_annotations_path = _data_dir + '/External_Annotations.xlsx'
_assessments_path = _data_dir + '/SelfAsessment.xlsx'
_features_path = _data_dir + '/features-total.csv'

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

# video_number starts from 0 to 15 for short movies
def get_assessments(user_id, video_number):
	global _number_of_users, _videos_per_user
	valid_inputs = 0 < user_id <= _number_of_users and 0 <= video_number < _videos_per_user

	assert valid_inputs, "Invalid Inputs!"

	global _assessments_cache
	if _assessments_cache is None:
		_assessments_cache = pd.read_excel(_assessments_path)
	_assessments_cache = _assessments_cache.drop(['Rep_Index', 'arousal', 'valence', 'dominance', 'liking', 'familiarity',
					'familiarity.1', 'arousal.1', 'valence.1', 'dominance.1', 'liking.1'], axis=1)

	df = _assessments_cache
	# Filter
	df = df[df['UserID'] == user_id]
	df = df[df['Video_Number'] == video_number]

	# Clean
	df = df
	df = df.reset_index()
	return df

def get_all_assessments():
	# make sure get_assessments() runs at least once
	get_assessments(5,5)
	return _assessments_cache

def get_features():
	df = pd.read_csv(_features_path, header=None)
	# user_video = df.loc[:, [0,1]].values
	# df = df.drop([0, 1], axis=1)
	return df

# Used for testing
if __name__ == "__main__":
	# Get data of a specific user and a specific video
	# df = get_annotations(5, 18)
	# print(df)

	# This line should raise the Exception
	# df = get_annotations(50, 18)
	# data = get_matlab_data('eeg', 5, 1)
	# print(data.shape)
	# data = get_matlab_data('ecg', 10, 2)
	# print(data.shape)
	# data = get_matlab_data('gsr', 10, 3)
	# print(data.shape)

	df = get_features()
	# print(user_video)
	# print(user_video.shape)
	# print(df)
	print(df.shape)

	# data = get_assessments(1, 6)
	# print(data.head())

	data = get_all_assessments()
	print(data.shape)
	print(data.head())
