import numpy as np
from util.constants import *


def normilizeSample(sample):
	# Normilize keypoint x and y coordinate by subtracting mean and dividing by std of sample
	mean = np.append(np.nanmean(sample[:, :, 0:2], axis=(0, 1)), 0)
	std = np.append(np.nanstd(sample[:, :, 0:2], axis=(0, 1)), 1)
	return (sample - mean) / std


def remove_missing_keypoints(sample):
	missing_removed = np.copy(sample)
	# Replace x and y coordinate with 0 if confidence is negative
	missing_removed[missing_removed[:, :, 2] < 0] = 0
	# Only replacing x and y coordinate of 0 with nan
	missing_removed[:, :, 0:2][missing_removed[:, :, 0:2] == 0] = np.nan
	return missing_removed


def keypoints_nan_to_zero(sample):
	nan_to_zero = np.copy(sample)
	# Only replacing x and y coordinate of nan to zero
	nan_to_zero[nan_to_zero[:, :, 2] == 0] = 0
	return nan_to_zero


def preprocess_sample(sample):
	pre_sample = remove_missing_keypoints(sample)
	return normilizeSample(pre_sample)


def preprocess(all_samples, verbose=False):
	if verbose:
		print(f"Preprocessing {len(all_samples)} samples")
	return np.array([preprocess_sample(s) for s in all_samples])


def preprocess_test(all_samples, verbose=False):
	if verbose:
		print(f"Preprocessing {len(all_samples)} samples")
	return np.array([keypoints_nan_to_zero(preprocess_sample(s)) for s in all_samples])


# Normalize with respect to two points (0: Nose, 1: Neck, 2: RShoulder, 5: LShoulder, 8; MidHip )
def rescale_samples(all_samples, index=[1, 8]):
	rescale_all_samples = np.copy(all_samples)
	for ind, sample in enumerate(all_samples):
		if not ((sample[:, index[0], 2] == 0) & (sample[:, index[1], 2] == 0)).all():
			rescale_all_samples[ind] /= np.sqrt(
				((sample[:, index[0], :2] - sample[:, index[1], :2]) ** 2).sum(-1)).mean(0)
		else:  # quickfix
			rescale_all_samples[ind] /= sample[:, :, :2].std()
	return rescale_all_samples
