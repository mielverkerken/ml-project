import numpy as np
from util.constants import *


def normilizeSample(sample):
	# Normilize keypoint x and y coordinate by subtracting mean and dividing by std of sample
	mean = np.append(np.nanmean(sample[:, :, 0:2], axis=(0, 1)), 0)
	std = np.append(np.nanstd(sample[:, :, 0:2], axis=(0, 1)), 1)
	return (sample - mean) / std


def remove_missing_keypoints(sample, prec=0):
	missing_removed = np.copy(sample)
	# Replace x and y coordinate with 0 if confidence is negative
	missing_removed[missing_removed[:, :, 2] < prec] = 0
	# Only replacing x and y coordinate of 0 with nan
	missing_removed[:, :, 0:2][missing_removed[:, :, 0:2] == 0] = np.nan
	return missing_removed


def keypoints_nan_to_zero(sample):
	nan_to_zero = np.copy(sample)
	# Only replacing x and y coordinate of nan to zero
	nan_to_zero[nan_to_zero[:, :, 2] == 0] = 0
	return nan_to_zero


def preprocess_sample(sample, prec=0):
	pre_sample = remove_missing_keypoints(sample, prec=prec)
	return normilizeSample(pre_sample)


def preprocess(all_samples, verbose=False, prec=0):
	if verbose:
		print(f"Preprocessing {len(all_samples)} samples")
	return np.array([preprocess_sample(s, prec=prec) for s in all_samples])


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


# Translate all keypoints to mean of body part
def absolute_to_relative(all_samples):
	"""
	Transform points from absolute coordinates to coordinates relative to the mean
	of the body-part it belongs to.

	Applies normalization w.r.t. the estimated size of the body-part.

	:param all_samples:
	:return:
"""
	# Split keypoints into body-parts
	body_part_count = 4
	start_ind = [0, 25, 95, 116, 137]  # pose, face, left-hand, right-hand
	# Compute mean for each body-part
	means = []
	for m in range(len(all_samples)):
		frame_count = len(all_samples[m])
		matrix = np.zeros((frame_count, body_part_count, 2))  # 2 = (x, y)
		for f in range(frame_count):
			for b in range(body_part_count):
				points = all_samples[m][f, start_ind[b]:start_ind[b + 1]]
				matrix[f, b, 0] = sum(points[:, 0]) / len(points)
				matrix[f, b, 1] = sum(points[:, 1]) / len(points)
		means.append(matrix)
		# Transform absolute to relative
		absolute_to_relative_all_samples = np.copy(all_samples)
	for m in range(len(all_samples)):
		frame_count = len(absolute_to_relative_all_samples[m])
		for f in range(frame_count):
			for b in range(body_part_count):
				points = absolute_to_relative_all_samples[m][f, start_ind[b]:start_ind[b + 1]]
				# Absolute to relative
				points[:, 0] -= means[m][f, b, 0]
				points[:, 1] -= means[m][f, b, 1]
				# Get max dist from mean
				m_dist = max(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2))
				# Normalize
				if m_dist > 0:
					points[:, 0] /= m_dist
					points[:, 1] /= m_dist
	return absolute_to_relative_all_samples
