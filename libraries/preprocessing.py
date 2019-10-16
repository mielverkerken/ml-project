import numpy as np
from util.constants import * 

def normalize2D(frame):
  frame_x = frame[:,0]
  frame_y = frame[:,1]
  norm_frame_x = (frame_x - np.nanmean(frame_x)) / np.nanstd(frame_x)
  norm_frame_y = (frame_y - np.nanmean(frame_y)) / np.nanstd(frame_y)
  return np.vstack((norm_frame_x, norm_frame_y)).T

def normilizeSample(sample):
  norm_sample = np.empty((len(sample), NUM_KEYPOINTS, 2))
  for i, frame in enumerate(sample):
    norm_sample[i] = normalize2D(frame)
  return norm_sample

def remove_missing_keypoints(sample):
  missing_removed = np.copy(sample)
  missing_removed[missing_removed == 0] = np.nan
  return missing_removed

def preprocess_sample(sample):
    pre_sample = remove_missing_keypoints(sample)
    return normilizeSample(pre_sample)