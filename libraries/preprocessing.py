import numpy as np
from util.constants import * 

def normilizeSample(sample):
  # Normilize keypoint x and y coordinate by subtracting mean and dividing by std of sample
  mean = np.append(np.nanmean(sample[:, :, 0:2], axis=(0,1)), 0)
  std = np.append(np.nanstd(sample[:, :, 0:2], axis=(0,1)), 1)
  return (sample - mean) / std

def remove_missing_keypoints(sample):
  missing_removed = np.copy(sample)
  missing_removed[missing_removed == 0] = np.nan
  return missing_removed

def preprocess_sample(sample):
    pre_sample = remove_missing_keypoints(sample)
    return normilizeSample(pre_sample)

def preprocess(all_samples, verbose=False):
  if verbose:
    print(f"Preprocessing {len(all_samples)} samples")
  return np.array([preprocess_sample(s) for s in all_samples])