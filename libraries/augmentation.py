import numpy as np
import random
from util.constants import *

def interpolate(x):
  out=[]
  for i in range(1,len(x)):
    noise=np.random.normal(size=x[i].shape)
    out.append((x[i]+x[i-1])/2+1.5*noise)
  return np.array(out)

# auguments normalized sample by mirroring it
def mirror_sample(sample):
  mirror_x_mat = np.array([[-1, 0, 0],[0, 1, 0], [0, 0, 1]])
  mirror = np.copy(sample)
  for i, frame in enumerate(sample):
    mirror[i] = mirror[i].dot(mirror_x_mat)
    # switching right and left hand
    mirror[i, 95:116], mirror[i, 116:137] = mirror[i, 116:137], mirror[i, 95:116].copy()
    # switching left and right side of body
    mirror[i][[2,3,4,9,10,11,15,17,22,23,24]], mirror[i][[5,6,7,12,13,14,16,18,19,20,21]] = mirror[i][[5,6,7,12,13,14,16,18,19,20,21]], mirror[i][[2,3,4,9,10,11,15,17,22,23,24]].copy()
    # swichting left and right side of face
    face_left_index = 25 + np.array((0,1,2,3,4,5,6,7,17,18,19,20,21,36,37,68,38,39,40,41,31,32,58,59,48,49,50,67,60,61))
    face_right_index = 25 + np.array((16,15,14,13,12,11,10,9,26,25,24,23,22,45,44,69,43,42,47,46,35,34,56,55,54,53,52,65,64,63))
    mirror[i][face_left_index], mirror[i][face_right_index] = mirror[i][face_right_index], mirror[i][face_left_index].copy()
  return mirror

def skip_fram_agmentation(all_samples, all_labels, all_persons, min_frames=8, max_per_label = 400):
  n_labels = len(np.unique(all_labels))
  samples_per_label = np.empty((n_labels,),dtype=object)
  persons_per_label = np.empty((n_labels,),dtype=object)
  for i in range(n_labels):
    samples_per_label[i] = []
    persons_per_label[i] = []
  for label, sample, person in zip(all_labels, all_samples, all_persons):
    samples_per_label[label].append(sample)
    persons_per_label[label].append(person)
  for i, samples in enumerate(samples_per_label):
    samples_per_label[i] = sorted(samples_per_label[i], key=len, reverse = True)
  
  for i, samples in enumerate(samples_per_label):
    len_orig = len(samples)
    for j in range(len_orig):
      if len(samples_per_label[i][j]) > min_frames:
        persons_per_label[i].append(persons_per_label[i][j])
        samples_per_label[i].append(np.delete(samples_per_label[i][j], random.randint(0, len(samples_per_label[i][j]) - 1), axis=0))
        persons_per_label[i].append(persons_per_label[i][j])
        samples_per_label[i].append(np.delete(samples_per_label[i][j], random.randint(0, len(samples_per_label[i][j]) - 1), axis=0))
  aug_samples = []
  aug_labels = []
  aug_persons = []
  for label, (samples, persons) in enumerate(zip(samples_per_label, persons_per_label)):
    count = 0
    for sample, person in zip(samples, persons):
      if count < max_per_label:
        aug_samples.append(sample)
        aug_labels.append(label)
        aug_persons.append(person)
      count += 1
  return aug_samples, aug_labels, aug_persons

def add_noise_hands(all_samples, all_labels, all_persons, snr = 100):
    np.random.seed(0)
    start_idx_hand = hand_left_offset
    end_idx_hand = hand_left_offset + hand_left_len + hand_right_len
    n_hand_keypoints = end_idx_hand - start_idx_hand
    augmented_samples = []
    for sample in all_samples:
        augmented_sample = sample.copy()
        random_noise = np.random.multivariate_normal(np.zeros(2), np.identity(2), (len(sample), n_hand_keypoints)) / snr
        augmented_sample[:, start_idx_hand:end_idx_hand, x_index:y_index+1] += random_noise
        augmented_samples.append(augmented_sample)
    return np.array(augmented_samples), all_labels, all_persons
    