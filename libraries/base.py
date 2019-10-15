from util.constants import *
import numpy as np


def get_frame_parts(frame):
  pose = frame[pose_offset:pose_offset+pose_len]
  face = frame[face_offset:face_offset+face_len]
  hand_L = frame[hand_left_offset:hand_left_offset+hand_left_len]
  hand_R = frame[hand_right_offset:hand_right_offset+hand_right_len]
  return pose, face, hand_L, hand_R

def get_sample_parts(data):
  separeted_data=[]
  for sample in data:
    separeted_sample=[]
    for frame in sample:
      separeted_sample.append(get_frame_parts(frame))
    separeted_data.append(separeted_sample)
      
  return np.array(separeted_data)

def get_missing_poses(pose):
  output = []
  tmp = np.concatenate((pose, np.array([pose_labels]).T), axis=1)
  for i,t in enumerate(tmp):
    if float(t[0]) == 0: output.append((i, t[3])) 
  return output


def get_value_arrays(framepart, samples):
  values_x = []
  values_y = []
  values_c = []
  frame_ids = []

  for sample in samples: 
    num_frames = sample.shape[0]
  
    for frame_index in range(num_frames):
      frame_ids.append(frame_index)

      pose, face, hand_L, hand_R = get_frame_parts(sample[frame_index])
      part_array = []
      if framepart == 'POSE':
        values_x.append(pose[:,x_index])
        values_y.append(pose[:,y_index])
        values_c.append(pose[:,c_index]) 
      elif framepart == 'FACE':
        values_x.append(face[:,x_index])
        values_y.append(face[:,y_index])
        values_c.append(face[:,c_index]) 
      elif framepart == 'HAND_L':
        values_x.append(hand_L[:,x_index])
        values_y.append(hand_L[:,y_index])
        values_c.append(hand_L[:,c_index]) 
      elif framepart == 'HAND_R':
        values_x.append(hand_R[:,x_index])
        values_y.append(hand_R[:,y_index])
        values_c.append(hand_R[:,c_index]) 
  return values_x, values_y, values_c, frame_ids

# Define distance
def dist(x,y):
  return np.sqrt(np.power(x-y,2).sum())


def nonzeromean(x):
  return x[x!=0].mean()