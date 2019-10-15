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

def sample_info(sample_id):
  label = all_labels[sample_id]
  gloss = label_to_gloss[label]
  person = all_persons[sample_id]
  print("corresponds to label %d (%s) with person %s" %(label, gloss, person))


def display_sample(sample_id):
  sample = all_samples[sample_id]
  tb = widgets.TabBar([str(i) for i in range(mat.shape[0])])
  for i in range(sample.shape[0]): 
    with tb.output_to(i):
      V.visualize(mat[i])


def get_value_arrays(framepart, sample_range):
  values_x = []
  values_y = []
  values_c = []
  labels = []
  frame_ids = []

  for sample_id in sample_range: 
    sample = all_samples[sample_id]
    label = all_labels[sample_id]
    person = all_persons[sample_id]
    label_descr = label_to_gloss[label]
  
    num_frames = sample.shape[0]
  
    for frame_index in range(num_frames):
      labels.append(label)
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
  return values_x, values_y, values_c, labels, frame_ids

# Define distance
def dist(x,y):
  return np.sqrt(np.power(x-y,2).sum())

# To measure if a finger is open or closed
def arclength(x,size):
  Sum=0
  for i in range(1,size):
    if x[i].all() and x[i-1].all():
       Sum+=dist(x[i],x[i-1])
  return dist(x[size-1],x[0])/(Sum+1e-5)
def nonzeromean(x):
  return x[x!=0].mean()