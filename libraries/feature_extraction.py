from util.constants import *
from libraries.base  import *
import numpy as np


def stats(func):
  def wrapper(sample):
    out = []
    for f in func(sample):
      diff1,diff2 = (float('nan'),float('nan')) if len(f) ==1 else (f[(len(f) - 1) // 2] - f[0],f[-1] - f[(len(f) - 1) // 2])
      out.extend([np.max(f), np.min(f), np.mean(f), np.std(f), diff1, diff2])
    return np.array(out)

  return wrapper


def get_arm_angles(pose):
  if np.isnan(pose[l_arm_should][0]) or np.isnan(pose[l_arm_elbow][0]) or np.isnan(pose[l_arm_wrist][0]):
     left_angle = float('NaN')
  else:
    p1 = pose[l_arm_should][:2]  
    p2 = pose[l_arm_elbow][:2]
    p3 = pose[l_arm_wrist][:2]

    v0 = np.array(p1) - np.array(p2)
    v1 = np.array(p3) - np.array(p2)

    left_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

  if np.isnan(pose[r_arm_should][0]) or np.isnan(pose[r_arm_elbow][0]) or np.isnan(pose[r_arm_wrist][0]):
    right_angle = float('NaN')
  else:
    p1 = pose[r_arm_should][:2] 
    p2 = pose[r_arm_elbow][:2]
    p3 = pose[r_arm_wrist][:2]

    v0 = np.array(p1) - np.array(p2)
    v1 = np.array(p3) - np.array(p2)

    right_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  return np.degrees(left_angle), np.degrees(right_angle)

def get_shoulder_angles(pose):
  if np.isnan(pose[neck][0]) or np.isnan(pose[l_arm_should][0]) or np.isnan(pose[l_arm_elbow][0]):
    left_angle = float('NaN')
  else:
    p1 = pose[neck][:2] 
    p2 = pose[l_arm_should][:2]
    p3 = pose[l_arm_elbow][:2]

    v0 = np.array(p1) - np.array(p2)
    v1 = np.array(p3) - np.array(p2)

    left_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  if np.isnan(pose[neck][0]) or np.isnan(pose[r_arm_should][0]) or np.isnan(pose[r_arm_elbow][0]):
    right_angle = float('NaN')
  else:
    p1 = pose[neck][:2] 
    p2 = pose[r_arm_should][:2]
    p3 = pose[r_arm_elbow][:2]

    v0 = np.array(p1) - np.array(p2)
    v1 = np.array(p3) - np.array(p2)

    right_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  return np.degrees(left_angle), np.degrees(right_angle)


@stats
def get_all_arm_angles(sample):
  arm_angles_left = []
  arm_angles_right = []
  for frame in sample:
    pose, _, _, _ = get_frame_parts(frame)
    arm_angles = get_arm_angles(pose)
    arm_angles_left.append(arm_angles[0])
    arm_angles_right.append(arm_angles[1])
  return arm_angles_left, arm_angles_right

@stats
def get_all_shoulder_angles(sample):
  shoulder_angles_left =[]
  shoulder_angles_right =[]
  for frame in sample:
    pose, _,_,_ = get_frame_parts(frame)
    shoulder_angles = get_shoulder_angles(pose)
    shoulder_angles_left.append(shoulder_angles[0])
    shoulder_angles_right.append(shoulder_angles[1])
  return shoulder_angles_left, shoulder_angles_right


def get_number_inflections(dy, threshold=1):
  number_of_ups_downs = 0
  val_pos = (dy[0] > 0)
  for val in dy:
    if (val_pos != (val > 0)) and abs(val) > threshold:
      val_pos = (val>0)
      number_of_ups_downs+=1
  return number_of_ups_downs

def get_hand_movement_raw(sample):
  derivative = [0,0,0,0]
  non_zero_count = [[] for n in range(hand_left_len)]
  for i in range(hand_left_len):
    non_zero_count[i] = [[] for n in range(4)]
    #first derivative with best fit line
    dy_L = np.diff(sample[:,hand_left_offset + i,y_index])
    dx_L = np.diff(sample[:,hand_left_offset + i,x_index])
    dy_R = np.diff(sample[:,hand_right_offset + i,y_index])
    dx_R = np.diff(sample[:,hand_right_offset + i,x_index]) 

    derivative[0] += dx_L
    derivative[1] += dx_R
    derivative[2] += dy_L
    derivative[3] += dy_R

  derivative[0] /= hand_left_len 
  derivative[1] /= hand_left_len 
  derivative[2] /= hand_left_len 
  derivative[3] /= hand_left_len 

  return(derivative[0], derivative[1], derivative[2], derivative[3])

@stats
def get_hand_movement(sample):
  return get_hand_movement_raw(sample)

def arclength(x):
  Sum = 0
  x = x[x[:,0] != x[:,0]]
  size = x.shape[0]
  if size == 0:
    return float('nan')
  for i in range(1, size):
    Sum += dist(x[i], x[i - 1])
  return dist(x[-1], x[0]) / (Sum + 1e-5)

@stats
def finger_openness(sample):
  # @title finger open/closed
  n_fingers = 10
  finger_openness_feature = np.zeros((n_fingers, len(sample)))
  for j, frame in enumerate(sample):
    _, _, hand_L, hand_R = get_frame_parts(frame)
    for k in range(int(n_fingers / 2)):
      # Left Hand
      finger_openness_feature[k, j] = arclength(hand_L[1 + 4 * k:5 + 4 * k])
      # Right Hand
      finger_openness_feature[k + 5, j] = arclength(hand_R[1 + 4 * k:5 + 4 * k])
  return finger_openness_feature


@stats
def shoulder_wrist_y(sample):
  R = []
  L = []
  for frame in sample:
    body, _, _, _ = get_frame_parts(frame)
    c_shoulder = body[[1, 2, 5], 1]
    c_shoulder = c_shoulder[c_shoulder == c_shoulder]
    c_shoulder = c_shoulder.mean()
    r_wrist = body[4, 1]
    l_wrist = body[7, 1]
    d_r = float('nan') if c_shoulder != c_shoulder or r_wrist != r_wrist else c_shoulder - r_wrist
    d_l = float('nan') if c_shoulder != c_shoulder or l_wrist != l_wrist else c_shoulder - l_wrist
    R.append(d_r)
    L.append(d_l)

  return np.array(R), np.array(L)


@stats
def head_hand(sample):
  R = []
  L = []
  head = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[f_head[:, 0] == f_head[:, 0]][:, 0:2]
    f_head = f_head.mean(axis=0)
    head[i] = f_head

  head = head[head[:, 0] == head[:, 0]]
  if len(head):
    head = np.mean(head)
  else:
    return float('nan')

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]][:, 0:2]
    r_hand = r_hand.mean(axis=0)
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]][:, 0:2]
    l_hand = l_hand.mean(axis=0)
    d_r = float('nan') if (r_hand != r_hand).any() else dist(head, r_hand)
    d_l = float('nan') if (l_hand != l_hand).any() else dist(head, l_hand)
    R.append(d_r)
    L.append(d_l)

  return np.array(R), np.array(L)


@stats
def var_hands(sample):
  R = []
  L = []
  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]][:, 0:2]
    r_hand = r_hand.var(axis=0)
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]][:, 0:2]
    l_hand = l_hand.var(axis=0)
    R.append(r_hand)
    L.append(l_hand)
  return np.array(R), np.array(L)


@stats
def chin_thumb(sample):
  R = []
  L = []
  chin = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[7:10, 0:2]
    f_head = f_head[f_head[:, 0] == f_head[:, 0]]
    f_head = f_head.mean(axis=0)
    chin[i] = f_head

  chin = chin[chin[:, 0] == chin[:, 0]]
  if len(chin):
    chin = np.mean(chin)
  else:
    return float('nan')

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[2:5, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    r_hand = r_hand.mean(axis=0)
    l_hand = l_hand[2:5, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    l_hand = l_hand.mean(axis=0)
    d_r = float('nan') if (r_hand != r_hand).any() else dist(chin, r_hand)
    d_l = float('nan') if (l_hand != l_hand).any() else dist(chin, l_hand)
    R.append(d_r)
    L.append(d_l)

  return np.array(R), np.array(L)


@stats
def mouth_index(sample):
  R = []
  L = []
  mouth = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[48:68, 0:2]
    f_head = f_head[f_head[:, 0] == f_head[:, 0]]
    f_head = f_head.mean(axis=0)
    mouth[i] = f_head

  mouth = mouth[mouth[:, 0] == mouth[:, 0]]
  if len(mouth):
    mouth = np.mean(mouth)
  else:
    return float('nan')

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[6:9, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    r_hand = r_hand.mean(axis=0)
    l_hand = l_hand[6:9, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    l_hand = l_hand.mean(axis=0)
    d_r = float('nan') if (r_hand != r_hand).any() else dist(mouth, r_hand)
    d_l = float('nan') if (l_hand != l_hand).any() else dist(mouth, l_hand)
    R.append(d_r)
    L.append(d_l)

  return np.array(R), np.array(L)


@stats
def thumb_pink(sample):
  R = []
  L = []
  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)

    r_hand1 = r_hand[2:5, 0:2]
    r_hand1 = r_hand1[r_hand1[:, 0] == r_hand1[:, 0]]
    r_hand1 = r_hand1.mean(axis=0)
    l_hand1 = l_hand[2:5, 0:2]
    l_hand1 = l_hand1[l_hand1[:, 0] == l_hand1[:, 0]]
    l_hand1 = l_hand1.mean(axis=0)

    r_hand2 = r_hand[18:21, 0:2]
    r_hand2 = r_hand2[r_hand2[:, 0] == r_hand2[:, 0]]
    r_hand2 = r_hand2.mean(axis=0)
    l_hand2 = l_hand[18:21, 0:2]
    l_hand2 = l_hand2[l_hand2[:, 0] == l_hand2[:, 0]]
    l_hand2 = l_hand2.mean(axis=0)
    d_r = float('nan') if (r_hand1 != r_hand1).any() or (r_hand2 != r_hand2).any() else dist(r_hand1, r_hand2)
    d_l = float('nan') if (l_hand1 != l_hand1).any() or (l_hand2 != l_hand2).any() else dist(l_hand1, l_hand2)
    R.append(d_r)
    L.append(d_l)
  return np.array(R), np.array(L)


@stats
def index_index(sample):
  out = []
  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[6:9, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    r_hand = r_hand.mean(axis=0)
    l_hand = l_hand[6:9, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    l_hand = l_hand.mean(axis=0)
    d = float('nan') if (r_hand != r_hand).any() or (l_hand != l_hand).any() else dist(r_hand, l_hand)
    out.append(d)
  return np.array(out)


@stats
def wrist_wrist_x(sample):
  out = []
  for frame in sample:
    body, _, _, _ = get_frame_parts(frame)
    r_wrist = body[4, 0]
    l_wrist = body[7, 0]
    d = float('nan') if l_wrist != l_wrist or r_wrist != r_wrist else l_wrist - r_wrist
    out.append(d)

  return [np.array(out)]

def generate_feature_matrix(all_samples):
  NUM_SAMPLES = len(all_samples)

  #FEATURE_MATRIX = np.array((NUM_SAMPLES, NUM_FEATURES))
  FEATURE_MATRIX = [[] for _ in range(NUM_SAMPLES)]

  for i, sample in enumerate(all_samples):
    FEATURE_MATRIX[i] = [[] for _ in range(NUM_FEATURES)]

    #angle features  
    if(len(sample)>1):
      arm_angles = get_all_arm_angles(sample)
      should_angles = get_all_shoulder_angles(sample)

      FEATURE_MATRIX[i][ARM_L_ANGLE_FEATURE] = arm_angles[:(len(arm_angles)//2)]
      FEATURE_MATRIX[i][ARM_R_ANGLE_FEATURE] = arm_angles[(len(arm_angles)//2):]

      FEATURE_MATRIX[i][SHOULD_ANGLE_L_FEATURE] = should_angles[:(len(should_angles)//2)]
      FEATURE_MATRIX[i][SHOULD_ANGLE_R_FEATURE] = should_angles[(len(should_angles)//2):]

    #hand movement features
    if(len(sample)>2):
      hand_movements = get_hand_movement(sample)

      FEATURE_MATRIX[i][HAND_MOVEMENT_L_VERT_FEATURE] = hand_movements[:(len(hand_movements)//4)] 
      FEATURE_MATRIX[i][HAND_MOVEMENT_R_VERT_FEATURE] = hand_movements[(len(hand_movements)//4)*1:(len(hand_movements)//4)*2] 
      FEATURE_MATRIX[i][HAND_MOVEMENT_L_HOR_FEATURE] = hand_movements[(len(hand_movements)//4)*2:(len(hand_movements)//4)*3]  
      FEATURE_MATRIX[i][HAND_MOVEMENT_R_HOR_FEATURE] = hand_movements[(len(hand_movements)//4)*3:(len(hand_movements)//4)*4]

      (dx_L, dx_R, dy_L, dy_R) = get_hand_movement_raw(sample)
      (side_L, side_R, ups_L, ups_R) = get_number_inflections(dx_L),get_number_inflections(dx_R),get_number_inflections(dy_L) ,get_number_inflections(dy_R)


      FEATURE_MATRIX[i][INFLECTIONS_L_VERT_FEATURE] = ups_L
      FEATURE_MATRIX[i][INFLECTIONS_R_VERT_FEATURE] = ups_R
      FEATURE_MATRIX[i][INFLECTIONS_L_HOR_FEATURE] = side_L
      FEATURE_MATRIX[i][INFLECTIONS_R_HOR_FEATURE] = side_R
    
    #finger openess features
    finger_openess = finger_openness(sample)
    FEATURE_MATRIX[i][FINGER_OPENNESS] = finger_openess
