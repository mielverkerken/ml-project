from util.constants import *
from libraries.base  import *
import numpy as np

def get_arm_angles(pose):
  p1 = pose[l_arm_should][:2] 
  p2 = pose[l_arm_elbow][:2]
  p3 = pose[l_arm_wrist][:2]

  v0 = np.array(p1) - np.array(p2)
  v1 = np.array(p3) - np.array(p2)

  left_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

  p1 = pose[r_arm_should][:2] 
  p2 = pose[r_arm_elbow][:2]
  p3 = pose[r_arm_wrist][:2]

  v0 = np.array(p1) - np.array(p2)
  v1 = np.array(p3) - np.array(p2)

  right_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  return np.degrees(left_angle), np.degrees(right_angle)

def get_shoulder_angles(pose):
  p1 = pose[neck][:2] 
  p2 = pose[l_arm_should][:2]
  p3 = pose[l_arm_elbow][:2]

  v0 = np.array(p1) - np.array(p2)
  v1 = np.array(p3) - np.array(p2)

  left_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

  p1 = pose[neck][:2] 
  p2 = pose[r_arm_should][:2]
  p3 = pose[r_arm_elbow][:2]

  v0 = np.array(p1) - np.array(p2)
  v1 = np.array(p3) - np.array(p2)

  right_angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  return np.degrees(left_angle), np.degrees(right_angle)
# To measure if a finger is open or closed
def arclength(x):
  Sum=0
  x=x[x!=0]
  size=len(x)
  for i in range(1,size):
    Sum+=dist(x[i],x[i-1])
  return dist(x[-1],x[0])/(Sum+1e-5)

def finger_openness(sample):
  # @title finger open/closed
  n_fingers = 10
  finger_openness_feature=np.zeros((len(sample), n_fingers))
  for j, frame in enumerate(sample):
    _, _, hand_L, hand_R= get_frame_parts(frame)
    for k in range(int(n_fingers / 2)):
      # Left Hand
      finger_openness_feature[ j, k] = arclength(hand_L[1 + 4 * k:5 + 4 * k])
      # Right Hand
      finger_openness_feature[ j, k + 5] = arclength(hand_R[1 + 4 * k:5 + 4 * k])
  return finger_openness_feature

def get_number_inflections(dy, threshold=1):
  number_of_ups_downs = 0
  val_pos = (dy[0] > 0)
  for val in dy:
    if (val_pos != (val > 0)) and abs(val) > threshold:
      val_pos = (val>0)
      number_of_ups_downs+=1
  return number_of_ups_downs


def get_hand_movement(sample):
  derivative = [0,0,0,0]
  for i in range(hand_left_len):
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

  # @title finger open/closed

  def stats(func):
    def wrapper(sample):
      out = []
      for f in func(sample):
        assert len(f) > 1
        diff1 = f[(len(f) - 1) // 2] - f[0]
        diff2 = f[-1] - f[(len(f) - 1) // 2]
        out.extend([np.max(f), np.min(f), np.mean(f), np.max(f), np.std(f), diff1, diff2])
      return np.array(out)

    return wrapper

  def arclength(x):
    Sum = 0
    x = x[x != 0]
    size = len(x)
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

def create_feature_matrix(all_samples, all_labels):
  NUM_FEATURES = 5
  NUM_SAMPLES = len(all_samples)

  #FEATURE_MATRIX = np.array((NUM_SAMPLES, NUM_FEATURES))
  FEATURE_MATRIX = [[] for _ in range(NUM_SAMPLES)]

  ARM_ANGLE_FEATURE = 0
  SHOULD_ANGLE_FEATURE = 1
  HAND_MOVEMENT_FEATURE = 2
  INFLECTIONS_FEATURE = 3
  FINGER_OPENNESS = 4

  for i, sample in enumerate(all_samples):
    FEATURE_MATRIX[i] = [[] for _ in range(NUM_FEATURES)]

    #angle features  
    arm_angles = []
    should_angles =[]
    for j, frame in enumerate(sample):
      pose, face, hand_L, hand_R = get_frame_parts(frame)
      arm_angles.append(list(get_arm_angles(pose)))
      should_angles.append(list(get_shoulder_angles(pose)))

    FEATURE_MATRIX[i][ARM_ANGLE_FEATURE] = arm_angles
    FEATURE_MATRIX[i][SHOULD_ANGLE_FEATURE] = should_angles

    #hand movement features
    if(len(sample)>1):
      (dx_L, dy_L, dx_R, dy_R ) = get_hand_movement(sample)
      (side_L, side_R, ups_L, ups_R) = get_number_inflections(dx_L),get_number_inflections(dx_R),get_number_inflections(dy_L) ,get_number_inflections(dy_R)
      FEATURE_MATRIX[i][HAND_MOVEMENT_FEATURE] = [dx_L, dy_L, dx_R, dy_R]
      FEATURE_MATRIX[i][INFLECTIONS_FEATURE] = [side_L, ups_L, side_R, ups_R]
    
    #finger openess features
    finger_openess = finger_openness(sample)
    FEATURE_MATRIX[i][FINGER_OPENNESS] = finger_openess

    return FEATURE_MATRIX