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
