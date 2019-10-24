from util.constants import *
from libraries.base  import *
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd

def stats(func):
  def wrapper(sample):
    out = []
    for f in func(sample):
      f=f[f==f]
      if len(f) == 0:
        out.extend([np.nan]*6)
      else:
        diff1,diff2 =(f[(len(f) - 1) // 2] - f[0],f[-1] - f[(len(f) - 1) // 2])
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
  return np.abs(np.degrees(left_angle)), np.abs(np.degrees(right_angle))

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
  return np.abs(np.degrees(left_angle)), np.abs(np.degrees(right_angle))


@stats
def get_all_arm_angles(sample):
  arm_angles_left = []
  arm_angles_right = []
  for frame in sample:
    pose, _, _, _ = get_frame_parts(frame)
    arm_angles = get_arm_angles(pose)
    arm_angles_left.append(arm_angles[0])
    arm_angles_right.append(arm_angles[1])
  return np.array(arm_angles_left), np.array(arm_angles_right)

@stats
def get_all_shoulder_angles(sample):
  shoulder_angles_left =[]
  shoulder_angles_right =[]
  for frame in sample:
    pose, _,_,_ = get_frame_parts(frame)
    shoulder_angles = get_shoulder_angles(pose)
    shoulder_angles_left.append(shoulder_angles[0])
    shoulder_angles_right.append(shoulder_angles[1])
  return np.array(shoulder_angles_left), np.array(shoulder_angles_right)


def get_number_inflections(dy, threshold=1):
  number_of_ups_downs = 0
  val_pos = (dy[0] > 0)
  for val in dy:
    if (val_pos != (val > 0)) and abs(val) > threshold:
      val_pos = (val>0)
      number_of_ups_downs+=1
  return number_of_ups_downs

def get_hand_movement_raw(sample):
  baricenter_L = []
  baricenter_R = []
  for frame in sample:
    bcl = frame[hand_left_offset:hand_left_offset+hand_left_len, x_index:y_index+1]
    bcl = bcl[bcl[:,0]==bcl[:,0]]
    if len(bcl)==0:
      bcl = np.array([np.nan]*2)
    else:
      bcl = bcl.mean(axis=0)
    baricenter_L.append(bcl)


    bcr = frame[hand_right_offset:hand_right_offset+hand_right_len, x_index:y_index+1]
    bcr = bcr[bcr[:,0]==bcr[:,0]]
    if len(bcr)==0:
      bcr = np.array([np.nan]*2)
    else:
      bcr = bcr.mean(axis=0)
    baricenter_R.append(bcr)

  dx_L,dy_L = np.diff(np.array(baricenter_L).T)
  dx_R,dy_R = np.diff(np.array(baricenter_R).T)

  return(dx_L, dx_R, dy_L, dy_R)
  
@stats
def get_hand_movement(sample):
  return get_hand_movement_raw(sample)

def arclength(x):
  Sum = 0
  x = x[x[:,0] == x[:,0]]
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
    if len(c_shoulder)==0:
      c_shoulder=float('nan')
    else:
      c_shoulder = c_shoulder.mean()
    r_wrist = body[4, 1]
    l_wrist = body[7, 1]
    d_r =  c_shoulder - r_wrist
    d_l =  c_shoulder - l_wrist
    R.append(d_r)
    L.append(d_l)

  return np.array(L), np.array(R)


@stats
def head_hand(sample):
  R = []
  L = []
  head = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[f_head[:, 0] == f_head[:, 0]][:, 0:2]
    if len(f_head):
      f_head = f_head.mean(axis=0)
      head[i] = f_head

  head = head[head[:, 0] == head[:, 0]]
  if len(head):
    head = np.mean(head)
  else:
    return np.array([[float('nan')], [float('nan')]])

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]][:, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]][:, 0:2]
    if len(r_hand) == 0:
      d_r = np.nan
    else:
      r_hand = r_hand.mean(axis=0)
      d_r = dist(head, r_hand)

    if len(l_hand) == 0:
      d_l = np.nan
    else:
      l_hand = l_hand.mean(axis=0)
      d_l =  dist(head, l_hand)

    R.append(d_r)
    L.append(d_l)

  return np.array(L), np.array(R)


@stats
def var_hands(sample):
    Rx=[]
    Ry=[]
    Lx=[]
    Ly=[]
    for frame in sample:
        _, _, r_hand,l_hand = get_frame_parts(frame)
        r_hand=r_hand[r_hand[:,0]==r_hand[:,0]][:,0:2]
        if len(r_hand):
          r_hand=r_hand.var(axis=0)
        else:
          r_hand=[np.nan]*2
        l_hand=l_hand[l_hand[:,0]==l_hand[:,0]][:,0:2]
        if len(l_hand):
          l_hand=l_hand.var(axis=0)
        else:
          l_hand=[np.nan]*2
        Rx.append(r_hand[0])
        Ry.append(r_hand[1])
        Lx.append(l_hand[0])
        Ly.append(l_hand[1])
    return [np.array(Lx),np.array(Ly),np.array(Rx),np.array(Ry)]

@stats
def chin_thumb(sample):
  R = []
  L = []
  chin = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[7:10, 0:2]
    f_head = f_head[f_head[:, 0] == f_head[:, 0]]
    if len(f_head):
      f_head = f_head.mean(axis=0)
      chin[i] = f_head

  chin = chin[chin[:, 0] == chin[:, 0]]
  if len(chin):
    chin = np.mean(chin)
  else:
    return np.array([[float('nan')], [float('nan')]])

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[2:5, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    if len(r_hand):
        r_hand = r_hand.mean(axis=0)
    else:
        r_hand=np.nan
    l_hand = l_hand[2:5, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    if len(l_hand):
      l_hand = l_hand.mean(axis=0)
    else:
      l_hand = np.nan
    d_r = dist(chin, r_hand)
    d_l = dist(chin, l_hand)
    R.append(d_r)
    L.append(d_l)

  return np.array(L), np.array(R)


@stats
def mouth_index(sample):
  R = []
  L = []
  mouth = np.zeros((len(sample), 2))
  for i, frame in enumerate(sample):
    _, f_head, _, _ = get_frame_parts(frame)
    f_head = f_head[48:68, 0:2]
    f_head = f_head[f_head[:, 0] == f_head[:, 0]]
    if len(f_head):
      f_head = f_head.mean(axis=0)
      mouth[i] = f_head

  mouth = mouth[mouth[:, 0] == mouth[:, 0]]
  if len(mouth):
    mouth = np.mean(mouth)
  else:
    return np.array([[float('nan')], [float('nan')]])

  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[6:9, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    if len(r_hand):
      r_hand = r_hand.mean(axis=0)
    else:
      r_hand = np.nan

    l_hand = l_hand[6:9, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    if len(l_hand):
      l_hand = l_hand.mean(axis=0)
    else:
      l_hand = np.nan

    d_r = dist(mouth, r_hand)
    d_l = dist(mouth, l_hand)
    R.append(d_r)
    L.append(d_l)

  return np.array(L), np.array(R)


@stats
def thumb_pink(sample):
  R = []
  L = []
  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)

    r_hand1 = r_hand[2:5, 0:2]
    r_hand1 = r_hand1[r_hand1[:, 0] == r_hand1[:, 0]]

    if len(r_hand1):
      r_hand1 = r_hand1.mean(axis=0)
    else:
      r_hand1 = np.nan
    l_hand1 = l_hand[2:5, 0:2]
    l_hand1 = l_hand1[l_hand1[:, 0] == l_hand1[:, 0]]
    if len(l_hand1):
      l_hand1 = l_hand1.mean(axis=0)
    else:
      l_hand1 = np.nan

    r_hand2 = r_hand[18:21, 0:2]
    r_hand2 = r_hand2[r_hand2[:, 0] == r_hand2[:, 0]]
    if len(r_hand2):
      r_hand2 = r_hand2.mean(axis=0)
    else:
      r_hand2 = np.nan

    l_hand2 = l_hand[18:21, 0:2]
    l_hand2 = l_hand2[l_hand2[:, 0] == l_hand2[:, 0]]
    if len(l_hand2):
      l_hand2 = l_hand2.mean(axis=0)
    else:
      l_hand2 = np.nan

    d_r =  dist(r_hand1, r_hand2)
    d_l =  dist(l_hand1, l_hand2)
    R.append(d_r)
    L.append(d_l)
  return np.array(L), np.array(R)


@stats
def index_index(sample):
  out = []
  for frame in sample:
    _, _, r_hand, l_hand = get_frame_parts(frame)
    r_hand = r_hand[6:9, 0:2]
    r_hand = r_hand[r_hand[:, 0] == r_hand[:, 0]]
    if len(r_hand):
      r_hand = r_hand.mean(axis=0)
    else:
      r_hand = np.nan
    l_hand = l_hand[6:9, 0:2]
    l_hand = l_hand[l_hand[:, 0] == l_hand[:, 0]]
    if len(l_hand):
      l_hand = l_hand.mean(axis=0)
    else:
      l_hand = np.nan
    d = dist(r_hand, l_hand)
    out.append(d)
  return [np.array(out)]


@stats
def wrist_wrist_x(sample):
  out = []
  for frame in sample:
    body, _, _, _ = get_frame_parts(frame)
    r_wrist = body[4, 0]
    l_wrist = body[7, 0]
    d = l_wrist - r_wrist
    out.append(d)

  return [np.array(out)]

def confidence_hands(sample):
  # Returns mean confidence of x and y coordinate over all frames of a sample. First value is for left hand, second for right hand.
  conf_left = np.nanmean(sample[:,np.arange(hand_left_offset, hand_left_offset+hand_left_len),c_index])
  conf_right = np.nanmean(sample[:,np.arange(hand_right_offset, hand_right_offset+hand_right_len),c_index])
  return [conf_left, conf_right]

def number_of_frames(sample):
  return [len(sample)]

@stats
def reverse_hand_movement(sample):
  (dx_L, dx_R, dy_L, dy_R)  = get_hand_movement_raw(sample)
  X = dx_L*dx_R
  Y = dy_L*dy_R
  return(X,Y)



def generate_feature_matrix(all_samples):
  NUM_SAMPLES = len(all_samples)

                  # regular statistical features                         #singular features       
  COLUMNS = (NUM_FEATURES-NUM_FEATURES_WITHOUT_STATS)*NUM_STATS   + NUM_FEATURES_WITHOUT_STATS   

  FEATURE_MATRIX = np.zeros((NUM_SAMPLES, COLUMNS))

  for i, sample in enumerate(tqdm(all_samples)):
    sample_row = []
    #expect 1 feature for num frames
    sample_row.extend(number_of_frames(sample))

    #expect 4 features for the inflection points
    if(len(sample)>2):
      (dx_L, dx_R, dy_L, dy_R) = get_hand_movement_raw(sample)
      (side_L, side_R, ups_L, ups_R) = get_number_inflections(dx_L),get_number_inflections(dx_R),get_number_inflections(dy_L) ,get_number_inflections(dy_R)
      sample_row.extend([ups_L, ups_R, side_L, side_R])
    else:
      sample_row.extend([float('NaN')]*4)

    #expect 2 features for hand confidence
    sample_row.extend(confidence_hands(sample))

    #expect 12 features for arm angles
    #expect 12 features for shoulder angles
    if(len(sample)>1):
      sample_row.extend(get_all_arm_angles(sample))
      sample_row.extend(get_all_shoulder_angles(sample))
    else:
      sample_row.extend([float('NaN')]*24)

    #expect 24 features for the hand movement
    if(len(sample)>2):
      sample_row.extend(get_hand_movement(sample))
    else:
      sample_row.extend([float('NaN')]*24)

    #expect 60 features for finger openness
    sample_row.extend(finger_openness(sample))

    #expect 12 features for shoulder wrist
    sample_row.extend(shoulder_wrist_y(sample))

    #expect 12 featurs for head hand
    sample_row.extend(head_hand(sample))

    #expect 24 featurs for hands variation
    sample_row.extend(var_hands(sample))

    #expect 12 features for chin thumb
    sample_row.extend(chin_thumb(sample))

    #expect 12 features for mouth index
    sample_row.extend(mouth_index(sample))

    #expect 12 features for thumb pink
    sample_row.extend(thumb_pink(sample))

    #expect 6 features for index index
    sample_row.extend(index_index(sample))

    #expect 6 features for index index
    sample_row.extend(wrist_wrist_x(sample))

    #expect 12 featurs for reverse hand movement
    if(len(sample)>1):
      sample_row.extend(reverse_hand_movement(sample))
    else:
      sample_row.extend([float('NaN')]*12)

    #transform to numpy array
    FEATURE_MATRIX[i] = np.array(sample_row)
  return FEATURE_MATRIX


def transform_to_panda_dataframe(MATRIX):
  df = pd.DataFrame()
  for feature_index, feature_col in enumerate(MATRIX.T):
    if feature_index < NUM_FEATURES_WITHOUT_STATS:
      df[ FEATURE_LIST[feature_index]] = feature_col
    else: 
      actual_feature_index = NUM_FEATURES_WITHOUT_STATS + ((feature_index - NUM_FEATURES_WITHOUT_STATS)//NUM_STATS)
      stat_index = (feature_index-NUM_FEATURES_WITHOUT_STATS) % NUM_STATS
      column_name = FEATURE_LIST[actual_feature_index] + '_' + STAT_LIST[stat_index]
      df[ column_name ] = feature_col
  return df