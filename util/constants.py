pose_labels = [ 
  "Nose",      #0
  "Neck",      #1
  "RShoulder", #2
  "RElbow",    #3
  "RWrist",    #4 
  "LShoulder", #5
  "LElbow",    #6
  "LWrist",    #7
  "MidHip",    #8
  "RHip",      #9
  "RKnee",     #10
  "RAnkle",    #11
  "LHip",      #12
  "LKnee",     #13
  "LAnkle",    #14
  "REye",      #15
  "LEye",      #16
  "REar",      #17
  "LEar",      #18
  "LBigToe",   #19
  "LSmallToe", #20
  "LHeel",     #21
  "RBigToe",   #22
  "RSmallToe", #23
  "RHeel",     #24
]

x_index = 0
y_index = 1
c_index = 2

neck = 1
l_arm_should = 2
l_arm_elbow = 3
l_arm_wrist = 4
r_arm_should = 5
r_arm_elbow = 6
r_arm_wrist = 7

NUM_KEYPOINTS = 137

pose_len = 25
face_len = 70
hand_left_len = 21
hand_right_len = 21


pose_offset = 0
face_offset = pose_offset + pose_len
hand_left_offset = face_offset + face_len 
hand_right_offset = hand_left_offset + hand_left_len

NUM_FEATURES = 44
NUM_FEATURES_WITHOUT_STATS=7

FEATURE_LIST = list([ 
## first features without stats
'NUM_FRAMES',
'INFLECTIONS_L_VERT_FEATURE',
'INFLECTIONS_R_VERT_FEATURE',
'INFLECTIONS_L_HOR_FEATURE',
'INFLECTIONS_R_HOR_FEATURE',
'CONF_HAND_L',
'CONF_HAND_R',
## next features with stats
'ARM_L_ANGLE_FEATURE',
'ARM_R_ANGLE_FEATURE',
'SHOULD_ANGLE_L_FEATURE',
'SHOULD_ANGLE_R_FEATURE',
'HAND_MOVEMENT_L_VERT_FEATURE',
'HAND_MOVEMENT_R_VERT_FEATURE',
'HAND_MOVEMENT_L_HOR_FEATURE',
'HAND_MOVEMENT_R_HOR_FEATURE',
'FINGER_OPENNESS_L_THUMB',
'FINGER_OPENNESS_L_INDEX',
'FINGER_OPENNESS_L_MIDDLE',
'FINGER_OPENNESS_L_RING',
'FINGER_OPENNESS_L_PINK',
'FINGER_OPENNESS_R_THUMB',
'FINGER_OPENNESS_R_INDEX',
'FINGER_OPENNESS_R_MIDDLE',
'FINGER_OPENNESS_R_RING',
'FINGER_OPENNESS_R_PINK',
'SHOULDER_WRIST_Y_L',
'SHOULDER_WRIST_Y_R',
'HEAD_HAND_L',
'HEAD_HAND_R',
'VAR_HANDS_L_HOR',
'VAR_HANDS_L_VERT',
'VAR_HANDS_R_HOR',
'VAR_HANDS_R_VERT',
'CHIN_THUMB_L',
'CHIN_THUMB_R',
'MOUTH_INDEX_L',
'MOUTH_INDEX_R',
'THUMB_PINK_L',
'THUMB_PINK_R',
'INDEX_INDEX',
'WRIST_WRIST',
'REV_HAND_X',
'REV_HAND_Y',
'MOUTH_DISTANCE'
])

NUM_STATS = 6
####################
STAT_LIST = list([
'MAX',
'MIN',
'MEAN',
'STD',
'DIF1',
'DIF2'
])
