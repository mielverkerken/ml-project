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

pose_len = 25
face_len = 70
hand_left_len = 21
hand_right_len = 21

pose_offset = 0
face_offset = pose_offset + pose_len
hand_left_offset = face_offset + face_len 
hand_right_offset = hand_left_offset + hand_left_len

NUM_FEATURES = 13
########################
ARM_L_ANGLE_FEATURE = 0
ARM_R_ANGLE_FEATURE = 1
SHOULD_ANGLE_L_FEATURE = 2
SHOULD_ANGLE_R_FEATURE = 3
HAND_MOVEMENT_L_VERT_FEATURE = 4
HAND_MOVEMENT_R_VERT_FEATURE = 5
HAND_MOVEMENT_L_HOR_FEATURE = 6
HAND_MOVEMENT_R_HOR_FEATURE = 7
INFLECTIONS_L_VERT_FEATURE = 8
INFLECTIONS_R_VERT_FEATURE = 9
INFLECTIONS_L_HOR_FEATURE = 10
INFLECTIONS_R_HOR_FEATURE = 11
FINGER_OPENNESS = 12

NUM_STATS = 6
####################
STAT_MAX = 0
STAT_MIN = 1
STAT_MEAN = 2
STAT_STD = 3
STAT_DIF1 = 4
STAT_DIF2 = 5


