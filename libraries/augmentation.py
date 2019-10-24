import numpy as np

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

