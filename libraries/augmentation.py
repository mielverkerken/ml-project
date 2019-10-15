import numpy as np

def interpolate(x):
  out=[]
  for i in range(1,len(x)):
    noise=np.random.normal(size=x[i].shape)
    out.append((x[i]+x[i-1])/2+1.5*noise)
  return np.array(out)

def mirror_sample(sample):
  mirror_x_mat = np.array([[-1, 0, 0],[0, 1, 0], [0, 0, 1]])
  mirror = np.copy(sample)
  for i, frame in enumerate(sample):
    mirror[i] = mirror[i].dot(mirror_x_mat)
    for j, keypoint in enumerate(frame):
      if mirror[i][j][0] != 0:
        mirror[i][j][0] += 455
  return mirror

