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
    mirror[i][[np.arange(95, 137)]] = mirror[i][[np.concatenate((np.arange(116, 137), np.arange(95, 116)))]]
  return mirror

