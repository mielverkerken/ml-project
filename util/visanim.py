import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML

a_hand = np.load('./util/a_hand.npy')
a_face = np.load('./util/a_face.npy')
a_body = np.load('./util/a_body.npy')



def _plot_bodypart(adjacency, x, y, index_range, index_offset, color, ax):
    for i in index_range:
        for j in index_range:
            if i != j and adjacency[i, j] == 1:
                # The y coordinate is flipped, because the origin lies in the top left corner in OpenPose
                ax.plot([x[i + index_offset], x[j + index_offset]], [-1*y[i + index_offset], -1*y[j + index_offset]], c=color)

def animate(i,sample, ax):
  ax.clear()
  ax.set_xlim(( 0, 455))
  ax.set_ylim((-256, 0))
  x = sample[i, :, 0]
  y = sample[i, :, 1]
  ax.set_title(f"Frame #{i}")
  _plot_bodypart(a_body, x, y, range(25), 0, 'red', ax)
  _plot_bodypart(a_face, x, y, range(70), 25, 'blue', ax)
  _plot_bodypart(a_hand, x, y, range(21), 95, 'green', ax)
  _plot_bodypart(a_hand, x, y, range(21), 116, 'magenta', ax)


def visanim(sample, fig, ax):
    anim = animation.FuncAnimation(fig, animate, frames=len(sample), interval=200, fargs=(sample,ax))
    return anim


def show_anim(fig, anim):  
  plt.close(fig) # otherwise first frame is also plotted below video
  HTML(anim.to_html5_video())
  HTML(anim.to_jshtml())