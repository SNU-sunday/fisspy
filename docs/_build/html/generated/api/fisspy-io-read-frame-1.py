import matplotlib.pyplot as plt
from fisspy.io import read
import fisspy
import fisspy.data.sample
data=read.frame(fisspy.data.sample.FISS_IMAGE,xmax=True)
plt.imshow(data[:,75],cmap=fisspy.cm.ca,origin='lower')
plt.title(r"NST/FISS 8542 $\AA$ Spectrogram")
plt.show()