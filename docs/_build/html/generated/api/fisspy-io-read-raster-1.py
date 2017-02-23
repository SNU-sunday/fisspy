import matplotlib.pyplot as plt
from fisspy.io import read
import fisspy
import fisspy.data.sample
raster=read.raster(fisspy.data.sample.FISS_IMAGE,0.3)
plt.imshow(raster,cmap=fisspy.cm.ha,origin='lower')
plt.title(r"NST/FISS 8542+0.3 $\AA$ Spectrogram")
plt.show()