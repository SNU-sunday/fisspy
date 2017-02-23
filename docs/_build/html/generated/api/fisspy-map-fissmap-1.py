from fisspy.io import read
from fisspy.map.map_factory import fissmap
import fisspy.data.smaple
raster=read.raster(fisspy.data.smaple.FISS_IMAGE,0,smooth=True)
header=read.getheader(fisspy.data.smaple.FISS_IMAGE)
fmap=fissmap(raster,header)
fmap.peek()