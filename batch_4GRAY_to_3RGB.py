import stitch as sti
import os
import amatools
import numpy as np
import matplotlib.pyplot as pp
             
folderpath = sti.getFolder()
filelist = os.listdir(folderpath)
for f in filelist:
    if f.endswith('_proj.tif'):
        im = amatools.AMATiff()
        im.open(folderpath+'/'+f)
        imdata = np.transpose(im.tiff_array[0:3,:,:],[1,2,0])
        pp.imsave(im.filename[0:-9]+'.tif',imdata,format='tiff')