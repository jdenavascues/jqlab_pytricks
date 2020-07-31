'''=============================================================================
JQPYTRICKS contains basic utilities useful for general purpose in image analysis
and manipulation
============================================================================='''

import numpy as np
import skimage.morphology as skimo
import skimage.measure as skime
from enthought.pyface.api import DirectoryDialog, OK

def getFolder():
    '''
GETFOLDER creates a GUI to obtain the path of the selected folder
Author: Jonathan McKenzie: http://jonathanmackenzie.net/
'''
    # GUI for getting a sample filepath
    dialog = DirectoryDialog(action="open", default_path='/Volumes/jd467/DATA/C O N F O C A L/ZEISS 700')
    dialog.open()
    if dialog.return_code == OK:		
        folderpath=dialog.path 
    return folderpath

def slice_boundingbox(BoundingBox):
    '''SLICE_BOUNDINGBOX obtains a slice object from the coordinates of a
    skimage.measure.regionprops Bounding Box'''
    
    s = np.s_[BoundingBox[0]:BoundingBox[2],BoundingBox[1]:BoundingBox[3]]

    return s
        
def normalize_img(img, bit_depth, bw=False):
    '''NORMALIZE_IMG takes a numpy array (intended to contain image data) and
    normalizes the signal values between 0 and (2^bit_depth)-1, with bit_depth
    taking the values 1, 8 or 16. Depending of the value of bit_depth and the
    argument bw, the output will be:
        
    bit_depth  bw         Max value       Data type
    -----------------------------------------------
     1         False             1        float16
     1         True              1        boolean
     8         either          255        uint8
    16         either        65535        uint16'''

    if bit_depth not in [1,8,16]:
        raise ValueError('The bit depth must take one of the values: 1, 8, 16')

    max_value = np.power(2,bit_depth)-1

    if bit_depth==1 and not bw:
        output = np.float32(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))
    elif bit_depth==1 and bw:
        output = np.bool(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))
    elif bit_depth==8:
        output = np.uint8(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))
    elif bit_depth==16:
        output = np.uint16(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))
    elif bit_depth==32:
        output = np.uint32(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))

    return output

def bwareaopen(bw, sz):
    '''BWAREAOPEN takes a b/w image and removes the connected areas smaller than
    the specified size. The output is a boolean array.
    
    ============================================================================
    INPUT DATA TYPE CHECKS NEED TO BE IMPLEMENTED
    ============================================================================
    '''
    L = skimo.label(bw)
    props = skime.regionprops(L,properties=['Area','BoundingBox','Coordinates','Image'])
    bw_open = np.zeros(bw.shape,dtype='int')
    for blob in props:
        if blob['Area'] >= sz:
            s = slice_boundingbox(blob['BoundingBox'])
            # check that the blob is not a background island:
            if np.sum(np.multiply(bw[s],blob['Image']))>0:
                # transfer the blob to the new image
                bw_open[blob['Coordinates'][:,0],blob['Coordinates'][:,1]] = 1

    return normalize_img(bw_open,1,bw=True)
