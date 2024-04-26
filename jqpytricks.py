'''=============================================================================
JQTRICKS contains basic functions I use for general purpose
============================================================================='''

import numpy as np
import skimage.measure as skime
from enthought.pyface.api import DirectoryDialog, OK

#def find_filename_roots(folder):
    # usar "regular expressions" library
    #import re
    #regex = re.compile('th.s')
    #l = ['this', 'is', 'just', 'a', 'test']
    #matches = [string for string in l if re.match(regex, string)]
    
    #or alternatively:
    #import fnmatch
    #lst = ['this','is','just','a','test']
    #filtered = fnmatch.filter(lst, 'th?s')

def CellCounterXML_reader(CellCounterXML_filepath):

    ''' this should be definitely better done with lxml library, however when
    parsing the XML files coming from CellCounter, all string conversion options
    I checked seemed to introduce characters (perhaps due to the UTF-8 enconding?)
    before the initial "<" so I settled for manual, ad hoc parsing.
    '''
    
    f = open(CellCounterXML_filepath)
    xml = f.readlines()
    f.close()
    
    xml = [x.lstrip() for x in xml]
    xml = [x for x in xml if not x.startswith("</")]
    
    CellType_idx = [idx for idx,i in enumerate(xml) if i[:6]=="<Type>"]
    Cells_idx = np.asarray([idx for idx,i in enumerate(xml) if i[:9]=="<MarkerX>"])

    '''
    # this code will boolean-filter the XML list
    boolean_filter = np.zeros(len(xml))
    boolean_filter[CellType_idx] = 1
    filtered_list = [i for indx,i in enumerate(xml) if boolean_filter[indx]]    
    '''

    # Get X and Y positions per group of cell type (zip iteration)

    CellTypes_count = []
    CellTypes_Xpos= []
    CellTypes_Ypos= []
    
    for first, second in zip(CellType_idx, CellType_idx[1:]):
        
        # extract the X values between >< chars from XML for Markers between two Type lines
        CellTypes_Xpos.append([int(xml[i].split('>')[1].split('<')[0]) for 
            i in Cells_idx[(Cells_idx > first) & (Cells_idx < second)]])
        
        # extract the Y values between >< chars from XML for Markers between two Type lines
        CellTypes_Ypos.append([int(xml[i].split('>')[1].split('<')[0]) for 
            i in Cells_idx[(Cells_idx > first) & (Cells_idx < second)]+1])

        # extract the cell type number and the number of cells of that type
        CellType = int(xml[first].split('>')[1].split('<')[0])
        ThisCellType_count = len(Cells_idx[(Cells_idx > first) & (Cells_idx < second)])
        CellTypes_count.append(np.ndarray.tolist(np.ones(ThisCellType_count)*CellType))
    
    flatten = lambda l: [item for sublist in l for item in sublist]

    CellType_Data = np.column_stack((flatten(CellTypes_count),
                                     flatten(CellTypes_Xpos),
                                     flatten(CellTypes_Ypos))).astype(int)
                                     
    return CellType_Data
            
def tifffile_opener_sifted(filepath, channels):
    from tifffile import imread
    
    im = imread(filepath)
    
    im2 = np.empty([im.shape[2],im.shape[3],im.shape[0]/channels,channels])
    
    for c in range(channels):
        im2[:,:,:,c] = np.transpose(im[c::channels,0,:,:],[1,2,0])
        
    del im
    return im2
    
def imshowflat(ndimage, projection_mode='max'):
    '''
    Takes a multidimensional image in the form XYZC, XYZ, XYC (up to four
    channels) stored in a numpy array and displays it as the projection along
    the Z axis.
    Projection can be:
    - of the maximum intensity (projection_mode='max', default)
    - of the average intensity (projection_mode='mean')
    '''
    from matplotlib.pyplot import imshow
    
    # for non-bidimensional arrays
    if len(ndimage.shape)<2:
        raise ValueError('This is not a 2D image; it might be a 1D vector')
    
    # for 3D arrays in the form [X, Y, Z] - just one channel
    if len(ndimage.shape)==3:
        if projection_mode=='max':
            imshow(np.max(ndimage, axis=2), cmap='gray')
        elif projection_mode=='mean':
            imshow(np.mean(ndimage, axis=2), cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')
    
    # for potential [X, Y, RGB] images
    if len(ndimage.shape)==3 and ndimage.shape[2]<4:
        imshow(ndimage)
    
    # for XYZ and 4 channels
    if len(ndimage.shape)==4 and ndimage.shape[3]==4:
        if projection_mode=='max':
            TL = np.max(ndimage[:,:,:,0], axis=2)
            TR = np.max(ndimage[:,:,:,1], axis=2)
            BL = np.max(ndimage[:,:,:,2], axis=2)
            BR = np.max(ndimage[:,:,:,3], axis=2)
            top = np.hstack([TL, TR])
            bottom = np.hstack([BL, BR])
            im = np.vstack([top, bottom])
            imshow(im, cmap='gray')
        elif projection_mode=='mean':
            TL = np.mean(ndimage[:,:,:,0], axis=2)
            TR = np.mean(ndimage[:,:,:,1], axis=2)
            BL = np.mean(ndimage[:,:,:,2], axis=2)
            BR = np.mean(ndimage[:,:,:,3], axis=2)
            top = np.hstack([TL, TR])
            bottom = np.hstack([BL, BR])
            im = np.vstack([top, bottom])
            imshow(im, cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')

    # for XYZ and 2-3 channels
    if len(ndimage.shape)==4 and ndimage.shape[3]>1 and ndimage.shape[3]<4:
        if projection_mode=='max':
            imshow(np.max(ndimage,axis=2), cmap='gray')
        elif projection_mode=='mean':
            imshow(np.mean(ndimage,axis=2), cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')    

def unique_rows(data):
    '''
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    '''    
    uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]),return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), indices

def getFolder():
    '''
GETFOLDER creates a GUI to obtain the path of the selected folder
Author: Jonathan McKenzie: http://jonathanmackenzie.net/
'''
    from os import getcwd
    # GUI for getting a sample filepath
    dialog = DirectoryDialog(action="open", default_path=getcwd())
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
    
    img = img.astype('float64')
    
    if bit_depth==1 and not bw:
        output = np.float32(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)))
    elif bit_depth==1 and bw:
        output = np.greater(((img-np.min(img))*max_value)/(np.max(img)-np.min(img)),0)
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
    L = skime.label(bw)
    props = skime.regionprops(L,['Area','BoundingBox','Coordinates','Image'])
    bw_open = np.zeros(bw.shape,dtype='int')
    for blob in props:
        if blob['Area'] >= sz:
            s = slice_boundingbox(blob['BoundingBox'])
            # check that the blob is not a background island:
            if np.sum(np.multiply(bw[s],blob['Image']))>0:
                # transfer the blob to the new image
                bw_open[blob['Coordinates'][:,0],blob['Coordinates'][:,1]] = 1

    return normalize_img(bw_open,1,True)
    
def RGBshuffle(folderpath, ending, ch1, ch2, ch3):
    '''RGBSHUFFLE shuffles colour channels in an RGB image'''
    import os
    import numpy as np
    import matplotlib.pyplot as pp
    order = []
    for o in [ch1, ch2, ch3]:
        if o == 'R':
            x = 0
        elif o == 'G':
            x = 1
        elif o == 'B':
            x = 2
        order.append(x)
    shuffle_list = [x for x in os.listdir(folderpath) if x.endswith(ending)]
    for s in shuffle_list:
        im = pp.imread(os.path.join(folderpath,s))
        if np.max(im)>255:
            raise ValueError('This image is not RGB 8-bit: ')
            print(s)
        else:
            im2 = np.ndarray(shape=im.shape, dtype = 'uint8')
            im2[:,:,0] = im[:,:,order[0]]
            im2[:,:,1] = im[:,:,order[1]]
            im2[:,:,2] = im[:,:,order[2]]
            im2[:,:,3] = 255
            pp.imsave(os.path.join(folderpath,s.split(ending)[0]+'_shf.tif'),im2,format='tiff')

def threshold_triangle(img,bit_depth):
    '''
---> NOW THIS IS AVAILABLE FROM SCIKIT-IMAGE! <---

    TRANSLATED FROM THE MATLAB UPLOAD BY B. PANNETON:
        
    Triangle algorithm
    
    This technique is due to Zack (Zack GW, Rogers WE, Latt SA (1977), 
    "Automatic measurement of sister chromatid exchange frequency", 
    J. Histochem. Cytochem. 25 (7): 741-53, )
    
    A line is constructed between the maximum of the histogram at 
    (b) and the lowest (or highest depending on context) value (a) in the 
    histogram. The distance L normal to the line and between the line and 
    the histogram h[b] is computed for all values from a to b. The level
    where the distance between the histogram and the line is maximal is the 
    threshold value (level). This technique is particularly effective 
    when the object pixels produce a weak peak in the histogram.
    
    Use Triangle approach to compute threshold (level) based on a
    1D histogram (lehisto). num_bins levels gray image. 
    
    INPUTS
        lehisto :   histogram of the gray level image
        num_bins:   number of bins (e.g. gray levels)
    OUTPUT
        level   :   threshold value in the range [0 1];

    Dr B. Panneton, June, 2010
    Agriculture and Agri-Food Canada
    St-Jean-sur-Richelieu, Qc, Canad
    bernard.panneton@agr.gc.ca
    '''
    from skimage.exposure import histogram
    try:
        bit_depth
        if bit_depth not in [8,16]:
            raise Warning('''The bit depth must take one of the values: 8, 16.
            The image will be considered as 8-bit.''')
            num_bins=256
        else:
            num_bins = np.power(2,bit_depth)
    except:
        print('The image will be considered as 8-bit.')
        num_bins=256
    # Find maximum of histogram (h) and its location along the x axis (xmax)
    try:
        data = img.flatten().data[img.flatten().mask==True]
        [H,bin_centres]=(histogram(data,nbins = num_bins));
    except:
        [H,bin_centres]=(histogram(img.flatten(),nbins = num_bins));
    if len(H)<num_bins:
       num_bins = len(H)
       print('''................................................................................
WARNING (jqtricks.threshold_triangle): Not all possible bins contain data,
num_bins will be reduced to reflect this.
''')
    xmax_idx = np.where(H==np.max(H))[0]
    h = np.mean(H[xmax_idx])
    # LET'S ASSUME THERE IS ONLY ONE xmax_idx:
    xmax_idx = xmax_idx[0]
    #xmax = np.mean(bin_centres[np.array(xmax_idx)])
    
    ## VISUAL CHECK
    #import matplotlib.pyplot as plt
    #n, bins, patches = plt.hist(img.flatten(), num_bins, normed=2, facecolor='green', alpha=0.5)
    #y = range((h+h/20).astype('int'))
    #plt.plot(np.ones(len(y)*xmax), y, 'r--')
    #plt.show()
    
    # Find location of first and last non-zero values.
    indi=np.where(H>(h/10000))[0]         # Values<h/10000 are considered zeros
    #indi=np.where(H>0)[0]
    fnz=indi[0]
    lnz=indi[-1]
    # Pick side as side with longer tail. Assume one tail is longer.
    lspan=xmax_idx-fnz
    rspan=lnz-xmax_idx
    if rspan>lspan:  # then flip lehisto
        H=np.fliplr(H.reshape([1,len(H)]))[0]
        bin_centres=np.fliplr(bin_centres.reshape([1,len(bin_centres)]))[0]
        a=num_bins-(lnz+1)
        b=num_bins-xmax_idx
        isflip=1
    else:
        isflip=0
        a=fnz
        b=xmax_idx+1
    # Compute parameters of the straight line from first non-zero to peak
    # To simplify, shift x axis by a (bin number axis)
    m=h/(b-a)
    # Compute distances
    x1=range((b-a).astype('int'))
    y1=H[x1+a]
    beta=y1+x1/m
    x2=beta/(m+1/m)
    y2=m*x2
    #from matplotlib.pyplot import plot
    #plot(x2,y2,'gD-')
    #plot(x1,y1,'mo-')
    L=np.sqrt(np.square(y2-y1)+np.square(x2-x1))
    # Obtain threshold as the location of maximum L.    
    level=np.where(L==np.max(L))[0]
    level=a+np.mean(level)
    # Flip back if necessary
    if isflip:
        level=num_bins-level+1
    return level

def threshold_entropy(image, nbins=256):
    """Return threshold value based on the Maximum Entropy method, similar to
    Otsu's method but maximizing inter-class entropy as S = -(sum)p*log2(p).
    """
    from skimage.exposure import histogram
    H, bin_centers = histogram(image, nbins)
    H = H.astype(float)
    if (H==0).sum() > 0:
        H[H==0] = np.min(H[H>0])/float(10)
        
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(H)
    weight2 = np.cumsum(H[::-1])[::-1]
    
    # class means for all possible thresholds
    entpy1 = np.cumsum(-(H/weight1)*np.log2(H/weight1))
    entpy2 = np.cumsum(-(H[::-1]/weight2[::-1])*np.log2(H[::-1]/weight2[::-1]))[::-1]

    # maximum entropy
    entpy_sum = entpy1 + entpy2

    idx = np.argmax(entpy_sum)
    threshold = bin_centers[:-1][idx]
    return threshold    

def readIJpointPicker(filename):
    """Read data from pointpicker output file
    """
    #Aleix's version:
    XYdata = np.loadtxt(filename, skiprows=1, usecols=(1, 2), unpack=True, dtype='uint16')
    XYdata = np.transpose(XYdata, [1,0])

    # Deprecated:
    #f = open(filename,'rU')
    #lines = f.readlines()[1:]
    #XYdata = []
    #sep = ' '
    #for line in range(len(lines)):
    #XYdata.append(lines[line].split(sep)[2:4])
    #XYdata = np.array(XYdata,dtype='uint16')
    #f.close()
    return XYdata
