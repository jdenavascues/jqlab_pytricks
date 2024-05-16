"""
=============================================================================
 JQTRICKS contains basic functions I use for general purpose
=============================================================================
"""


# utilities
import os
import requests
from tkinter import filedialog

# data IO
import csv
import pandas as pd

# analysis
import math
import numpy as np
import alphashape
import collections
import tifffile as tf

import skimage.measure as skime
import skimage.morphology as skimo
import skimage.filters as skifi
import skimage.segmentation as skisg
import skimage.feature as skife
from skimage.exposure import histogram

import scipy.ndimage as ndi
from scipy.stats import mode, chisquare
from scipy.spatial import Delaunay

# plotting
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely
import shapely.geometry as sg


"""
-----------------------------------------------------------------------------
 GENERAL PURPOSE FUNCTIONS
-----------------------------------------------------------------------------
"""

def allin(l, s):
    return all(map(lambda l: l in s, l))


def nonin(l, s):
    return all(map(lambda l: l not in s, l))


def inany(s, l):
    """Finds if any string in a list is a substring of any string in another list."""
    func = lambda x: bool(re.search(
        re.sub("\\W+", "", s, 0, re.IGNORECASE),
        re.sub("\\W+", "", x, 0, re.IGNORECASE),
        re.IGNORECASE
        ))
    return any(map(func, l))
    
    
def setdiff_rows(a1, a2):
    import numpy as np
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    diff_rows = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    return diff_rows


def unique_rows(data):
    """
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """    
    uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]),return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), indices

# def unique_rows(A):    
#     unique_idc = A.view(np.dtype( (np.void, A.dtype.itemsize * A.shape[1]) ) )
#     _, idx = np.unique(unique_idc, return_index=True)
#     A_unique_rows = A[idx]
#     return A_unique_rows






"""
-----------------------------------------------------------------------------
 CELL COUNTER PLUGIN FUNCTIONS
-----------------------------------------------------------------------------
"""

"""
 Small utilities
----------------------------------------------------
"""

def angle(a, b):
    import numpy as np
    return np.arctan2(np.cross(a, b), np.dot(a, b))


def polygon_area(x,y, order_convex=False):
    # this adds the area of all the triangles in the polygon
    # but first, polygon vertices need to be in angle order
    import numpy as np
    def wise_hull_vertices(vertices): 
        # vertices is a numpy array
        centroid = np.mean(vertices,axis=0) 
        hull_list = vertices.tolist() 
        hull_list.sort(key=lambda p: math.atan2(p[1]-centroid[1],p[0]-centroid[0]))
        return np.array(hull_list)
    # order the x, y coordinates
    v = np.array([x, y]).T
    if order_convex:
        v = wise_hull_vertices(v) 
    x, y = v[:,0], v[:,1]
    area = 0.5 * np.abs( np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)) ) 
    return area


def dist_2p(A):
    ''' A is assumed to be a 2x2 array - no check is done '''
    from math import sqrt, pow
    s_l = sqrt(pow( A[0][0]-A[1][0], 2) + pow( A[0][1]-A[1][1], 2 ))
    return s_l


def side_centr(A):
    ''' A is assumed to be a 2x2 array - no check is done '''        
    import numpy as np
    centre = [abs(A[0][0]-A[1][0])/2 + np.min([A[0][0],A[1][0]]), \
              abs(A[0][1]-A[1][1])/2 + np.min([A[0][1],A[1][1]])]    
    return np.array(centre)


"""
 I/O functions
----------------------------------------------------
"""


def readIJpointPicker(filename):
    """
    Read data from pointpicker output file.
    """
    XYdata = np.loadtxt(filename, skiprows=1, usecols=(1, 2), unpack=True, dtype='int')
    XYdata = np.transpose(XYdata, [1,0])
    return XYdata


def CellCounterXML_reader(CellCounterXML_filepath, ):
    """
    This should be definitely better done with lxml library, however when
    parsing the XML files coming from CellCounter, all string conversion options
    I checked seemed to introduce characters (perhaps due to the UTF-8 enconding?)
    before the initial "<" so I settled for manual, ad hoc parsing.
    """

    # Open XML file and basic manual parsing
    f = open(CellCounterXML_filepath)
    xml = f.readlines()
    f.close()    
    xml = [x.lstrip() for x in xml]
    xml = [x for x in xml if not x.startswith("</")]
    CellType_idx = [idx for idx,i in enumerate(xml) if i[:6]=="<Type>"]
    Cells_idx = np.asarray([idx for idx,i in enumerate(xml) if i[:9]=="<MarkerX>"])

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
    CellType_Data = np.column_stack(
        (flatten(CellTypes_count),
         flatten(CellTypes_Xpos),
         flatten(CellTypes_Ypos))
    ).astype(int)
    
    return CellType_Data


def pp_numbers2(fpath) :
    # Adapted from Aleix's GitHub repo
    # https://github.com/aleixpuigb/PhD_scripts/blob/main/counting_2pos.py
    import pandas as pd
    countings = pd.read_table(fpath, usecols=['Type', 'X', 'Y'])
    ISC_GFP = (sum(countings.Type == 1))
    EB_GFP = (sum(countings.Type == 2))
    EE_GFP = (sum(countings.Type == 5))
    preEE_GFP = (sum(countings.Type == 9))
    EC_GFP = (sum(countings.Type == 6))
    GFP_num = ISC_GFP + EB_GFP + EE_GFP + EC_GFP + preEE_GFP
    total_num = float(len(countings.index))
    t_ISC_GFP = (sum(countings.Type == 1)*100/total_num)
    t_EB_GFP = (sum(countings.Type == 2)*100/total_num)
    t_EE = (sum(countings.Type == 3)*100/total_num)
    t_EC = (sum(countings.Type == 4)*100/total_num)
    t_EE_GFP = (sum(countings.Type == 5)*100/total_num)
    t_EC_GFP = (sum(countings.Type == 6)*100/total_num)
    t_ISC = (sum(countings.Type == 7)*100/total_num)
    t_EB = (sum(countings.Type == 8)*100/total_num)
    t_preEE_GFP = (sum(countings.Type == 9)*100/total_num)
    exp = [fpath.split('/')[-1]]
    g_ISC_GFP = ISC_GFP*100/GFP_num
    g_EB_GFP = EB_GFP*100/GFP_num
    g_EE_GFP = EE_GFP*100/GFP_num
    g_EC_GFP = EC_GFP*100/GFP_num
    g_preEE_GFP = preEE_GFP*100/GFP_num
    
    totaldata = pd.DataFrame(data = (),
                             columns=['Sample','ISC_GFP_%', 'EB_GFP_%',
                                      'EE_GFP_%', 'preEE_GFP_%', 'EC_GFP_%',
                                      'ISC_%' , 'EB_%', 'EE_%', 'EC_%'])
    totaldata.loc[0] = np.hstack([exp, t_ISC_GFP, t_EB_GFP,
                                  t_EE_GFP, t_preEE_GFP, t_EC_GFP,
                                  t_ISC, t_EB, t_EE, t_EC])
    GFPdata = pd.DataFrame(data = (),
                           columns=['Sample', 'ISC_GFP_%', 'EB_GFP_%',
                                    'EE_GFP_%', 'preEE_GFP_%', 'EC_GFP_%'])
    GFPdata.loc[0] = np.hstack([exp, g_ISC_GFP, g_EB_GFP,
                                g_EE_GFP, g_preEE_GFP, g_EC_GFP])
    
    return totaldata, GFPdata


"""
 Point-cloud functions
----------------------------------------------------
"""


def delaunay_distances(XYdata, pixel_size, THR=8):
    # THR is the threshold to eliminate non-neighbouring links
    # obtain triangulation, get sides of triangles
    # usual parameter values:
    # THR = 20
    # pixel_size = 1
    D = Delaunay(XYdata)
    V = D.simplices
    Vsorted = []
    A = V[:, 0]
    B = V[:, 1]
    C = V[:, 2]
    Vsorted.extend(zip(A, B))
    Vsorted.extend(zip(B, C))
    Vsorted.extend(zip(C, A))
    Vsorted = np.sort(Vsorted,axis=1)
    # uniquify sides
    V_NR = unique_rows(Vsorted)
    # remove sides too close to opposing vertex (convex hull)
    excluded = []
    for side in D.convex_hull:
        for vertex in D.vertices:
            if len(np.setdiff1d(vertex, side)) == 1:
                A = np.vstack([D.points[np.setdiff1d(vertex, side)],side_centr(D.points[side])])
                dist_vtx2hull = dist_2p(A)
                if dist_vtx2hull < dist_2p(D.points[side])/THR:
                    excluded.append(side)
    relevant_sides = setdiff_rows(V_NR,np.sort(np.array(excluded),axis=1))
    # get distances
    coord = D.points[relevant_sides]
    distances = [dist_2p(x) for x in coord]*pixel_size
    relevant_sides = relevant_sides[(np.asarray(distances) < pixels), :]
    
    return XYdata, relevant_sides


def extended_delaunay(XYdata, alpha = 0.02):
    # Obtain a Delaunay triangulation with ordered hull, calculated alphashape and areas
    # obtain triangulation, get sides of triangles
    from scipy.spatial import Delaunay
    from alphashape import alphashape
    import shapely
    import numpy as np
    
    D = Delaunay(XYdata)
    V = D.simplices
    Vsorted = []
    A = V[:, 0]
    B = V[:, 1]
    C = V[:, 2]
    Vsorted.extend(zip(A, B))
    Vsorted.extend(zip(B, C))
    Vsorted.extend(zip(C, A))
    Vsorted = np.sort(Vsorted,axis=1)
    D.aristae = unique_rows(Vsorted)
    # new property of the original hull
    D.hull_area = polygon_area(
        D.points[np.unique(D.convex_hull.flatten())][:,0],
        D.points[np.unique(D.convex_hull.flatten())][:,1])
    # get the alpha shape and store
    ashp = alphashape(XYdata, alpha=alpha)
    while (ashp.geom_type=='MultiPolygon') or (ashp.area/D.hull_area < 0.75):
        alpha = alpha*0.95
        ashp = alphashape(XYdata, alpha=alpha)
    D.a_s = ashp
    # get the coordinates of the alpha shape for the new hull, and area
    D.alpha_shape = {'coords' : np.array(ashp.boundary.coords.xy),
                     'alpha' : alpha}
    D.alpha_area = ashp.area
    # remove the aristae that are now outside the alpha shape
    aristalist = np.split(D.points[D.aristae], len(D.aristae), axis=0)
    aristalist = [x.squeeze() for x in aristalist]
    aristae_keep = [shapely.within(sg.LineString(x).centroid, D.a_s) for x in aristalist]
    D.aristae_alpha = D.aristae[aristae_keep]

    return D


def get_neighbour_dist(D, pixel_micron):
    """Obtain the list of distances (in microns) between points in a Delaunay triangulation."""
    # if D is a list of 2-3 points
    if (not hasattr(D, 'simplices')) & (hasattr(D, 'shape')):
        if D.shape[1] == 2:
            R = np.roll(D, -1, axis=0)
            coord = np.vstack((D, R)).reshape((D.shape[0],2,2),order='F')
        else:
            raise ValueError
    # if D is a triangulation object
    else:
        # if D has not been produced by the extended_delaunay wrapper
        if not hasattr(D, 'aristae'):
            V = D.simplices
            Vsorted = []
            A = V[:, 0]
            B = V[:, 1]
            C = V[:, 2]
            Vsorted.extend(zip(A, B))
            Vsorted.extend(zip(B, C))
            Vsorted.extend(zip(C, A))
            Vsorted = np.sort(Vsorted,axis=1)
            Vsorted = np.sort(Vsorted,axis=1)
            D.aristae = unique_rows(Vsorted)
        # list of coords
        coord = D.points[D.aristae]
    distances = np.array([dist_2p(x) for x in coord])*pixel_micron
    return distances
    
    
def connected_comp(l):
    # maybe based in https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph?
    # Algorithm to find connected components:
    # 1. take first set A from list
    # 2. for each other set B in the list do if B has common element(s) with A join B into A; remove B from list
    # 3. repeat 2. until no more overlap with A
    # 4. put A into outpup
    # 5. repeat 1. with rest of list
    out = []
    while len(l)>0:
        first, rest = l[0], l[1:]
        first = set(first)
    
        lf = -1
        while len(first)>lf:
            lf = len(first)
    
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
    
        out.append(first)
        l = rest
    
    return [list(out[x]) for x in range(len(out))]


def clique_celltypes(TXYdata, single, multiple, marker_types):
    """
    Finds cell type composition of cliques of labelled cells.
    
    Parameters
    ----------------------------
    - TXYdata : (N,3) numpy array
        `TXYdata` contains the Type [:,0] and XY position [:,1:3] for N cells.
    - single : list of integers
        `single` is the list of indices in TXYdata that are isolated, single labelled cells.
    - multiple : list of lists of integers
        `multiple` is a list that contains, for each clique of labelled cells, the list of
        indices in TXYdata of the labelled cells that comprise the clique.
    - marker_types : dataframe
        `marker_types` contains the correspondence of the CellCounter marker Types
        (integers) to the cell type, the criterion whereby this cell type was defined, and
        whether it is genetically labelled or not, and the nature of the label, respectively,
        in the columns:
        - 'marker_type' (integer)
        - 'cell_type'   (string)
        - 'defined_as'  (string)
        - 'labelled'    (boolean)
        - 'label_id'    (string)
    
    Returns
    -------
    - A dataframe containing the size and cell type composition of the clones/clusters.
    """
    
    # MarkerType integer for each cell in a clique of labelled cells
    type_cliques = [TXYdata[x,0] for x in multiple]
    # Add singleton labelled cells if they exist
    if single.any():
        single_types = TXYdata[single,0]
        single_types = np.split(single_types, len(single_types))
        type_cliques = type_cliques + single_types
    # turn numpy arrays to list
    type_cliques = [list(x.astype(int)) for x in type_cliques]
    # read cell types from the 'meta'dataframe provided
    celltypes_considered = list(marker_types.cell_type[marker_types.labelled])
    # create dataframe for composition
    cols = ['Size'] + celltypes_considered
    clique_composition = pd.DataFrame(columns=cols, dtype=int)
    # find correspondence between Type (as integer) and Cell Type (string)
    # and count
    marker_to_celltype = lambda x: marker_types[marker_types.marker_type==x].cell_type[0]
    for c in complete_types:
        t = list(map(marker_to_celltype, c))
        row = [len(t)] + [t.count(x) for x in celltypes_considered]
        clique_composition.loc[len(clique_composition)] = row
        
    return clique_composition


def clique_data(source, marker_types):
    """
    Finds individual mutant clones or cell clusters within a point cloud.
    
    Parameters
    ----------------------------
    - source : string containing a path or XYT dataframe
        `source` can be a path to a raw text file containing cell type markers and their X,
        Y positions in an image, as produced by ImageJ CellCounter, or a pandas dataframe
        containing the same information a CellCounter txt file would.
    - marker_types : dataframe containing...
        `marker_types` contains the correspondence between markers in `txtfile`...
        -- 'marker_type': integers
        -- 'GFP' or other molecular label : boolean
    - col : string naming a column in `marker_types` (default: 'GFP')
        `col` is the column that contains the information of whether that cell is genetically
        labelled.
    
    Returns
    -------
    - Triangulation out of `pointcloud_utils.extended_delaunay()`.
    - A dataframe containing the size and cell type composition of the clones/clusters.
    - Numpy array (N,) with the indices of points in the triangulation.
    - Numpy array (M,2) with labelled edges as pairs of indices of points in the triangulation.
    """
    from pointcloud_utils import extended_delaunay
    import pandas as pd
    import numpy as np
    
    if os.path.isfile(source):
        point_cloud = pd.read_table(txtfile, usecols=['Type', 'X', 'Y'])
    elif isinstance(source, pd.DataFrame):
        point_cloud = source
    else:
        raise ValueError('''
`clique_data` needs a `source` parameter that is either the path to a CellCounter
txt file or the pandas DataFrame obtained from reading such a file.''')
    
    xyd = np.array([point_cloud.Y, point_cloud.X]).T
    extD = extended_delaunay(xyd)
    # to allow easy indexing
    TXYdata = np.array(point_cloud)
    # to avoid a FutureWarning about downcasting behaviour in `replace`:
    pd.set_option("future.no_silent_downcasting", True) 
    edge_marker_types = pd.DataFrame(
        {'v1' : TXYdata[ extD.aristae_alpha[:,0], 0 ],
         'v2' : TXYdata[ extD.aristae_alpha[:,1], 0 ]}
    )
    # turn marker_type numbers into True/False labelled cells
    labelled_vertices = edge_marker_types.replace(
        marker_types.set_index('marker_type')['labelled']
    )
    # get edges where both members are labelled
    labelled_edges = extD.aristae_alpha[labelled_vertices.v1 & labelled_vertices.v2, :]
    # get cliques of labelled vertices
    cliques = connected_comp(labelled_edges.tolist())
    # get isolated labelled cells
    labelled_vertices = np.where(
        point_cloud.Type.replace(
            marker_types.set_index('marker_type')['labelled']
        )
    )[0]
    singletons = np.array(list(
        set( labelled_vertices ) - set( np.unique(labelled_edges) )
        )
    )
    # get composition
    clique_composition = clique_celltypes(TXYdata, single, multiple, marker_types)
    
    return (clique_composition, point_cloud, extD, singletons, cliques)
    
    
"""
-----------------------------------------------------------------------------
 IMAGE IO FUNCTIONS
-----------------------------------------------------------------------------
"""



def getLIFMetaData(folderpath, filename):
    """
    Reads MetaData folder after LIF exporting to TIFF from within the
    Leica software
    """
    import os
    flist = os.listdir(folderpath)
    fname_root = filename.split('.')[0]
    
    for f in flist:
        if f.startswith(fname_root):
            datafile = f
    imgMetaData = datafile.readlines()
    pixel_size = imgMetaData[20].split('Voxel="')[1].split('"')[0]
    return float(pixel_size)


def tifffile_opener_sifted(filepath, channels):
    im = tf.imread(filepath)
    im2 = np.empty([im.shape[2],im.shape[3],im.shape[0]/channels,channels])
    for c in range(channels):
        im2[:,:,:,c] = np.transpose(im[c::channels,0,:,:],[1,2,0])
    return im2


def RGBshuffle(folderpath, ending, ch1, ch2, ch3):
    """RGBSHUFFLE shuffles colour channels in an RGB image"""
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
        im = plt.imread(os.path.join(folderpath,s))
        if np.max(im)>255:
            raise ValueError('This image is not RGB 8-bit: ')
            print(s)
        else:
            im2 = np.ndarray(shape=im.shape, dtype = 'uint8')
            im2[:,:,0] = im[:,:,order[0]]
            im2[:,:,1] = im[:,:,order[1]]
            im2[:,:,2] = im[:,:,order[2]]
            im2[:,:,3] = 255
            plt.imsave(os.path.join(folderpath,s.split(ending)[0]+'_shf.tif'),im2,format='tiff')


def normalize_img(img, bit_depth, bw=False):
    """
    NORMALIZE_IMG takes a numpy array (intended to contain image data) and
    normalizes the signal values between 0 and (2^bit_depth)-1, with bit_depth
    taking the values 1, 8 or 16. Depending of the value of bit_depth and the
    argument bw, the output will be:
        
    bit_depth  bw         Max value       Data type
    -----------------------------------------------
     1         False             1        float16
     1         True              1        boolean
     8         either          255        uint8
    16         either        65535        uint16
    """
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
    
    
"""
-----------------------------------------------------------------------------
 IMAGE VISUALIZATION FUNCTIONS
-----------------------------------------------------------------------------
"""


def imshowflat(ndimage, projection_mode='max'):
    """
    Takes a multidimensional image in the form XYZC, XYZ, XYC (up to four
    channels) stored in a numpy array and displays it as the projection along
    the Z axis.
    Projection can be:
    - of the maximum intensity (projection_mode='max', default)
    - of the average intensity (projection_mode='mean')
    """
    # for non-bidimensional arrays
    if len(ndimage.shape)<2:
        raise ValueError('This is not a 2D image; it might be a 1D vector')
    # for 3D arrays in the form [X, Y, Z] - just one channel
    if len(ndimage.shape)==3:
        if projection_mode=='max':
            plt.imshow(np.max(ndimage, axis=2), cmap='gray')
        elif projection_mode=='mean':
            plt.imshow(np.mean(ndimage, axis=2), cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')
    # for potential [X, Y, RGB] images
    if len(ndimage.shape)==3 and ndimage.shape[2]<4:
        plt.imshow(ndimage)
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
            plt.imshow(im, cmap='gray')
        elif projection_mode=='mean':
            TL = np.mean(ndimage[:,:,:,0], axis=2)
            TR = np.mean(ndimage[:,:,:,1], axis=2)
            BL = np.mean(ndimage[:,:,:,2], axis=2)
            BR = np.mean(ndimage[:,:,:,3], axis=2)
            top = np.hstack([TL, TR])
            bottom = np.hstack([BL, BR])
            im = np.vstack([top, bottom])
            plt.imshow(im, cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')
    # for XYZ and 2-3 channels
    if len(ndimage.shape)==4 and ndimage.shape[3]>1 and ndimage.shape[3]<4:
        if projection_mode=='max':
            plt.imshow(np.max(ndimage,axis=2), cmap='gray')
        elif projection_mode=='mean':
            plt.imshow(np.mean(ndimage,axis=2), cmap='gray')
        else:
            raise ValueError('Projection mode is incorrectly defined')    


def visualize_delaunay(XYdata, relevant_sides, image_name):        
    P = XYdata
    X,Y = P[:,0],P[:,1]
    fig = plt.figure(figsize=(20,20))
    axes = plt.subplot(1,1,1)
    im = plt.imread(image_name)
    plt.imshow(im)
    plt.scatter(X, Y, marker='o', color='w')
    plt.axis([0,512,0,512])
    lines = matplotlib.collections.LineCollection(P[relevant_sides], color='y')
    plt.gca().add_collection(lines)
    plt.axis([0,512,0,512])
    plt.show()


def visualize_pointcloud(D, RGB, fs: int=5):
    # D is the output of extended_delaunay
    # lines between points
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection
    
    lines_col = LineCollection(D.points[D.aristae],
                               color='r', linewidth = 0.5, alpha = 0.8)
    lines_in = LineCollection(D.points[D.aristae_alpha],
                              color='w', linewidth = 0.5, alpha = 0.8)
    # convex hull vertices
    hull_v = D.points[D.convex_hull.flatten(),:]
    # alpha shape vertices
    alpha_v = D.alpha_shape['coords'].T
    alpha_segs = np.zeros(( (alpha_v.shape[0]-1) , 2 , 2))
    alpha_segs[:, ::2, :] = np.roll(alpha_v[:-1, np.newaxis, :], -1, axis=0)
    alpha_segs[:, 1::2, :] = alpha_v[:-1,np.newaxis, :]
    alines_col = LineCollection(alpha_segs,
                                color='w', linewidth = 1)
    #poly = np.dstack([ashp_x, ashp_y]).squeeze()
    alpha_patch = Polygon(alpha_v, alpha = 0.2)
    plt.figure(figsize=(fs,fs))
    plt.imshow(np.rot90(np.flip(RGB, axis=1)))
    #plt.scatter(D.points[D.aristae[:,0]], D.points[V_NR,1], marker='o', color='c')
    plt.scatter(hull_v[:,0], hull_v[:,1], marker='o', color = 'k', edgecolor = 'w', s = 30)
    plt.gca().add_collection(lines_col)
    plt.gca().add_collection(lines_in)
    plt.gca().add_collection(alines_col)
    plt.gca().add_patch(alpha_patch)
    plt.axis([0,512,0,512]) 
    plt.show()


def check_points_2_fov(lsm_list, txt_list, alpha: float=0.02, fs: int=5):
    """
    Creates visualisation of a confocal image overlaid with a point cloud.
    
    Parameters
    ----------------------------
    - lsm_list : list of strings
        The strings contain the full paths to the confocal images, which must be LSM files.
        The TIFF array containing the pixel data is expected to be arranged as either:
            - TCZYX, if it contains multiple fields of view (stored in the Time
              dimension), which will likely mean that there is only one file in the list,
              or
            - CZYX, if there are multiple file paths in the list and each contains just
              one field of view.
    - txt_list : list of strings
        The strings contain the full paths to the TXT files containing counter type and XY
        positions of cells in the field of view. These files are produced with the
        PointPicker plugin of ImageJ, so they will be expected to have 3 columns named 'Type',
        'X' and 'Y' to store these data. Other dimensions of the data (Z, C) are ignored here.
    - alpha : float (default is 0.02)
        This is the parameter of the alpha shape, which determines the radius used to decide
        which edges of the triangulation of the point cloud stay in the hull. For alpha=0,
        the alpha shape is the convex hull of the point cloud.
    - fs : int (default is 5)
    
    Returns
    -------
    Plots a `matplotlib.pyplot object` per individual field of view, containing the Z
    projection of the confocal stack (by maximum intensity) of the first three channels as an
    RGB image (expected to be some antibody staining, GFP and DAPI/Hoechst), the full
    Delaunay triangulation of all the points (in red), the outline and shade of the alpha
    shape polygon in thick white, and the triangulation edges inside the alpha shade in thin
    white. If there are not enough points to make any triangulation (1-3), the points are
    plotted as thick dots.
    """

    if len(lsm_list)==1:
        print('\nTesting multi-FoV file:\t' + lsm_list[0].split('/')[-1])
        I = tf.imread(lsm_list[0])
        # get the FoV number from naming convention '_00.txt'
        ixes = np.array([ int(re.search(r'_(\d{1,2})\.txt', x).group(1)) for x in txt_list ])
        for ctx, fov in enumerate(ixes):
            imx = fov-1
            print('\nOverlaying FoV #' + str(fov))
            fpath = txt_list[ctx]
            print('Adding points from file:\t' + fpath.split('/')[-1])
            # Turn TCZYX FoV into an XY3 RGB-like array
            im = np.moveaxis( np.max(I[imx,:,0:3,:,:],axis = 0), 0, -1 )
            point_cloud = pd.read_table(fpath, usecols=['Type', 'X', 'Y'])
            print('Number of points: ' + str(len(point_cloud.Y)))
            if len(point_cloud.Y)>3:
                # overlay triangulation
                xyd = np.array([point_cloud.Y, point_cloud.X]).T
                visualize_pointcloud(extended_delaunay(xyd, alpha = alpha), im, fs)
            else:
                # overlay individual points
                impos = np.zeros(im.shape[:2])
                impos[point_cloud.Y, point_cloud.X] = 1
                # scaled so disk radius will be 3 for a 512x512 image, 6 if 1024x1024
                radius = np.floor(im.shape[0]/150)
                impos = skimo.dilation(impos, skimo.disk( radius ))
                impos = np.dstack((impos,) * 3)
                overlay = np.add(im*(impos==0),
                                 (impos*255).astype('uint8'))
                plt.figure(figsize = (fs,fs))
                plt.imshow(overlay)
                
    if len(lsm_list)>1:
        # get FoV numbers for which there is point data
        ct_ixes = np.array([ int(re.search(r'_(\d{1,2})\.txt', x).group(1)) for x in txt_list ])
        # get FoV numbers for which there is image data
        im_ixes = np.array([ int(re.search(r'_(\d{1,2})\.lsm', x).group(1)) for x in lsm_list ])
        for ctx, fov in enumerate(ct_ixes):
            print('Overlaying points file:\t' + txt_list[ctx].split('/')[-1])
            fpath = txt_list[ctx]
            imx = np.where(im_ixes==fov)[0][0]
            print('\nTesting single-FoV file:\t' + lsm_list[imx].split('/')[-1])
            I = tf.imread(lsm_list[imx])
            # Turn TCZYX FoV into an XY3 RGB-like array
            im = np.moveaxis( np.max(I[:,0:3,:,:],axis = 0), 0, -1 )
            point_cloud = pd.read_table(fpath, usecols=['Type', 'X', 'Y'])
            print('Number of points:\t' + str(len(point_cloud.Y)))
            if len(point_cloud.Y)>3:
                # overlay triangulation
                xyd = np.array([point_cloud.Y, point_cloud.X]).T
                visualize_pointcloud(extended_delaunay(xyd, alpha = alpha), im, fs)
            else:
                # overlay individual points
                impos = np.zeros(im.shape[:2])
                impos[point_cloud.Y, point_cloud.X] = 1
                # scaled so disk radius will be 3 for a 512x512 image, 6 if 1024x1024
                radius = np.floor(im.shape[0]/150)
                impos = skimo.dilation(impos, skimo.disk( radius ))
                impos = np.dstack((impos,) * 3)
                overlay = np.add(im*(impos==0),
                                 (impos*255).astype('uint8'))
                plt.figure(figsize = (fs,fs))
                plt.imshow(overlay)
                
                
def check_points_2_fov_special(lsm_list, txt_list_special, alpha: float=0.02, fs: int=5):
    """
    Creates visualisation of a confocal image overlaid with a point cloud.
    
    Parameters
    ----------------------------
    - lsm_list : list of strings
        The strings contain the full paths to the confocal images, which must be LSM files.
        The TIFF array containing the pixel data is expected to be arranged as either:
            - TCZYX, if it contains multiple fields of view (stored in the Time
              dimension), which will likely mean that there is only one file in the list,
              or
            - CZYX, if there are multiple file paths in the list and each contains just
              one field of view.
    - txt_list_special : list of strings
        The strings contain the full paths to the TXT files containing counter type and XY
        positions of cells in the field of view. These files are produced with the
        PointPicker plugin of ImageJ, so they will be expected to have 3 columns named 'Type',
        'X' and 'Y' to store these data. Other dimensions of the data (Z, C) are ignored here.
        NOTE: this parameter is named 'special' because the files do not contain the
        positions of _every_ cell, but only some of them - therefore there needs to be
        additional image analysis on the image files contained in `lsm_list` to obtain
        the rest of the point cloud.
    - alpha : float (default is 0.02)
        This is the parameter of the alpha shape, which determines the radius used to decide
        which edges of the triangulation of the point cloud stay in the hull. For alpha=0,
        the alpha shape is the convex hull of the point cloud.
    - fs : int (default is 5)
    
    Returns
    -------
    Plots a `matplotlib.pyplot object` per individual field of view, containing the Z
    projection of the confocal stack (by maximum intensity) of the first three channels as an
    RGB image (expected to be some antibody staining, GFP and DAPI/Hoechst), the full
    Delaunay triangulation of all the points (in red), the outline and shade of the alpha
    shape polygon in thick white, and the triangulation edges inside the alpha shade in thin
    white. If there are not enough points to make any triangulation (1-3), the points are
    plotted as thick dots.
    """

    if len(lsm_list)==1:
        print('\nTesting multi-FoV file:\t' + lsm_list[0].split('/')[-1])
        with tf.TiffFile(lsm_list[0]) as tif:
            px_size = float(tif.lsm_metadata['VoxelSizeX'])*1e6
            I = tif.asarray()
        # get the FoV number from naming convention '_00.txt'
        ixes = np.array([ int(re.search(r'_(\d{1,2})\.txt', x).group(1)) for x in txt_list_special ])
        for ctx, fov in enumerate(ixes):
            imx = fov-1
            print('\nOverlaying FoV #' + str(fov))
            fpath = txt_list_special[ctx]
            print('Adding points from file:\t' + fpath.split('/')[-1])
            # Turn TCZYX FoV into an XY3 RGB-like array
            im = np.moveaxis( np.max(I[imx,:,0:3,:,:],axis = 0), 0, -1 )
            point_cloud = pd.read_table(fpath, usecols=['Type', 'X', 'Y'])
            xyd = np.array([point_cloud.Y, point_cloud.X]).T
            print('Adding nuclear centroids from corresponding image')
            dapi_centroids = extract_nuclear_pointcloud(im[:,:,2], px_size, xyd)
            xyd = np.vstack([xyd, dapi_centroids])
            visualize_pointcloud(extended_delaunay(xyd, alpha = alpha), im, fs)
                
    if len(lsm_list)>1:
        # get FoV numbers for which there is point data
        ct_ixes = np.array([ int(re.search(r'_(\d{1,2})\.txt', x).group(1)) for x in txt_list_special ])
        # get FoV numbers for which there is image data
        im_ixes = np.array([ int(re.search(r'_(\d{1,2})\.lsm', x).group(1)) for x in lsm_list ])
        for ctx, fov in enumerate(ct_ixes):
            print('Overlaying points file:\t' + txt_list[ctx].split('/')[-1])
            fpath = txt_list_special[ctx]
            imx = np.where(im_ixes==fov)[0][0]
            print('\nTesting single-FoV file:\t' + lsm_list[imx].split('/')[-1])
            with tf.TiffFile(lsm_list[imx]) as tif:
                px_size = float(tif.lsm_metadata['VoxelSizeX'])*1e6
                I = tif.asarray()
            # Turn TCZYX FoV into an XY3 RGB-like array
            im = np.moveaxis( np.max(I[:,0:3,:,:],axis = 0), 0, -1 )
            point_cloud = pd.read_table(fpath, usecols=['Type', 'X', 'Y'])
            xyd = np.array([point_cloud.Y, point_cloud.X]).T
            print('Adding nuclear centroids from corresponding image')
            dapi_centroids = extract_nuclear_pointcloud(im[:,:,2], px_size, xyd)
            xyd = np.vstack([xyd, dapi_centroids])
            visualize_pointcloud(extended_delaunay(xyd, alpha = alpha), im, fs)
            
            
def compare_cellcounters(P1, P2, RGB, root, colours=['tomato','lightskyblue'], fs=5, all=False):
    """
    Visualise the positions from CellCounter XML files on the same image.
    
    Parameters
    ----------------------------
    - P1 : (N, 2) np.array or similar container
        `P1` contains the X, Y positions from a CellCounter XML file with N positions marked.
        P1 is obtained with:
        `cellcounter = pandas.read_csv(count, sep='\t', usecols=[0,2,3])`
        `P1 = numpy.array([cellcounter.Y, cellcounter.X]).T`
    - P2 : (M, 2) np.array or similar container
        `P2` contains the X, Y positions from a CellCounter XML file with M positions marked.
        P2 is obtained like P1.
    - RGB is a numpy array with shape (y, x, 3) suitable for use with matplotlib.pyplot.imshow().
        `RGB` contains the array data of the image with which to compare the point clouds.
    - root: string
        `root` is the string that will be used to save the visualisations as PNG images.
    - colours : list of strings (default ['tomato','lightskyblue'])
        `colours` should contain the string names of two named matplotlib colours or any
        single-letter string that matplotlib can interpret as a colour. The list of colours can
        be longer than 2, but only the first two will be used.
        'tomato','lightskyblue' and 'lime' is a good options for red, blue and green,
        respectively, with dark backgrounds (e.g. fluorescence data).
    - fs : int (default is 5)
        `fs` is the figure size passed on to matplotlib.pyplot.
    
    Returns
    -------
    Plots and saves a figure with points P1 and P2 overlayed on RGB, using colours[0] and
    colours[1], respectively, to mark the positions and their Delaunay triangulation. If
    `all=True`, it also plots the individual overlay of P1 and P2 on RGB.
    Images will be saved with file names `root` + a suffix (`'_P1'`, `'_P2'`, `'_P1P2'`,
    depending on the point cloud(s) that are overlayed onto RGB.
    """
    # P1, P2 are XY of CellCounters
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay
    from matplotlib.collections import LineCollection
    from copy import copy
    # colours
    c0 = colours[0]
    c1 = colours[1]
    # triangles
    D1 = Delaunay(P1)
    D2 = Delaunay(P2)
    # lines between points
    lines_D1 = LineCollection(D1.points[D1.simplices],
                              color=c0, linewidth = 0.5,
                              linestyle = 'dashed',alpha = 0.75)
    lines_D2 = LineCollection(D2.points[D2.simplices],
                              color=c1, linewidth = 0.75,
                              linestyle = 'dotted', alpha = 0.75)
    lines_D1c = copy(lines_D1)
    lines_D2c = copy(lines_D2)

    if all:
        plt.figure(figsize=(fs,fs))
        plt.imshow(RGB)
        # marker positions
        plt.scatter(P1[:,0], P1[:,1], marker='o', color = c0, alpha = 0.5, s = 30)
        plt.gca().add_collection(lines_D1)
        plt.axis([0,512,0,512]) 
        plt.show()
        plt.savefig(root+'_P1')
    
        plt.figure(figsize=(fs,fs))
        plt.imshow(RGB)
        plt.scatter(P2[:,0], P2[:,1], marker='o', color = c1, alpha = 0.5, s = 15)
        plt.gca().add_collection(lines_D2)
        plt.axis([0,512,0,512]) 
        plt.show()
        plt.savefig(root+'_P2')

    plt.figure(figsize=(fs,fs))
    plt.imshow(RGB)
    # marker positions
    plt.scatter(P1[:,0], P1[:,1], marker='o', color = c0, alpha = 0.5, s = 30)
    plt.scatter(P2[:,0], P2[:,1], marker='o', color = c1, alpha = 0.3, s = 15)
    plt.gca().add_collection(lines_D1c)
    plt.gca().add_collection(lines_D2c)
    plt.axis([0,512,0,512]) 
    plt.show()
    plt.savefig(root+'_P1P2')


"""
-----------------------------------------------------------------------------
 IMAGE ANALYSIS FUNCTIONS
-----------------------------------------------------------------------------
"""


def slice_boundingbox(BoundingBox):
    """
    SLICE_BOUNDINGBOX obtains a slice object from the coordinates of a
    skimage.measure.regionprops Bounding Box
    """
    s = np.s_[BoundingBox[0]:BoundingBox[2],BoundingBox[1]:BoundingBox[3]]
    return s


def bwareaopen(bw, sz):
    """
    BWAREAOPEN takes a b/w image and removes the connected areas smaller than
    the specified size. The output is a boolean array.
    ============================================================================
    INPUT DATA TYPE CHECKS NEED TO BE IMPLEMENTED
    ============================================================================
    """
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


def extract_nuclear_pointcloud(dapi, px_size: float, xyd = None):
    """
    Obtains point cloud of the centroids of nuclei from a 2D DNA dye image.

    Parameters
    ----------------------------
    - dapi : 2D numpy array
        The array contains the fluorescence data, expected to be from a DNA dye.
    - px_size : float
        The pixel size in X and Y (assumed identical).
    - alpha : float (default is 0.02)
        This is the parameter of the alpha shape, which determines the radius used to decide
        which edges of the triangulation of the point cloud stay in the hull. For alpha=0,
        the alpha shape is the convex hull of the point cloud.
    - xyd : numpy array (N, 2)
        The array contains the Y, X positions of N points that may be redundant with some
        in the point cloud generated by image analysis. These will be removed by proximity
        (which involves some guesswork, as everything else). If no `xyd` parameter is passed,
        there will be no subtraction of points.
    
    Returns
    -------
    An (M, 2) numpy array with the Y, X positions of blob centroids.
    """
    if xyd is None:
        xyd = np.empty((0,2))
    # Local thresholding and basic morphological filtering
    radius = np.ceil(100*px_size)
    footprint = skimo.disk(radius)
    mask = dapi >= skifi.rank.otsu(dapi, footprint)
    mask = skisg.clear_border(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = skimo.binary_erosion(mask, skimo.disk(1))
    # Find areas of FoV without tissue using variance
    maskmean = ndi.uniform_filter(dapi, size=radius)
    masksqmn = ndi.uniform_filter(dapi**2, size=radius)
    maskvar = masksqmn - maskmean**2
    maskvar = maskvar<skifi.threshold_otsu(maskvar)
    maskvar = skimo.binary_opening(maskvar, skimo.disk(np.floor(radius/2)) )
    # Remove blobs outside tissue
    mask = mask * (maskvar==0)
    # Marker-controlled watershed
    distim = ndi.distance_transform_edt(mask)
    distim = skifi.gaussian(distim, sigma=1)
    markers = skife.peak_local_max(distim)
    markim = np.zeros(distim.shape)
    markim[markers[:,0], markers[:,1]] = 1
    markim = skime.label(markim)
    ws = skisg.watershed(mask, markim)
    # Background is one of the objects, with label value not predictable, so:
    ws = ws!=mode(ws.flatten()).mode
    # Now get the centroids
    labelled = skime.label(ws)
    regs = skime.regionprops(labelled)
    centroids = np.array([np.round(x.centroid) for x in regs]).astype(xyd.dtype)
    centrim = np.zeros(ws.shape).astype('bool')
    centrim[centroids[:,0], centroids[:,1]] = True
    # Remove the esg[ts]FO cells
    area_thresh = np.ceil(skifi.threshold_otsu(np.array([x.area for x in regs])))
    radius_thresh = np.ceil(2 * np.sqrt(area_thresh/np.pi))
    gfp_mask = np.zeros(ws.shape)
    gfp_mask[xyd[:,0], xyd[:,1]] = 1
    gfp_mask = skimo.binary_dilation(gfp_mask, skimo.disk(radius_thresh))
    centrim = centrim & ~gfp_mask
    
    return np.array(np.where(centrim)).T
    
    
def threshold_triangle(img,bit_depth):
    """
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
    """
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
    """
    ---> NOW THIS IS AVAILABLE FROM SCIKIT-IMAGE! <---
    
    Return threshold value based on the Maximum Entropy method, similar to
    Otsu's method but maximizing inter-class entropy as S = -(sum)p*log2(p).
    """
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
    # find threshold value
    idx = np.argmax(entpy_sum)
    threshold = bin_centers[:-1][idx]
    return threshold    