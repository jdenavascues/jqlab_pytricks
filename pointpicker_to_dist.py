###
### Get image data from exported MetaData
###

def getLIFMetaData(folderpath, filename):
    '''Will read MetaData folder after LIF exporting to TIFF from within the
    Leica software'''
    
    import os
    flist = os.listdir(folderpath)
    fname_root = filename.split('.')[0]
    
    for f in flist:
        if f.startswith(fname_root):
            datafile = f
    
    imgMetaData = datafile.readlines()
    pixel_size = imgMetaData[20].split('Voxel="')[1].split('"')[0]
    return float(pixel_size)

###
### Select relevant distances and obtain distance values
###

import numpy as np

def dist_2p(A):
    ''' A is assumed to be a 2x2 array - no check is done '''    
    from math import pow, sqrt
    s_l = sqrt(pow(A[0][0]-A[1][0],2) + pow(A[0][1]-A[1][1],2))       
    return s_l

def side_centr(A):
    ''' A is assumed to be a 2x2 array - no check is done '''        
    centre = [abs(A[0][0]-A[1][0])/2 + np.min([A[0][0],A[1][0]]), \
              abs(A[0][1]-A[1][1])/2 + np.min([A[0][1],A[1][1]])]    
    return np.array(centre)
    
def unique_rows(A):
    
    unique_idc = A.view(np.dtype( (np.void, A.dtype.itemsize * A.shape[1]) ) )
    _, idx = np.unique(unique_idc, return_index=True)
    A_unique_rows = A[idx]

    return A_unique_rows
    
def setdiff_rows(a1,a2):
    
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    diff_rows = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

    return diff_rows

def delaunay_distances(filename,pixel_size):

    import numpy as np
    from scipy.spatial import Delaunay
            
    THR = 8 #threshold to eliminate non-neighbouring links
    
    # read data from pointpicker output file
    f = open(filename,'rU')
    lines = f.readlines()[1:]
    XYdata = []
    sep = '   '
    for line in range(len(lines)):
        XYdata.append(lines[line].split(sep)[2:4])
    XYdata = np.array(XYdata,dtype='uint16')
    f.close()

    # obtain triangulation, get sides of triangles        
    D = Delaunay(XYdata)
    V = D.vertices
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
    
    # export to csv files
    import csv
    fl = open(filename+'.csv', 'w')
    writer = csv.writer(fl)
    for value in distances:
        writer.writerow([value])
    fl.close()    
    
    return XYdata, relevant_sides

###
### Generate visual ouput
###

def visualize_delaunay(XYdata, relevant_sides, image_name):

    import matplotlib
    import matplotlib.pyplot as plt
        
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
    #plt.show()
    plt.savefig(image_name+'.png', dpi=144, format='png')