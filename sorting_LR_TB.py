def label_sortedLRTB(bw, nrows):
    '''
    LABEL_SORTEDLRTB returns a labeled image where objects are ordered in label
    from left to right and then from top to bottom.
    This is based on an ad hoc solution which and requires and assumes:
        - previous knowledge of the number of rows
        - same number of objects per row
    Which is good for our purposes but it is not generally applicable.
    The trick can be made more generalized.
    '''
    import numpy as np
    import skimage.measure as skime
    
    # First we label as usual and get the regions properties (centroids)
    L = skime.label(bw)
    R = skime.regionprops(L)
    centroids = np.round(np.array([c.centroid for c in R])).astype(int)

    # we simplify the wiggliness of the centroid positions in y by fixing it to 
    # one value per row using histogram
    [dummy, ys] = np.histogram(centroids[:,0],nrows-1)
    ys = np.round(ys).astype(int)
    centroids[:,0] = np.repeat(ys,centroids.shape[0]/nrows)
    order = np.zeros(centroids.shape[0]).astype(int)

    # Now for the value of y in each row, we get the indices in x sorted.
    # store them in a vector.
    # Each of this sorting operations will start with 0 so all indices get added
    # the number of objects sorted in the previous rows:
    for line in range(nrows):
        # this gets the order in x of the objects in row 'line'
        indices = centroids[centroids[:,0]==ys[line],:].argsort(axis=0)[:,1]
        # this makes sure these order numbers are after the previous rows
        indices = indices+(centroids.shape[0]/nrows*line)
        # now this is stored in a vector
        order[0+(centroids.shape[0]/nrows)*line:(centroids.shape[0]/nrows)*(line+1)] = indices

    # 'order' now stores the indices in 'R' (and 'L') in the order they should be,
    # both in x and y as they have been both taken into account.
    
    # Now we re-create the labelled image simply by placing the objects into an
    # empty image, but with the value corresponding to their sorting order.
    sortL = np.zeros(L.shape)
    for o in range(len(order)):
        idx = order[o]
        sortL[R[idx].coords[:,0], R[idx].coords[:,1]] = o+1
        
    return sortL
    
    # to visualise the changes:
    # from skimage.io import imshow
    # imshow(np.hstack([L, sortL]))
    
