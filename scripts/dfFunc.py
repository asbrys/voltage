"""
Functions used for calculating normalised fluorescence activity and blobs of significant activity
Contains:
    1. Loading functions
    2. Basic array manipulation
    3. Filtering and rolling stats
    4. Combined filtering/normalising functions
    5. Extracting continuous activity episodes

"""

import numpy as np, tifffile, imagecodecs, copy, scipy, pandas as pd, ray, bottleneck as bn
from scipy.ndimage import label, generate_binary_structure
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cuNd
except:
    pass



            ### 1 - LOADING STUFF ###

def tiff_read(filename):
    with open(filename, 'rb') as fh:
        data = fh.read()
    return imagecodecs.tiff_decode(data)


def loadTiff(FILE):
    return tifffile.imread(FILE, imread=tiff_read)



                ### 2 - ARRAY MANIPULATION ###


def normArray(data, Axis=None, Type=None, setMinMax=None):
    """
    Normalise array. Standard based on mean and std.
    If type = minmax then between 0 and 1 across stated axis.
    Input: Dict w keys: array, mean, std
    """
    if Type=='minmax':
        #print('normalising based on array minmax', end=' | ')
        if setMinMax:
            (Min, Max) = setMinMax
        else: 
            Min, Max = np.min(data, axis=Axis), np.max(data, axis=Axis)
        nArray = (data - Min)/(Max - Min)

    else:
        #print('normalising array based on mean / std', end=' | ')
        nArray = (data['array'] - data['mean'])/data['std']

    return nArray


def baselineTiff(Stack, Median, CUTOFF, NEWVALUE):
    """Finds x-y pixel locations where median is under a certain cutoff value. Then
    applies a new value to the pixel in the Tiff stack across every frame.
    Stack = Tiff stack to be baselined
    Median = median of TIff stack
    """
    (xvals,yvals)=np.where(Median < CUTOFF)
    for X,Y in zip(xvals,yvals):Stack[:,X,Y]=NEWVALUE
    return Stack


def pixelScale(Stack, Pix=None):
    """Between-pixel normalisation of tiff stack between 0 and 1.
    Gets pixel-wise min and max values, then normalises across time"""
    if Pix==True: # within-pixel normalisation
        return normArray(Stack,Axis=0)
    elif Pix==False: # across array normalisation
        return normArray(Stack,Axis=None)
    else: print('pixel or array-wide norm?')


def percentile(Stack, P):
    """
    Naive normalisation of Tif movie according to the percentile value of data in each pixel.
    Stack: type = np array. Dimensions (t,r,c) where t is number of frames, r is y pixels and c is x pixels. 
    P: type = int/float. Percentile value at which to normalise array 
    """
    base = np.percentile(Stack, P, axis = 0)
    base = np.expand_dims(base, axis = 0)
    return Stack - base


def subArray(d,N): 
    """
    Subtract a value N from an array.
    Use this to compress entries of normalised and thresholded arrays so that they diverge from 0.
    """
    return np.where(d>0,d-N,0)



                ### 3 - FILTERING, SAMPLING, ROLLING STATS ###


def gFilt(Stack, Wind, Gpu=False, Mode='reflect'): 
    """MultiD Gaussian filter. Window == dimensions of filter"""

    if Gpu:
        try:
            #print("filtering w GPU..", end=' ')
            filtCp = cuNd.gaussian_filter(cp.asarray(Stack), cp.asarray(Wind), mode=Mode)
            filt = cp.asnumpy(filtCp) 
            del filtCp
            return filt
        
        except:
            #print("filtering w/o GPU...", end='')
            return scipy.ndimage.gaussian_filter(Stack, Wind, mode = Mode)

    else:
        #print("filtering w/o GPU...", end='')
        return scipy.ndimage.gaussian_filter(Stack,Wind,mode=Mode)


def upSample(Stack, ZM, ORD):
    """Upsample using interpolation. ZM determines scaling size"""
    return scipy.ndimage.zoom(Stack, ZM, order=ORD)


def rollStd(Stack, Window):
    """
    Get rolling std of tiff movie. Try to use bottleneck module which is much faster than pandas.
    Input:
    1) 2d array (T, XY) where T is frames and XY is flattened x-y spatial pixels
    2) Length of rolling window in frames
    """
    try:
        stdPix = bn.move_std(Stack, window=Window, axis=0, ddof=1)
    except:
        print('using pandas for rolling std...', end=' | ')
        stdPix = pd.DataFrame(Stack).rolling(Window,center=True,axis='rows').std().to_numpy()
    stdPix = padArray(stdPix)
    return stdPix


def rollMean(Stack, Window, axis=0):
    """
    Get rolling mean of tiff movie (Default rolls down columns) 
    Inputs for default axis: 
    1) 2d array (T, XY) where T is frames and XY is flattened x-y spatial pixels
    2) Length of rolling window in frames
    """
    meanPix = bn.move_mean(Stack, Window, axis=axis)
    if axis==1:
        return padArray(meanPix.T).T
    else:
        return padArray(meanPix)


def padArray(arr):
    """
    Pads a NxM array containing with rows of NaNs at the start (or end) of axis 0
    with the 1st (or last) entries in each column. Used with rolling mean/std calculations.
    """
    v1,v2 = np.where(np.nan_to_num(arr[:,0],-1e9)>0)[0][[0,-1]] # choose value
    arr[:v1,:], arr[v2:,:] = arr[v1,:], arr[v2,:] # pad
    return arr



                #### 4 - COMBINED SCALING / NORMALISING FUNCTIONS ####


def scaleNorm(Stack, Scale=False, Pixel=False, Perc=None, Reverse=False):

    # Scale all values between 0 and 1
    if Scale: 
        Stack = pixelScale(Stack, Pix=Pixel) # within or across pixel scaling
        if Reverse: 
            Stack = 1-Stack
        if Perc: # get Df/F relative to percentile value
            percStack = np.percentile(Stack, Perc, axis=0)
            Stack = (Stack - percStack)/percStack
    
    else:
        percStack = np.percentile(Stack, Perc, axis=0) # get perc across axis 0
        Stack = -(Stack - percStack)/percStack # normalise relative to perc. Then invert.
    
    return Stack


def cellNon(Stack, Inds, SET=0):
    """
    Get locations of movie (eg for cell vs non-cell), fix these values (default 0)
    Input:
    1. Movie stack array (TxYxX)
    2. Indices 
    3. Set value 
    """

    # Copy array (weird results previously not doing this... test!!)
    stackI = copy.deepcopy(Stack)

    stackI[:,Inds[0],Inds[1]] = SET # Set non-cell and cell locations to value

    return stackI 



                #### 5 - Extracting contiguous activity episodes ####


def getContigData(nTiff, Info):
    """
    Get locations of contiguous activity episodes above given threshold and calculate their AUCs.
    Input: 
    1. Normalised array, 
    2. Threshold
    """

    # Get contiguous activity episodes. Returns: array w unique activity labels, n labels
    print("getting continuous activity episodes...", end=' | ')
    contigData = label(np.where(nTiff > Info['std'], 1, 0), \
                             structure = generate_binary_structure(3,3))
    
    # Get AUCs of contiguous activity episodes
    print("getting Aucs...")
    Aucs = getAucs(nTiff, contigData[0], contigData[1], Info)

    return {'labArray':contigData[0], 
            'nLabels':contigData[1], 
            'aucs':Aucs}


def getAucs(normArr, labArray, NLABELS, Info): 
    """
    Function to integrate across activity epochs in normalised array.
    """

    if Info['gpu']:
        print('using GPU...')
        actDist = cpTrapz(normArr, labArray, NLABELS)

    elif Info['par']:
        ray.shutdown()
        ray.init()
        normArr_ray, labArray_ray = ray.put(normArr), ray.put(labArray)
        actDist = ray.get([arrSearch.remote(labArray_ray,normArr_ray,N,Info['std']) \
                           for N in range(NLABELS+1)[1:]])
    else: 
        actDist = [np.trapz(normArr[labArray==int(N)]) for N in range(NLABELS+1)[1:]]

    return actDist


def remBaseActivity(normPixArr, actLabelArr, actDist, UPPER):
    """
    Finds locations of pixels associated w episodes whose integrated value is
    below threshold. Sets these pixel values equal to 0.
    Inputs: 1) matrix of all pixel values, 2) matrix of labelled discrete activity episodes
    3) Distribution of all activity values (indexed from 1!), 4) Percentile for activity value cutoff 
    """
    print('removing activity episodes below cutoff in cell-only and non-masked movies...')
    normPixArr[np.isin(actLabelArr,np.where(np.array(actDist)<UPPER)[0]+1)] = 0  # Activity labels indexed from 1 !!!!

    return normPixArr


                ### PARALLELISED FUNCTIONS ###


@ray.remote
def arrSearch(Arr1,Arr2,N,stdDev):
    """Find """
    return np.trapz(Arr2[Arr1==N]-stdDev)


                ### GPU FUNCTIONS ###


def cpTrapz2(normDat, labDat, Nlabels):
    """
    Integrate across activity epochs using Cupy
    Input:
    1) Normalised array
    2) Array with labelled epochs
    3) Number of labels
    """
    cp_normDat, cp_labDat = cp.asarray(normDat), cp.asarray(labDat)
    actDist = [cp.trapz(cp_normDat[cp_labDat==int(N)]) for N in range(Nlabels+1)[1:]]
    adist = [float(X) for X in actDist]
    del cp_normDat, cp_labDat
    return adist


def cpTrapz(normDat, labDat, Nlabels):
    """
    Vectorise AUC calculation.
    Split tiled dot product into columns to avoid memory issues....
    """
    
    #labDat = labDat.flatten()
    Aucs = []
    print(Nlabels,' total activity episodes...', end='  ')
    normDat, labDat = cp.asarray(normDat), cp.asarray(labDat)

    for L in np.arange(Nlabels)+1:
        Aucs.append(cp.sum(normDat[cp.where(labDat==L)]))
    
    Aucs = [AUC.get().item() for AUC in Aucs]

    """
    for i,block in enumerate(np.array_split(np.arange(Nlabels).astype(np.int16)+1,20000)):
        Labels = np.tile(block, (len(labDat),1)).astype(np.int16)
        boolLab = np.equal(labDat[:,None], Labels).astype(bool)
        Aucs.append(np.dot(normDat.flatten(), boolLab))
        print(i)
        del Labels, boolLab
    """
    #Labels = np.tile(np.arange(Nlabels).astype(np.int16)+1, (len(labDat),1)).astype(np.int16)
    #boolLab = np.equal(labDat[:,None], Labels).astype(bool)
    #Aucs = np.dot(normDat.flatten(), boolLab)

    return Aucs








