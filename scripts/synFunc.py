""""
Functions to perform more synaptic input analysis.
"""

#######################################################################################################
#   0 - Imports
#######################################################################################################

import numpy as np, copy, os, glob, cv2, seaborn as sns, pandas as pd, itertools
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib as mpl, ray, scipy
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import pdist, cdist
from math import erf, sqrt
from sklearn.cluster import KMeans
from sklearn import metrics as skmetrics, mixture
from scripts import vFunc as vF, clustSigFunc as cSF, dfFunc as dF
try: 
    import cupy as cp, cupyx as cpx
    from cupyx.scipy.spatial.distance import jensenshannon as cpJs, kl_divergence as cpKl
    from cupyx.scipy.spatial.distance import pdist as cpPd
    GPU = True
except ImportError as impErr:
    print("[Error]: {}.".format(impErr.args[0]))
    GPU = False


#######################################################################################################
#   1 - Functions to flatten / normalise continuous activity episodes
#######################################################################################################

def normEpoch(Activity):
    """
    Normalise 2D activity so total summed = 1.
    Return flattened normalised array
    """
    try: 
        Sum = cp.sum(Activity)
    except: 
        Sum = np.sum(Activity)
    return (Activity/Sum).flatten()


def tempHeatmap(allActivity, labAct, Nlabels, Binary = False):
    """
    Get either binary, or summed/normalised, 2D heatmaps for each activity episode.  
    Binary just returns a 1 or 0 if there is any activity in that region. 
    Try to use cupy first, and then fall back on numpy...

    Return:
    1. Array of normalised heatmaps of activity. Each heatmap is length M pixels x N pixels 
    """
    try:
        allActivity = cp.asarray(allActivity)
        labAct = cp.asarray(labAct)
        print("(using gpu for temporal heatmap)")
        
        if Binary:
            hM = cp.asarray([cp.any(labAct == int(x + 1), axis=0).flatten()\
                            for x in range(Nlabels)]).T*int(1) 
        else:
            hM = cp.asarray([normEpoch(cp.sum(cp.where(labAct == int(x + 1), allActivity, 0),\
                            axis = 0)) for x in range(Nlabels)]).T 
        hM = cp.asnumpy(hM)
    
    except:
        if Binary:
            hM = np.asarray([np.any(labAct == int(x + 1), axis = 0)\
                            for x in range(Nlabels)]).T * int(1) 
        else:
            hM =  np.asarray([normEpoch(np.sum(np.where(labAct == int(x + 1), allActivity, 0),\
                            axis = 0)) for x in range(Nlabels)]).T 
    
    del allActivity, labAct
    gpuClear()

    return hM


def getNormEpochs(allActivity, LAB_DATA = False, BINARY = False):
    """
    
    Normalise each blob of activity in a recording. 
    Consider doing this before clustering. I.e., if 2 blobs have the same location but just scaled intensity profiles
    they are probably from the same synaptic input!

    Input:
    1. Normalised + thresholded Tiff movie
    2. (optional) preprocessed labelled activity data (labelled array, Nlabels)
    
    Output:
    1. MxN activity array: M is flattened Ypix x Xpix, N number of activity epochs.
    """

    # Get labelled contiguous activity episodes
    if LAB_DATA:
            (labAct, Nlabels) = LAB_DATA
    else:
        print("getting continuous activity episodes + labels...", end=' ')
        labAct, Nlabels = label(np.where(allActivity > 0, 1, 0), \
                                structure = generate_binary_structure(3, 3))

    # temporally sum, normalise and flatten each episode
    print(Nlabels, ' activity episodes')
    print("getting normalised activity episodes...", end=' ')
    
    actM = tempHeatmap(allActivity, labAct, Nlabels, Binary = BINARY)
    
    del allActivity

    return actM, labAct


#######################################################################################################
#    2 - Functions to get similarity measure between activity episodes & background correction
#######################################################################################################

def getDistance(actM, MET = 'correlation'):
    """
    Get distance between all activity episodes.
    Input: 1. MxN matrix of flattened activity episodes
    Returns: 
    1. flattened 1D array of all pair-wise distances between heatmaps
    """

    print("getting distance")

    try:
        dist = cpPd(actM.T, metric = MET)
        dist = cp.asnumpy(dist)
        gpuClear()

    except:
        dist = pdist(actM.T, metric = MET)

    return dist


def getSm(distance, N):
    """convert flattened distance into symmetric sim matrix"""
    sM = np.ones((N,N))
    sM[np.triu_indices(N,k=1)] = distance
    sM = 1 - sM
    sM = sM + sM.T
    np.fill_diagonal(sM, 1)
    return sM


def bgHmCorrection(spatialDims, cellInds, actM, sM, Plot = True, Percentile = 99):
    """
    Find the distribution of mean heatmap correlations within non-cell regions. 
    Use this to remove weakly corrrelated heatmaps. These should mostly comprise background activity. 
    """
    # Mask for cell regions
    cellMask = np.zeros(spatialDims)
    cellMask[cellInds[0], cellInds[1]] = 1
    cellMask = cellMask.reshape(int(spatialDims[0] * spatialDims[1]))  

    cellOverlaps = np.dot((actM.T > 0), cellMask)       # Zero entries = no cell overlap
    nonOverlapInds, overlapInds = np.where(cellOverlaps == 0)[0], \
                                np.where(cellOverlaps != 0)[0] # Overlap/non-overlap indices 

    cutOff = np.percentile(np.mean(sM[nonOverlapInds, :], axis=1), 100 - Percentile)

    if Plot:
        fig,ax=plt.subplots(1,1,figsize=(8,3))

        _ = ax.hist(np.mean(sM[nonOverlapInds, :], axis = 1), bins = 100, alpha = 0.5)
        _ = ax.hist(np.mean(sM[overlapInds, :], axis = 1), bins = 200, alpha = 0.5)

        ax.plot(cutOff, 10, 'o', color='red')
        ax.legend(['cutoff', 'not cell','cell'])

        fig.tight_layout()
        fig.savefig('synFigs/hmDistanceDistribution.png',dpi=400)

        plt.close()

    return cutOff


def trimHeatmaps(cutOff, actM, sM, labAct, METRIC = 'corrrelation', \
                 SPLITS = 3, GPU = True):
    """
    Remove weakly correlated heatmaps for background correction
    """
    
    delBins = np.where(np.mean(sM, axis = 1) > cutOff)[0] # Bins of activity matrix to delete
    del sM
    
    # Delete bins of activity matrix and re-calculate distances
    actM = np.delete(actM, delBins, axis = 1)   
    thDist = getDistance(actM, MET = METRIC)

    delBins = delBins + 1           # Labelled data begins at 1

    # Convert entries corresponding to bins to zero
    labAct = ((~np.in1d(labAct, delBins).reshape(labAct.shape)) * labAct).astype(np.int16)

    # Now re-label existing labels starting from 1 to correspond to new actM bins
    labels = np.unique(labAct)[np.unique(labAct) > 0]   # Sorted list of existing labels
    newLabels = np.arange(len(labels)) + 1              # New labels 

    if GPU:
        
        newLabAct = None

        Splits = int(len(newLabels)/SPLITS)

        labAct = cp.asarray(labAct).astype(cp.int16)
        newLabels, labels = cp.asarray(newLabels).astype(cp.int16), \
                            cp.asarray(labels).astype(cp.int16)

        for new_labels, old_labels in zip(cp.array_split(newLabels, Splits), \
                                            cp.array_split(labels, Splits)):
            
            newSum = ((labAct[:, :, :, None] == old_labels).astype(cp.int16) \
                      * new_labels).astype(cp.int16)
            
            if newLabAct is None:
                newLabAct = newSum 
            else:
                newLabAct = cp.sum(cp.concatenate([newLabAct, newSum], axis = 3), \
                                   axis = 3)[:,:,:,None].astype(cp.int16)
            #print('current ', new_labels[-1])
            del newSum

        del labAct

        newLabAct = cp.asnumpy(newLabAct[:,:,:,0])

        return actM, thDist, newLabAct
    
    else:
        for I, (L1, L2) in enumerate(zip(labels, newLabels)):
            labAct[np.where(labAct == L1)] = L2
            print(I)
        return actM, thDist, labAct
    
    
def hmActivityCover(actM, cellInds, rowData, Dims = (32, 128)):
    """
    Calculate the percentage of all the heatmap activity accounted for down to a certain hierarchy depth.
    Get both total, and just 'cell' regions
    Inputs:
    1. Matrix of flattened heatmap activity
    2. 
    """
    
    Activity = {'cell': {},'total': {}}

    LEVEL = len(rowData)
    
    # Get total summated activity
    cellSum = np.sum(actM.reshape(Dims[0], Dims[1], actM.shape[1])[cellInds[0], cellInds[1], :])

    for I, LVL in enumerate(range(LEVEL)):
        lvlBins = [rowData[LVL]['sortedBins'][L1:L2] for L1,L2 in \
                        zip(rowData[LVL]['ensLocSorted'][:-1], rowData[LVL]['ensLocSorted'][1:])]
        
        Activity['cell'][int(LEVEL-I)] = (np.sum([np.sum(actM[:, Bins].reshape(Dims[0], Dims[1], actM[:,Bins].shape[1])[cellInds[0], cellInds[1], :]) \
                                    for Bins in lvlBins])/cellSum) # Reshape activity matrix and select cell locations only

        Activity['total'][int(LEVEL-I)] = (np.sum([np.sum(actM[:,Bins]) for Bins in lvlBins])/np.sum(actM))
    
    del actM, rowData
    
    return Activity


#######################################################################################################
#    3 - Synapse specific plotting functions
#######################################################################################################

def heatmapCluster(rowData, actMap, Dim=None):
    """
    Plot heatmaps of every activity episode associated with each cluster.
    Input:
    1. Rowdata from addSimEns script containing sorted bins and ensemble
    locations for a given hierachy level.
    2. Normalised + flattened activity matrix ('actM')
    """

    # Get bins for each ensemble
    allClusters = [rowData['sortedBins'][X1:X2]\
        for X1,X2 in zip(rowData['ensLocSorted'][:-1],rowData['ensLocSorted'][1:])]

    if not Dim:
        print('need to specify image dimensions')
        return

    for NCL, cluster in enumerate(allClusters):

        Nsheets = len(cluster)//48 + 1

        for Sheet in range(Nsheets):

            fig, ax = plt.subplots(12, 4,figsize=(15,20))

            for I, AX in enumerate(ax.flatten()):
                try:
                    AX.pcolor(actMap[:,cluster[I+(48*Sheet)]].reshape(Dim))
                except: pass

            fig.tight_layout()
            fig.savefig('synFigs/Clust_'+str(NCL)+'_sheet_'+str(Sheet+1)+'.png',dpi=400)
            plt.close()
        print('cluster {} done'.format(NCL), end = ' | ')
    
    return allClusters


def plotMeanEnsHm(actM, rowData, Inline = False, Dim = (32,128), Abase = 0.2, Save = None, \
                  Combined = True, colors = False, clusterLabels = None, clusterList = None,\
                  labelText = False, plotDiffuse = False, ALPHA = 0.35, UPSAMPLE = 5, \
                    filt = (2, 2)):
    """
    Plot the mean heatmaps associated with a given input cluster, or an arbitrary list of heatmaps if
    clusterLabels and clusterList are not None
    Input:
    1. 2D matrix of activity episdoes (rows = unfolded pixels, columns = episodes)
    2. 
    3. clusterlabels: labels of clusters to plot
    4. clusterList: 
    """
        
    # Plotting info
    Info = {'bgr': os.path.basename(glob.glob('background/*.png')[0]),\
                'AlphHm': ALPHA, 'upSampHm': UPSAMPLE, 'inlineHm': Inline, 'filtHm': filt}
    if not Save: 
        Save = '_'

    # Get lists containing the bins (for actM) of each cluster and their labels
    if clusterList is not None:
        assert clusterLabels is not None, 'Need to specify clusterList!'
        clusterBins = getLabelBins(clusterLabels, clusterList, rowData)
    else:
        if plotDiffuse: # Include the weakest cluster
            clusterBins = [rowData['sortedBins'][X1:X2] for X1, X2 in \
                           zip(rowData['ensLocSorted'], rowData['ensLocSorted'][1:] + [-1])]
            clusterLabels = [rowData['bestRow'][rowData['sortedBins'][X1]]\
                            for X1 in rowData['ensLocSorted']]
            
        else:           # Exclude the weakest cluster
            clusterBins = [rowData['sortedBins'][X1:X2] for X1, X2 in \
                           zip(rowData['ensLocSorted'][:-1], rowData['ensLocSorted'][1:])]
            clusterLabels = [rowData['bestRow'][rowData['sortedBins'][X1]]\
                            for X1 in rowData['ensLocSorted'][:-1]]
        
    if len(clusterBins) < 1:    # If no activity at a certain hierarchy
        return

    if Combined:
        
        # For each cluster, get flattened activity episodes and find pixel-wise mean
        meanClusters = [np.mean(actM[:, Cluster], axis = 1).reshape(Dim) for Cluster in clusterBins]

        # Get maximum activity across pixels for each cluster, normalise across all clusters
        maxClustValues = np.asarray([np.max(np.sum(actM[:, Cluster], axis = 1)) \
                             for Cluster in clusterBins])
        adjustedClustNorm = maxClustValues / np.max(maxClustValues)

        # Change colours if you want a different heatmap colour for each cluster
        if colors:
            colors = [sns.color_palette(palette = C, as_cmap = True) for X in range(5) \
                  for C in ['Greens', 'Oranges', 'Blues', 'Reds', 'Purples']]
        else:
            if plotDiffuse:
                colors = [cm.jet for X in range(len(meanClusters))][:-1] + [cm.spring]
            else:
                colors = [cm.jet for X in range(len(meanClusters))]
            
        # Turn pixel-wise mean for each cluster into rgba heatmap
        meanFrames = [vF.getActivityImage(mCluster[None, :, :], Info['upSampHm'], Info['filtHm'], \
                                        Cm, Info['AlphHm'], Abase, norm = Norm) for mCluster, Norm, Cm \
                                        in zip(meanClusters, adjustedClustNorm, colors)]
        
        # Combine heatmaps onto the same image
        # Create 2 combined arrays: 1 for overlap & 1 for non-overlap regions
        combinedImage1 = addOverlaps(meanFrames)
        nonOverlapFrames, _, _ = overlapFunc(meanFrames, overlap = False)
        combinedImage2 = multiframeAdd(nonOverlapFrames)

        # Combine overlap and non-overlap regions
        combinedImage = cv2.add(combinedImage1, combinedImage2)

        # Add background
        bg = vF.getBgImage(Info['bgr'], Info['upSampHm'], Save = None, bgFilter = (2, 2))
        bg_fr = vF.combineImages(bg, combinedImage)

        # Add number for cluster onto each plotted heatmap
        if labelText:

            assert clusterLabels is not None, 'need labels to plot!'

            nLabels = len(clusterLabels)

            for I, (clustFrame, text) in enumerate(zip(meanFrames, clusterLabels)):
                # Get location to place text
                (y, x) = np.where(clustFrame[:, :, 3] > 0)
                y, x = int(np.quantile(y, 0.25)), int(np.quantile(x, 0.2))

                # Place text
                if plotDiffuse:
                    if I + 1 == nLabels:
                        textColor = (0, 0, 255, 255)
                    else:
                        textColor = (255, 255, 255, 255)
                else:
                    textColor = (255, 255, 255, 255)
                bg_fr = cv2.putText(bg_fr, str(int(text)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, \
                            0.3, textColor, 1, cv2.LINE_AA)

        cv2.imwrite('synFigs/hMap_'+Save+'combinedClust.png', bg_fr)

    else:
        for CLUST, Cluster in enumerate(clusterBins):

            # Get mean image associated with given cluster
            meanIm = np.mean(actM[:,Cluster],axis=1).reshape(Dim)

            vF.pngHeatmap(meanIm[None,:,:], Info, Abase, cm.jet,\
                        'synFigs/hMap_'+Save+'cluster'+str(CLUST)+'.png')


def plotSubClusters(LABEL, parentTree, clusterList, actM, rowData, separate = True, Dim = (32, 128)):
    """
    Plot the heatmaps of a cluster and all of its children subclusters. 
    Can either plot separately or combine on the same image (which might be messy)
    """

    loc = [(R, np.where(row == LABEL)[0][0]) for R,row in \
                    enumerate(parentTree) if len(np.where(row == LABEL)[0]) > 0][0]          # Get location of labels
    
    subclustTree = branchingNans(parentTree, loc, extractValues=True)               # Get subcluster tree

    if separate:
        subClusters = []
        for row in subclustTree:
            if np.nansum(row) == 0: break
            subClusters.append([C for C in row if ~np.isnan(C)])
        
        for I, clusters in enumerate(subClusters):
            plotMeanEnsHm(actM, rowData, labelText = True, Save = 'subClust_' + str(LABEL) + '_' + str(I) + '_',\
                    clusterList = clusterList, clusterLabels = clusters, Dim = Dim)

    else:
        subClusters = [int(val) for row in subclustTree for val in row if ~np.isnan(val)]    # Convert to list w/o nans

        plotMeanEnsHm(actM, rowData, labelText = True, Save = 'subClust_'+str(LABEL)+'_', Dim = Dim,\
                    clusterList = clusterList, clusterLabels=[int(SCL) for SCL in subClusters])


def multiframeAdd(multiFrames, Wt=None):
    """
    Add multiple image frames and weight each frame per pixel.
    Input frame is (MxNx4): ie, xPix x yPix x BGRA
    Weight is MxN 
    """
    if Wt is None: 
        Wt = 1
    else:
        Wt = Wt[:,:,None].astype(np.float64) # add dim for RGBA

    combinedImage = np.zeros_like(multiFrames[0])
    for frame in multiFrames:
        wtFrame = Wt*frame.astype(np.float64)
        combinedImage = cv2.add(combinedImage,wtFrame)
    
    return combinedImage


def addOverlaps(arrays):
    """
    Add overlapping heatmap regions and weight each overlap by the number of
    overlapping regions.
    """

    # remove non-overlapping regions from arrays, get n overlaps/pixel and weights
    overlapArr, mask, allAct = overlapFunc(arrays, overlap=True)
    pixOverlaps = np.sum(allAct,axis=0)*mask
    pixWt = np.nan_to_num(1/pixOverlaps,0,posinf=0)
    
    # Combine images
    combinedImage = multiframeAdd(overlapArr, pixWt)
    
    return combinedImage


def overlapFunc(arrays,overlap=True):
    """
    Find locations of non-overlapping activity. 
    Get locations of each BGRA array w alpha > 0. Sum, select locs with sum = 1. 
    """
    allAct = np.asarray([ar[:,:,3]>0 for ar in arrays]).astype(int) # Find alpha>0
    if overlap:
        mask = np.sum(allAct,axis=0) > 1  # bool mask
    else:
        mask = np.sum(allAct,axis=0) == 1  
    arrays = (allAct*mask)[:,:,:,None]*arrays # apply mask
    return arrays, mask, allAct


def getLabelBins(clusterLabels, clusterList, rowData):
    """Get the bins for activity maps corresponding to a list of cluster labels"""

    bins = []

    for L in clusterLabels:
        ROW, COL = getLabelLocation(L, None, clustList=clusterList)[0]
        rowD = rowData[ROW]
        clusterBins = [rowD['sortedBins'][X1:X2]\
            for X1,X2 in zip(rowD['ensLocSorted'][:-1],rowD['ensLocSorted'][1:])][COL]
        bins.append(clusterBins)
    
    return bins


def getClusterMovies(thTiff, rowData, labAct, Dim = (32,128), Std = None, \
                     SPD = 0.4, vBase = 0.1, vAlph = 0.5, fName = None, Short = None, \
                        clusterLabels = None, clusterList = None):
    """
    This creates a concatenated movie of the activity episodes associated with clusters. Clusters can either be 
    at 1 level of the hierarchy (leave clusterlabels and cluster List as none) or any arbitrary list of clusters
    if these are specified.
    Input:
    1. Thresholded tiff of activity
    2. Row data from addSimEns function
    3. Labelled activity episodes
    4. Also need to specify Std of threshold etc
    """

    #  **** Get movie info ****
    Info = {'Speed':SPD,                  # Speed of video
            'hPad':0.1,                    # pad activity epochs on either side (seconds)
            'vAlph': vAlph,                  # Alpha value for movie
            'vAbase':vBase,                    # Alpha threshold for movie activity
            'hRange':[Std, Std+0.25],   # Lower and upper range for cMap (change for diverging depol + hyper)       
            'vFilt':(2,2),                  # Gaussian filter for each activity frame
            'fps':417,
            'bgr': os.path.basename(glob.glob('background/*.png')[0])
            }
    
    if not fName: 
        fName = 'synFigs/clusterMovTest'
    
    # Get activity colormap
    cNorm = mpl.colors.Normalize(vmin=Info['hRange'][0],vmax=Info['hRange'][1]) # Dynamic range
    cMap = cm.ScalarMappable(norm=cNorm, cmap=cm.jet) # Colormap

    # Get padding
    padDur = 0.5 # seconds for padding duration between activity episodes
    pad = np.zeros((int(padDur*Info['fps']*Info['Speed']),Dim[0],Dim[1]))

    # Get bins for each cluster
    if clusterLabels is None:
        clusterBins = [rowData['sortedBins'][X1:X2]\
                for X1,X2 in zip(rowData['ensLocSorted'][:-1],rowData['ensLocSorted'][1:])]  
    else:
        assert clusterList is not None, 'cluster list is None: need to provide the cluster \
                                        list to plot each cluster label!'
        clusterBins = getLabelBins(clusterLabels, clusterList, rowData)

    for CLUST, Cluster in enumerate(clusterBins):

        clMovie = np.zeros((int(padDur*Info['fps']*Info['Speed']),Dim[0],Dim[1]))

        # ? shorten movie
        if Short: 
            Cluster = Cluster[:int(Short)]
        
        for Bin in Cluster:

            # Careful: labelled data starts at 1! Add 1 to bin to get corresponding label
            Epoch = np.where(labAct == Bin + 1, thTiff, 0) # Return tiff movie of this section
            Epoch = Epoch[np.where(Epoch > 0)[0], :, :]    # temporal slice

            # Update cluster movie
            clMovie = np.concatenate([clMovie, Epoch, pad],axis=0)
        
        # now turn the cluster into a movie
        vF.pngVidConvert(clMovie, Info, cMap, fps=Info['fps']*Info['Speed'],\
                         fName=fName+'_clust'+str(CLUST)+'.mp4')


def plotDend(lD):

    import scipy.cluster.hierarchy as sch

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    dg = sch.dendrogram(lD, no_labels = True, ax = ax)

    fig.tight_layout()

    return (fig,ax), dg


#######################################################################################################
#    4 -  Monte carlo heatmap functions
#######################################################################################################

def hmMonteCarlo(Hm, Runs, Dim = (32,128), batchSize = (20,50), shufflePlot = False):
    """
    Randomly permute heatmaps, calculate distances, get percentile value.

    Return:
    1. N x M array: N = number of runs, M = shuffled distance calculation for each blob-blob pair for given run
    2. 
    """
 
    # A0: n heatmaps; A1: n row pixels; A2: n column pixels
    A0, (A1, A2) = Hm.shape[1], Dim                             # axis dimensions for convenience
    assert int(A1 * A2) == Hm.shape[0], "Rows x column must = unfolded heatmap pixels!"
    Hm = Hm.T.reshape(A0, A1, A2)                               # reshape heatmaps into 3D array

    (zLoc, yLoc, xLoc) = np.where(Hm > 0)                       # Get Hm non-zero locations

    # Rows = heatmaps, True = pixels corresponding to zLoc/yLoc/xLoc locations 
    # Ie, for heatmap index 0, with 5 pixels > 0, boolLoc[0, :5] = True
    # Then, for heatmap index 1, with 5 pixels > 0, boolLoc[1, 5:12] = True
    boolLoc = zLoc[None, :] == np.linspace(0, A0 - 1, A0).astype(int)[:, None]

    # Randomly shuffle x/y locations independently across each heatmap 
    print("getting random heatmap locations..", end = '')
    #print(A0, A1, A2, yLoc.shape, boolLoc.shape)
    newY = getNewLocVect(yLoc, boolLoc, A1, Runs, batchSize = batchSize[0])
    newX = getNewLocVect(xLoc, boolLoc, A2, Runs, batchSize = batchSize[0])
    del boolLoc

    # Specify replacement shuffled Hm values, array to store results
    hmInputs = Hm[zLoc, yLoc, xLoc]       # replacement values
    allDistances = np.empty((Runs, int((A0 * (A0 - 1)) / 2))).astype(np.float16)

    # Batch split info
    Nsplit = Runs // batchSize[1]                    # number of arrays after splitting
    splitLengths = [len(X) for X in np.array_split(newY, Nsplit)] # length of each split array

    if shufflePlot:
        sPlot = np.empty((Runs, Dim[0], Dim[1]))  # for plotting some shuffled frames
    del Hm

    print("getting random distances..")
    for sRun, (split1, split2) in enumerate(zip(np.array_split(newY, Nsplit), np.array_split(newX, Nsplit))):

        batchDistance = cp.empty((splitLengths[sRun], int((A0 * (A0 - 1)) / 2))).astype(np.float16)

        for Run, (nY, nX) in enumerate(zip(split1, split2)):

            newHm = np.zeros((A0, A1, A2))      # new heatmaps

            newHm[zLoc.astype(int), nY.astype(int), nX.astype(int)] = hmInputs  # Populate new Hm with shuffled values

            batchDistance[Run, :] = cpPd(newHm.reshape((A0, int(A1 * A2))), \
                                  metric = 'correlation')                     # get distances
            
            if shufflePlot:
                sInd = int(np.sum(splitLengths[:sRun])+Run)
                sPlot[sInd,:,:] = newHm[shufflePlot, :, :] 
            
        batchDistance = cp.asnumpy(batchDistance)

        allRuns = int(np.sum(splitLengths[:sRun]))
        allDistances[allRuns:int(allRuns+splitLengths[sRun]),:] = batchDistance
        del batchDistance
    
    del newHm
    gpuClear()

    if shufflePlot:
        return allDistances, sPlot
    else:
        return allDistances, None


def getNewLocVect(axis, boolLoc, Dim, Runs, batchSize = 20):
    """
    This script gets new independently randomly shuffled locations of heatmaps across either the x or y axis 
    """

    # Loc: row is a heatmap, and values different (x or y) pixel location. 
    # Ie, if yLoc entered, for heatmap index 0 with 5 +ve pixels, Loc[0][:5] contain y location of heatmap
    # And for heatmap index 1 with 7 +ve pixels, Loc[1][5:12] contain y location of hm
    axis = axis + 1                             # Add 1 because we want to turn non-pixel locations into Nans
    Loc = (axis[None, :].astype(np.int16) * boolLoc[:, :]).astype(np.float32)
    Loc[Loc == 0] = np.nan                      # turn zeros to nans
    Loc = Loc - 1                               # Return pixels to original values
    del boolLoc
    
    ### Divide the section below into smaller batches for GPU
    allLocs = np.empty((Runs, Loc.shape[1]))
    allBatches = [batchSize for X in range(Runs // batchSize)] +\
        [Runs % batchSize for X in range(1) if Runs % batchSize != 0] 

    for RUN, batch in enumerate(allBatches):

        Vori0 = np.nanmin(Loc, axis = 1)            # Minimum pixel location for each heatmap 
        Vnorm = Loc - Vori0[:, None]                # Normalised pixel locations

        # Get bounds of minimum and maximum locations to place new heatmap
        Length = len(Vori0)
        dimVect = np.full(Length, Dim) 
        #print(Dim, dimVect.shape, dimVect, Length)

        MinV = np.tile(np.zeros(Length), (batch, 1))
        MaxV = np.tile((dimVect - np.nanmax(Vnorm, axis = 1)), (batch, 1))

        # Create new origin with dimensions of each heatmap and total runs
        #print(np.where(np.isnan(MaxV)))
        newOri = np.random.randint(MinV, high = MaxV)
        del MinV, MaxV, Vori0, dimVect

        newOri = newOri.astype(np.int16)
        Vnorm = Vnorm.astype(np.float16)

        try:
            Vnorm = cp.asarray(Vnorm)
            newOri = cp.asarray(newOri.astype('int16'))
            newLoc = cp.nansum(Vnorm + newOri[:, :, None], axis = 1).astype(np.float16)
            newLoc = cp.asnumpy(newLoc)
            del Vnorm, newOri
            gpuClear()
        except:
            newLoc = np.nansum(Vnorm + newOri[:, :, None], axis = 1)

        # Get start and end indices for array and append
        Start = (batchSize * (Runs // batchSize)) - (((Runs // batchSize) - (RUN)) * batchSize)
        End = Start + [Runs % batchSize if RUN == Runs // batchSize else batchSize][0]

        allLocs[Start: End] = newLoc

    return allLocs


def shuffleVideo(sPlot, fps=5, cRange=0.15, vAbase=0.01):
    """
    Create video of a shuffled heatmap
    """

    Info = {'Speed':1,                  # Speed of video
            'hPad':0.1,                    # pad activity epochs on either side (seconds)
            'vAlph': 0.7,                  # Alpha value for movie
            'vAbase':vAbase,                    # Alpha threshold for movie activity
            'hRange':[0, cRange],   # Lower and upper range for cMap (change for diverging depol + hyper)       
            'vFilt':(2,2),                  # Gaussian filter for each activity frame
            'fps':fps,
            'bgr': os.path.basename(glob.glob('background/*.png')[0])
            }

    cNorm = mpl.colors.Normalize(vmin=Info['hRange'][0],vmax=Info['hRange'][1]) # Dynamic range
    cMap = cm.ScalarMappable(norm=cNorm, cmap=cm.jet) # Colormap

    vF.pngVidConvert(sPlot, Info, cMap, fps=Info['fps']*Info['Speed'],\
                            fName='synFigs/monteShuffle.mp4')


def getThreshDistance(actM, Runs, batchSize, dist, spatialDims, percentile = 10, \
                      shufflePlot = False, sInfo = None):
    """
    Threshold distance values between heatmaps by randomly shuffling the spatial location of each heatmap,
    recalculating the distance between shuffled heatmaps, and using these values as a distribution.  
    Choose distances BELOW a certain cutoff. I.e., percentile = 10 chooses the closest 10% distances.

    Return:
    1. Thresholded distances. Either real distance if < thresh, or max distance
    2. N x M shuffled distance matrix

    """
    distPerm, sPlot = hmMonteCarlo(actM, Runs, batchSize = batchSize, Dim = spatialDims, \
                                      shufflePlot = shufflePlot)

    dPerc = np.percentile(distPerm, percentile, axis = 0)  # Take lower because it's a distance measure...

    thDist = np.where(dist < dPerc, dist, np.max(dist))

    if shufflePlot:
        try:
            shuffleVideo(sPlot, fps = sInfo['fps'], \
                            cRange = sInfo['cRange'], vAbase = sInfo['vAbase'])
        except:
            print('shuffle video not created')
    
    return thDist, distPerm, sPlot



#######################################################################################################
#    5 -  Manipulating & plotting similarity matrices
#######################################################################################################


def sortSim(sM, cMr, Sorthigh=True):
    """
    Generate a similarity matrix where each ensemble is grouped together
    Input: 
    1. Original similarity matrix
    2. ROW (1xM) of cluster matrix for given hierachy level 
    """
    (_,M) = sM.shape
    sMorg = np.zeros((M,M))
    triInds = np.triu_indices(M,k=1)

    # gets bins of cluster matrix row sorted into the same cluster, and group according to cluster size
    #sortedBins = sortSize(-cMr)
    sortedBins = sortMeanSim(cMr, sM, Sorthigh=Sorthigh)

    # Re-index similarity matrix by sorted bins
    sMorg[triInds] = sM[sortedBins[triInds[0]],sortedBins[triInds[1]]]
    sMorg = sMorg + sMorg.T
    np.fill_diagonal(sMorg,1) 

    return sMorg


def unsortUnqInds(arr):
    """Returns unsorted start/end indices of unique values in a SORTED array"""
    unsortInds = [I + 1 for I, X in enumerate(arr[:-1]) if arr[I + 1] != X]
    unsortInds.insert(0, 0)
    return unsortInds


def sortMeanSim(row, sM, Sorthigh=True):
    """
    Return bin indices sorted by unique values, with unique values sorted by
    mean similarity within each unique value
    """
    # Get unique vals, counts, bool array for counts > 1, and row locations
    uVals, uCounts = np.unique(row,return_counts=True)
    uLocs = np.where(uCounts>1,True,False) 
    boolLocs = uVals[uLocs][:,None]==row[None,:] # M unique values X N activity

    # Get mean similarity values for each unique value, sort, append for counts < 2
    meanSims = [np.sum([sM[INDS] for INDS in itertools.combinations(np.where(uBool)[0],2)])/\
            ((np.count_nonzero(uBool)*(np.count_nonzero(uBool)-1))/2 ) for uBool in boolLocs] 
    
    # Sort means in ascending ort descending depending on measure
    if Sorthigh:
        sortedMeans =  np.argsort(meanSims)[::-1]
    else:
        sortedMeans =  np.argsort(meanSims)

    sortedUvals = np.append(uVals[uLocs][sortedMeans], uVals[~uLocs])

    # sort bins according to similarity-sorted unique values
    sortedBins = np.where(row[None,:]==sortedUvals[:,None])[1]

    return sortedBins


def plotSim(sM, sMorig, ROW, rows, cM = None, bnds = None, addClust = True, \
            LW = 1, Sorthigh = False, N_ENS = -2, COL = 'green'):
    """Plot unsorted and sorted similarity matrix"""

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    if bnds:
        sns.heatmap(sM, ax = ax, vmin = bnds[0], vmax = bnds[1])
    else:
        sns.heatmap(sM, ax = ax)
    
    if addClust:

        Row = cM[rows[ROW]]

        sortedBins = sortMeanSim(Row, sMorig, Sorthigh = Sorthigh)

        ensLocSorted = unsortUnqInds(Row[sortedBins])

        for p1,p2 in zip(ensLocSorted[:-1], ensLocSorted[1:]):
            ax.plot([p1,p2], [p1,p1], color = COL, lw = LW) 
            ax.plot([p1,p2], [p2,p2], color = COL, lw = LW)
            ax.plot([p1,p1], [p1,p2], color = COL, lw = LW) 
            ax.plot([p2,p2], [p1,p2], color = COL, lw = LW)
    
    else:
        Row, sortedBins, ensLocSorted = None, None, None

    ax.set_xticks([]), ax.set_yticks([])

    fig.savefig('synFigs/simPlot_' + str(ROW) + '.png',dpi = 400)

    plt.close()

    return {'bestRow': Row, 'sortedBins': sortedBins, 'ensLocSorted': ensLocSorted}


def clustMat(lD):
    """
    Create matrix where row = cluster level, col = bins, entry is cluster number 
    for each bin.   
    """

    print("getting matrix of cluster hierarchies")

    if GPU:

        lD = cp.asarray(lD)

        # Get size of cluster matrix, populate 1st row
        N = int(int(cp.max(lD[:, :2]) + 1) / 2) + 1 
        cM = cp.zeros((N, N)).astype(cp.uint16) 
        cM[0] = cp.linspace(1, N, N).astype(cp.uint16) 

        # Populate each row
        for i, lvl in enumerate(lD[:, :2]):

            cM[i+1] = cM[i] 

            # Find entries of current row corresponding to joined clusters, and
            # set these entries in following row as next cluster number   
            cM[i+1][cp.nonzero(cp.in1d(cM[i], lvl + 1))[0]] = N + i + 1

        cM = cp.asnumpy(cM)
        del lD
        cp._default_memory_pool.free_all_blocks()
    
    else:
        N = int(int(np.max(lD[:,:2])+1)/2)+1 
        cM = np.zeros((N, N)).astype(np.uint16) 
        cM[0] = np.linspace(1, N, N).astype(np.uint16) 

        for i, lvl in enumerate(lD[:, :2]):
            cM[i+1] = cM[i]  
            cM[i+1][np.nonzero(np.in1d(cM[i], lvl + 1))[0]] = N + i + 1

    return cM



#######################################################################################################
#    6 -  Cluster metrics & significance functions
#######################################################################################################

def testWholeTree(lD, cM, rowData, actM, nData = 1000, Dots = 1e4, \
                  Thresh = 90, norm = True, Dim = (32,128)):
    """
    Function to sequentially test entire cluster tree.

    Return:
    1. List of significant cluster labels across the  hierarchy
    """

    ####################
    ###  Get information
    ####################

    clLabels, parentTree = clustInfo(lD = lD, cM = cM, rowData = rowData)       # List of all clusters
    _, nodes = scipy.cluster.hierarchy.to_tree(lD, rd = True)                   # List of nodes
    del lD

    allClList = pd.Series([int(entry) for row in clLabels[::-1] for entry in row]).unique()  # List of clusters to test
    trueClustList = []                                                                      # True cluster list

    #####################
    ###  Test 1st cluster
    #####################

    sigList, discardList = testClusterChildren(allClList[0], cM, parentTree, nodes, \
                            actM, nData = nData, Dots = Dots, Thresh = Thresh, \
                                norm = norm, Dim = Dim)   # Test children of largest/tightest cluster

    [trueClustList.append(X) for X in sigList]                              # Store significant clusters
    allClList = [entry for entry in allClList if entry not in discardList]  # Remove label + lineage from further testing

    ############################
    ###  Test remaining clusters
    ############################

    while len(allClList) > 0: 

        print('\nmoving onto cluster {}. Remaining labels are\n {}\n'.format(allClList[0],allClList))   

        highParents, keepList, discardList = testAdjacentClusters(allClList[0], clLabels, \
                        parentTree, rowData, actM, Dots = Dots, nData = nData, Thresh = Thresh, \
                        norm = norm, Dim = Dim)       # Check for overlaps with other clusters   
        
        if len(highParents) != len(keepList):                               # Overlaps exist: no further testing 
                
                ######
                # Consider changing. Eg, don't necessarily discard all children??  
                ######
                print('\nnot testing children of cluster', allClList[0])  

                labelLoc = getLabelLocation(allClList[0], parentTree)

                discardList = [int(entry) for row in branchingNans(parentTree, labelLoc, extractValues=True) \
                        for entry in row if ~np.isnan(entry)] + discardList                 # Add label + children to discard list

                allClList = [int(entry) for entry in allClList if entry not in discardList] # Discard overlaps, label + children

        else:                                                                               # No overlap: test children
            
            print('\ntesting children of cluster', allClList[0])
            
            sigList, discardList = testClusterChildren(allClList[0], cM, parentTree, nodes, \
                                    actM, nData = nData, Dots = Dots, Thresh = Thresh, \
                                    norm = norm, Dim = Dim)    # Test children

            [trueClustList.append(int(X)) for X in sigList]                                 # Store significant clusters

            allClList = [int(entry) for entry in allClList if entry not in discardList]     # Remove label + lineage from further testing
    
    return trueClustList, parentTree, clLabels 


def testAdjacentClusters(LABEL, clLabels, parentTree, rowData, actM, Thresh = 95, nData = 100, \
                         Dots = 1e4, norm = True, Dim = (32,128)):
    """
    This function tests for significance between a cluster and the highest parents of any adjacent clusters. 
    Inputs:
    1. Label of next cluster to test
    2. cluster labels
    3. parent Tree
    4. rowData
    5. activity matrix
    Return:
    1. highParents: list of adjacent high parents
    2. Keeplist: high parents that are significantly separated
    3. Discard list: high parents that are NOT separated  
    """

    highParents = getHighParents(LABEL, clLabels, parentTree)   # Adjacent parents of cluster

    hpBins = [getLvlBins(LBL, rowData, getLevel = False)[1][0] for LBL in highParents] # Parent bins
    lbBins = getLvlBins(LABEL, rowData, getLevel = False)[1]                           # Label bins

    hpHm = getHms(actM, binList = hpBins, norm = norm, Dim = Dim)                   # Parent heatmaps
    lbHm = getHms(actM, binList = lbBins, norm = norm, Dim = Dim)[0]                # Label heatmaps

    overlaps = np.asarray([hmOverlap(lbHm, adjHm) for adjHm in hpHm])  # Overlaps? (bool)

    keepList = np.asarray(highParents)[~overlaps].astype(int).tolist()  # List of clusters to keep. Included non-overlaps
    discardList = []                                                    # Discard list

    for overlapHm, LBL in zip(np.array(hpHm)[overlaps], np.array(highParents)[overlaps]): # Test significance...
        
        combinedHm = overlapHm + lbHm                       # Combine parent and label heatmap

        SCALE = int(Dots / (np.mean(combinedHm[np.where(combinedHm > 0)]) * \
                            len(np.where(combinedHm > 0)[0])))      # Get coordinate point scale
        
        [combinedDots, clusterDots, adjDots] = [getDots(Hm, Scale=SCALE) for \
                    Hm in [combinedHm, lbHm, overlapHm]]            # Get coordinate points for testing
        
        clustSig, _, _ = subclustSig(combinedDots, clusterDots, adjDots, nData=nData) # Test significance
        
        print('Clusters {} are separate. Adjacent cluster sig between {} & {} is {}'.format(keepList, LABEL, int(LBL), clustSig))
        
        if clustSig >= Thresh: 
            keepList.append(int(LBL))
        else:
            discardList.append(int(LBL))
    
    return highParents, keepList, discardList


def testClusterChildren(LABEL, cM, parentTree, nodes, actM, DEPTH = 21, nData = 100, \
                        Dots = 1e4, Thresh = 95, norm = True, Dim = (32,128)):
    """
    This function tests for significance of a cluster's children.
    """

    labelLocation = getLabelLocation(LABEL, parentTree)

    _, sigList = clusterBranchSig(labelLocation, cM, parentTree,nodes, actM, \
                                  nData = nData, Dots = Dots, Threshold = Thresh, \
                                  norm = norm, Dim = Dim) # Lowest significant children
    
    discardList = [int(entry) for row in branchingNans(parentTree, labelLocation, extractValues = True) \
                   for entry in row if ~np.isnan(entry)]   # Discard all labels in lineage for future testing
    
    return sigList, discardList


def clusterBranchSig(labelLocation, cM, parentTree, nodes, actM, nData = 100, \
                     Threshold = 95, Dots = 1e4, norm = True, Dim = (32,128)):
    """
    Main function to get significance of all clusters extending from a root node. 
    Get the label of a cluster, get a list of subclusters, test for significance of all subcluster pairs, 
    and then repopulate the list with NaNs if not significant.
    Inputs:
    1. Rootlabel
    2. Cluster matrix
    3. list of all nodes
    4. Flattened activity matrix
    Return:
    1. Cluster list containing ID of significant child subclusters
    """

    clList = branchingNans(parentTree, labelLocation, extractValues = True) # Get list of subclusters

    for LVL in range(len(clList[1:])):                    # Iterate over child rows 
        for PAIR in range(int(len(clList[1:][LVL])/2)):   # Iterate over double entries in row 

            level = clList[1:][LVL]    # Redefine rows and entries since we are repopulating w nans
            child_pairs = np.array_split(level, len(level)/2)[PAIR]

            if np.isnan(child_pairs[0]): 
                continue  # Move to next cluster if children are NaN (outside depth)
            
            parent = int(clList[LVL][PAIR])     # Parent from the row above
            
            # TEST SIGNIFICANCE
            (percentile, _, _), _ = childSig(nodes, cM, actM, parent, nData=nData, \
                                             Dots=Dots, norm=norm, Dim=Dim)

            print('populating')
            if percentile < Threshold:                      # If not significant, populate children as Nan
                clList = branchingNans(clList, (LVL, PAIR)) # LVL = parent level, PAIR = parent column

    while np.nansum(clList[-1])==0: 
        clList.pop()    # remove any end rows filled with nans
    
    keepList = np.unique(getClosestParents(clList)).astype(int) # Get unique nearest parent values

    return clList, keepList


def childSig(nodes, cM, actM, parentLabel, Dim = (32,128), norm = True, nData = 100, Dots = 1e4):
    """
    Get child clusters of a parent and calculate separation significance based on mean
    heatmaps of activity bins. Dots is the approx number of points generated for the parent Hm
    Inputs:
    1. nodes of cluster hierarchy
    2. cluster matrix
    3. Activity matrix
    3. Label of parent cluster obtained from cluster matrix

    Outputs:
    1. significance of cluster and associated scores
    2. 
    """
    
    # Get labels of joined clusters. Linkage index is reduced by 1. Add 1 to re-index using cluster matrix. 
    c1, c2 = getChildren(nodes, int(parentLabel))

    assert np.max(np.where(cM==c2)[0])==np.max(np.where(cM==c1)[0]),\
        'final row of cluster matrix where child clusters form are not equal!'

    allBins = [np.where(lastRow(cM, c2, row = True) == Child)[0] for Child in [c1, c2]]        # Get bins of each child
    allBins.append(np.where(lastRow(cM, parentLabel, row = True) == parentLabel)[0])           # Add bins of parent

    # Get mean heatmaps of child and parent clusters
    Hms = getHms(actM, binList = allBins, norm = norm, Dim = Dim)                               # Get activity Hm & convert into co-ordinates
    SCALE = int(Dots / (np.mean(Hms[2][np.where(Hms[2] > 0)]) * len(np.where(Hms[2] > 0)[0])))

    # Convert mean heatmap into pixels with spatial co-ordinates
    allCoords = [getDots(Hm, Scale = SCALE) for Hm in Hms]

    print('\ngetting significance for parent ', parentLabel)      # Get significance
    (percentile, siScores, siOriginal) = subclustSig(allCoords[2], allCoords[0], allCoords[1], nData = nData)

    print('\npercentile for parent {} and children {} & {} is {}'.format(parentLabel, c1, c2, percentile))
    return (percentile, siScores, siOriginal) , (c1, c2)


def subclustSig(parentClust, sc1, sc2, nData=100, PAR=True, GPU=True):
    """"
    Get significance of the separation of 2 child clusters from a parent by generating random Gaussian 
    data and calculating a distance metric.
    Inputs:
    1. Coordinate points corresponding to parent cluster
    2 & 3. Coordinate points corresponding to child clusters
    4. Number of surrogate datasets to generate during analysis
    Return:
    1. Percentile score (eg, 0.9 means 90% of surrogate scores are lower)
    2. Surrogate data scores
    3. Original data score
    """

    siOrig = getSilh(sc1, c2 = sc2, GPU = GPU)              # Silhouette score of child clusters

    gaussParams = getGaussParams(parentClust)               # Gaussian Parameters of parent

    gaussData = genGauss(gaussParams, nData)                # Surrogate data

    Km = KMeans(n_clusters = 2, n_init = 'auto')            # K-means to separate surrogate data

    if GPU == True:
        #print('getting labels', end=' | ')
        if PAR == True:
            #print('using Ray', end=' | ')
            remoteKfit = ray.remote(KmFit)
            Labels = ray.get([remoteKfit.remote(Km, data) for data in gaussData])
        else:
            Labels = [Km.fit(data).labels_ for data in gaussData]
        #print('using GPU for silhouette scores')
        Scores = np.asarray([getSilh(data, labels = label, GPU = GPU) for data, label in zip(gaussData, Labels)])

    elif PAR == True:
        #print('using Ray')
        remoteKfit = ray.remote(KmFit)
        Labels = ray.get([remoteKfit.remote(Km, data) for data in gaussData])

        remoteSilh = ray.remote(getSilh)
        Scores = ray.get([remoteSilh.remote(data,labels=label) \
                          for data, label in zip(gaussData, Labels)])
        ray.shutdown()

    else:
        Labels = [Km.fit(data).labels_ for data in gaussData]
        Scores = np.asarray([getSilh(data,labels=label) for data, label in zip(gaussData, Labels)])

    return scipy.stats.percentileofscore(Scores, siOrig), Scores, siOrig


def KmFit(km, data): 
    return km.fit(data).labels_


def getSilh(c1, c2 = None, labels = None, GPU = False):
    """
    Calculate the silhouette score for 2 clusters
    Input:
    1. cluster 1 & 2. Array: N obs x M features.
    2. Labels (list o rarray) of each cluster in c1. Only if c2 is None. 
    Return:
    1. Silhouette score
    """
    # Get silhouette score from known data
    if c2 is not None:
        if labels is not None:
            print('Cannot pre-specify labels if 2 clusters are given!')
            return
        else:
            data = np.concatenate([c1,c2])
            labels = [1 for X in range(c1.shape[0])]+[2 for X in range(c2.shape[0])]
    
    # Get silhouette score from generated data
    else:
        if labels is None:
                print('Need labels provided if only one cluster given!')
                return None
        else:
            data = c1

    #return skmetrics.silhouette_score(data,labels)
    #print(len(data), len(labels), 'LENGTHS')
    return silhScore(data, labels, GPU=GPU) 


def silhScore(data, labels, GPU = False):
    """
    Vectorised function to get silhouette score. 
    Get distance matrix of all points. Between-cluster distances are off-diagonal blocks. 
    Much faster than sklearn for this data.
    """

    if GPU:
        coords, labels = cp.asarray(data), cp.asarray(labels)

        (c1, c2) = [coords[labels == L] for L in cp.unique(labels)]
        L1, L2 = len(c1), len(c2)

        allP = cp.concatenate([c1, c2])                              # Concatenate all points
        Dist = cpx.scipy.spatial.distance.cdist(allP, allP)          # Full distance matrix
        cp.fill_diagonal(Dist, cp.nan)                               # Fill diagonals

        #del coords, labels, allP

        a = cp.concatenate([cp.nanmean(Dist[:L1, :L1],axis=1),\
                            cp.nanmean(Dist[L1:, L1:],axis=1)])     # Within cluster distances

        b = cp.concatenate([cp.mean(Dist[:L1, L1:],axis=1),\
                            cp.mean(Dist[L1:, :L1],axis=1)])        # Between cluster distances

        ss = cp.mean((b - a)/(cp.max(cp.asarray([a, b]), axis = 0)))

        #del a, b
        #sF.gpuClear()

        return float(ss.get())
    
    else:
        (c1, c2) = [data[labels==L] for L in np.unique(labels)]
        L1 = len(c1)

        allP = np.concatenate([c1,c2])                          # Concatenate all points
        Dist = cdist(allP,allP)                                 # Full distance matrix
        np.fill_diagonal(Dist,np.nan)                           # Fill diagonals

        a = np.concatenate([np.nanmean(Dist[:L1, :L1],axis=1),\
                            np.nanmean(Dist[L1:, L1:],axis=1)]) # Within cluster distances

        b = np.concatenate([np.mean(Dist[:L1, L1:],axis=1),\
                            np.mean(Dist[L1:, :L1],axis=1)])    # Between cluster distances (off-diagonal blocks)

    return np.mean((b-a)/(np.max([a,b],axis=0)))            # Return silhouette score


def getGaussParams(coords):
    """
    Get mean and covariance matrix of co-ordinates. nObs is the number of observations in the dataset. 
    Input:
    1. NxM coordinates. N is observations, M is each variables (i.e., M=2 for 2D coordinates)
    Returns:
    1. Mean vector
    2. Covariance matrix
    """
    coV = np.cov(coords, rowvar = False) 
    mean = np.mean(coords, axis = 0)
    return {'mean':mean, 'cov':coV, 'nObs':coords.shape[0]}


def genGauss(params, n):
    """
    Generate n multivariate Gaussian surrogate datasets. Same number of observations in each dataset. 
    """
    return [np.random.multivariate_normal(params['mean'], params['cov'], size=params['nObs'])\
        for X in range(n)]


def getClustMetrics(data, cM, Range=10, Metric='silhouette'):
    """
    Compute clustering metrics across top N cluster hierarchies
    Input:
    1. Data. Either similarity matrix (silhouette score) or activity matrix
    2. Cluster matrix (rows = cluster hierarchies, columns = cluster labels)
    """

    allMetrics = []
    topLevel = int(cM.shape[0]-1)

    for cLevel in range(topLevel-Range, topLevel):
        if Metric == 'silhouette':
            assert data.shape[0] == data.shape[1], \
                'data needs to be symmetric similarity matrix'
            allMetrics.append(skmetrics.silhouette_score(data, cM[cLevel,:], \
                metric='precomputed'))
        elif Metric == 'calinski_harabasz':
            if data.shape[0] == data.shape[1]:
                print('careful: data is symmetric, need to use activity marray for this metric!')
            allMetrics.append(skmetrics.calinski_harabasz_score(data.T, cM[cLevel,:]))
        elif Metric == 'davies_bouldin':
            if data.shape[0] == data.shape[1]:
                print('careful: data is symmetric, need to use activity marray for this metric!')
            allMetrics.append(skmetrics.davies_bouldin_score(data.T, cM[cLevel,:]))
        else:
            print('unknown metric')

    return allMetrics


def clustSig(data, labels, n_simulations = 10, covariance_method = 'soft_thresholding'):
    """
    This function is taken from the sigclust package.
    It creates N multidimensional Gaussians with statistics based on the data points corresponding
    to each of 2 labels, specified in 'labels'. Based on this, calculates the probability that the 
    data points form separate clusters.
    Input:
    1 - data (2d Array). Rows = entries, columns = values
    2 - labels (1d array, length = number of data rows). Each entry is 1 of 2 values. 
    3 - number of simulations for boot strapping. 
    """

    nRows, nColumns = data.shape
    eigenvalues = cSF.getEv_cp(data, covariance_method)
    print('calculated eigenvalues', end = ' | ')

    sample_cluster_index = cSF.compute_cluster_index(data, labels)

    ray.shutdown()
    ray.init()

    simulated_cluster_indices = ray.get([cSF.simulate_cluster_indexPar.remote(i,nRows,nColumns,eigenvalues) \
                                            for i in range(n_simulations)])

    print('finished simulating cluster indices', end = ' | ')

    p_value = np.mean(sample_cluster_index >= simulated_cluster_indices)
    z_score = (sample_cluster_index - np.mean(simulated_cluster_indices)) / \
        np.std(simulated_cluster_indices, ddof=1)

    print('p-value and z-score are: {}'.format((p_value, z_score)))
    return p_value, z_score


def getHms(actM, binList = None, rowData = None, norm = True, Dim = (32, 128)):
    """
    Get list of mean heatmaps from either list, or across a hierarchy level from 'rowData' 
    Each list contains bins corresponding to locations of activity matrix. +/- normalise heatmaps.
    Inputs:
    1. Activity matrix
    2. Either a binlist or row from rowdata
    3. Normalise?
    Returns:
    1. Heatmap list
    """

    if (binList is None and rowData is None) or (binList is not None and rowData is not None):
        print('Need to specify either rowData or binlist!')
        return

    if rowData is not None:
        binList = [rowData['sortedBins'][L1:L2] for L1,L2 in \
            zip(rowData['ensLocSorted'][:-1], rowData['ensLocSorted'][1:])]

    meanHms = [np.mean(actM[:, bins], axis=1).reshape(Dim) for bins in binList]

    if norm:
        meanHms = [dF.normArray(Hm, Type='minmax') for Hm in meanHms]

    return meanHms


def hmOverlap(hm1, hm2):
    """
    Check if 2 heatmaps are either adjacent or overlap with each other.
    Input:
    1 & 2. Heatmaps (n x M array)
    Return:
    1. Bool
    """

    # Convert heatmap to binary coordinate points
    points = [getDots(np.nan_to_num(HM/HM), Scale=1) for HM in [hm1, hm2]]

    # shift the points in the 1st heatmap by 1 pixel in each direction to check for adjacency
    sPoints = shiftPoints(points[0])

    # Check for any overlap between heatmap 2 and all the shifted points
    overlap = np.sum([np.sum([np.sum((P == shift).all(axis=1)) for P in points[1]]) for shift in sPoints])

    return overlap > 0


def getDots(Hm, Scale = 100):
    """Convert a normalised 2D heatmap with intensity values into co-ordinates for each pixel, 
    with number of co-ordinates proportional to heatmap intensity
    Inputs:
    1. Heatmap
    2. Scale: multiply heatmap values by this and convert to an Int
    Return:
    1. Nx2 array, where each N is a (y, x) co-ordinate
    """

    if Scale:
        dots = np.asarray([[xLoc,yLoc] for xLoc, yLoc in \
            zip(np.where(Hm>0)[0],np.where(Hm>0)[1]) for X in range(int(Hm[xLoc,yLoc]*Scale))])
    
    else:
        dots = np.asarray([[xLoc,yLoc] for xLoc, yLoc in \
            zip(np.where(Hm>0)[0],np.where(Hm>0)[1])])
    
    return dots


def shiftPoints(points):
    """
    Shift all 2D heatmap points by 1 pixel in each x-y direction. 
    Input:
    1. List of [y,x] locations of points
    Return:
    1. List of length 4, each entry is array of [y,x] locations of points
    """
    shiftedPoints = []
    for axis in [0,1]: 
        for shift in [1,-1]:
            newPoints = copy.deepcopy(points)
            newPoints[:,axis] = newPoints[:,axis] + shift
            shiftedPoints.append(newPoints)
    
    return shiftedPoints


def htTest(g1, g2):
    """
    Calculate hotelling's T^2 test between 2 gaussian distributions.
    Input:
    1. Dict of group1 info: mean, covariance matrix, n Obs
    2. Dict of group2 info as above
    """

    [n1, cov1, mean1] = [g1[KEY] for KEY in ['nObs','cov','mean']]
    [n2, cov2, mean2] = [g2[KEY] for KEY in ['nObs','cov','mean']]
    
    # Calculate the pooled covariance of the two groups and difference in means 
    pooled_cov = ((n1-1) * cov1 + (n2-1) * cov2) / (n1 + n2-2)
    mean_diff = mean1 - mean2
    
    # Calculate Hotelling's T² statistic
    t2_stat = (n1 * n2) / (n1 + n2) * mean_diff.T.dot(np.linalg.inv(pooled_cov)).dot(mean_diff)
    
    # Determine the degrees of freedom for the numerator (number of variables) and denominator
    df1 = len(mean1)
    df2 = n1 + n2 - df1-1
    
    # Convert T² statistic to an F statistic, get p-value
    f_stat = t2_stat * (df2 / df1) * (n1 + n2-2) / (n1 + n2)
    p_value = 1 - scipy.stats.f.cdf(f_stat, df1, df2)
    
    return t2_stat, p_value


def clusterSigTest(Labels, rowData, actM, nDots = 1000, Dim = (32, 128), norm = False):
    """ 
    Testing function. Enter any 2 arbitrary clusters, test for significance and 
    return all the data. 
    """
    assert len(Labels) == 2, print("This function only tests 2 labels!")

    labelBins = [getLvlBins(L, rowData, getLevel = False)[1][0] for L in Labels]
    Hms = getHms(actM, binList = labelBins, norm = norm, Dim = Dim) 

    combinedHm = Hms[0] + Hms[1]

    SCALE = int(nDots / (np.mean(combinedHm[np.where(combinedHm > 0)]) * len(np.where(combinedHm > 0)[0])))

    Dots = [getDots(Hm, Scale = SCALE) for Hm in [Hms[0], Hms[1], Hms[0] + Hms[1]]] 

    siOrig = getSilh(Dots[0], c2 = Dots[1], GPU = True)

    gaussParams = getGaussParams(Dots[2])        # Gaussian Parameters of parent

    gaussData = genGauss(gaussParams, 100)       # Surrogate data

    Km = KMeans(n_clusters = 2, n_init = 'auto')        # K means

    remoteKfit = ray.remote(KmFit)
    Labels = ray.get([remoteKfit.remote(Km,data) for data in gaussData])

    Scores = np.asarray([getSilh(data, labels = label, GPU = True) for data, label in zip(gaussData, Labels)])

    return scipy.stats.percentileofscore(Scores, siOrig), Dots, Hms, gaussData, siOrig, Scores



#######################################################################################################
#    6 -  Cluster hierarchy wrangling functions
#######################################################################################################

def getSubclusters(cM, Depth, nodes, rootLabel):
    """
    Get labels of all subclusters of a given node up to certain depth. 
    Do not use this repeatedly. Re-calculate sub-trees using 'branchingNans' function.
    Inputs:
    1. Cluster matrix
    2. Depth of search
    3. all nodes
    4. root label of cluster (using cluster matrix label)
    """

    # Get cluster matrix row search depth
    depthRow = cM.shape[0] - Depth
    depthEntries = np.unique(cM[depthRow:]).tolist()
    depthEntries.remove(rootLabel)

    Index = -1
    clusterList = [int(rootLabel)]
    Loop = None

    # Create a flattened list which sequentially adds on children, until no more entries are present
    while Loop is None:
        try:
            clusterList, Index, Loop, depthEntries = getChildClusters(nodes, \
                                    clusterList, Index, cM, depthRow, depthEntries)
        except Exception as err:
            print(err)
            pass
    
    # Extend subclusters to length of hierarchy, and turn into nested list
    clusterList, Levels = extendClusterList(clusterList)
    _, clusterList = clustListToDict(Levels, clusterList, todict=False)

    clusterList = [entry for entry in clusterList if np.nansum(entry)>0]

    return clusterList, Levels


def getChildClusters(nodes, clusterList, Index, cM, depthRow, depthEntries):
    """
    Get children of nodes from a node list and add to list if they join together at a level above a given depth =
    ?? over complicated. Need to double check.....

    Inputs:
    1. all nodes from cluster hierachy analysis
    2. List of clusters descending from root
    3. Index of currently analysed cluster from list
    4. cluster matrix
    5. depthRow
    Return:
    1. Clusterlist
    2. Index (int, for search)
    3. Loop (bool) to signify end of search
    """
    """
    if np.isnan(clusterList[Index+1]): 
        
        # if the last row is filled with nans then stop the loop
        if Index+1 in 2**np.linspace(0,10,11):
            index = int(np.where(2**np.linspace(0,10,11) == Index+1 )[0])
            if np.nansum(clusterList[-index:]) == 0:
                return clusterList, Index, True

        clusterList += [np.nan,np.nan]
        Index += 1
        return clusterList, Index, None
    """
    if len(depthEntries) == 0:
        return clusterList, Index, True, depthEntries
    
    elif np.isnan(clusterList[Index+1]): 
        clusterList += [np.nan,np.nan]
        Index += 1
        return clusterList, Index, None, depthEntries
    
    else:
        # Get child nodes
        #if Index==10: print('z')
        child = getChildren(nodes, clusterList[Index+1])

        # Get the last row for the children. Make sure this row is the same. 
        childRows = [lastRow(cM, Ch) for Ch in child]
        assert childRows[0]==childRows[1], 'child rows are not equal!'

        # Append the child to the list if they are at or above our depth of interest
        if childRows[0] >= depthRow:
            clusterList += child
            for C in child:
                try: depthEntries.remove(C)
                except: pass
        else:
            clusterList += [np.nan,np.nan]

        # Advance Index by 1
        Index += 1

        return clusterList, Index, None, depthEntries


def extendClusterList(subClusterList, totalDepth=30):
    """
    Extend a list of clusters (obtained via 'getSubclusters' function) to a length that
    corresponds to the total number of clusters across a given number of hierachies. 
    Input:
    1. Subcluster list
    2. totalDepth. This just refers to the maximum depth for the level search. 20 is alot.
    Output:
    1. Extended subcluster list
    """
    # Get total number of levels by subtracting powers of 2 from the length of subclusters
    try:
        Levels = np.where(np.cumsum(2**np.linspace(0,totalDepth-1,totalDepth))-len(subClusterList)>=0)[0][0] + 1
    except Exception as Error:
        [print('\nCan\'t extend list: probably need more depth\n', Error)]
        return

    # Get the difference in length and extend with Nones
    lengthDiff = np.sum(2**np.linspace(0, Levels - 1, Levels)) - len(subClusterList)
    
    if lengthDiff > 0:
        subClusterList += [np.nan for X in range(int(lengthDiff))]
        subClusterList = np.asarray(subClusterList)

    return subClusterList, Levels


def getChildren(nodes, label):
    """Return labels of child nodes using cluster matrix indexing"""
    return int(nodes[label-1].right.id + 1), int(nodes[label-1].left.id + 1)


def lastRow(cM, label, row=False): 
    """Return row (or index) of uppermost row in cluster matrix containing a given label"""
    if row:
        return cM[np.max(np.where(cM==label)[0])]
    else:
        return np.max(np.where(cM==label)[0])


def clustListToDict(Levels, subClusters, todict=False):
    """
    Convert a list of subclusters into a nested dictionary or list of lists corresponding to cluster hierarchies. 
    Each key (cluster) containslength 2 array, with each entry containing a dict corresponding to child cluster.
    Input:
    1. Number of levels of cluster hierarchies
    2. Flattened list of subclusters
    3. Convert to dict or list of lists
    """

    # Get cluster indices, list of cluster lists for each level, and then reverse it
    cIs = np.insert(np.cumsum(2**np.linspace(0, Levels - 1, Levels)), 0, 0).astype(int) # Cluster indices
    clList = [subClusters[start:stop] for start, stop in zip(cIs[:-1],cIs[1:])]

    if todict is False:
        return None, clList
    
    else:

        rvList = [l for l in clList[::-1]]

        ## Get list of dict entries for each level
        level = rvList[0]
        while len(level) > 1:
            allD = []
            for level in rvList:
                lvlDict = {K:{} for K in level}
                allD.append([{K:V} for K,V in lvlDict.items()])

        # Convert list of dicts into a nested dictionary
        for lv1, lv2 in zip(allD[:-1], allD[1:]):

            # split lv1 into array with double entries
            lv1Dicts = np.array_split(np.asarray(lv1),int(len(lv1)/2))

            # For each array entry, put dicts in the dict within the corresponding entry in above level
            for parentNode, childDict in enumerate(lv1Dicts):
                lv2[parentNode][[K for K in lv2[parentNode].keys()][0]] = childDict

        # Select only the final entry
        allD = allD[-1]
        
        return allD, clList


def branchingNans(parentTree, Location, extractValues = False):
    """
    Take a list corresponding to the entries of a cluster hierarchy, and a location within the hierarchy.
    Convert all lower branching entries into nans.
    Inputs:
    1. List of hierachy entries. Each entry of list must be 2x the size of the previous entry.
    2. Root location (Row, Column) to then specify Nan entries. 
    Outputs:
    1. List of hierarchy entries filled with Nans where relevant.
    """

    R, C = Location         # Row, column entry

    tailLength = int(len(parentTree) - (R + 1))  # Number of distal rows to fill

    if extractValues:
        extractedValues = copy.deepcopy([np.asarray([parentTree[R][C]])]) # Top value of tree

        for ROW in np.arange(tailLength) + 1:
            nFills = int(2**ROW)                           # Number of new entries doubles for each row
            C = int(2*C)                                   # Beginning of column index of row to fill
            nextRow = parentTree[R + ROW][C:C + nFills]
            extractedValues.append(nextRow)

        return extractedValues
    
    else:

        nanList = copy.deepcopy(parentTree)

        for ROW in np.arange(tailLength) + 1:
            nFills = 2**ROW     # Number of new entries for each row
            nanList[R + ROW][(nFills*(C+1))-nFills:nFills*(C+1)] = [np.nan for x in range(nFills)]
        
        return nanList


def getUpperClusters(Label, Tree, clustLabels):
    """
    For a given cluster label, get it's successive parent cluster labels until a parent is incorporated
    into the diffuse cluster.
    Inputs:
    1. Label of child cluster (int)
    2. Tree (list of lists): List of hierarchies containing cluster labels 
    3. Cluster Labels: list of lists 
    Return:
    1. Label of highest parent that is not within diffuse cluster
    """

    diffClust = False

    while diffClust is False:
        # Get row and entry of current cluster within Tree
        treeIndex = [[row, np.where(Label == lvl)[0][0]] for row, lvl in enumerate(Tree) \
                    if len(np.where(Label == lvl)[0])>0][0]
        
        # Get parent label within Tree
        parentLabel = Tree[treeIndex[0]-1][treeIndex[1]//2]

        # If parent label not in clustLabels, set True
        if sum([len(Ar[0]) for Ar in [np.where(np.asarray(lbls)==int(parentLabel)) for lbls in clustLabels]]) == 0:
            diffClust = True
        
        # Else keep false, and relabel
        else:
            diffClust = False
            Label = parentLabel
    
    return Label


def getLvlBins(LABEL, rowData, getLevel=True):
    """
    For a given cluster label, get the level in the cluster matrix at which it forms (ie its lowest level).
    Get the activity bins for this cluster (labelBins), and all other clusters at the same level (levelBins), 
    sorted by correlation. This does NOT include the final cluster of each level (ie the most diffuse cluster).
    Input:
    1. Label number
    2. Row data
    Outputs:
    1. LOWLVL: Lowest level of hierarchy
    2. labelBins: Bins for the labelled cluster (None if just a list of all clusters selected)
    3. levelBins: list of bins of all clusters (+/- labelled cluster)
    4. labels: Labels corresponding to all clusters (+/- original cluster)
    """

    LOWLVL = np.min(np.where(np.asarray([level['bestRow'] for level in rowData]) == LABEL)[0])

    # Get bins & heatmaps at this level. Split into the main heatmap and all others 
    if getLevel:
        labelBins = None
        levelBins = [rowData[LOWLVL]['sortedBins'][L1:L2] for L1,L2 in \
                zip(rowData[LOWLVL]['ensLocSorted'][:-1], rowData[LOWLVL]['ensLocSorted'][1:])]

    else:
        labelBins = [rowData[LOWLVL]['sortedBins'][L1:L2] for L1,L2 in \
                zip(rowData[LOWLVL]['ensLocSorted'][:-1], rowData[LOWLVL]['ensLocSorted'][1:]) if \
                    rowData[LOWLVL]['bestRow'][rowData[LOWLVL]['sortedBins'][L1]] == LABEL]
        
        levelBins = [rowData[LOWLVL]['sortedBins'][L1:L2] for L1,L2 in \
                zip(rowData[LOWLVL]['ensLocSorted'][:-1], rowData[LOWLVL]['ensLocSorted'][1:]) if \
                    rowData[LOWLVL]['bestRow'][rowData[LOWLVL]['sortedBins'][L1]] != LABEL]
    
    # Get labels associated with other bins in the level
    labels = np.asarray([rowData[LOWLVL]['bestRow'][Bins[0]] for Bins in levelBins])
    
    return LOWLVL, labelBins, levelBins, labels


def clustInfo(lD = None, cM = None, rowData = None):
    """
    Get useful lists that contain cluster hierarchy information.
    Return:
    1. clLabels: List of lists w cluster labels at all levels, but 'diffuse' cluster excluded. Ie List 0 = empty.
    This differs to 'parentTree' because there is only 1 new cluster label at each level. 
    2. parentTree: list of lists of parent labels and their children. Includes the diffuse cluster. 
    """
    
    if rowData is not None:
        try:
            clLabels = [[int(level['bestRow'][level['sortedBins'][L]]) for L in \
                    level['ensLocSorted']][:-1] for level in rowData]
            #clLabels = [[int(level['bestRow'][level['sortedBins'][L]]) for L in \
            #        level['ensLocSorted']] for level in rowData]
            parentTree = None
        except:
            print('need to specify rowData if trying to get cluster labels')
    
    if lD is not None:
        try:
            _, nodes = scipy.cluster.hierarchy.to_tree(lD, rd=True)
            parentTree, _ = getSubclusters(cM, len(rowData), nodes, rowData[-1]['bestRow'][0])
            if rowData is None: 
                clLabels = None
        except:
            print('need to specify linkage data and cluster matrix if trying to get parent tree')
    
    try: 
        return clLabels, parentTree
    except:
        print('need to specify Labels or Tree')


def getHighParents(LABEL, clLabels, parentTree):
    """
    For a cluster label, this function return the highest parents of all other clusters within the hierachy that are 
    NOT within the label's subgroup.
    Inputs:
    1. Label of interest
    2. Cluster labels. From 'clustInfo' fnct. Complete list of clusters (minus diffuse) at each level.
    3. parentTree. From 'clustInfo' fnct. Branching list of distinct clusters up to stated depth.
    """

    # Get highest level (from top = 0) at which the label of interest exists in the parentTrees list
    H_LVL = [row for row, cl in enumerate(parentTree)\
                    if len(np.where(np.array(cl) == LABEL)[0]) > 0][0]
    H_LOC = np.where(np.array(parentTree[H_LVL]) == LABEL)[0][0]

    removedTree = branchingNans(parentTree, (H_LVL, H_LOC)) # Get parent list with label lineage excluded
    removedTree[H_LVL][H_LOC] = np.nan                      # Also remove label

    # Only select labels within the lowest level that are WITHIN the removedTree list
    lowLabels = [value for value in clLabels[0] if value in \
                 [entry for tree in removedTree for entry in tree if np.isnan(entry) == False]]
  
    highParents = np.unique([getUpperClusters(LBL, parentTree, clLabels) \
                             for LBL in lowLabels])         # Get the highparents of the remaining clusters in list

    return highParents


def getClosestParents(parentTree):
    """
    This function searches the bottom of a tree and find the closest parent if any pairs of entries
    are NaN. Otherwise, returns the non-Nan values at the bottom.
    Inputs:
    1. parentTree: list of lists
    Outputs:
    1. list of close parents
    """

    if len(parentTree) == 1:
        #print('Cluster tree has only 1 entry!')
        return parentTree[0][0]

    ROW = int(-1)
    closeParents= [] 

    for ROW_INDEX, child_pairs in enumerate(np.array_split(parentTree[ROW], len(parentTree[ROW])/2)):
        if np.isnan(child_pairs[0]):
            TEST_ROW, TEST_INDEX = ROW, ROW_INDEX
            while np.isnan(child_pairs[0]):

                TEST_ROW = int(TEST_ROW - 1)
                CH_INDEX = [1 if TEST_INDEX % 2 > 0 else 0][0]
                TEST_INDEX = TEST_INDEX//2 

                if len(parentTree[TEST_ROW]) == 1:   # Return root label if top of tree
                    child_pairs = [parentTree[TEST_ROW][0], None]
                    CH_INDEX = 0
                else:
                    child_pairs = np.array_split(parentTree[TEST_ROW], len(parentTree[TEST_ROW])/2)[TEST_INDEX]
            closeParents.append(child_pairs[CH_INDEX])
        else:
            [closeParents.append(C) for C in child_pairs]

    return closeParents


def getLabelLocation(LABEL, parentTree, clustList=False):
    """ Get the location (tuple) of a label within the 'parentTree' list"""
    
    if clustList:
        LOC = [[Index, np.where(np.array(R)==LABEL)[0][0]] for Index, R in \
                        enumerate(clustList) if len(np.where(np.array(R)==LABEL)[0])>0]

    else:
        LOC = [[Index, np.where(R==LABEL)[0][0]] for Index, R in \
                        enumerate(parentTree) if len(np.where(R==LABEL)[0])>0][0]
    return LOC


#######################################################################################################
#    7 -  Gaussian mixture models
#######################################################################################################

def getPeakLoc(actM, dim):
    """
    Get x-y co-ordinates of peak activity for each flattened activity array
    Input:
    1. activity matrix (MxN. M rows = flattened pixels, N columns = activity episodes)
    2. dim = tuple length 3 (X, Y, N_Frames) 
    """
    # Get location of activity episode peaks
    maxPoints = np.where(actM == np.max(actM,axis=0),1,0)
    Yind, Xind, _ = np.where(maxPoints.reshape(dim)==1)

    # Reshape data for use by mixture model module
    peakLocs = np.asarray([Xind,Yind]).T

    return peakLocs


def getGmm(data, Nmix, Rstate=1):
    """
    Get Gaussian mixture models for 2D data for a range of component numbers.
    Calculate information criteria for each model.
    Inputs:
    1. Locations of data points
    2. Number of mixtures
    3. Random state (? use here)
    """
    # Get mixture models
    nComponents = np.arange(1, Nmix)
    Gmms = [mixture.GaussianMixture(N, covariance_type='full', random_state=Rstate).fit(data)
          for N in nComponents]
    Labels = [model.predict(data) for model in Gmms]
    Probs = [model.predict_proba(data) for model in Gmms]

    # Get scoring: bayesian & aikake info critera
    Bic = [model.bic(data) for model in Gmms]
    Aic = [model.aic(data) for model in Gmms]

    return {'Gmms':Gmms, 'Labels':Labels, 'Probs':Probs,\
            'Bic': Bic,'Aic': Aic}


#######################################################################################################
#    8 - Random functions
#######################################################################################################

def gpuClear():
    try: cp._default_memory_pool.free_all_blocks()
    except: pass


"""

def normEpoch(Activity):
    Normalise 2D activity between Min and Max value. 
    Min=minimum nonzero value.Input: 1. YxX array with 0 entries except for activity locations.
    Return:1. Normalised flattened array.

    MIN = np.min(Activity[np.nonzero(Activity)])
    MAX = np.max(Activity)
    return (np.where(Activity>0,Activity-MIN,0)/(MAX-MIN)).flatten() 




# Old script for moving heatmap 

def rollHmap(actm, Rolls, Dim=(32,128)):
    
    randomly roll non-zero activity within frame X number of times
    

    permAct = np.empty((Rolls, actm.shape[0],actm.shape[1],actm.shape[2]),dtype=np.float16)
    
    # Get random tensors 
    lRand = int(Rolls*Dim[0]*Dim[1]) # unravelled length of random tensors
    low = np.asarray([0 for X in range(lRand)]).reshape(Rolls, Dim[0], Dim[1])
    highY = np.asarray([Dim[0] for X in range(lRand)]).reshape(Rolls, Dim[0], Dim[1])
    highX = np.asarray([Dim[1] for X in range(lRand)]).reshape(Rolls, Dim[0], Dim[1])

    yShifts = np.random.randint(low, high=highY)
    xShifts = np.random.randint(low, high=highX)
    
    for Roll, Ys, Xs in zip(range(Rolls), yShifts, xShifts):
        print(Ys.shape, Xs.shape)
        permAct[Roll,:,:] = np.roll(actm, (Ys,Xs), axis=(1,2))
    
    return permAct

    
    for Run, (nY, nX) in enumerate(zip(np.array_split(newY), np.array_split(newX))):  

        newHm = np.empty((A0, A1, A2))      # new Hm  

        newHm[zLoc.astype(int), nY.astype(int), nX.astype(int)] = hmInputs  # Populate w shuffled values

        allDistances[Run,:] = cpPd(newHm.reshape((A0,int(A1*A2))), \
                                  metric='correlation')                     # get distances
        if shufflePlot:
            sPlot[Run,:,:] = newHm[shufflePlot, :, :] 
            
    allDistances = cp.asnumpy(allDistances)

def getHmEdge():
    
    Take a heatmap, convert into single dots as co-ordinates, then select only the outermost dots as co-ordinate.
    NOT NEEDED ANYMORE!!!!
    
    yEdge = np.asarray([[[Y, np.min(points[np.where(points[:,0] == Y)[0]][:,1])], [Y, np.max(points[np.where(points[:,0] == Y)[0]][:,1])]] \
                for Y in np.unique(points[:,0])])
    yEdge = yEdge.reshape(yEdge.shape[0]*yEdge.shape[1], yEdge.shape[2])

    xEdge = np.asarray([[[np.min(points[np.where(points[:,1] == X)[0]][:,0]), X], [np.max(points[np.where(points[:,1] == X)[0]][:,0]), X]] \
                for X in np.unique(points[:,1])])
    xEdge = xEdge.reshape(xEdge.shape[0]*xEdge.shape[1], xEdge.shape[2])

    xEdge = np.delete(xEdge, np.where((xEdge[10] == yEdge).all(axis=1))[0], axis=0)

    return np.concatenate([xEdge, yEdge])


"""