""""
Functions to streamline analysis.
Need to improve this to better manage memory issues....
"""
from  scripts import dfFunc as dF
import numpy as np


def filterTiff(Stack, Info):
    """
    Filter & get rolling mean/std of raw movie.
    Input: 1. Raw movie, 2. Info with filtering paramaters
    Output: 1. Filtered movie, 2. Rolling mean, 3. Rolling std
    """

    if Info['filt']: 
        Stack = dF.gFilt(Stack, Info['filt'], Gpu=Info['gpu'])

    # Reshape array
    T, Y, X = Stack.shape
    Stack = Stack.reshape(T, Y * X)

    # Get rolling mean/std
    print("Getting rolling mean/std of movie...", end=' ')
    rMean = dF.rollMean(Stack, Info['fps'] * Info['sec']).reshape(T, Y, X) 
    rStd = dF.rollStd(Stack, Info['fps'] * Info['sec']).reshape(T, Y, X)

    return {'filtTiff': Stack.reshape(T, Y, X), 
            'rMean': rMean, 'rStd': rStd}


def getActivity(fData, Info, Inv = True):
    """
    *** Function to extract significant "blobs" of activity from recording ***
    Does this following:
    1. Normalises the recording (pixel-wise) based on rolling mean/std
    2. Get spatio-temporal activity episodes > threshold
    3. Also calculate summed intensity of each episode
    (time consuming... need to optimise or remove in future)
    4. Do a background correction: remove blobs of activity < a threshold set by non-cell areas
    
    Return:
    1. Normalised + thresholded recording (ie, 0 if < Z-score threshold)
    2. Labelled data (dict): {'labArray': recording with label number of each significant blob
                              'nabels': number of significant blobs
                              'aucs': summed intensity of each significant blob}
    3. Labelled data (as above) but just for non-cell regions (?? remove this)
    4. Pixel indices corresponding to cell locations (?? remove this)

    """

    normTiff = (fData['filtTiff'] - fData['rMean']) / fData['rStd']             # Normalise recording
    del fData

    # Invert depending on indicator & threshold for std (DO THIS AFTER NORMALISATION!)
    if Inv: 
        normTiff = -normTiff

    normTiff = np.where(normTiff > Info['std'], normTiff, 0)                    # Threshold recording

    contDataAll = dF.getContigData(normTiff, Info)                              # Get blob of activity > threshold    

    # Get blobs of activity in non-cell regions (for background correction)
    normTiffNonCell = dF.cellNon(normTiff, np.where(Info['meanImage'] > Info['cutoff']))
    contDataNonCell = dF.getContigData(normTiffNonCell, Info)
    del normTiffNonCell, contDataNonCell['labArray']

    try:                                                                        # Set percentile threshold for bg correction    
        UPPER = np.percentile(contDataNonCell['aucs'], Info['perc'])
    except:                                                             
        UPPER = -1e9

    normTiff = dF.remBaseActivity(normTiff, contDataAll['labArray'],\
                                  contDataAll['aucs'], UPPER)                   # Background correction
    
    print('{} and {} continuous episodes in total and non-cell regions\
          '.format(contDataAll['nLabels'], contDataNonCell['nLabels']))
    
    return {'threshTiff': normTiff, 
            'labDataAll': contDataAll,
            'labDataNonCell': contDataNonCell,
            'cellInds': np.where(Info['meanImage'] > Info['cutoff'] )}

