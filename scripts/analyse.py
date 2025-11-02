""""
Functions to streamline analysis.
Need to improve this to better manage memory issues.

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
    Normalised activity array & extract episodes above threshold.
    Perform background correction by removing activity epochs with summed value that 
    fall below a threshold based on the non-cell region. 
    """

    # Get normalised activity array
    normTiff = (fData['filtTiff'] - fData['rMean']) / fData['rStd']
    del fData

    # Invert depending on indicator & threshold for std (DO THIS AFTER NORMALISATION!)
    if Inv: 
        normTiff = -normTiff
    normTiff = np.where(normTiff > Info['std'], normTiff, 0)

    # Get continuous activity epochs for all areas
    contDataAll = dF.getContigData(normTiff, Info)

    # Get continuous activity epochs for non-cell areas
    normTiffNonCell = dF.cellNon(normTiff, np.where(Info['meanImage'] > Info['cutoff']))
    contDataNonCell = dF.getContigData(normTiffNonCell, Info)
    del normTiffNonCell, contDataNonCell['labArray']

    # Set threshold for Auc activity epoch cutoff
    try:    
        UPPER = np.percentile(contDataNonCell['aucs'], Info['perc'])
    except: # Get all activity if something funny going on
        UPPER = -1e9

    # Remove continuous activity episodes with AUC < cutoff
    normTiff = dF.remBaseActivity(normTiff, contDataAll['labArray'],\
                                  contDataAll['aucs'], UPPER)
    
    print('{} and {} continuous episodes in total and non-cell regions\
          '.format(contDataAll['nLabels'], contDataNonCell['nLabels']))
    
    return {'threshTiff': normTiff, 
            'labDataAll': contDataAll,
            'labDataNonCell': contDataNonCell,
            'cellInds': np.where(Info['meanImage'] > Info['cutoff'] )}

