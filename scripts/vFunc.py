"""
Functions to manipulate videos and images
"""
import numpy as np, os, cv2, glob
from  scripts import dfFunc as dF,pltFunc as pF, synFunc as sF
import matplotlib.cm as cm,matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scripts import synFunc as sF

## This is a colormap that gives diverging blue-red for combined depol and hyperpol voltages
cdict = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.6, 1, 1),
                   (0.75, 1, 1),
                   (1.0, 1, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.5, 0, 0),
                   (0.6, 0.6, 0.6),
                   (0.75, 0.4, 0.4),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 1, 1),
                   (0.25, 0.75, 0.75),
                   (0.5, 0.5, 0.5),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
          'alpha':  ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))}
br_cmap = mcolors.LinearSegmentedColormap('br', cdict)

#######################################################################################################
### 1 - IMAGE MANIPULATION FUNCTIONS
#######################################################################################################
def bgraConvert(Array, cMap, ALPHA, alphaBase = None):
    """
    Convert 2D array normalised between 0 and 1 into an RGBA 8-bit colormap.
    Then float 32 for png conversion.
    Use ALPHA value to blend with b/g image.
    Set full transparency for specified alphaBase cutoff value.
    Convert to BGRA for use with cv2 imwrite function into PNG files
    """

    # Do not introduce an alpha value if this is already included in the colormap 
    if cMap.cmap(0.5)[3]<1.0:   # alpha < 1.0 is transparent
        colArray = (cMap.to_rgba(Array)*255).astype(np.float32)
    else:                       # alpha of 1.0 is opaque
        colArray = (cMap.to_rgba(Array,alpha=ALPHA)*255).astype(np.float32)
    
    # make fully transparent if below given cutoff value
    if alphaBase is not None:
        yvals,xvals = np.where(np.abs(Array)<=alphaBase)
        for Y,X in zip(yvals,xvals):colArray[Y,X,3] = 0

    colArray = cv2.cvtColor(colArray, cv2.COLOR_RGBA2BGRA)

    return colArray


def combineImages(background, foreground):
    """
    Combines a background with an image containing an alpha channel
    """
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0
    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)
    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background


#######################################################################################################
### 2 - VIDEO CREATION FUNCTIONS ###
#######################################################################################################


def getBg(Info):
    """
    Get background image for videos from Info file
    Convert to right dtype, upsample and filter (set params for now...)
    """
    Bgr = cv2.imread('background/'+Info['bgr'], cv2.IMREAD_UNCHANGED) 
    if Bgr.dtype=='uint16': 
        Bgr = (Bgr/256).astype('uint8') 
    Bgr = dF.upSample(Bgr,2,0) 
    Bgr = dF.gFilt(Bgr,(1,1)) 
    return cv2.cvtColor(Bgr,cv2.COLOR_GRAY2BGRA) 


def pngVidConvert(Stack, Info, cMap, fps=None, Range=None, fName='test.mp4'):
    """
    Convert [T, X, Y] array with pixel intensities into RGBA colormap.
    Write png files for each frame with alpha value. 
    Convert into mp4 movie with specified framerate.
    Add background image and save.
    Inputs: 
    3D [Time, X, Y] array, colormap, alpha, frame rate, case cutoff alpha, Frame range to make video
    """

    os.system("mkdir -p tempPng")   # create dir for png files

    # Create full movie if no range is specified
    if Range is None: 
        print('making full movie...', end=' ')
        Range=(0, Stack.shape[0]) 

    for IND, i in enumerate(np.linspace(Range[0],Range[1],(Range[1]-Range[0])+1)[:-1].astype(int)): 

        Bgr = getBg(Info)

        # Select activity frame, upsample, filter
        Frame = Stack[i,:,:]                    
        Frame = dF.upSample(Frame,2,0)         
        Frame = dF.gFilt(Frame,Info['vFilt'])   

        # Convert frame to BGRA and combine with background image
        Frame = bgraConvert(Frame,cMap,Info['vAlph'],alphaBase=Info['vAbase'])  
        Frame = combineImages(Bgr, Frame)                              

        # Save image into temp directory for movie creation
        cv2.imwrite('tempPng/test'+str(IND).zfill(len(str(Range[1]-Range[0])))+'.png', Frame) 

    # Create  movie from saved png files
    os.system("ffmpeg -y -r "+str(fps)+" -i tempPng/test%0$(ls -l tempPng/*.png|wc -l|tr -d '\n'|wc -m)d.png\
              -c:v libx264 -crf 0 "+str(fName))  # rgba vcodec png copy /-c:v copy -pix_fmt rgba -crf 0 
    
    os.system("rm -r tempPng")

    print('\n\n **** Overlay movie created **** \n\n')


def slowVids(fD, cD, Info):
    """
    Function to create slowed videos for high activity depolarising episodes. 
    """

    # Get movie parameters
    [FPS, SPD, PAD] = [Info[K] for K in ['fps','hSpeed','hPad']]
    print('Movie is slowed down by {} and will be played at {}fps'.format(SPD, FPS*SPD))

    # Get normalised, thresholded movie
    depolThAct = normThresh(fD, Info['std'], Inv=True)

    # create high activity depolarising vids
    for i, epoch in enumerate(Info['highAct']):

        # select entries corresponding to epoch in thresholded array
        epochArr = np.where(cD['labDataAll']['labArray']==epoch[2], depolThAct, 0)

        # Get padding around high activity epochs
        RANGE = (int(max(0,epoch[0]-(FPS*PAD))),int(min(epoch[1]+(FPS*PAD),epochArr.shape[0])))

        # Name for video with all the information
        Name = 'activityMaps/Depol_'+str(i)+'_'+Info['name']+str(Info['std'])+\
                "_"+''.join([str(F) for F in Info['filt']])+"_"+str(Info['perc'])+\
                    "perc_"+str(RANGE[0]/FPS)+"secOnset_"+str(SPD)+"speed.mp4"
        
        # Get activity colormap
        cNorm = mpl.colors.Normalize(vmin=Info['hRange'][0],vmax=Info['hRange'][1]) # Dynamic range
        cMap = cm.ScalarMappable(norm=cNorm, cmap=Info['CmHm']['depol']) # Colormap
        
        # Convert video
        pngVidConvert(epochArr, Info, cMap, fps=FPS*SPD , Range=RANGE, fName=Name)
    
    # create high activity hyperpolarising vids if analysed
    #if hActH:
    #    for i,ran in enumerate(hActH):

    #        Name = 'activityMaps/highAct/Hyper_highAct_'+str(i)+'_'

    #        RANGE = (int(max(0,ran[0]-(FPS*PAD))),int(min(ran[1]+(FPS*PAD),d.shape[0])))

    #        createVid(d,RANGE,Info,Name)


#######################################################################################################
### 3 - HEATMAP IMAGE FUNCTIONS 
#######################################################################################################


def pngHeatmap(Activity, Info, aBase, Colormap, Save = None):
    """
    Plot heatmap of summated activity overlaid on background image.
    Input:
    1. Activity array to summate over time
    2. Background image
    3. Alpha of overlying activity
    4. Threshold for full transparency (between 0 and 1)
    """
    [Bgr, Alpha, ZM, Plot, actFilter] = [Info[K] for K in \
                              ['bgr', 'AlphHm', 'upSampHm','inlineHm','filtHm']]

    # Get activity frame, background image, and combine
    fr = getActivityImage(Activity, ZM, actFilter,Colormap, Alpha, aBase)
    bg = getBgImage(Bgr, ZM, Save=Save)
    bg_fr = combineImages(bg,fr)

    # Save and plot
    if Save:
        cv2.imwrite(Save, bg_fr)

    if Plot:
        fig,ax = pF.Fig(1,1,4,4)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGRA2RGBA)# Convert back to RGBA for inline plotting
        ax.imshow(combineImages(bg,fr))
        fig.tight_layout()


def getActivityImage(Activity, ZM, actFilter, Colormap, Alpha, aBase, norm = 1):
    """
    Temporally sum and normalise activity from (TxMxN) array, and then turn this 
    into a bgra colormap. 
    """

    # Sum activity across time, normalise (0-1), upsample & filter
    sumAct = np.sum(Activity,axis = 0)
    sumAct = dF.normArray(sumAct, Type='minmax')
    
    if norm != 1:
        sumAct = norm * sumAct
    
    sumAct = dF.upSample(sumAct, ZM, 0)
    sumAct = dF.gFilt(sumAct, actFilter)

    # Create colormap from 0-1, convert to RGBA colormap & combine with b/g
    cNorm = mpl.colors.Normalize(vmin = 0, vmax = 1) 
    cMap = cm.ScalarMappable(norm = cNorm, cmap = Colormap)

    fr = bgraConvert(sumAct, cMap, Alpha, alphaBase = aBase)

    return fr


def getBgImage(Bgr, ZM, Save=None, bgFilter=(2,2)):
    """
    Loads the background png image, upsample, filter and convert to bgra
    """

    # Load b/g image, upsample, filter, convert to BGRA format 
    bg = cv2.imread('background/'+Bgr,cv2.IMREAD_UNCHANGED)
    if bg.dtype=='uint16': 
        bg = (bg/256).astype('uint8')
    bg = dF.upSample(bg,ZM,0)
    bg = dF.gFilt(bg,bgFilter)
    bg = cv2.cvtColor(bg,cv2.COLOR_GRAY2BGRA)

    if Save: 
        cv2.imwrite('figures/background.png',bg)
    
    return bg


def dehypHmap(fD, cD, Info): 
    """
    Main plotting function for depolarising (+/- hyperpolarising) overall heatmaps.
    """
    
    SAVE = 'figures/'+Info['name']+'alphTh_'

    # Plot all activity above threshold
    if Info['plotHm']['All']:

        # Plot all thresholded depolarising data
        depolThAct = normThresh(fD, Info['stdHm'], Inv=True)

        for TH in np.round(Info['ThHm'],2):
            pngHeatmap(depolThAct, Info, TH, Info['CmHm']['depol'],\
                        Save=SAVE+str(TH)+'_allDepol_stdTh_'+str(Info['stdHm'])+'.png') 
        del depolThAct

        # Plot all thresholded hyperpolarising data
        """
        hypThAct = normThresh(fD, Info['stdHm'], Inv=False)

        for TH in np.round(Info['ThHm'],2):
            pngHeatmap(hypThAct, Info, TH, Info['CmHm']['hyp'],\
                        Save=SAVE+str(TH)+'_allHyperpol_stdTh_'+str(Info['stdHm'])+'.png') 
            
        del hypThAct
        """

    # Now plot activity associatd with continuous episodes 
    if Info['plotHm']['Epochs']:

        for TH in np.round(Info['ThHm'],2):
            pngHeatmap(cD['threshTiff'], Info,TH, Info['CmHm']['depol'],\
                    Save=SAVE+str(TH)+'_thEpochStd_'+str(Info['std'])+'_Depol.png') 


def normThresh(d, Std, Inv=True):
    """
    Get all normalised activity, +/- inverted, thresholded for given STD value 
    """
    
    # Normalise
    normTh = (d['filtTiff'] - d['rMean'])/d['rStd']
    
    # Invert depending on indicator
    if Inv:
        normTh = -normTh 
    
    # Threshold
    normTh = np.where(normTh>Std, normTh, 0) # threshold
    
    return normTh


def plotMultiHm(heatMaps, Inline = False, Dim = (32,128), Abase = 0.2, Save = None, Norm = True):
    """
    Plot multiple individual heatmaps onto the same background.
    Inputs:
    1. List or array of flattened heatmaps
    2. Dim: spatial dimensions of imaging data
    """

    heatMaps = [Hm.reshape(Dim) for Hm in heatMaps]
        
    # Plotting info
    Info = {'bgr': os.path.basename(glob.glob('background/*.png')[0]),\
                'AlphHm': 0.35, 'upSampHm': 5, 'filtHm': (2,2)}

    # Get the maximum total activity at a single pixel for each cluster. Use this to normalise.
    if Norm:
        maxClustValues = np.asarray([np.max(Hm) for Hm in heatMaps])
        adjustedClustNorm = maxClustValues/np.max(maxClustValues)
    else:
        adjustedClustNorm = np.ones(len(heatMaps))
        
    # Turn pixel-wise mean for each cluster into rgba heatmap
    meanFrames = [getActivityImage(Hm[None, :, :], Info['upSampHm'], Info['filtHm'], \
                                    cm.jet, Info['AlphHm'], Abase, norm = Norm) for Hm, Norm \
                                    in zip(heatMaps, adjustedClustNorm)]
    
    # Combine heatmaps onto the same image
    # Create 2 combined arrays: 1 for overlap & 1 for non-overlap regions
    combinedImage1 = sF.addOverlaps(meanFrames)
    nonOverlapFrames, _, _ = sF.overlapFunc(meanFrames, overlap = False)
    combinedImage2 = sF.multiframeAdd(nonOverlapFrames)

    # Combine overlap and non-overlap regions
    combinedImage = cv2.add(combinedImage1, combinedImage2)

    # Add background
    bg = getBgImage(Info['bgr'], Info['upSampHm'], Save = None, bgFilter = (2, 2))
    bg_fr = combineImages(bg, combinedImage)

    if Save is not None:
        cv2.imwrite('hMap_'+Save+'combinedHm.png', bg_fr)

    if Inline:
        fig,ax = pF.Fig(1, 1, 4, 4)
        fr = cv2.cvtColor(bg_fr, cv2.COLOR_BGRA2RGBA)# Convert back to RGBA for inline plotting
        ax.imshow(bg_fr)
        fig.tight_layout()





#######################################################################################################
### 4 - REDUNDANT
#######################################################################################################

### Now redundant
def overlayMovie(Temp, Bg, fName):
    """Code is just kept for future reference"""
    os.system("ffmpeg -y -i "+str(Temp)+" -i background/"+str(Bg)+" \
            -filter_complex \"[0:v]format=rgba,geq=r='r(X,Y)':a='1*alpha(X,Y)'[zork]; \
            [1:v][zork]overlay\" -c:v prores_ks -crf 0 "+fName) #libx264


def createVid(movArray, Range, Info, Name): #,ALPH=0.7,ABASE=None,Color='jet',UPPER=1, minmax=None):
    """"
    Collect information needed to create a video, then execute video convert.
    """
  
    # Specify colormap for video
    cDict = {'jet': cm.jet,'blue_red':br_cmap}
    cNorm = mpl.colors.Normalize(vmin=Info['hRange'][0],vmax=Info['hRange'][1]) # Dynamic range
    cMap=cm.ScalarMappable(norm=cNorm, cmap=cDict[Color]) # Colormap

    # Save name for file
    Save = Name+str(Info['std'])+"_"+str(Info['sec'])+"sec_preF"+\
        ''.join([str(F) for F in Info['filt']])+"_"+str(Info['perc'])+"perc.mp4"

    # Convert video
    pngVidConvert(movArray, Range, cMap, fName=Save) #, cMap, alpha=ALPH, fps=FPS*SPEED, A_BASE=ABASE, Range=Range, fName=Save)



## OLD CODE ###


"""

                    ### IMAGE AND VIDEO FUNCTIONS ###

def bgraConvert(Array, cMap, ALPHA, alphaBase = None):
    Convert 2D array normalised between 0 and 1 into an RGBA 8-bit colormap.
    Convert into float 32 for png conversion.
    Set full transparency for specified cutoff value.
    Convert to BGRA for use with cv2 imwrite function into PNG files
    
    colArray = (cMap.to_rgba(Array,alpha=ALPHA)*255).astype(np.float32)
    if alphaBase is not None:
        yvals,xvals = np.where(Array<=alphaBase)
        for Y,X in zip(yvals,xvals):colArray[Y,X,3] = 0
    colArray = cv2.cvtColor(colArray, cv2.COLOR_RGBA2BGRA)
    return colArray

def combineImages(background, foreground):
    Combines a background with an image containing an alpha channel
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0
    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)
    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background

def pngVidConvert(Stack,cMap,alpha=None,fps=None,A_BASE=None,RANGE=None, fName=None, Bg=None):
    Convert [T, X, Y] array with pixel intensities into RGBA colormap.
    Write png files for each frame with alpha value. Convert into mp4 movie with specified framerate.
    Add background image and save
    Inputs: 3D [Time, X, Y] array, colormap, alpha, frame rate, case cutoff alpha, Frame range to make video
    

    os.system("mkdir -p testPng")   # create dir for png files

    if RANGE is None: RANGE=Stack.shape[0]  # N frames for movie
    if fName is None: fName="overlayVid.mp4" # output filename

    for i in range(RANGE): 
        Bgr = cv2.imread('background/'+Bg, cv2.IMREAD_UNCHANGED) # load background
        if Bgr.dtype=='uint16': Bgr = (Bgr/256).astype('uint8') # convert to right dtype
        Bgr = cv2.cvtColor(Bgr,cv2.COLOR_GRAY2BGRA) # convert bg to rgba format
        Frame = Stack[i,:,:] # select frame
        Frame = bgraConvert(Frame,cMap,alpha,alphaBase=A_BASE) # convert to rgba -> bgra
        Frame = combineImages(Bgr, Frame) # combine images
        cv2.imwrite('testPng/test'+str(i).zfill(len(str(RANGE)))+'.png', Frame) # write png files

    # Create  movie from activity
    os.system("ffmpeg -y -r "+str(fps)+" -i testPng/test%0$(ls -l testPng/*.png|wc -l|tr -d '\n'|wc -m)d.png\
              -c:v libx264 -crf 0 "+str(fName))  # rgba vcodec png copy /-c:v copy -pix_fmt rgba -crf 0 
    
    os.system("rm -r testPng")
    print('\n\n **** Overlay movie created **** \n\n')


### Now redundant
def overlayMovie(Temp, Bg, fName):
    Code is just kept for future reference
    os.system("ffmpeg -y -i "+str(Temp)+" -i background/"+str(Bg)+" \
            -filter_complex \"[0:v]format=rgba,geq=r='r(X,Y)':a='1*alpha(X,Y)'[zork]; \
            [1:v][zork]overlay\" -c:v prores_ks -crf 0 "+fName) #libx264



"""