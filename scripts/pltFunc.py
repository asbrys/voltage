import numpy as np
import matplotlib.pyplot as plt, seaborn as sns, cv2

                   ### PLOTTING FUNCTION STUFF ####


def Fig(rows, cols, Xsize, Ysize):
    """Return axis and figure with specified dimensions etc """
    return plt.subplots(rows, cols, figsize=(Xsize,Ysize))


def Hist(ax, Data, Range=None, Bins=None):
    if (Range is not None) or (Bins is not None):
        if Range is not None: ax.hist(Data.flatten(), range=Range, bins=Bins)
        else: ax.hist(Data.flatten(), bins=Bins)
    else: ax.hist(Data.flatten())


def medCutoff(medStack, Bgr, cRange, save = False):
    """
    Use this to determine a cutoff value for median intensity pixel 
    values above which to perform analysis
    """

    fig, ax = Fig(1, 5, 13, 3)

    ax[0].imshow(cv2.imread('background/' + Bgr, cv2.IMREAD_UNCHANGED))

    for AX, CUT in enumerate(cRange):

        ax[AX+1].imshow(np.where(medStack > CUT, medStack, 0))

        ax[AX+1].set_title(CUT,fontsize=10, weight='demi')

    fig.tight_layout()

    if save: 
        fig.savefig(save['fname'],dpi=save['dpi'])


def plotAveTrace(Stack, medStack, cut = None, save = False, LW = 0.05, size = (6, 1.2)):
    """
    Plot spatially averaged trace over time.
    Plot across all pixels, and across median-filtered pixels
    """
    if cut:
        fig, ax = Fig(3, 1, size[0], size[1])
        XpN,YpN = np.where(medStack < cut)
        Xp,Yp = np.where(medStack > cut)
        ax[1].plot(np.mean(Stack[:, Xp, Yp], axis = 1), color = 'k', lw = LW)
        ax[2].plot(np.mean(Stack[:,XpN,YpN],axis=1),color='k',lw=LW)
        ax[0].plot(np.mean(Stack.reshape(Stack.shape[0],Stack.shape[1]*Stack.shape[2]),axis=1),lw=LW,color='k')
        [A.tick_params(axis='both',which='major',labelsize=8) for A in ax]
        [A.set_xticks([]) for A in ax[:2]]
        ax[0].set_title('all pixels',fontsize=7),ax[1].set_title('cell',fontsize=7),ax[2].set_title('non-cell',fontsize=7)

    else:
        fig , ax = Fig(1, 1, size[0], size[1])
        ax.plot(np.mean(Stack.reshape(Stack.shape[0], Stack.shape[1] * Stack.shape[2]), axis = 1), lw = LW, color = 'k')
        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)

    fig.tight_layout()

    if save: 
        fig.savefig(save['fname'], dpi = save['dpi'])


def plotPixResults(d, Info, save = None, size = (8, 3.5)):
    """
    Plot raw, normalised, mean and std time series 
    for the 10 brightest pixels (or 10 mid-brightest pixels).

    Input:
    1. data with: (TxYxX) gauss filtered stack, rolling mean, and rolling std stack
    2. Info with mean image, std
    """
    fig, ax = Fig(4, 5, size[0], size[1])

    # Define arrays and parameters to use
    normS = (d['filtTiff'] - d['rMean'])/d['rStd']
    xPlot = np.arange(len(d['filtTiff'] [:, 0]))  # length of time axis

    # Get x,y location of 10 brightest pixels (from median image)
    (x, y) = np.unravel_index(np.argsort(Info['meanImage'].flatten())[-10:], Info['meanImage'].shape)
    
    # Plot raw time series for these 10 pixels + mean/std in rows 1 and 3
    for i, AX in enumerate(np.concatenate([ax[0], ax[2]])):
        AX.plot(d['filtTiff'][:, x[i], y[i]], \
                lw = 0.15, color = 'C0', alpha = 0.7)
        AX.plot(d['rMean'][:, x[i], y[i]], \
                lw = 0.5, color = 'k')
        AX.fill_between(xPlot, d['rMean'][:,x[i],y[i]] + Info['std'] * d['rStd'][:, x[i], y[i]],\
                        d['rMean'][:, x[i], y[i]] - Info['std'] * d['rStd'][:, x[i], y[i]],\
                            color = 'green', alpha = 0.3)
    
    # Plot normalised time series in rows 2 and 4
    for i, AX in enumerate(np.concatenate([ax[1], ax[3]])):
        AX.plot(normS[:,x[i], y[i]], \
                lw = 0.15, color = 'k')
        AX.plot(xPlot, [Info['std']] * len(xPlot),\
                 '--', color = 'red')
        AX.plot(xPlot, [-Info['std']]*len(xPlot),\
                 '--', color='red')
        AX.set_ylim([-3.5,3.5])
    
    # Set plotting features
    [A.set_xticks([]) for A in ax.flatten()]
    [A.spines[E].set_visible(False) for A in ax.flatten() for E in ['top','right']]
    [A.tick_params(axis='both',which='major',labelsize=8) for A in ax.flatten()]

    [AX.set_ylabel("Raw trace \nmean/std", fontstyle = "italic", fontweight = 'demi', \
                   fontsize = 8) for AX in [ax[0][0], ax[2][0]]]
    [AX.set_ylabel("Norm trace \n(z-score)", fontstyle = "italic", fontweight = 'demi', \
                   fontsize = 8) for AX in [ax[1][0], ax[3][0]]]

    fig.tight_layout()

    # Save
    if save: 
        fig.savefig(save,dpi=300)


def heatRow(Arrays,Save=None):
    X = len(Arrays)
    fig,ax=Fig(1,X,X*2.5,3)
    [ax[i].imshow(Arr) for i, Arr in enumerate(Arrays)]
    [A.tick_params(axis='both',which='major',labelsize=8) for A in ax.flatten()]
    fig.tight_layout()
    if Save:fig.savefig(Save, dpi=300)


def plotActDist(d, Info, type='strip', Save=None):
    """
    Plot distribution of activity episodes within all and non-cell regions
    """

    if len(d['labDataNonCell']['aucs'])<1: d['labDataNonCell']['aucs'].append(0)

    # Cutoff based on non-cell regions
    CUT = np.percentile(d['labDataNonCell']['aucs'], Info['perc'])

    fig,ax = Fig(1,1,8,1.5)

    if type=='strip':
        sns.stripplot(data=[d['labDataAll']['aucs'], d['labDataNonCell']['aucs']], \
                      orient='h', ax=ax)
        ax.plot([CUT-1e-9, CUT+1e-9], [-0.25,1.25],\
                '--', color='red', zorder=3)
        ax.set_title(str(len(d['labDataAll']['aucs']))+' total activity episodes', fontsize=9)

    elif type=='violin':
        sns.violinplot(data=[d['labDataAll']['aucs'], d['labDataNonCell']['aucs']], \
                       orient='h', ax=ax)

    # Set plot features
    ax.set_yticklabels([L for L in ['all','non-cell']], weight='demi')
    ax.tick_params(axis='both',which='major',labelsize=8) 
    fig.tight_layout()

    if Save: 
        fig.savefig(Save, dpi=300)


def plotHighAct(d, Save=None, N=10):
    """
    Plot frame locations of all activity epochs, and X most intense activity episodes, 
    plot, and save frame locations.
    """

    # Get labels for highest X activity periods, sorted from lowest to highest
    Labels = np.argsort(d['labDataAll']['aucs'])[-N:]+1
    
    # Store [start frame, end frame, label] of high activity frames here
    hAct = [] 
    
    # First plot location of all activity episodes
    fig,ax=Fig(1,1,10,0.5)
    ax.scatter(np.where(d['threshTiff']>0)[0], [1]*len(np.where(d['threshTiff']>0)[0]),\
               s=1, color='k')

    # Now plot high-activity episodes in red & save frame locations
    print('start_end frames for high activity...')
    for L in Labels:
        Loc = np.where(d['labDataAll']['labArray']==L)[0]
        ax.scatter(Loc, len(Loc)*[1], \
                   s=1, color='red')
        print(Loc[0],Loc[-1],end=' | ') # Print frame locations
        hAct.append([Loc[0], Loc[-1], L])

    # Plotting features
    ax.tick_params(axis='both',which='major',labelsize=8) 
    fig.tight_layout()
    
    if Save:
        fig.savefig(Save, dpi=300)
        
    return hAct


def logPlotAct(Arr1,Arr2,UL,Bins,ALPH=0.3,Save=None):
    fig,ax=Fig(1,1,3,1.75)
    ax.hist(Arr1,bins=np.logspace(0,UL,Bins),density=True,alpha=0.3)
    ax.hist(Arr2,bins=np.logspace(0,UL,Bins),density=True,alpha=0.3)
    ax.tick_params(axis='both',which='major',labelsize=8) 
    ax.set_xscale("log")
    fig.tight_layout()
    if Save:fig.savefig(Save,dpi=300)

    








