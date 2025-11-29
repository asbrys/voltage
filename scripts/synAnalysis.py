"""
Class for synaptic clustering analysis
"""
from scripts import synFunc as sF
import  numpy as np, pickle as pkl, scipy
import scipy.cluster.hierarchy as sch, scipy.spatial.distance as ssd


class synClust:

    """Class to perform hierarchical clustering on synaptic input activity"""

    def __init__(self, dataPath):
        """
        Initiate synaptic analysis class with preprocessed data:
        1. Load data. The input 
        2. Get normalised heatmaps of activity, summed over temporal frames: 
            2D array: M rows x N heatmaps, where M is unfolded 2D pixels (n pixel row x column of recording) 
        3. Get spatial dimensions of recording
        """
        self.dataPath = dataPath                                                        

        Data = pkl.load(open(dataPath, 'rb'))  
                                                
        self.actM, self.labAct = sF.getNormEpochs(Data['threshTiff'], BINARY = False)   # Normalised heatmaps of significant activity
        self.spatialDims = Data['threshTiff'].shape[1:]                                 # Spatial dimensions of recording
        self.cellInds = Data['cellInds']                                                # Pixel values of cell (use for bg correction)
        del Data
    

    def getDistance(self, Runs, batchSize, percentile = 10, shufflePlot = False, \
                    sInfo = {'fps': 5,'cRange': 0.15,'vAbase': 0.01}, Metric = 'correlation'):
        """
        1. Get distance between all normalised activity episodes. 
        2. Get threshold for significance similarities between heatmaps. 
        Can also create a demo video of the heatmap being shuffled around as a sanity check....
        """
        
        self.dist = sF.getDistance(self.actM, MET = Metric)

        self.thDist, self.distPerm, self.sPlot = sF.getThreshDistance(self.actM, \
            Runs, batchSize, self.dist, self.spatialDims, percentile = percentile, \
                shufflePlot = shufflePlot, sInfo = sInfo)


    def getClusters(self, METRIC = 'correlation'):
        """
        Get linkage data. 
        Get matrix of cluster hierarchies (rows = cluster levels, col = heatmap. Entry = cluster grouping)
        Get similarity matrix
        """
        self.lD = sch.linkage(self.thDist, method = 'ward',\
                              metric = METRIC, optimal_ordering = False)

        # note: clusters defined in cM are offset by 1 compared to lD. Ie cM cluster 2 = lD cluster 1, etc 
        self.cM = sF.clustMat(self.lD).astype(np.int32)

        self.sM = ssd.squareform(self.thDist)

    
    def bgCorrection(self, Correct = False, Percentile = 99, SPLITS = 3):
        """Remove weakly correlated heatmaps from activity matrix and recalculate distances"""

        # Get cutoff for background correction
        self.bgCutoff = sF.bgHmCorrection(self.spatialDims, self.cellInds,\
                                          self.actM, self.sM, Plot = True, Percentile = Percentile)

        if Correct:
            self.actM, self.thDist, self.labAct = sF.trimHeatmaps(self.bgCutoff, self.actM, \
                                                self.sM, self.labAct, METRIC = 'correlation',\
                                                SPLITS = SPLITS)


    def plotSortSim(self, rows, addEns = True, plList = None, bounds = None, nEns = -2, \
                    sHigh = False, base = False, save = True, Par = False):
        """
        1. Sort similarity matrices by within-cluster distance 
        2. Plot sorted matrices with clusters highlighted
        3. Get heatmaps corresponding to each cluster within 'row data'
        4. % of total activity accounted for by each cluster hierarchy level
        """

        self.sortSm = [sF.sortSim(self.sM, self.cM[ROW], Sorthigh = sHigh) \
                       for ROW in rows]    
        
        if not plList:
            plList = np.linspace(0, len(self.sortSm) - 1, len(self.sortSm)).astype(int)

        sortedSm = np.asarray(self.sortSm)[plList]

        self.rowData = [sF.plotSim(simM, self.sM, CL, rows, cM = self.cM, bnds = bounds) \
                           for CL, simM in zip(plList, sortedSm)]
        
        self.activityCover = sF.hmActivityCover(self.actM, self.cellInds, \
                                                self.rowData, Dims = self.spatialDims)


    def plotDend(self, save = True):
        """Plot dendrogram of clustering results"""

        self.dPlot, self.dg = sF.plotDend(self.lD)

        if save:
            if type(save) != str:
                save = 'dend'

            self.dPlot[0].savefig('synFigs/' + save + '.png', dpi = 400)
    

    def plotHmaps(self, plList = None, Abase = 0.1, colors = False, addLabels = True):
        """Plot heatmaps of mean activity associated with each ensemble"""
        
        if not plList:              # List of hierarchies to plot
            plList = np.linspace(0,len(self.rowData)-1,len(self.rowData)).astype(int)

        for LEVEL in plList:

            #labels, LBLTXT = None, False
            
            #if addLabels:

                ############################
                # plus or minus the weakest cluster
                ############################
                #labels = [self.rowData[LEVEL]['bestRow'][self.rowData[LEVEL]['sortedBins'][X1]]
                #            for X1, X2 in zip(self.rowData[LEVEL]['ensLocSorted'][:-1], self.rowData[LEVEL]['ensLocSorted'][1:])]
                
                #labels = [self.rowData[LEVEL]['bestRow'][self.rowData[LEVEL]['sortedBins'][X1]]
                #            for X1, X2 in zip(self.rowData[LEVEL]['ensLocSorted'], self.rowData[LEVEL]['ensLocSorted'][1:] + [-1])]
                
                #LBLTXT = True

            sF.plotMeanEnsHm(self.actM, self.rowData[LEVEL], Abase = Abase, colors = colors, \
                             Save = 'level'+str(LEVEL)+'_', Dim = self.spatialDims,\
                             labelText = addLabels, clusterLabels = None)  # labels
    

    def getClustMovies(self, Level, Std = 2.5, Speed = 0.4,\
        vBase = 0.1, vAlph = 0.5, fName = None, Short = None):
        """
        Get movie of concatenated activity associated with each cluster
        """
        threshRec = pkl.load(open(self.dataPath, 'rb'))['threshTiff']
        sF.getClusterMovies(threshRec, self.rowData[Level], self.labAct, \
                  Dim = self.spatialDims, Std = Std, SPD = Speed, vBase = vBase, vAlph = vAlph, \
                    fName = fName, Short = Short)
        del threshRec



    def getClusterSignificance(self, Threshold, normalise = False, labelText = True, nDots = 1e3, \
                               nData = 1e3, plotFilt = (2, 2), upSample = 5):
        """
        Search through tree and identify spatially separated clusters:
        1. Get list of parent clusters and overlaps

        Inputs:
        1. normaliseHm: Bool. Normalise heatmaps during comparison? (False produces more clusters generally)
        2. Threshold (float). Significance threshold for Hm comparison.
        """

        # Get hierarchy nodes and lists of all clusters
        clLabels, clTree = sF.clustInfo(lD = self.lD, cM = self.cM, rowData = self.rowData) 

        # Get a list of parent cluster and identify any significant overlaps
        print("\n\n*** Getting parents clusters ***\n\n")
        parentClusters = sF.getParentClusters(clLabels)                     
        parentClustArr = sF.parentClustOverlap(parentClusters, self.rowData, self.actM, \
                                               self.spatialDims, norm = normalise, TH = Threshold,\
                                                nDots = nDots)
        self.parentClustArr = sF.removeSingleOverlaps(parentClustArr)

        # Get significant children from isolated parent clusters
        print("\n\n*** Getting significant children from isolated parents ***\n\n")
        self.sigClusters1 = sF.separateParents(self.lD, self.parentClustArr, self.cM, \
                                   clTree, self.actM, self.spatialDims, nDots = nDots, nData = nData,\
                                    norm = normalise, TH = Threshold)
        
        # Get overlapping parents -> sort into significant and overlapping children
        print("\n\n*** Sorting overlapping parents into significant children ***\n\n")
        _, nodes = scipy.cluster.hierarchy.to_tree(self.lD, rd = True) 
        self.sigClusters2 = sF.sortOverlappingParents(self.parentClustArr, self.cM, clTree, \
                                    self.actM, self.rowData, nodes, self.spatialDims, nDots = nDots,\
                                    nData = nData)
        
        # Get the final list of clusters (Joined clusters listed: [1, 2, [3, 4], 5]) where [3,4] are joined
        self.finalClusters = self.sigClusters1 + [C[0] for C in self.sigClusters2 if len(C) == 1] \
                                    + [C for C in self.sigClusters2 if len(C) > 1]
        
        # Plot heatmaps of all significant clusters
        print("\nplotting all significant clusters")
        sF.plotMeanEnsHm(self.actM, self.rowData, Dim = self.spatialDims, \
                        plotClusters = self.finalClusters, clusterLabels = clLabels, \
                        labelText = labelText, filt = plotFilt, UPSAMPLE = upSample,\
                        Save = 'norm' + str(normalise) + '_Labels_thresh' + str(Threshold) + '_')





        #self.sigClusters, self.clTree, self.clLabels = sF.testWholeTree(self.lD, self.cM, \
        #                                self.rowData, self.actM, norm = normaliseHms, \
        #                                Dim = self.spatialDims, Thresh = Threshold)
        
        #print('\n Detected significant clusters are: {}'.format(self.sigClusters))

        #print('\n Now plotting mean heatmaps of significant clusters')
        
        #sF.plotMeanEnsHm(self.actM, self.rowData, labelText = labelText, Dim = self.spatialDims,\
        #         Save = 'norm' + str(normaliseHms) + '_Labels_thresh' + str(Threshold) + '_', \
        #         clusterLabels = self.clLabels, plotClusters = [int(tCl) for tCl in self.sigClusters], \
        #            UPSAMPLE = 5)


    #def getClustScore(self, Score='CI'):
    #    """Get score for each cluster hierarchy using given metric"""
    #
    #    if Score == 'CI':
    #        self.cI = sF.getCindex(self.cM, self.sM)
    
