"""
Clustering algorithms used for MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage    

class BaseClustering(ABC):
    @abstractmethod
    def cluster(self, points):
        pass

class HierarchicalClustering(BaseClustering):
    """Hierarchical clustering using scipy
    """
    def __init__(self, method='single', metric='euclidean'):
        self.method = method
        self.metric = metric
    
    def cluster(self, points, cut_method="first_gap"):
        Z = linkage(points, method=self.method, metric=self.metric)
        num_points = len(points)
        
        # Method used in original MAPPER paper
        if cut_method == "first_gap":
            edge_dist = [a[2] for a in Z]
            histo_freq, bin_edges = np.histogram(edge_dist, bins="auto")
            try:
                # Index of first empty bin
                i = list(histo_freq).index(0)
                # Threshold distance is midpoint of this first empty bin
                threshold = 0.5*(bin_edges[i] + bin_edges[i+1])
                cluster_memberships = list(fcluster(Z, threshold, criterion="distance"))
            except:
                return [list(range(num_points))]
        
        clusters = []
        for c in np.unique(cluster_memberships):
            pic = np.where([cluster_memberships[i] == c for i in range(num_points)])[0]
            clusters.append(list(pic))

        return clusters 
