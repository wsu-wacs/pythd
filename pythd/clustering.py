"""
Clustering algorithms used for MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage    

class BaseClustering(ABC):
    """
    Abstract base class for all clustering methods.
    """
    
    @abstractmethod
    def cluster(self, points):
        """Run the clustering and return the obtained clusters.
        
        Parameters
        ----------
        points : numpy.ndarray
            The points to cluster. Its shape should be (n, k), where
            n is the number of points and k the dimensionality of the data.
        
        Returns
        -------
        list
            A list of clusters. Each cluster is a list of integers indexing the
            original points array.
        """
        pass

class HierarchicalClustering(BaseClustering):
    """Hierarchical clustering using scipy
    
    This class runs hierarchical clustering using the scipy linkage function.
    
    Parameters
    ----------
    method : str
        The type of hierarchical clustering to use; default is "single" for single-linkage.
        Valid choices are:
        * 'single' : single-linkage
        * 'complete' : complete-linkage
        * 'average'
        * 'weighted'
        * 'centroid' : uses cluster centroids
        * 'median' : uses cluster medoids
        * 'ward' : Ward variance minimization
    metric : str
        The metric to use. Default is 'euclidean'
        Valid choices include:
        * 'euclidean' : Euclidean distance
        * 'cityblock' : Manhattan (taxicab) distance
        * 'seuclidean' : Standardized Euclidean, also known as variance-normalized Euclidean
        * 'sqeuclidean' : The square of the Euclidean distance
        * 'cosine' : Cosine distance (1 minus the cosine similarity)
        * 'correlation' : Correlation distance (1 minus the correlation coefficient)
        * 'hamming' : Hamming distance; number of points the vectors disagree in
        * 'jaccard' : Jaccard distance; proportion of points the vectors disagree in
        * 'chebyshev' : Chebyshev distance; maximum 1-norm of the vectors' components
        
    """
    def __init__(self, method='single', metric='euclidean'):
        if not isinstance(method, str):
            raise TypeError(f"Clustering method must be a string, not {type(method)}")
        self.method = method
        
        if not isinstance(metric, str):
            raise TypeError(f"Distance metric must be a string, not {type(metric)}")
        self.metric = metric
    
    def cluster(self, points, cut_method="first_gap"):
        """Run the clustering and return the obtained clusters.
        
        Parameters
        ----------
        points : numpy.ndarray
            The points to cluster. Its shape should be (n, k), where
            n is the number of points and k the dimensionality of the data.
        cut_method : str
            The method to use to obtain the clusters from the dendrogram.
            Valid choices are:
            * 'first_gap' : The method used in the original MAPPER paper. Compute a
              histogram of the cluster differences, and then choose the distance 
              corresponding to the first empty bin and cut there.
        
        Returns
        -------
        list
            A list of clusters. Each cluster is a list of integers indexing the
            original points array.
        """
        if isinstance(points, list):
            points = np.array(points)
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Points given to clustering method must be a numpy array, not {type(points)}")

        Z = linkage(points, method=self.method, metric=self.metric)
        num_points = points.shape[0]
        
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
