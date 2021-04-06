"""
Clustering algorithms used for MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math

import numpy as np
import numba
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import OPTICS

@numba.jit(nopython=True, parallel=True, cache=True)
def auto_num_bins(x):
    """Reimplementation of numpy's automatic bin number algorithm"""
    sturges = int(np.log2(x.size)) + 1
    fd = 0

    ps = np.percentile(x, [75, 25])
    iqr = ps[1] - ps[0]
    if iqr != 0.0:
        ptp = x.max() - x.min()
        h = 2.0 * iqr * (x.size ** (-1.0 / 3.0))
        fd = int(ptp / h)
    
    return max(sturges, fd)

@numba.jit(nopython=True, parallel=True, cache=True)
def first_gap_cluster(Z, num_points):
    edge_dist = np.array([a[2] for a in Z])
    histo_freq, bin_edges = np.histogram(edge_dist, bins=auto_num_bins(edge_dist))
    ids = np.where(histo_freq == 0)[0] # indices of empty bins

    if ids.shape[0] > 0:
        # Index of first empty bin
        i = ids[0]
        # Threshold distance is midpoint of this first empty bin
        threshold = 0.5*(bin_edges[i] + bin_edges[i+1])
        return (True, threshold)
    else:
        return (False, 0.0)

@numba.jit(nopython=True, cache=True, parallel=True)
def extract_clusters(cluster_memberships, num_points):
    clusters = []
    for c in np.unique(cluster_memberships):
        pic = np.where(np.array([cluster_memberships[i] == c for i in range(num_points)]))[0]
        clusters.append(list(pic))
    return clusters

def labels_to_clusters(labels):
    # Convert labels to cluster arrays
    clusters = defaultdict(list)
    noise_index = max(labels) + 1
    for i, lab in enumerate(labels):
        if lab < 0: # noise in algorithms like DBSCAN
            clusters[noise_index].append(i)
            noise_index += 1
        else:
            clusters[lab].append(i)
    
    return list(clusters.values())

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

class ScikitLearnClustering(BaseClustering):
    """Used for clustering algorithms that follow scikit-learn's interface
    
    Parameters
    ----------
    cls
        The clustering algorithm class. Instances of it should have
        a fit_predict method.
    *args
        Positional arguments that will be passed to the constructor of cls
    **kwargs
        Keyword arguments that will be passed to the constructor of cls
    """
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        
        self.reset()
    
    def reset(self):
        self.clust = self.cls(*self.args, **self.kwargs)
    
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
        if isinstance(points, list):
            points = np.array(points)
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Points given to clustering method must be a numpy array, not {type(points)}")

        n = points.shape[0]
        if n == 1:
            return [[0]]
        else:
            labels = self.clust.fit_predict(points)
            return labels_to_clusters(labels)
        

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
        
        self.cut_method = "first_gap"
    
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

        num_points = points.shape[0]
        
        if self.metric == 'precomputed':
            points = squareform(points)
            
        Z = linkage(points, method=self.method, metric=self.metric)
        
        # Method used in original MAPPER paper
        self.cut_method = cut_method
        if cut_method == "first_gap":
            res, val = first_gap_cluster(Z, num_points)
            if res:
                cluster_memberships = fcluster(Z, val, criterion="distance")
                clusters = extract_clusters(cluster_memberships, num_points)
                return clusters 
            else:
                return [list(range(num_points))]

    def get_dict(self):
        """Get a dictionary representation of the clustering settings.
        
        The dictionary is suitable for JSON serialization."""
        return {
            "type": type(self).__name__,
            "method": self.method,
            "metric": self.metric
        }
