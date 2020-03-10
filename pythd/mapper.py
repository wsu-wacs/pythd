"""
Pure Python implementation of MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
import itertools
import functools
import numpy as np

from .complex import SimplicialComplex
from .filter import BaseFilter
from .cover import BaseCover
from .clustering import BaseClustering
from .utils import create_igraph_network, create_networkx_network

class MAPPERResult:
    """
        Represents the result of a MAPPER - the set of nodes with memberships.
        
        This is the 0-skeleton of the complex.
        
        Attributes
        ----------
        nodes
            The 0 simplices in the complex.
        complex : SimplicialComplex
            The simplicial complex constructed from this cover. Contains the
            k-skeleton, where k is the highest order passed in to compute_k_skeleton
            
        Parameters
        ----------
        complex : SimplicialComplex
            The 0-skeleton of the simplicial complex, constructed by the clustering
            run in the MAPPER class.
    """
    def __init__(self, complex):
        self.complex = complex
        self.nodes = {}
        for simplex, data, node_dict in self.complex.get_k_simplices(0, True):
            self.nodes[simplex[0]] = (data, node_dict)
    
    def compute_0_skeleton(self):
        """
        Get the 0-skeleton of the MAPPER.
        
        Returns
        -------
        SimplicialComplex
            The simplicial complex. 
        """
        return self.complex

    def compute_1_skeleton(self):
        """
        Compute the 1-skeleton of the MAPPER.
        
        This will compute the 1-skeleton of the MAPPER and return the nodes and edges.
        """
        for n1, n2 in itertools.combinations(sorted(self.nodes.keys()), 2):
            points1 = self.nodes[n1][1]["points"]
            points2 = self.nodes[n2][1]["points"]
            common = points1.intersection(points2)
            if len(common) > 0:
                self.complex.add_simplex((n1,n2))
        
        return self.complex
    
    def compute_k_skeleton(self, k=1):
        """
        Compute the k-skeleton of the MAPPER and return the resulting simplicial complex
        
        Parameters
        ----------
        k : int
            The maximum size of simplices to compute. Should be non-negative.
        
        Returns
        -------
        pythd.complex.SimplicialComplex
            The simplicial complex object
        """
        if k < 0:
            raise ValueError(f"Invalid value of k for k-skeleton: {k}")
        if k == 0:
            return self.compute_0_skeleton()
        elif k == 1:
            return self.compute_1_skeleton()
        
        self.compute_k_skeleton(k-1)
        km1_simps = self.complex.get_k_simplices(k=k-1)
        
        # It takes k+1 (k-1)-simplices to make up a k-simplex.
        # Consider all possible combinations of these and check their intersections.
        for subsets in itertools.combinations(km1_simps, k+1):
            # First we check if this is even a candidate k-simplex
            simplex = functools.reduce(lambda a,b: a|b, [frozenset(s) for s in subsets])
            if len(simplex) == (k+1):
                # This could be a k-simplex, now check for overlap in clusters
                clusters = [self.nodes[n][1]["points"] for n in simplex]
                common = functools.reduce(lambda a,b: a&b, clusters)
                if len(common) > 0:
                    self.complex.add_simplex(sorted(simplex))
        
        return self.complex
    
    def get_complex(self):
        """
        Get the simplicial complex associated to the MAPPER result.
        
        Returns
        -------
        SimplicialComplex
            The simplicial complex associated with the MAPPER result
        """
        return self.complex
    
    def get_igraph_network(self):
        """
        Get the 1-skeleton of the MAPPER as an igraph network.
        
        This requires the igraph package to be installed.
        """
        self.compute_1_skeleton()
        nodes = self.complex.get_k_simplices(k=0, include_data=True)
        edges = self.complex.get_k_simplices(k=1)
        return create_igraph_network(nodes, edges)
    
    def get_networkx_network(self):
        """
        Get the 1-skeleton of the MAPPER as a networkx network.
        
        This requires the networkx package to be installed.
        """
        self.compute_1_skeleton()
        nodes = self.complex.get_k_simplices(k=0, include_data=True)
        edges = self.complex.get_k_simplices(k=1)
        return create_networkx_network(nodes, edges)

class MAPPER:
    """
    This class represents the MAPPER algorithm.
    
    Attributes
    ----------
    filter : BaseFilter
        A filter function which computes a lower-dimensional representation of the data
    cover : BaseCover
        A covering of the range of the filter function
    clustering : BaseClustering
        A class which can cluster subsets of the data and return flat clusters
    """
    def __init__(self, filter=None, cover=None, clustering=None):
        self.set_filter(filter)
        self.set_cover(cover)
        self.set_clustering(clustering)
    
    def set_filter(self, filter):
        """Set the filter function"""
        if not isinstance(filter, BaseFilter):
            raise TypeError(f"Incorrect filter type: {type(filter)}")
        self.filter = filter
        return self
    
    def set_cover(self, cover):
        """Set the cover function"""
        if not isinstance(cover, BaseCover):
            raise TypeError(f"Incorrect cover type: {type(cover)}")
        self.cover = cover
        return self
    
    def set_clustering(self, clustering):
        """Set the clustering method"""
        if not isinstance(clustering, BaseClustering):
            raise TypeError("Incorrect clustering type: {type(clustering)}")
        self.clustering = clustering
        return self
    
    def run(self, points, f_x=None, rids=None):
        """
        Run MAPPER on the given data
        
        Parameters
        ----------
        points : numpy.ndarray or list
            The dataset in the shape (num_points, num_features). The columns should
            have the same dimension as the input to the filter function.
        f_x : numpy.ndarray
            (Optional) pre-computed filter values. For filter functions that take a long time,
            it may be more helpful to precompute the filter values and pass them here. This is
            also useful if you want to use a coloring based on filter values and do not wish
            to recompute them.
        rids : list
            Identifier numbers for each point. Used to keep track of membership in a larger
            dataset when running MAPPER on a subset of the dataset. 
        
        Returns
        -------
        MAPPERResult
            A MAPPERResult object which can be used to construct a simplicial complex.
        """
        # allow a list to be passed if it can be converted to a numpy array 
        if isinstance(points, list):
            points = np.array(points)
        if not isinstance(points, np.ndarray):
            raise TypeError(f"points array should be a numpy array, not {type(points)}")
        
        if rids is None:
            rids = list(range(points.shape[0]))
        elif not isinstance(rids, list):
            raise TypeError(f"RIDs should be given as a list, not {type(rids)}")

        if f_x is None:
            f_x = self.filter(points)
        elif isinstance(f_x, list):
            f_x = np.array(f_x)
        if not isinstance(f_x, np.ndarray):
            raise TypeError(f"Passed filter values should be a numpy array, not {type(f_x)}")

        d = self.cover.get_open_set_membership_dict(f_x)
        
        num_nodes = 0
        complex = SimplicialComplex()
        
        for open_set_id, members in d.items():
            memb_np = np.array(members)
            if len(members) == 1:
                clusters = [[0]] # Just the one point 
            else:
                clusters = self.clustering.cluster(points[members])
            
            for clust_id, cluster in enumerate(clusters):
                pointset = frozenset(memb_np[cluster]) # indexes within points array
                points_orig = frozenset([rids[idx] for idx in pointset]) # RIDs of points within points array
                complex.add_simplex((num_nodes,), points=pointset, points_orig=points_orig)
                num_nodes += 1
        
        return MAPPERResult(complex)
