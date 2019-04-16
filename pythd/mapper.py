"""
Pure Python implementation of MAPPER

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
import itertools
import functools
import numpy as np

def create_igraph_network(nodes, edges):
    """
    Convert a 1-skeleton to an igraph network.
    """
    import igraph
    g = igraph.Graph()
    for vid, points in nodes.items():
        g.add_vertex(name=str(vid), points=points)
    g.add_edges(edges)
    return g

def create_networkx_network(nodes, edges):
    """
    Convert a 1-skeleton to a networkx network.
    """
    import networkx as nx
    g = nx.Graph()
    for vid, points in nodes.items():
        g.add_node(vid, name=str(vid), points=points)
    g.add_edges_from(edges)
    return g

class MAPPERResult:
    """
        Represents the result of a MAPPER - the set of nodes with memberships.
        
        This is the 0-skeleton of the complex.
    """
    def __init__(self, nodes):
        self.nodes = nodes
    
    def compute_0_skeleton(self):
        """
        Get the 0-skeleton of the MAPPER.
        """
        return self.nodes

    def compute_1_skeleton(self):
        """
        Compute the 1-skeleton of the MAPPER.
        
        This will compute the 1-skeleton of the MAPPER and return the nodes and edges.
        """
        edges = []
        for n1, n2 in itertools.combinations(self.nodes.keys(), 2):
            common = self.nodes[n1].intersection(self.nodes[n2])
            if len(common) > 0:
                edges.append((n1, n2))
        
        return (self.nodes, edges)
    
    def compute_k_skeleton(self, k=1):
        """
        Compute the k-skeleton of the MAPPER
        
        Parameters
        ----------
        k : int
            The maximum size of simplices to compute. Should be non-negative.
        
        Returns
        -------
        tuple
            A tuple where the first element is 0-simplices, the next one 1-simplices,
            and so on up to k-simplices.
        """
        if k < 0:
            raise ValueError(f"Invalid value of k for k-skeleton: {k}")
        if k == 0:
            return self.compute_0_skeleton()
        elif k == 1:
            return self.compute_1_skeleton()
        
        km1_skel = self.compute_k_skeleton(k-1)
        km1_simps = km1_skel[-1]
        
        k_simps = []
        
        # It takes k+1 (k-1)-simplices to make up a k-simplex.
        # Consider all possible combinations of these and check their intersections.
        for subsets in itertools.combinations(km1_skel[-1], k+1):
            # First we check if this is even a candidate k-simplex
            sets = [set(s) for s in subsets]
            simplex = functools.reduce(lambda a,b: a|b, [set(s) for s in subsets])
            if len(simplex) == (k+1):
                # This could be a k-simplex, now check for overlap in clusters
                clusters = [self.nodes[n] for n in simplex]
                common = functools.reduce(lambda a,b: a&b, clusters)
                if len(common) > 0:
                    k_simps.append(tuple(simplex))
        
        return km1_skel + (k_simps,)
            
            
    
    def get_igraph_network(self):
        """
        Get the 1-skeleton of the MAPPER as an igraph network.
        
        This requires the igraph package to be installed.
        """
        nodes, edges = self.compute_1_skeleton()
        return create_igraph_network(nodes, edges)
    
    def get_networkx_network(self):
        """
        Get the 1-skeleton of the MAPPER as a networkx network.
        
        This requires the networkx package to be installed.
        """
        nodes, edges = self.compute_1_skeleton()
        return create_networkx_network(nodes, edges)

class MAPPER:
    def __init__(self, filter=None, cover=None, clustering=None):
        self.set_filter(filter)
        self.set_cover(cover)
        self.set_clustering(clustering)
    
    def set_filter(self, filter):
        self.filter = filter
        return self
    
    def set_cover(self, cover):
        self.cover = cover
        return self
    
    def set_clustering(self, clustering):
        self.clustering = clustering
        return self
    
    def run(self, points, f_x=None):
        """
        Run MAPPER on the given data
        
        Parameters
        ----------
        points : numpy.ndarray
            The dataset in the shape (num_points, num_features). The columns should
            have the same dimension as the input to the filter function.
        """
        if f_x is None:
            f_x = self.filter(points)
        d = self.cover.get_open_set_membership_dict(f_x)
        
        nodes = {}
        num_nodes = 0
        
        for open_set_id, members in d.items():
            memb_np = np.array(members)
            if len(members) == 1:
                clusters = [[0]] # Just the one point 
            else:
                clusters = self.clustering.cluster(points[members])
            
            for clust_id, cluster in enumerate(clusters):
                nid = num_nodes
                num_nodes += 1
                nodes[nid] = set(memb_np[cluster])
        
        return MAPPERResult(nodes)
