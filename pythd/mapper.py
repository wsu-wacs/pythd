"""
Pure Python implementation of MAPPER

TODO:
* Separate computation of intermediate structure (clustering, etc.)
  from computation of k-skeleton

Original code by Xiu Huan Yap <yap.4@wright.edu>
Rewritten and modified by Kyle Brown <brown.718@wright.edu>
"""
import itertools
import numpy as np

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
    
    def compute_1_skeleton(self, points):
        """
            Compute the MAPPER structure as a topological network
        """
        f_x = self.filter(points) # filter values
        d = self.cover.get_open_set_membership_dict(f_x)
        
        nodes = {}
        num_nodes = 0
        
        for set_id, members in d.items():
            memb_np = np.array(members)
            if len(members) == 1:
                clusters = [[0]] # Just the one point 
            else:
                clusters = self.clustering.cluster(points[members])
            
            for clust_id, cluster in enumerate(clusters):
                nid = num_nodes
                num_nodes += 1
                nodes[nid] = set(memb_np[cluster])

        edges = []
        for n1, n2 in itertools.combinations(nodes.keys(), 2):
            common = nodes[n1].intersection(nodes[n2])
            if len(common) > 0:
                edges.append((n1, n2))
        
        return (nodes, edges)
