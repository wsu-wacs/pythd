import numpy as np

class ClusterNode:
    """
    A class representing a node in a hierarchical clustering.
    
    Meant to be more useful than the ClusterNode provided by scipy, mainly for evaluation.
    
    Attributes
    ----------
    ids : set
        The set of data point IDs in the cluster
    parent : ClusterNode
        The parent of this cluster, or None if it has no parent
    children : list
        The list of children of this cluster
    X : numpy.ndarray
        The original data (optional)
    y : numpy.ndarray
        Labels for the original data (optional)
    dist : float
        The distance between the children clusters, depending on the form of
        hierarchical clustering used
    """
    def __init__(self, ids=frozenset(), parent=None, children=[], dist=0.0):
        self.has_parent = parent is not None
        self.parent = parent
        self.children = children
        self.ids = ids
        self.dist = dist
    
    def get_data(self, X):
        ids = list(sorted(self.ids))
        return X[ids, :]
    
    def get_labels(self, y):
        ids = list(sorted(self.ids))
        return y[ids]
    
    """
    Get level sets by cluster distance
    """
    def get_dist_level(self, dist):
        to_visit = [self]
        changed = any([n.dist >= dist for n in to_visit])
        
        while changed:
            to_visit = [n.children if ((len(n.children) > 0) and n.dist >= dist) else [n] for n in to_visit]
            to_visit = [item for sublist in to_visit for item in sublist]
            changed = any([n.dist >= dist for n in to_visit])
        return to_visit
            
    """
    Get level sets by depth in the tree
    """
    def get_depth_level(self, depth):
        if depth == 0:
            return [self]
        
        to_visit = self.children
        depth -= 1
        while depth > 0:
            to_visit = [n.children if len(n.children) > 0 else [n] for n in to_visit]
            to_visit = [item for sublist in to_visit for item in sublist]
            depth -= 1
        return to_visit
    
    """
    Build a clustering tree from a scipy linkage matrix
    
    Parameters
    ----------
    Z : numpy.ndarray
        The linkage matrix obtained from a call to scipy's linkage function
    X : numpy.ndarray
        The original data (optional)
    y : numpy.ndarray
        Labels for the original data (optional)
    """
    @classmethod
    def FromLinkageMatrix(cls, Z):
        n = Z.shape[0] + 1 # num points in original data
        # base clusters (original data points)
        clusters = {i: ClusterNode(ids=frozenset([i])) for i in range(n)}
        
        for i in range(Z.shape[0]):
            # Clusters to merge
            cid1 = int(Z[i, 0])
            c1 = clusters[cid1]
            cid2 = int(Z[i, 1])
            c2 = clusters[cid2]
            # New cluster
            ncid = n + i
            nids = c1.ids | c2.ids
            nc = ClusterNode(ids=nids, children=[c1, c2])
            nc.dist = Z[i, 2]
            clusters[ncid] = nc
            # Update parents
            c1.parent = nc
            c2.parent = nc
    
        return nc
    
    """
    Assign labels to a flat clustering, such as that returned from get_dist_level
    """
    @classmethod
    def ComputeLabels(cls, clustering, n):
        y = np.zeros(n, dtype=np.uint)
        for i in range(n):
            for j, c in enumerate(clustering):
                if i in c.ids:
                    y[i] = j + 1
        return y