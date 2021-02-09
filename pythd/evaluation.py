class ClusterNode:
    """
    A class representing a node in a hierarchical clustering.
    
    Meant to be more useful than the ClusterNode provided by scipy, mainly for evaluation.
    """
    def __init__(self, ids=frozenset(), parent=None, children=[], X=None, y=None):
        self.has_parent = parent is not None
        self.parent = parent
        self.children = children
        self.ids = ids
        # original data
        self.X = X
        self.has_data = X is not None
        self.y = y
        self.has_labels = y is not None
    
    def get_data(self):
        if not self.has_data:
            raise ValueError("Data not provided for the cluster tree.")
        
        ids = list(sorted(self.ids))
        return self.X[ids, :]
    
    def get_labels(self):
        if not self.has_data:
            raise ValueError("Labels not provided for the cluster tree.")
        
        ids = list(sorted(self.ids))
        return self.y[ids]
    
    def get_level(self, depth):
        if depth == 0:
            return [self]
        
        to_visit = self.children
        depth -= 1
        while depth > 0:
            # TODO: propagate root nodes if no children
            to_visit = [n.children for n in to_visit]
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
    def FromLinkageMatrix(cls, Z, X=None, y=None):
        n = Z.shape[0] + 1 # num points in original data
        # base clusters (original data points)
        clusters = {i: ClusterNode(ids=frozenset([i]), X=X, y=y) for i in range(n)}
        
        for i in range(Z.shape[0]):
            # Clusters to merge
            cid1 = int(Z[i, 0])
            c1 = clusters[cid1]
            cid2 = int(Z[i, 1])
            c2 = clusters[cid2]
            # New cluster
            ncid = n + i
            nids = c1.ids | c2.ids
            nc = ClusterNode(ids=nids, children=[c1, c2], X=X, y=y)
            clusters[ncid] = nc
            # Update parents
            c1.parent = nc
            c2.parent = nc
    
        return nc
