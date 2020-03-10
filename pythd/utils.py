def create_igraph_network(nodes, edges):
    """
    Convert a 1-skeleton to an igraph network.
    
    This function requires the python-igraph package to be installed.
    
    Parameters
    ----------
    nodes : list
        List of 0-simplices with associated data and dict
    edges : list
        List of 1-simplices
    
    Returns
    -------
    igraph.Graph
        The igraph Graph constructed from the 1-skeleton
    """
    import igraph
    g = igraph.Graph()
    for vid, data, dict in nodes:
        g.add_vertex(name=str(vid[0]), data=data, **dict)
    g.add_edges(edges)
    return g

def create_networkx_network(nodes, edges):
    """
    Convert a 1-skeleton to a networkx network.
    
    This function requires the networkx package to be installed.

    Parameters
    ----------
    nodes : list
        List of 0-simplices with associated data and dict
    edges : list
        List of 1-simplices
    
    Returns
    -------
    networkx.Graph
        The networkx Graph constructed from the 1-skeleton
    """
    import networkx as nx
    g = nx.Graph()
    for vid, data, dict in nodes:
        g.add_node(vid[0], name=str(vid[0]), data=data, **dict)
    g.add_edges_from(edges)
    return g

class open_or_use:
    def __init__(self, fname, f, mode):
        if f is None:
            self.fname = fname
            self.needs_close = True
            self.mode = mode
        else:
            self.f = f
            self.needs_close = False
    def __enter__(self):
        if self.needs_close:
            self.f = open(self.fname, self.mode)
            return self.f
        else:
            return self.f
    def __exit__(self, type, value, traceback):
        if self.needs_close:
            self.f.close()
