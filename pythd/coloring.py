"""
Node and edge colorings for MAPPER

By Kyle Brown <brown.718@wright.edu>
"""
from matplotlib import cm
import numpy as np

def create_node_coloring(values, complex):
    """Create coloring dictionary based on average value in each node.
    
    Parameters
    ----------
    values : iterable
        An iterable of floats representing the values associated to points.
    complex : tuple
        The simplicial complex on which to create the coloring.
    """
    values = np.array(values)
    minv = values.min()
    maxv = values.max()
    if maxv == minv:
        div = 1.0 / maxv
        minv = 0.0
    else:
        div = 1.0 / (maxv - minv)
    cmap = cm.get_cmap("jet", 64)
    
    colors = {}
    for n, pts in complex[0].items():
        avg_val = values[list(pts)].mean()
        colors[n] = cmap((avg_val - minv) * div)
    return colors

def create_node_density_coloring(complex):
    """Create a node density based coloring from a simplicial complex.
    """
    nodes = complex[0]
    node_counts = {n: len(pts) for n, pts in nodes.items()}
    min_n = min(node_counts.values())
    max_n = max(node_counts.values())
    if max_n == min_n:
        div = 1.0 / max_n
        min_n = 0.0
    else:
        div = 1.0 / (max_n - min_n)
    cmap = cm.get_cmap("jet", 64)
    colors = {n: cmap(float(node_counts[n] - min_n) * div) for n in node_counts.keys()}
    return colors