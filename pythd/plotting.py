"""
Functionality for plotting MAPPER results

By Kyle Brown <brown.718@wright.edu>
"""
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def _draw_node(ax, x, y, radius=0.1, color="#FF0000"):
    circ = patches.Circle((x, y), radius=radius, color=color)
    ax.add_patch(circ)

def _draw_edge(ax, from_xy, to_xy):
    patch = patches.ConnectionPatch(from_xy, to_xy, "data")
    ax.add_patch(patch)

def draw_2_skeleton(complex, layout):
    """draw the 2-skeleton of a MAPPER output
    
    Parameters
    ----------
    complex: tuple
        The 2-skeleton (or higher) of a MAPPER, as returned by calling compute_k_skeleton()
        with k=2 or higher.
    layout: object
        An object which implements the __getitem__ method which is used to get the
        positions of the nodes in the network
    """
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))
    ax.set_axis_off()
    
    nodes, edges, faces = complex
    
    min_x = 1e20
    max_x = 1e-20
    min_y = 1e20
    max_y = 1e-20
    
    for a,b in edges:
        from_xy = (layout[a][0], -layout[a][1])
        to_xy = (layout[b][0], -layout[b][1])
        _draw_edge(ax, from_xy, to_xy)

    for n, pts in nodes.items():
        x, y = (layout[n][0], -layout[n][1])
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
        _draw_node(ax, x, y)
    
    ax.set_xlim(min_x-0.1, max_x+0.1)
    ax.set_ylim(min_y-0.1, max_y+0.1)
    ax.set_aspect(1.0)