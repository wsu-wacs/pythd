"""
Functionality for plotting MAPPER results

By Kyle Brown <brown.718@wright.edu>
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.lines as lines

from . import coloring as _coloring

def _draw_node(ax, x, y, radius=2, color="#FF0000"):
    """Draw a single node in a MAPPER complex"""
    circ = patches.Circle((x, y), radius=radius, facecolor=color, edgecolor="black", zorder=1.0)
    ax.add_patch(circ)

def _draw_edge(ax, from_xy, to_xy, color="#000000"):
    """Draw a single edge in a MAPPER complex"""
    line = lines.Line2D([from_xy[0], to_xy[0]], [from_xy[1], to_xy[1]], linewidth=1.0, color=color, zorder=0.0)
    ax.add_line(line)

def _draw_face(ax, coords):
    """Draw a single face (2-simplex) in a MAPPER complex"""
    patch = patches.Polygon(coords, closed=True, alpha=0.5, color="blue", zorder=-1.0)
    ax.add_patch(patch)

def _draw_nodes(ax, nodes, layout, coloring="density", radius=0.1):
    """Draw all the nodes in a MAPPER complex"""
    min_x = 1e20
    max_x = 1e-20
    min_y = 1e20
    max_y = 1e-20
    
    colors = {n[0][0]: "red" for n in nodes}

    if coloring == "density":
        colors = _coloring.create_node_density_coloring(nodes)
    elif isinstance(coloring, dict):
        colors = coloring
    else:
        raise TypeError(f"Unknown coloring type: {type(coloring)}")

    for n, _, __ in nodes:
        id = n[0]
        x, y = (layout[id][0], -layout[id][1])
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
        _draw_node(ax, x, y, color=colors[id], radius=radius)
    
    return (min_x, max_x, min_y, max_y)

def _draw_edges(ax, edges, layout):
    """Draw all the edges in a MAPPER complex"""

    for a,b in edges:
        from_xy = (layout[a][0], -layout[a][1])
        to_xy = (layout[b][0], -layout[b][1])
        _draw_edge(ax, from_xy, to_xy)

def _draw_faces(ax, faces, layout):
    """Draw all the faces (2-simplices) in a MAPPER complex"""
    for face in faces:
        coords = np.array([[layout[n][0], -layout[n][1]] for n in face])
        _draw_face(ax, coords)

def draw_topological_network(complex, layout, node_coloring="density", node_radius=0.1):
    """Draw the 1-skeleton of a MAPPER output

    Parameters
    ----------
    complex: tuple
        The 1-skeleton (or higher) of a MAPPER, as returned by calling compute_k_skeleton()
        with k=1 or higher.
    layout: object
        An object which implements the __getitem__ method which is used to get the
        positions of the nodes in the network
    node_coloring : str or dict
        An optional node coloring to use. If a dict, should be a coloring dictionary
        mapping from 0 simplices to colors.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))
    ax.set_axis_off()

    nodes = complex.get_k_simplices(k=0, include_data=True)
    edges = complex.get_k_simplices(k=1)

    _draw_edges(ax, edges, layout)
    min_x, max_x, min_y, max_y = _draw_nodes(ax, nodes, layout, coloring=node_coloring, radius=node_radius)
    
    ax.set_xlim(min_x-0.2, max_x+0.2)
    ax.set_ylim(min_y-0.2, max_y+0.2)
    ax.set_aspect(1.0)

def draw_2_skeleton(complex, layout, node_coloring="density", node_radius=0.1):
    """Draw the 2-skeleton of a MAPPER output
    
    Parameters
    ----------
    complex: tuple
        The 2-skeleton (or higher) of a MAPPER, as returned by calling compute_k_skeleton()
        with k=2 or higher.
    layout: object
        An object which implements the __getitem__ method which is used to get the
        positions of the nodes in the network
    node_coloring : str or dict
        An optional node coloring to use. If a dict, should be a coloring dictionary
        mapping from 0 simplices to colors.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))
    ax.set_axis_off()
    
    nodes = complex.get_k_simplices(k=0, include_data=True)
    edges = complex.get_k_simplices(k=1)
    faces = complex.get_k_simplices(k=2)
    
    _draw_faces(ax, faces, layout)
    _draw_edges(ax, edges, layout)
    min_x, max_x, min_y, max_y = _draw_nodes(ax, nodes, layout, coloring=node_coloring, radius=node_radius)
    
    ax.set_xlim(min_x-2*node_radius, max_x+2*node_radius)
    ax.set_ylim(min_y-2*node_radius, max_y+2*node_radius)
    ax.set_aspect(1.0)

def plot_2d_point_cloud(data, show=True, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], ".")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if show:
        plt.show()