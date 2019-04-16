"""
Functionality for plotting MAPPER results

By Kyle Brown <brown.718@wright.edu>
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.lines as lines

def _draw_node(ax, x, y, radius=0.1, color="#FF0000"):
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

def _draw_nodes(ax, complex, layout, coloring="density"):
    """Draw all the nodes in a MAPPER complex"""
    nodes = complex[0]

    min_x = 1e20
    max_x = 1e-20
    min_y = 1e20
    max_y = 1e-20
    
    colors = {n: "red" for n in nodes.keys()}
    if coloring == "density":
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

    for n, pts in nodes.items():
        x, y = (layout[n][0], -layout[n][1])
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
        _draw_node(ax, x, y, color=colors[n])
    
    return (min_x, max_x, min_y, max_y)

def _draw_edges(ax, complex, layout):
    """Draw all the edges in a MAPPER complex"""
    edges = complex[1]

    for a,b in edges:
        from_xy = (layout[a][0], -layout[a][1])
        to_xy = (layout[b][0], -layout[b][1])
        _draw_edge(ax, from_xy, to_xy)

def _draw_faces(ax, complex, layout):
    """Draw all the faces (2-simplices) in a MAPPER complex"""
    faces = complex[2]
    
    for face in faces:
        coords = np.array([[layout[n][0], -layout[n][1]] for n in face])
        _draw_face(ax, coords)

def draw_topological_network(complex, layout, node_coloring="density"):
    """Draw the 1-skeleton of a MAPPER output

    Parameters
    ----------
    complex: tuple
        The 1-skeleton (or higher) of a MAPPER, as returned by calling compute_k_skeleton()
        with k=1 or higher.
    layout: object
        An object which implements the __getitem__ method which is used to get the
        positions of the nodes in the network
    """
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))
    ax.set_axis_off()

    _draw_edges(ax, complex, layout)
    min_x, max_x, min_y, max_y = _draw_nodes(ax, complex, layout, coloring=node_coloring)
    
    ax.set_xlim(min_x-0.2, max_x+0.2)
    ax.set_ylim(min_y-0.2, max_y+0.2)
    ax.set_aspect(1.0)

def draw_2_skeleton(complex, layout, node_coloring="density"):
    """Draw the 2-skeleton of a MAPPER output
    
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
    
    _draw_faces(ax, complex, layout)
    _draw_edges(ax, complex, layout)
    min_x, max_x, min_y, max_y = _draw_nodes(ax, complex, layout, coloring=node_coloring)
    
    ax.set_xlim(min_x-0.2, max_x+0.2)
    ax.set_ylim(min_y-0.2, max_y+0.2)
    ax.set_aspect(1.0)