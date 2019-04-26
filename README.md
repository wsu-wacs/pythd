This repository contains a pure Python implementation of MAPPER. In the future,
it will also implement topological hierarchical decompositions (THDs) on top of
the MAPPER implementation. It is developed with support for Python 3, and may
not run on Python 2.7 or earlier.

# Required Packages

The following Python packages are required to use pythd:
- numpy
- scipy
- matplotlib

The following packages are needed for graph export functionality:
- igraph
- networkx

# Basic Example

The following example will create a test dataset, and run MAPPER on it:
```python
import pythd
from matplotlib import pyplot as plt

# Two intersecting circles with some noise
dataset = (pythd.datagen.DatasetGenerator()
                .circle(center=[-4.0, 0.0], radius=4.0, noise=0.06, num_points=200)
                .circle(center=[4.0, 0.0], radius=4.0, noise=0.06, num_points=200)).get()

# Setup MAPPER
filt = pythd.filter.IdentityFilter() # use x,y coordinates as filter
f_x = filt(dataset) # filter values
cover = pythd.cover.IntervalCover.EvenlySpacedFromValues(f_x, 4, 0.5)
clustering = pythd.clustering.HierarchicalClustering()
mapper = pythd.mapper.MAPPER(filter=filt, cover=cover, clustering=clustering)
res = mapper.run(dataset, f_x=f_x) # run clustering step of MAPPER
network = res.compute_k_skeleton(k=1) # compute the one skeleton
```

To visualize the graph, there are three methods. The first uses igraph:
```python
import igraph
g = res.get_igraph_network()
layout = g.layout_kamada_kawai() # graph layout from igraph
igraph.plot(g, layout=layout)
```

The second uses networkx:
```python
import networkx as nx
g = res.get_networkx_network()
nx.draw(g)
```

The third is the recommended approach; there is support for drawing 2-simplices (faces) as well
as the nodes and edges, and support for node coloring. You'll need to provide the layout,
however, either computed with igraph or generated on your own:
```python
# re-using the layout computed from igraph above
coloring = pythd.coloring.create_node_coloring(f_x[:,0], network) # x-coordinate as coloring
pythd.plotting.draw_topological_network(network, layout, node_coloring=coloring)
```
