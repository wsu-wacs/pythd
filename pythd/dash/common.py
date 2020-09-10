def networkx_network_to_cytoscape_elements(network, nrows=1):
    elements = []
    maxn = max([len(network.nodes[n]['points']) for n in network.nodes])

    for n in network.nodes:
        node = network.nodes[n]
        elements.append({'data': {'id': n, 
                         'npoints': len(node['points']),
                         'density': len(node['points']) / maxn}})

    for src, dst in network.edges:
        elements.append({'data': {'source': src, 'target': dst}})

    return elements

