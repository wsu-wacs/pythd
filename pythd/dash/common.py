"""
Common functionality for all MAPPER dashboards
"""
import io, base64

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ..filter import *

__all__ = ['get_filter', 'networkx_network_to_cytoscape_elements', 'contents_to_dataframe']

def get_filter(name, metric, n_components=2, component_list=[0], eccentricity_method='mean'):
    """
    Get a filter object from parameters
    """
    if name == 'tsne':
        return ScikitLearnFilter(TSNE, n_components=n_components, metric=metric)
    elif name == 'pca':
        return ScikitLearnFilter(PCA, n_components=n_components)
    elif name == 'identity':
        return IdentityFilter()
    elif name == 'component':
        return ComponentFilter(components)
    elif name == 'eccentricity':
        return EccentricityFilter(metric=metric, method=method)

def networkx_network_to_cytoscape_elements(network, df):
    """
    Convert a networkx network produced by running MAPPER into an elements list for Cytoscape

    Parameters
    ----------
    network
        The network object
    df : pandas.DataFrame
        The dataframe the MAPPER was built on

    Returns
    -------
    list
        The Cytoscape elements list converted from the network.
        The nodes are all listed before the edges.
    """
    elements = []
    maxn = max([len(network.nodes[n]['points']) for n in network.nodes])

    for n in network.nodes:
        node = network.nodes[n]
        d = {
                'data': {'id': n,
                         'npoints': len(node['points']),
                         'density': len(node['points']) / maxn
            }}

        d['data'].update({
            col: df.loc[node['points'], col].mean()
            for col in df.columns
        })
        elements.append(d)

    for src, dst in network.edges:
        elements.append({'data': {'source': src, 'target': dst}})

    return elements

def contents_to_dataframe(contents):
    """
    Convert the value of a Dash upload component (CSV or zipped CSV) into a Pandas dataframe

    Parameters
    ----------
    contents : str or bytes
        The contents of the uploaded file (base64 encoded)
        The file should be a CSV (or a zipped CSV) with the following components:
            * The first row is a header that gives column names
            * The first column is an index
            * All other columns are numeric data

    Returns
    -------
    pandas.DataFrame
        The converted dataframe
    """
    contents = contents.split(',')
    content_type = contents[0].split(';')
    contents = contents[1]
    contents = base64.b64decode(contents, validate=True)

    if 'zip' in content_type[0]:
        with io.BytesIO(contents) as f:
            df = pd.read_csv(f, header=0, index_col=0, compression='zip')
    else:
        with io.StringIO(contents.decode('utf-8')) as f:
            df = pd.read_csv(f, header=0, index_col=0)
    return df

