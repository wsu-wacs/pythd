"""
Common functionality for all MAPPER dashboards
"""
import io, base64, pickle
from uuid import uuid4

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ..filter import *
from .config import *

__all__ = ['get_filter', 'networkx_network_to_cytoscape_elements', 'contents_to_dataframe',
           'make_dataframe_token', 'load_cached_dataframe', 'summarize_dataframe']

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
                         'points': list(node['points']),
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

def contents_to_dataframe(contents, no_index=False, no_header=False):
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

    compression = 'zip' if ('zip' in content_type[0]) else 'infer'
    index_col = None if no_index else 0
    header = None if no_header else 0

    with io.BytesIO(contents) as f:
        df = pd.read_csv(f, header=header, index_col=index_col, compression=compression)
    return df

def make_dataframe_token(df):
    """
    Pickle the dataframe to the cache dir and return the file path

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe, loaded into memory

    Returns
    -------
    pathlib.Path
        The path object to the pickled dataframe
    """
    fname = DATA_DIR / '{}.pkl'.format(uuid4())
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    return fname

def load_cached_dataframe(fname):
    """
    Load a previously cached dataframe

    Parameters
    ----------
    fname : pathlib.Path or str
        The path to the pickled dataframe

    Returns
    -------
    pandas.DataFrame
        The un-pickled dataframe
    """
    with open(fname, 'rb') as f:
        df = pickle.load(f)
    return df

def summarize_dataframe(df):
    """
    Make a summary dataframe from a given dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to summarize

    Returns
    -------
    pandas.DataFrame
    """
    return pd.DataFrame({
            'column': df.columns,
            'mean': df.mean(axis=0),
            'median': df.median(axis=0),
            'min': df.min(axis=0),
            'max': df.max(axis=0)
        },
        index=df.columns)

