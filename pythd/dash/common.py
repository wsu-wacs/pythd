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
           'make_dataframe_token', 'load_cached_dataframe', 'summarize_dataframe',
           'handle_upload_options', 'make_datatable_info', 'get_comparison_groups', 'normalize_dataframe',
           'get_header']

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
        sub_df = df.iloc[node['points'], :]
        d = {
                'data': {'id': n,
                         'points': list(node['points']),
                         'npoints': len(node['points']),
                         'density': len(node['points']) / maxn
            }}

        d['data'].update({
            col: sub_df.loc[:, col].mean()
            for col in df.columns
        })
        elements.append(d)

    for src, dst in network.edges:
        elements.append({'data': {'source': src, 'target': dst}})

    return elements

def decode_contents(contents):
    contents = contents.split(',')
    content_type = contents[0].split(';')
    contents = contents[1]
    contents = base64.b64decode(contents, validate=True)
    return (contents, content_type[0])

def contents_to_dataframe(contents, remove_cols=None, no_index=False, no_header=False):
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
    contents, content_type = decode_contents(contents)

    compression = 'zip' if ('zip' in content_type) else 'infer'
    index_col = None if no_index else 0
    header = None if no_header else 0

    with io.BytesIO(contents) as f:
        df = pd.read_csv(f, header=header, index_col=index_col, compression=compression)

    if remove_cols is not None:
        df.drop(columns=remove_cols, inplace=True)

    # Remove NA rows
    df.dropna(inplace=True)

    # Process non-numerical columns
    # for now, just one-hot encode (TODO: add option to select beteween this and converting to ints)
    new_cols = []
    drop_cols = []
    for ci, dtype in enumerate(list(df.dtypes)):
        if dtype.kind in ['O','S']:
            cname = df.columns[ci]
            dummies = pd.get_dummies(df[cname], prefix=cname)
            new_cols.append(dummies)
            drop_cols.append(cname)
    df.drop(columns=drop_cols, inplace=True)
    if len(new_cols) > 0:
        new_cols = pd.concat(new_cols, axis=1)
        df = pd.concat((df, new_cols), axis=1)
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

    Includes the following information:
        * Mean, median, minimum, and maximum values of the column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to summarize

    Returns
    -------
    pandas.DataFrame
    """
    q1 = df.quantile(0.25, axis=0)
    q3 = df.quantile(0.75, axis=0)
    
    return pd.DataFrame({
            'column': df.columns,
            'mean': df.mean(axis=0),
            'quantile25': q1,
            'median': df.median(axis=0),
            'quantile75': q3,
            'iqr': q3 - q1,
            'min': df.min(axis=0),
            'max': df.max(axis=0)
        },
        index=df.columns)

def handle_upload_options(options):
    """
    Process upload options from the list of checked values returned from Dash

    Parameters
    ----------
    options : list
        The list of selected options

    Returns
    -------
    dict
        A dictionary suitable for passing as kwargs to contents_to_dataframe
    """
    all_options = ['no_index', 'no_header']
    return {o: o in options for o in all_options}

def make_datatable_info(df):
    """
    Convert a pandas dataframe to the format expected by dash DataTable

    Creates the columns and data objects and determines the types of the columns
    """
    columns = []
    for i, c in enumerate(df.columns):
        dtype = df.dtypes[i]
        d = {'name': c, 'id': c}
        # column type
        if dtype.kind in ['i', 'u', 'f']:
            d['type'] = 'numeric'
        elif dtype.kind in ['S', 'U', 'O']:
            d['type'] = 'text'
        elif dtype.kind in ['m', 'M']:
            d['type'] = 'datetime'
        else:
            d['type'] = 'any'

        if dtype.kind == 'f':
            d['format'] = {
                'specifier': '.4f'
            }

            
        columns.append(d)

    data = df.to_dict('records')
    return columns, data

def get_comparison_groups(group1, group2, groups):
    """
    Get the RIDs and group names for groups to compare

    Parameters
    ----------
    group1 : str
        The name of the first group
    group2 : str
        The name of the second group
    groups : dict
        The groups dict from the THD

    Returns
    -------
    """
    all_rids = frozenset(groups['0.0.0']['rids'])
    d = {'all': 'All of source data', 'rest': 'Rest of source data'}
    g1name = d.get(group1, group1)
    g2name = d.get(group2, group2)

    if group1 == 'all':
        group1 = all_rids
    if group2 == 'all':
        group2 = all_rids

    # Handle "rest of data" selections
    if group1 == 'rest':
        group2 = frozenset(groups.get(group2, {}).get('rids', []))
        group1 = all_rids - group2
    elif group2 == 'rest':
        group1 = frozenset(groups.get(group1, {}).get('rids', []))
        group2 = all_rids - group1
    else:
        group1 = frozenset(groups.get(group1, {}).get('rids', []))
        group2 = frozenset(groups.get(group2, {}).get('rids', []))

    return group1, group2, g1name, g2name

def _norm_maxabs(ser):
    mv = ser.abs().max()
    if mv > 0.0:
        return ser / mv
    return ser

def _norm_minmax(ser):
    minv = ser.min()
    maxv = ser.max()
    d = maxv - minv
    if d > 0.0:
        return (ser - minv) / d
    return ser

def _norm_standard(ser):
    mu = ser.mean()
    std = ser.std()
    if std > 0.0:
        ser = (ser - mu) / std
    return ser

def _norm_robust(ser):
    q2 = ser.median()
    iqr = ser.quantile(0.75) - ser.quantile(0.25)
    if iqr > 0.0:
        ser = (ser - q2) / iqr
    return ser

def normalize_dataframe(df, method):
    column_normalizers = {
        'maxabs': _norm_maxabs,
        'minmax': _norm_minmax,
        'standard': _norm_standard,
        'robust': _norm_robust
    }

    cn = column_normalizers.get(method, lambda ser: ser)
    for column in df.columns:
        df[column] = cn(df[column])

    return df

def get_header(contents):
    contents, content_type = decode_contents(contents)
    if content_type == 'zip':
        pass # TODO: handle
    else:
        contents = contents.decode(encoding='utf-8')
        with io.StringIO(contents) as f:
            line = next(f)
    return list(map(lambda s: s.replace('"', ''), line.split(',')))
