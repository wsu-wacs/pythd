import io, base64
import pandas as pd

__all__ = ['networkx_network_to_cytoscape_elements', 'contents_to_dataframe']

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

    if 'zip' in content_type:
        with io.BytesIO(contents) as f:
            df = pd.read_csv(f, header=0, index_col=0, compression='zip')
    else:
        with io.StringIO(contents.decode('utf-8')) as f:
            df = pd.read_csv(f, header=0, index_col=0)
    return df
