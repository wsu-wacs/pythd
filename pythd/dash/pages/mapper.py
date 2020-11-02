from pathlib import Path
import base64, io, json

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash_table import DataTable
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app
from ..common import *
from ..layout_common import *
from ...cover import IntervalCover
from ...clustering import HierarchicalClustering 
from ...mapper import MAPPER

################################################################################
# Layout
################################################################################
layout = html.Div(style=dict(height='100%'), children=[
    # Main content div
    html.Div(style=dict(display='grid', 
                        gridTemplateColumns='20% auto',
                        gridTemplateRows='70% auto',
                        height='100%'), 
             children=[
        # Left bar
        html.Div(style=dict(gridColumn='1 / 2', gridRow='1 / 3', borderRightStyle="solid"), children=[
            make_upload_div(), 
            make_columns_div(name='mapper-columns'),
            make_filter_div(), html.Hr(),
            make_cover_div(), html.Hr(),
            make_clustering_div(), html.Hr(),
            make_network_settings_div(),
            # Other settings
            html.Button('Run MAPPER', id='mapper-button', n_clicks=0)
        ]),

        # Network view
        make_network_view_div(style=dict(gridColumn='2 / 3', gridRow='1 / 2', paddingLeft='5px',
                                         paddingBottom='10px')),

        # Node information
        make_node_info_div(style=dict(gridColumn='2 / 3', gridRow='2 / 3', 
                                      borderTopStyle='solid'))
    ]),
    # Hidden divs for callback outputs
    html.Div(id='bit-bucket-1', style=dict(display='none')),
    # Hidden divs to store information
    html.Div(id='columns-store', style=dict(display='none'), children=json.dumps({})),
    html.Div(id='mapper-file-store', style=dict(display='none'), children='')
])

################################################################################
# Callbacks
################################################################################
# JavaScript callback to hide/show extra parameters for specific filters
app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='selectFilter'),
        Output('bit-bucket-1', 'children'),
        [Input('filter-dropdown', 'value')]
)

@app.callback(Output('mapper-upload-div', 'children'),
              [Input('mapper-upload', 'contents')],
              [State('mapper-upload', 'filename')])
def on_mapper_upload_change(contents, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    display = 'Selected file: {}'.format(filename)
    return display

@app.callback([Output('network-coloring-column-dropdown', 'options'),
               Output('network-coloring-column-dropdown', 'value'),
               Output('mapper-data-info-span', 'children'),
               Output('mapper-file-store', 'children'),
               Output('mapper-columns-dropdown', 'options'),
               Output('mapper-columns-dropdown', 'value')],
              [Input('mapper-upload-button', 'n_clicks')],
              [State('mapper-upload', 'contents'),
               State('mapper-upload', 'filename'),
               State('mapper-upload-check', 'value')])
def on_mapper_upload_click(n_clicks, contents, filename, options):
    ctx = dash.callback_context
    if (not ctx.triggered) or (contents == '') or (n_clicks == 0):
        return (dash.no_update,)*6

    options = handle_upload_options(options)
    df = contents_to_dataframe(contents, **options)

    info = '{}; {} rows, {} columns'.format(Path(filename).name, df.shape[0], df.shape[1])
    columns = [{'label': col, 'value': col} for col in df.columns]
    fpath = make_dataframe_token(df)

    return columns, columns[0]['value'], info, str(fpath), columns, [c['value'] for c in columns]

@app.callback([Output('mapper-graph', 'stylesheet'),
               Output('network-coloring-params-div', 'style'),
               Output('mapper-color-info-span', 'children')],
              [Input('network-coloring-dropdown', 'value'),
               Input('network-coloring-column-dropdown', 'value')],
              [State('mapper-graph', 'stylesheet'),
               State('columns-store', 'children')])
def on_network_coloring_change(coloring, column, stylesheet, columns):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*3

    div = []
    info = ''
    hide_show = dict(display='none')

    if coloring == 'density':
        stylesheet = []
        stylesheet.append(colorings['density'])
        info = 'density coloring'
    elif coloring == 'column':
        hide_show = dict(display='initial')
        if columns:
            columns = json.loads(columns)
            minv, maxv = columns[column]
            stylesheet = []
            stylesheet.append({
                'selector': 'node',
                'style': {
                    'background-color': 'mapData({}, {}, {}, blue, red)'.format(column, minv, maxv)
                }
            })
            info = '{} ({:.2f}, {:.2f}); '.format(column, minv, maxv)
    return stylesheet, hide_show, info

@app.callback([Output('mapper-graph', 'elements'),
               Output('columns-store', 'children'),
               Output('mapper-network-info-span', 'children')],
             [Input('mapper-button', 'n_clicks')],
             [State('mapper-file-store', 'children'),
              State('mapper-columns-dropdown', 'value'),
              State('filter-dropdown', 'value'),
              # Cover parameters
              State('cover-interval-input', 'value'),
              State('cover-overlap-input', 'value'),
              # Clustering Parameters
              State('cluster-method-dropdown', 'value'),
              State('cluster-metric-dropdown', 'value'),
              # Filter-specific parameters
              # tSNE parameters
              State('tsne-components-input', 'value'),
              # PCA parameters
              State('pca-components-input', 'value'),
              # Component filter parameters
              State('filter-component-input', 'value'),
              # Eccentricity parameters
              State('eccentricity-method-dropdown', 'value')
])
def on_run_mapper_click(n_clicks, fname, columns, filter_name, 
                        num_intervals, overlap,
                        clust_method, metric,
                        tsne_components, pca_components,
                        component_list, eccentricity_method):
    ctx = dash.callback_context
    if (not ctx.triggered) or (not fname):
        return (dash.no_update,) * 3
    
    elements = []

    df = load_cached_dataframe(fname)
    sub_df = df.loc[:, columns]

    n_components = tsne_components if filter_name == 'tsne' else pca_components
    filt = get_filter(filter_name, metric, n_components, component_list, eccentricity_method)
    f_x = filt(sub_df.values)

    cover = IntervalCover.EvenlySpacedFromValues(f_x, int(num_intervals), float(overlap) / 100)
    clust = HierarchicalClustering(method=clust_method, metric=metric)
    mapper = MAPPER(filter=filt, cover=cover, clustering=clust)
    result = mapper.run(sub_df.values, f_x=f_x)
    network = result.get_networkx_network()
    elements = networkx_network_to_cytoscape_elements(network, df)

    columns = {}
    for col in df.columns:
        vals = [d['data'][col] for d in elements if 'id' in d['data']]
        columns[col] = (min(vals), max(vals))

    info = '{} nodes, {} edges'.format(len(network.nodes), len(network.edges))

    return elements, json.dumps(columns), info

@app.callback(
        [Output('mapper-node-summary', 'columns'),
         Output('mapper-node-summary', 'data'),
         Output('mapper-node-data', 'columns'),
         Output('mapper-node-data', 'data')],
        [Input('mapper-graph', 'tapNodeData')],
        [State('mapper-file-store', 'children')])
def on_network_action(tapNodeData, fname):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,) * 4

    keys = frozenset(['id', 'npoints', 'points', 'density'])
    keys = frozenset(tapNodeData.keys()) - keys

    df = load_cached_dataframe(fname).iloc[tapNodeData['points'], :]

    summ_df = summarize_dataframe(df)
    summ_columns = [{'name': c, 'id': c} for c in summ_df.columns]
    summ_data = summ_df.to_dict('records')

    columns = [{'name': c, 'id': c} for c in df.columns]
    data = df.to_dict('records')
    return summ_columns, summ_data, columns, data

