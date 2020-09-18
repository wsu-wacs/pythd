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
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app
from ..common import *
from ...filter import *
from ...cover import IntervalCover
from ...clustering import HierarchicalClustering 
from ...mapper import MAPPER

################################################################################
# Constants & Variables
################################################################################
colorings = {
    'density': {
        'selector': 'node',
        'style': {
            'background-color': 'mapData(density, 0, 1, blue, red)'
        }
    }
}

################################################################################
# Layout
################################################################################
def make_filter_params():
    return html.Div(id='filter-params-div', children=[
        html.Div(id='tsne-params-div', children=[
            html.Span('Num. components: '),
            dcc.Input(id='tsne-components-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=2)
        ], style=dict(display='none')),
        html.Div(id='pca-params-div', children=[
            html.Span('Num. components: '),
            dcc.Input(id='pca-components-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=2)
        ]),
        html.Div(id='identity-params-div', children=[
        ], style=dict(display='none')),
        html.Div(id='component-params-div', children=[
             html.Span('Component List: '),
             dcc.Input(id='filter-component-input')
        ], style=dict(display='none')),
        html.Div(id='eccentricity-params-div', children=[
            html.Span('Method: '),
            dcc.Dropdown(id='eccentricity-method-dropdown',
                searchable=False,
                value='mean',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Medoid', 'value': 'medoid'}
                ]),
        ], style=dict(display='none'))
    ])

def make_column_dropdown(columns, name='column-dropdown'):
    return dcc.Dropdown(id=name,
                        searchable=False,
                        value=columns[0],
                        options=[
                            {'label': col, 'value': col}
                            for col in columns.keys()])

layout = html.Div(style=dict(height='100%'), children=[
    # Main content div
    html.Div(style=dict(display='grid', 
                        gridTemplateColumns='20% auto',
                        height='100%'), 
             children=[
        # Left bar
        html.Div(style=dict(gridColumn='1 / 2', borderRightStyle="solid"), children=[
            # Dataset settings
            html.H4('Data'),
            dcc.Upload(id='mapper-upload',
                children=html.Div([
                    html.Div(id='mapper-upload-div', children='Drop a file here or click to select file.'),
                    html.Button('Select file...', id='mapper-upload-button', n_clicks=0)
            ])),
            html.Hr(),
            # Filter settings
            html.H4('Filter'),
            dcc.Dropdown(id='filter-dropdown',
                searchable=False,
                value='tsne',
                options=[
                    {'label': 'tSNE', 'value': 'tsne'},
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 'Identity', 'value': 'identity'},
                    {'label': 'Component', 'value': 'component'},
                    {'label': 'Eccentricity', 'value': 'eccentricity'}
                ]),
            make_filter_params(),
            html.Hr(),
            # Cover settings
            html.H4('Cover'),
            html.Span('Num. Intervals: '),
            dcc.Input(id='cover-interval-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=5),
            html.Br(),
            html.Span('Percent Overlap: '),
            dcc.Input(id='cover-overlap-input',
                      debounce=True,
                      inputMode='numeric',
                      min=0.0,
                      max=100.0,
                      value=15.0),
            html.Hr(),
            # Clustering settings
            html.H4('Clustering'),
            html.Span('Method: '),
            dcc.Dropdown(id='cluster-method-dropdown',
                searchable=False,
                value='single',
                options=[
                    {'label': 'Single Linkage', 'value': 'single'},
                    {'label': 'Complete Linkage', 'value': 'complete'},
                    {'label': 'Average Linkage', 'value': 'average'},
                    {'label': 'Weighted', 'value': 'weighted'},
                    {'label': 'Centroid', 'value': 'centroid'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Ward', 'value': 'ward'}
                ]),
            html.Span('Metric: '),
            dcc.Dropdown(id='cluster-metric-dropdown',
                searchable=False,
                value='euclidean',
                options=[
                    {'label': 'Euclidean', 'value': 'euclidean'},
                    {'label': 'Manhattan', 'value': 'manhattan'},
                    {'label': 'Standardized Euclidean', 'value': 'seuclidean'},
                    {'label': 'Cosine', 'value': 'cosine'},
                    {'label': 'Correlation', 'value': 'correlation'},
                    {'label': 'Chebyshev', 'value': 'chebyshev'}
                ]),
            html.Hr(),
            # Network layout and coloring
            html.H4('Network View'),
            html.Span('Layout Algorithm: '),
            dcc.Dropdown(id='layout-method-dropdown',
                value='cose',
                options=[
                    {'label': 'COSE', 'value': 'cose'}
                ]),
            html.Span('Node Coloring: '),
            dcc.Dropdown(id='network-coloring-dropdown',
                value='density',
                options=[
                    {'label': 'Point Density', 'value': 'density'},
                    {'label': 'Column', 'value': 'column'}
                ]),
            html.Div(id='network-coloring-params-div', style=dict(display='none'), children=[
                html.Span('Column: '),
                dcc.Dropdown(id='coloring-column-dropdown', options=[])
            ]),
            # Other settings
            html.Button('Run MAPPER', id='mapper-button', n_clicks=0)
        ]),

        # Network view
        html.Div(style=dict(gridColumn='2 / 3', paddingLeft='5px', paddingBottom='10px'), children=[
            html.Div(style=dict(borderBottomStyle='solid'), children=[
                html.Span(id='data-info-span', children='No file loaded.'),
                html.Span(id='network-info-span', style=dict(float='right'), children=[]),
                html.Span(id='color-info-span', style=dict(float='right'), children=[])
            ]),
            cyto.Cytoscape(id='mapper-graph',
                layout=dict(name='cose'),
                style=dict(width='100%', height='100%'),
                stylesheet=[colorings['density']],
                elements=[])
        ])
    ]),
    # Hidden divs for callback outputs
    html.Div(id='bit-bucket-1', style=dict(display='none')),
    # Hidden divs to store information
    html.Div(id='columns-store', style=dict(display='none'), children=json.dumps({}))
])

################################################################################
# Functions
################################################################################
def get_filter(name, metric, *args):
    if name == 'tsne':
        n_components = int(args[0])
        return ScikitLearnFilter(TSNE, n_components=n_components, metric=metric)
    elif name == 'pca':
        n_components = int(args[1])
        return ScikitLearnFilter(PCA, n_components=n_components)
    elif name == 'identity':
        return IdentityFilter()
    elif name == 'component':
        components = args[2]
        return ComponentFilter(components)
    elif name == 'eccentricity':
        method = args[3]
        return EccentricityFilter(metric=metric, method=method)

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

@app.callback([Output('mapper-upload-div', 'children'),
               Output('coloring-column-dropdown', 'options'),
               Output('coloring-column-dropdown', 'value'),
               Output('data-info-span', 'children')],
              [Input('mapper-upload', 'contents')],
              [State('mapper-upload', 'filename')])
def on_mapper_upload_change(contents, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*4
    
    display = 'Uploaded file: {}'.format(filename)
    df = contents_to_dataframe(contents)
    info = '{}; {} rows, {} columns'.format(Path(filename).name, df.shape[0], df.shape[1])
    columns = [{'label': col, 'value': col} for col in df.columns]
    return display, columns, columns[0]['value'], info

@app.callback([Output('mapper-graph', 'stylesheet'),
               Output('network-coloring-params-div', 'style'),
               Output('color-info-span', 'children')],
              [Input('network-coloring-dropdown', 'value'),
               Input('coloring-column-dropdown', 'value')],
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
               Output('network-info-span', 'children')],
             [Input('mapper-button', 'n_clicks')],
             [State('mapper-upload', 'contents'),
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
def on_run_mapper_click(n_clicks, contents, filter_name, 
                        num_intervals, overlap,
                        clust_method, metric,
                        *args):
    ctx = dash.callback_context
    if (not ctx.triggered) or (not contents):
        return (dash.no_update,) * 3
    
    elements = []

    df = contents_to_dataframe(contents)
    filt = get_filter(filter_name, metric, *args)
    f_x = filt(df.values)
    cover = IntervalCover.EvenlySpacedFromValues(f_x, int(num_intervals), float(overlap) / 100)
    clust = HierarchicalClustering(method=clust_method, metric=metric)
    mapper = MAPPER(filter=filt, cover=cover, clustering=clust)
    result = mapper.run(df.values, f_x=f_x)
    network = result.get_networkx_network()
    elements = networkx_network_to_cytoscape_elements(network, df)

    columns = {}
    for col in df.columns:
        vals = [d['data'][col] for d in elements if 'id' in d['data']]
        columns[col] = (min(vals), max(vals))

    info = '{} nodes, {} edges'.format(len(network.nodes), len(network.edges))

    return elements, json.dumps(columns), info
