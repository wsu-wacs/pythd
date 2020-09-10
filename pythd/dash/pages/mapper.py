from pathlib import Path
import base64, io

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app
from ..common import networkx_network_to_cytoscape_elements
from ...filter import IdentityFilter, ComponentFilter, ScikitLearnFilter
from ...cover import IntervalCover
from ...clustering import HierarchicalClustering 
from ...mapper import MAPPER

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
        html.Div(id='identity-params-div', children=[
        ], style=dict(display='none')),
        html.Div(id='component-params-div', children=[
             html.Span('Component List: '),
             dcc.Input(id='filter-component-input')
        ], style=dict(display='none')),
        html.Div(id='eccentricity-params-div', children=[
        ], style=dict(display='none'))
    ])

layout = html.Div([
    # Main content div
    html.Div(style=dict(display='grid', 
                        gridTemplateColumns='20% auto',
                        gridTemplateRows='100%'), 
             children=[
        # Left bar
        html.Div(style=dict(), children=[
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
            # Network layout
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
                    {'label': 'Point Density', 'value': 'density'}
                ]),
            # Other settings
            html.Button('Run MAPPER', id='mapper-button', n_clicks=0)
        ]),

        # Network view
        html.Div(style=dict(), children=[
            cyto.Cytoscape(id='mapper-graph',
                layout=dict(name='cose'),
                style=dict(width='100%', height='100%'),
                elements=[])
        ])
    ]),
    html.Div(id='bit-bucket-1', style=dict(display='none'))
])

################################################################################
# Functions
################################################################################
def contents_to_dataframe(contents):
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

def get_filter(name, metric, *args):
    if name == 'tsne':
        n_components = int(args[0])
        return ScikitLearnFilter(TSNE, n_components=n_components, metric=metric)
################################################################################
# Callbacks
################################################################################
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
    
    return 'Uploaded file: ' + filename

@app.callback(Output('mapper-graph', 'stylesheet'),
             [Input('network-coloring-dropdown', 'value')])
def on_network_coloring_change(coloring):
    stylesheet = []
    if coloring == 'density':
        stylesheet.append({
            'selector': 'node',
            'style': {
                'background-color': 'mapData(npoints, 0, 1 blue, red)'
            }
        })
    return stylesheet

@app.callback(Output('mapper-graph', 'elements'),
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
              # Component filter parameters
              State('filter-component-input', 'value'),
])
def on_run_mapper_click(n_clicks, contents, filter_name, 
                        num_intervals, overlap,
                        clust_method, metric,
                        *args):
    ctx = dash.callback_context
    if (not ctx.triggered) or (not contents):
        return dash.no_update
    
    elements = []

    df = contents_to_dataframe(contents)
    filt = get_filter(filter_name, metric, *args)
    f_x = filt(df.values)
    cover = IntervalCover.EvenlySpacedFromValues(f_x, int(num_intervals), float(overlap) / 100)
    clust = HierarchicalClustering(method=clust_method, metric=metric)
    mapper = MAPPER(filter=filt, cover=cover, clustering=clust)
    result = mapper.run(df.values, f_x=f_x)
    network = result.get_networkx_network()
    elements = networkx_network_to_cytoscape_elements(network, nrows=df.shape[0])

    return elements
