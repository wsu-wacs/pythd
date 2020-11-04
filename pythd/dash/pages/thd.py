from pathlib import Path
import json

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash_table import DataTable
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app
from ..common import *
from ..layout_common import *
from ...filter import *
from ...complex import SimplicialComplex
from ...cover import IntervalCover
from ...clustering import HierarchicalClustering 
from ...mapper import MAPPER
from ...thd import THD
from ...utils import MinMaxScaler

################################################################################
# Layout
################################################################################
layout = html.Div(style=dict(height='100%'), children=[
    # Main content div
    html.Div(style=dict(display='grid',
                        gridTemplateColumns='20% auto',
                        gridTemplateRows='25% 25% auto auto',
                        height='100%'),
             children=[
        # Left bar
        html.Div(style=dict(gridColumn='1 / 2', gridRow='1 / 5', borderRightStyle='solid'), children=[
            make_upload_div(name='thd-upload'),
            make_columns_div(name='thd-columns'),
            html.H3('MAPPER Settings'),
            make_filter_div(name='thd-filter'),
            make_clustering_div(name='thd-cluster'),
            make_network_settings_div(name='thd-mapper'),
            make_cover_div(name='thd-cover'),
            # THD Settings
            html.Div([
                html.H3('THD Settings'),
                html.Span('Contract Amount: '),
                dcc.Input(id='thd-contract-input', debounce=True, inputMode='numeric',
                          min=0.001, max=1.0, value=0.1),
                html.Br(),
                html.Span('Group Threshold: '),
                dcc.Input(id='thd-threshold-input', debounce=True, inputMode='numeric',
                          min=1, value=100),
                html.Br(),
                html.Button('Run THD', id='thd-button', n_clicks=0),

                html.H3('THD Tree Settings'),
                html.Span('Tree coloring: '),
                dcc.Dropdown(id='thd-tree-color-dropdown',
                             searchable=False,
                             value='none',
                             options=[
                                 {'label': 'None', 'value': 'none'},
                                 {'label': 'Density', 'value': 'density'},
                                 {'label': 'Column', 'value': 'column'}
                             ]),
                html.Div(id='thd-tree-coloring-div', style=dict(display='none'), children=[
                    html.Span('Column:'),
                    dcc.Dropdown(id='thd-tree-column-dropdown',
                                 searchable=False,
                                 value='',
                                 options=[])
                ])
            ]),
            # THD Tree view
            html.Div(style=dict(borderTopStyle='solid'), children=[
                cyto.Cytoscape(id='thd-tree',
                    layout=dict(name='preset', fit=True), 
                    style=dict(width='100%', height='400px'),
                    userPanningEnabled=False,
                    userZoomingEnabled=False,
                    stylesheet=[],
                    elements=[])
            ])
        ]),

        make_network_view_div(name='thd-mapper',
                              style=dict(gridColumn='2 / 3', gridRow='1 / 3',
                                         paddingLeft='5px', paddingBottom='10px')),
        make_node_info_div(name='thd-mapper',
                           style=dict(gridColumn='2 / 3', gridRow='3 / 4',
                                      borderTopStyle='solid')),

        html.Div(style=dict(gridColumn='2 / 3', gridRow='4 / 5'), children=[
            html.H3('Group Selection'),
            html.Div(style=dict(display='grid', gridTemplateColumns='50% 50%'), children=[
                html.Div(style=dict(gridColumn='1 / 2'), children=[
                    html.H4('Summary'),
                    DataTable(id='thd-group-summary',
                              page_size=10)
                ]),
                html.Div(style=dict(gridColumn='2 / 3'), children=[
                ])
            ])
        ])

    ]),
    # Hidden divs for storage
    html.Div(id='thd-store', style=dict(display='none'), children=json.dumps({})),
    html.Div(id='thd-columns-store', style=dict(display='none'), children=json.dumps({})),
    html.Div(id='thd-file-store', style=dict(display='none'), children=''),
    # Hidden divs for callback outputs
    html.Div(id='thd-bitbucket-1', style=dict(display='none'))
])

################################################################################
# Functions
################################################################################
def serialize_thd(thd):
    d = thd.get_dict()
    return json.dumps(d)

def deserialize_thd(s):
    d = json.loads(s)
    return d

################################################################################
# Callbacks
################################################################################
# JavaScript callback to hide/show extra parameters for specific filters
app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='selectFilter'),
        Output('thd-bitbucket-1', 'children'),
        [Input('thd-filter-dropdown', 'value')]
)

@app.callback([Output('thd-mapper-graph', 'stylesheet'),
               Output('thd-mapper-coloring-params-div', 'style'),
               Output('thd-mapper-color-info-span', 'children')],
              [Input('thd-mapper-coloring-dropdown', 'value'),
               Input('thd-mapper-coloring-column-dropdown', 'value')],
              [State('thd-mapper-graph', 'stylesheet'),
               State('thd-columns-store', 'children')])
def on_thd_mapper_coloring_change(coloring, column, stylesheet, columns):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*3
    
    info = ''
    hide_show = dict(display='none')

    if coloring == 'none':
        stylesheet = []
        info = 'no coloring'
    elif coloring == 'density':
        stylesheet = [colorings['density']]
        info = 'density coloring'
    elif coloring == 'column':
        hide_show = dict(display='initial')
        if columns:
            columns = json.loads(columns)
            minv, maxv = columns[column]
            stylesheet = [{
                'selector': 'node',
                'style': {
                    'backgroundColor': 'mapData({}, {}, {}, blue, red)'.format(column, minv, maxv)
                }
            }]
            info = '{} ({:.2f}, {:.2f}); '.format(column, minv, maxv)
    return stylesheet, hide_show, info

@app.callback(Output('thd-upload-div', 'children'),
              [Input('thd-upload', 'filename')])
def on_thd_upload_change(filename):
    ctx = dash.callback_context
    if (not ctx.triggered) or (filename == ''):
        return ''

    display = 'Selected file: ' + str(filename)
    return display

@app.callback([Output('thd-mapper-coloring-column-dropdown', 'options'),
               Output('thd-mapper-coloring-column-dropdown', 'value'),
               Output('thd-mapper-data-info-span', 'children'),
               Output('thd-file-store', 'children'),
               Output('thd-columns-dropdown', 'options'),
               Output('thd-columns-dropdown', 'value')],
              [Input('thd-upload-button', 'n_clicks')],
              [State('thd-upload', 'contents'),
               State('thd-upload', 'filename'),
               State('thd-upload-check', 'value')])
def on_thd_upload_click(n_clicks, contents, filename, options):
    """
    Called when a data file is uploaded
    """
    ctx = dash.callback_context
    if (not ctx.triggered) or (contents == '') or (n_clicks == 0):
        return (dash.no_update,)*6

    options = handle_upload_options(options)
    df = contents_to_dataframe(contents, **options)

    info = '{}; {} rows, {} columns'.format(Path(filename).name, df.shape[0], df.shape[1])
    columns = [{'label': col, 'value': col} for col in df.columns]
    fpath = make_dataframe_token(df)

    return columns, columns[0]['value'], info, str(fpath), columns, [c['value'] for c in columns]

@app.callback([Output('thd-mapper-graph', 'elements'),
               Output('thd-columns-store', 'children'),
               Output('thd-group-summary', 'columns'),
               Output('thd-group-summary', 'data')],
              [Input('thd-tree', 'tapNodeData')],
              [State('thd-store', 'children'),
               State('thd-file-store', 'children')])
def on_thd_node_select(tapNodeData, groups, fname):
    """
    Called when a node in the THD tree is clicked
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*4

    elements = []
    columns = {}
    summ_columns = []
    summ_data = []

    group_name = tapNodeData['id']
    groups = deserialize_thd(groups).get('groups', {})
    if group_name in groups:
        group = groups[group_name]

        network = group['network']
        complex = SimplicialComplex()
        for s in network['simplices']:
            complex.add_simplex(s['simplex'], data=s['data'], **s['dict'])
        network = complex.get_networkx_network()

        df = load_cached_dataframe(fname)
        elements = networkx_network_to_cytoscape_elements(network, df)
        for col in df.columns:
            vals = [d['data'][col] for d in elements if 'id' in d['data']]
            columns[col] = (min(vals), max(vals))

        df = df.iloc[group['rids'], :]
        summ_df = summarize_dataframe(df)
        summ_columns = [{'name': c, 'id': c} for c in summ_df.columns]
        summ_data = summ_df.to_dict('records')

    return elements, json.dumps(columns), summ_columns, summ_data

@app.callback([Output('thd-tree', 'elements'),
               Output('thd-store', 'children'),
               Output('thd-tree-column-dropdown', 'options'),
               Output('thd-tree-column-dropdown', 'value')],
              [Input('thd-button', 'n_clicks')],
              [State('thd-file-store', 'children'),
               State('thd-columns-dropdown', 'value'),
               State('thd-filter-dropdown', 'value'),
               # Cover parameters
               State('thd-cover-interval-input', 'value'),
               State('thd-cover-overlap-input', 'value'),
               # Clustering parameters
               State('thd-cluster-method-dropdown', 'value'),
               State('thd-cluster-metric-dropdown', 'value'),
               # THD Parameters
               State('thd-contract-input', 'value'),
               State('thd-threshold-input', 'value'),
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
def on_run_thd_click(n_clicks, fname, columns, filter_name,
                     num_intervals, overlap,
                     clust_method, metric,
                     contract_amount, group_threshold,
                     tsne_components, pca_components,
                     component_list, eccentricity_method):
    """
    Called when a new THD is run
    """
    ctx = dash.callback_context
    if (not ctx.triggered) or (fname == ''):
        return (dash.no_update,)*4

    elements = []

    df = load_cached_dataframe(fname)
    sub_df = df.loc[:, columns]
    n_components = tsne_components if filter_name == 'tsne' else pca_components
    filt = get_filter(filter_name, metric, int(n_components), component_list, eccentricity_method)
    f_x = filt(sub_df.values)

    cover = IntervalCover.EvenlySpacedFromValues(f_x, int(num_intervals), float(overlap) / 100)
    clust = HierarchicalClustering(method=clust_method, metric=metric)
    thd = THD(sub_df, filt, cover, clust, float(contract_amount), int(group_threshold),
              full_df=df)

    group = thd.run()
    g = group.as_igraph_graph(True)
    avail_cols = frozenset(g.vs[0].attributes().keys()) & frozenset(df.columns)
    layout = g.layout_reingold_tilford()
    layout.scale(150)

    nrows = [v['num_rows'] for v in g.vs]
    rowsc = MinMaxScaler(min(nrows), max(nrows))
    cvs = {}
    for col in avail_cols:
        values = [v[col][1] for v in g.vs]
        cvs[col] = MinMaxScaler(min(values), max(values))

    for i, v in enumerate(g.vs):
        d = {
                'data': {'id': v['id'], 'nrows': v['num_rows'], 
                         'density': rowsc.scale(v['num_rows'])},
                'position': {'x': layout[i][0], 'y': layout[i][1]}
        }
        d['data'].update({col: cvs[col].scale(v[col][1])
                          for col in avail_cols})

        elements.append(d)

    for e in g.es:
        src = g.vs[e.source]
        tgt = g.vs[e.target]
        elements.append({'data': {'source': src['id'], 'target': tgt['id']}})

    columns = [{'label': col, 'value': col} for col in df.columns]

    return elements, serialize_thd(thd), columns, columns[0]['value']

@app.callback(
        [Output('thd-mapper-node-summary', 'columns'),
         Output('thd-mapper-node-summary', 'data'),
         Output('thd-mapper-node-data', 'columns'),
         Output('thd-mapper-node-data', 'data')],
        [Input('thd-mapper-graph', 'tapNodeData')],
        [State('thd-file-store', 'children')])
def on_thd_network_action(tapNodeData, fname):
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

@app.callback(
        [Output('thd-tree', 'stylesheet'), 
         Output('thd-tree-coloring-div', 'style')],
        [Input('thd-tree-color-dropdown', 'value'),
         Input('thd-store', 'children'),
         Input('thd-tree-column-dropdown', 'value')],
        [State('thd-tree', 'stylesheet')])
def handle_thd_tree_coloring(color_value, thd, column, stylesheet):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*2

    hideshow = dict(display='none')

    if color_value == 'none':
        return [], hideshow

    thd = json.loads(thd)
    if thd == {}:
        return stylesheet, hideshow

    if color_value == 'density':
        stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'background-color': 'mapData(density, 0, 1, blue, red)'
                }
            }
        ]
        return stylesheet, hideshow
    elif color_value == 'column':
        hideshow = dict(display='initial')
        if column == '':
            stylesheet = []
        else:
            stylesheet = [
                {
                    'selector': 'node',
                    'style': {
                        'background-color': 'mapData({}, 0, 1, blue, red)'.format(column)
                    }
                }
            ]
    else:
        stylesheet = []

    return stylesheet, hideshow

