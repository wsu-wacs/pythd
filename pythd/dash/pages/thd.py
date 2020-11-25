from pathlib import Path
from urllib.parse import urlencode
import json

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash_table import DataTable
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.express as px

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
                        gridTemplateRows='25% 25% auto auto'),
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
                make_dropdown(cid='thd-tree-color-dropdown',
                             options=[
                                 {'label': 'None', 'value': 'none'},
                                 {'label': 'Density', 'value': 'density'},
                                 {'label': 'Column', 'value': 'column'}]),
                html.Div(id='thd-tree-coloring-div', style=dict(display='none'), children=[
                    html.Span('Column:'),
                    make_dropdown(cid='thd-tree-column-dropdown')])
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
            html.Div(id='group-selection-summary'),
            html.Div(style=dict(display='grid', gridTemplateColumns='50% 50%'), children=[
                html.Div(style=dict(gridColumn='1 / 2'), children=[
                    html.H4('Summary'),
                    DataTable(id='thd-group-summary',
                              page_size=10,
                              **DATATABLE_STYLE)
                ]),
                html.Div(style=dict(gridColumn='2 / 3'), children=[
                ])
            ])
        ]),

    ]),
    html.Hr(),
    # Group comparison
    html.Div([
        html.H2('Group Comparison'),
        html.Div(style=dict(display='grid', gridTemplateColumns='25% 25% 25% 25%'), children=[
            html.Div(style=dict(gridColumn='1 / 3'), children=[
                html.H3('First Group'),
                dcc.Dropdown(id='group1-dropdown', clearable=False),
                html.Br()
            ]),
            html.Div(style=dict(gridColumn='3 / 5'), children=[
                html.H3('Second Group'),
                dcc.Dropdown(id='group2-dropdown'),
                html.Br()
            ]),
            html.Div(style=dict(gridColumn='2 / 4', textAlign='center'), children=[
                html.Button('Compare', id='group-compare-button', n_clicks=0)
            ]),
            html.Div(style=dict(gridColumn='1 / 5'), children=[
                html.Hr(),
                html.H2('Scatter Plots')
            ]),
            html.Br(),

            html.Div(style=dict(gridColumn='1 / 3'), children=[
                html.Span('Horizontal axis:'),
                make_dropdown(cid='thd-group-viz-column1')
            ]),
            html.Div(style=dict(gridColumn='3 / 5'), children=[
                html.Span('Vertical axis:'),
                make_dropdown(cid='thd-group-viz-column2')
            ]),
            html.Br(),

            dcc.Graph(id='thd-group-viz1', style=dict(gridColumn='1 / 3')),
            dcc.Graph(id='thd-group-viz2', style=dict(gridColumn='3 / 5'))
        ])
    ]),

    html.Br(), html.Br(), html.Hr(),
    # Hidden divs for storage
    html.Div(id='thd-store', style=dict(display='none'), children=json.dumps({})),
    html.Div(id='thd-columns-store', style=dict(display='none'), children=json.dumps({})),
    html.Div(id='thd-file-store', style=dict(display='none'), children=''),
    html.Div(id='group-compare-url', style=dict(display='none'), children=''),
    # Hidden divs for callback outputs
    html.Div(id='thd-bitbucket-1', style=dict(display='none')),
    html.Div(id='thd-bitbucket-2', style=dict(display='none'))
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

# Javascript callback to open the compare groups page
app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='compareGroups'),
        Output('thd-bitbucket-2', 'children'),
        [Input('group-compare-url', 'children')]
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
               Output('thd-columns-dropdown', 'value'),
               Output('thd-group-viz-column1', 'options'),
               Output('thd-group-viz-column1', 'value'),
               Output('thd-group-viz-column2', 'options'),
               Output('thd-group-viz-column2', 'value')],
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
        return (dash.no_update,)*10

    options = handle_upload_options(options)
    df = contents_to_dataframe(contents, **options)

    info = '{}; {} rows, {} columns'.format(Path(filename).name, df.shape[0], df.shape[1])
    columns = [{'label': col, 'value': col} for col in df.columns]
    fpath = make_dataframe_token(df)

    return (columns, columns[0]['value'], info, str(fpath), 
            columns, [c['value'] for c in columns],
            columns, columns[0]['value'], columns, columns[-1]['value'])

@app.callback([Output('thd-mapper-graph', 'elements'),
               Output('thd-columns-store', 'children'),
               Output('thd-group-summary', 'columns'),
               Output('thd-group-summary', 'data'),
               Output('group-selection-summary', 'children')],
              [Input('thd-tree', 'tapNodeData')],
              [State('thd-store', 'children'),
               State('thd-file-store', 'children')])
def on_thd_node_select(tapNodeData, groups, fname):
    """
    Called when a node in the THD tree is clicked
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*5

    elements = []
    columns = {}
    summ_columns = []
    summ_data = []
    summ_div = []

    group_name = tapNodeData['id']
    thd = deserialize_thd(groups)
    print(thd)
    groups = thd.get('groups', {})
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
        summ_columns, summ_data = make_datatable_info(summarize_dataframe(df))

        summ_div = [
                html.Div('Group name: ' + group_name),
                html.Div('Num. points: {}'.format(df.shape[0]))
        ]

    return elements, json.dumps(columns), summ_columns, summ_data, summ_div

@app.callback([Output('thd-tree', 'elements'),
               Output('thd-store', 'children'),
               Output('thd-tree-column-dropdown', 'options'),
               Output('thd-tree-column-dropdown', 'value'),
               Output('group1-dropdown', 'options'),
               Output('group2-dropdown', 'options'),
               Output('group1-dropdown', 'value'),
               Output('group2-dropdown', 'value')],
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
        return (dash.no_update,)*8

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
    layout = g.layout_reingold_tilford(root=[0])
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

    options = [{'label': 'All of data', 'value': 'all'}, {'label': 'Rest of data', 'value': 'rest'}]
    options += [{'label': g.get_name(), 'value': g.get_name()} for g in group]

    return (elements, serialize_thd(thd), columns, columns[0]['value'], 
            options, options, options[2]['value'], options[1]['value'])

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
    summ_columns, summ_data = make_datatable_info(summarize_dataframe(df))
    columns, data = make_datatable_info(df)

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

@app.callback(Output('group-compare-url', 'children'),
              [Input('group-compare-button', 'n_clicks')],
              [State('thd-file-store', 'children'),
               State('group1-dropdown', 'value'),
               State('group2-dropdown', 'value'),
               State('thd-store', 'children')])
def on_group_compare_click(n_clicks, fname, group1, group2, groups):
    ctx = dash.callback_context
    if (not ctx.triggered) or (n_clicks == 0):
        return dash.no_update

    groups = deserialize_thd(groups).get('groups', {})
    if not ('0.0.0' in groups):
        return dash.no_update
    
    g1rel = group1 in ['all', 'rest']
    g2rel = group2 in ['all', 'rest']
    if g1rel and g2rel:
        return dash.no_update

    group1, group2, g1name, g2name = get_comparison_groups(group1, group2, groups)

    query={
        'file': fname,
        'g1': ','.join(map(str, list(group1))),
        'g2': ','.join(map(str, list(group2))),
        'name1': g1name,
        'name2': g2name
    }
    url = '/compare?' + urlencode(query)
    return url

@app.callback([Output('thd-group-viz1', 'figure'),
               Output('thd-group-viz2', 'figure')],
              [Input('thd-group-viz-column1', 'value'),
               Input('thd-group-viz-column2', 'value')],
              [State('group1-dropdown', 'value'),
               State('group2-dropdown', 'value'),
               State('thd-store', 'children'),
               State('thd-file-store', 'children')])
def on_scatter_column_select(column1, column2, group1, group2, groups, fname):
    ctx = dash.callback_context
    if (not ctx.triggered) or (column1 == '') or (column2 == ''):
        return (dash.no_update,)*2

    groups = deserialize_thd(groups).get('groups', {})
    if not ('0.0.0' in groups):
        return (dash.no_update,)*2

    g1rel = group1 in ['all', 'rest']
    g2rel = group2 in ['all', 'rest']
    if g1rel and g2rel:
        return (dash.no_update,)*2

    group1, group2, g1name, g2name = get_comparison_groups(group1, group2, groups)
    group1 = list(group1)
    group2 = list(group2)

    df = load_cached_dataframe(fname)
    df1 = df.iloc[group1, :]
    df2 = df.iloc[group2, :]
    df = None

    fig1 = px.scatter(df1, x=column1, y=column2)
    fig2 = px.scatter(df2, x=column1, y=column2)
    return fig1, fig2

