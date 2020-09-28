from pathlib import Path

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app
from ..common import *
from ..layout_common import *
from ...filter import *
from ...cover import IntervalCover
from ...clustering import HierarchicalClustering 
from ...mapper import MAPPER
from ...thd import THD

################################################################################
# Layout
################################################################################
layout = html.Div(style=dict(height='100%'), children=[
    # Main content div
    html.Div(style=dict(display='grid',
                        gridTemplateColumns='20% auto',
                        gridTemplateRows='80% auto',
                        height='100%'),
             children=[
        # Left bar
        html.Div(style=dict(gridColumn='1 / 2', gridRow='1 / 3', borderRightStyle='solid'), children=[
            make_upload_div(name='thd-upload'),
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
                html.Button('Run THD', id='thd-button', n_clicks=0)
            ]),
            # THD Tree view
            html.Div(style=dict(gridColumn='1 / 2', gridRow='2 / 3', 
                                borderTopStyle='solid'), children=[
                cyto.Cytoscape(id='thd-tree',
                    layout=dict(name='cose'), # TODO: tree layout
                    style=dict(width='100%', height='400px'),
                    stylesheet=[],
                    elements=[])
            ])
        ]),

        make_network_view_div(name='thd-mapper',
                              style=dict(gridColumn='2 / 3', gridRow='1 / 2',
                                         paddingLeft='5px', paddingBottom='10px')),

    ])
])

################################################################################
# Callbacks
################################################################################
@app.callback([Output('thd-upload-div', 'children'),
               Output('thd-mapper-coloring-column-dropdown', 'options'),
               Output('thd-mapper-coloring-column-dropdown', 'value'),
               Output('thd-mapper-data-info-span', 'children')],
              [Input('thd-upload', 'contents')],
              [State('thd-upload', 'filename')])
def on_thd_upload_change(contents, filename):
    """
    Called when a data file is uploaded
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,)*4

    display = 'Uploaded file: {}'.format(filename)
    df = contents_to_dataframe(contents)
    info = '{}; {} rows, {} columns'.format(Path(filename).name, df.shape[0], df.shape[1])
    columns = [{'label': col, 'value': col} for col in df.columns]
    return display, columns, columns[0]['value'], info

