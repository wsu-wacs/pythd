from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State, ClientsideFunction

from ..app import app

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
    html.Div(style=dict(display='grid', gridTemplateColumns='20% auto'), 
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
            # Other settings
            html.Button('Run MAPPER', id='mapper-button', n_clicks=0)
        ]),

        # Network view
        html.Div(style=dict(), children=[
            cyto.Cytoscape(id='mapper-graph',
                layout=dict(name='preset'),
                style=dict(width='100%', height='100%'),
                elements=[])
        ])
    ]),
    html.Div(id='bit-bucket-1', style=dict(display='none'))
])

def make_filter_settings_div(filter_name):
    r=[]
    if filter_name == 'component':
        r = [html.Span('Component List: '),
             dcc.Input(id='filter-component-input')]
    return html.Div(children=r)

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

    p = Path(filename)
    if p.suffix == '.zip':
        pass # zip file
    elif p.suffix == '.csv':
        pass # csv file

    return 'Uploaded file: ' + filename


