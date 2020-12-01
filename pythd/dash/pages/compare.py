"""
Dynamic page for comparing two groups
"""
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash.dependencies import Input, Output, State, ClientsideFunction

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import scipy as sp
from scipy.stats import ks_2samp
import pandas as pd

from ..app import app
from ..common import *
from ..layout_common import *
from ..config import *

def make_group_comparison_page(fname, g1, g2, name1, name2):
    df = load_cached_dataframe(fname)
    df1 = df.iloc[g1, :]
    df2 = df.iloc[g2, :]
    df = None

    cols1, data1 = make_datatable_info(summarize_dataframe(df1))
    cols2, data2 = make_datatable_info(summarize_dataframe(df2))

    d = {
        'column': [],
        'ks_statistic': [],
        'p_value': [],
        'group1_mean': [],
        'group2_mean': []
    }
    for c in df1.columns:
        d1 = df1[c]
        d2 = df2[c]
        K, p = ks_2samp(d1, d2)

        d['column'].append(c)
        d['ks_statistic'].append(K)
        d['p_value'].append(p)
        d['group1_mean'].append(d1.mean())
        d['group2_mean'].append(d2.mean())

    summ_df = pd.DataFrame(d)
    summ_cols, summ_data = make_datatable_info(summ_df)

    res = [
        html.H1('Group Comparison'),
        html.Span('Select feature:'),
        make_dropdown('compare-dropdown', options=[{'label': c, 'value': c} for c in df1.columns]),
        dcc.Graph(id='group-compare-boxplot'),

        html.H2('Group Summaries'),
        html.Div(style=dict(display='grid', gridTemplateColumns='50% 50%'), children=[
            html.Div(style=dict(gridColumn='1 / 2'), children=[
                html.H3('Group 1 ({})'.format(name1)),
                html.Span('{} rows'.format(df1.shape[0])),
                DataTable(page_size=10, columns=cols1, data=data1, **DATATABLE_STYLE)
            ]),
            html.Div(style=dict(gridColumn='2 / 3'), children=[
                html.H3('Group 2 ({})'.format(name2)),
                html.Span('{} rows'.format(df2.shape[0])),
                DataTable(page_size=10, columns=cols2, data=data2, **DATATABLE_STYLE)
            ])
        ]),

        html.H2('KS Test Results'),
        DataTable(page_size=10, columns=summ_cols, data=summ_data, **DATATABLE_STYLE),

        html.Div(id='compare-g1-store', style=dict(display='none'), children=json.dumps(g1)),
        html.Div(id='compare-name1-store', style=dict(display='none'), children=name1),
        html.Div(id='compare-g2-store', style=dict(display='none'), children=json.dumps(g2)),
        html.Div(id='compare-name2-store', style=dict(display='none'), children=name2),
        html.Div(id='compare-fname-store', style=dict(display='none'), children=fname)
    ]

    return res

@app.callback(Output('group-compare-boxplot', 'figure'),
              [Input('compare-dropdown', 'value')],
              [State('compare-fname-store', 'children'),
               State('compare-g1-store', 'children'),
               State('compare-g2-store', 'children'),
               State('compare-name1-store', 'children'),
               State('compare-name2-store', 'children')])
def on_feature_select(value, fname, g1, g2, name1, name2):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    g1 = json.loads(g1)
    g2 = json.loads(g2)

    df = load_cached_dataframe(fname)
    d1 = df.iloc[g1, :]
    d1 = d1[value]
    d2 = df.iloc[g2, :]
    d2 = d2[value]

    fig = go.Figure()
    fig.add_trace(go.Box(name=name1, y=d1))
    fig.add_trace(go.Box(name=name2, y=d2))

    return fig.to_dict()

