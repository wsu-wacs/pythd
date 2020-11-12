"""
Dynamic page for comparing two groups
"""
import dash
import dash_core_components as dcc
import dash_html_components as html

from ..app import app
from ..common import *
from ..config import *

def make_group_comparison_page(fname, g1, g2):
    df = load_cached_dataframe(fname)
    df1 = df.iloc[g1, :]
    df2 = df.iloc[g2, :]
    df = None

    res = [
        html.H1('Group Comparison'),

        html.H2('Group Summaries'),
        html.Div(style=dict(display='grid', gridTemplateColumns='50% 50%'), children=[
            html.Div(style=dict(gridColumn='1 / 2', children=[
            ]),
            html.Div(style=dict(gridColumn='2 / 3', children=[
            ])
        ])
    ]

    return res

