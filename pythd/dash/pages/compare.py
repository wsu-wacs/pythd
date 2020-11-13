"""
Dynamic page for comparing two groups
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

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
        DataTable(page_size=10, columns=summ_cols, data=summ_data, **DATATABLE_STYLE)
    ]

    return res

