from urllib.parse import parse_qs
from argparse import ArgumentParser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from .app import app
from .config import *

from .pages import mapper as page_mapper
from .pages import thd as page_thd
from .pages.compare import make_group_comparison_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Span('[ '),
    dcc.Link('home', href='/'),
    html.Span(' ] [ '),
    dcc.Link('mapper', href='/mapper'),
    html.Span(' ] [ '),
    dcc.Link('thd', href='/thd'),
    html.Span(' ]'),
    html.Hr(),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
             [Input('url', 'pathname')],
             [State('url', 'search'),
              State('url', 'hash')])
def display_page(pathname, search, hashv):
    if pathname == '/':
        return []
    elif pathname == '/mapper':
        return page_mapper.layout
    elif pathname == '/thd':
        return page_thd.layout
    elif pathname == '/compare':
        if search[0] == '?':
            search = search[1:]
        qs = {k: v[0] for k,v in parse_qs(search).items()}
        qs['g1'] = [int(i) for i in qs.get('g1', '').split(',')]
        qs['g2'] = [int(i) for i in  qs.get('g2', '').split(',')]
        return make_group_comparison_page(qs['file'], qs['g1'], qs['g2'], qs['name1'], qs['name2']) 
    else:
        return 'page not found: {}'.format(pathname)

if __name__ == '__main__':
    parser = ArgumentParser(description='Python THD Dashboard')
    parser.add_argument('-d', '--debug', action='store_true', help='Run Dash in debug mode')
    args = parser.parse_args()

    # Initialize uploaded data cache
    DATA_DIR.mkdir(exist_ok=True)
    for p in DATA_DIR.glob('*.pkl'):
        p.unlink()

    app.run_server(debug=args.debug)

