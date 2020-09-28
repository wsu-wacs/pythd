import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from .app import app

from .pages import mapper as page_mapper
from .pages import thd as page_thd

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
             [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return []
    elif pathname == '/mapper':
        return page_mapper.layout
    elif pathname == '/thd':
        return page_thd.layout
    else:
        return 'page not found: {}'.format(pathname)

if __name__ == '__main__':
    app.run_server(debug=True)

