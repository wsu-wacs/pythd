"""
Common layouts for all MAPPER dashboards
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash_table import DataTable

__all__ = ['make_filter_params', 'make_upload_div', 'make_filter_div', 'make_columns_div',
           'make_cover_div', 'make_clustering_div', 'make_network_settings_div',
           'make_network_view_div', 'colorings', 'make_node_info_div', 'DATATABLE_STYLE', 'make_dropdown',
           'make_misc_settings_div']

colorings = {
    'density': {
        'selector': 'node',
        'style': {
            'background-color': 'mapData(density, 0, 1, blue, red)'
        }
    }
}

DATATABLE_STYLE = {
    'style_table': {
        'overflowX': 'auto'
    },

    'style_cell': {
        'height': 'auto',
        'whiteSpace': 'normal'
    },

    'style_data': {
        'whiteSpace': 'normal',
        'height': 'auto'
    }
}

def make_dropdown(cid, clearable=False, multi=False, searchable=False, options=[], value=None, **kwargs):
    """
    Make a dropdown with the most common settings used across the dashboard.

    The remaining parameters are the same as the dash_core_components.Dropdown method

    Parameters
    ----------
    cid : str
        The id of the dropdown

    Returns
    -------
    dash_core_components.Dropdown
    """
    if value is None and len(options) > 0:
        value = options[0].get('value')

    return dcc.Dropdown(id=cid, clearable=clearable, multi=multi, searchable=searchable,
                        options=options, value=value, **kwargs)

def make_filter_params():
    """
    Make layout divs for MAPPER filter parameters.

    Returns
    -------
    dash_html_components.Div
        The filter parameters div
    """
    return html.Div(id='filter-params-div', children=[
        html.Div(id='tsne-params-div', children=[
            html.Span('Num. components: '),
            dcc.Input(id='tsne-components-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=2)
        ], style=dict(display='none')),
        html.Div(id='umap-params-div', children=[
            html.Span('Num. components: '),
            dcc.Input(id='umap-components-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=2)
        ], style=dict(display='none')),
        html.Div(id='pca-params-div', children=[
            html.Span('Num. components: '),
            dcc.Input(id='pca-components-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=2)
        ]),
        html.Div(id='identity-params-div', children=[
        ], style=dict(display='none')),
        html.Div(id='component-params-div', children=[
             html.Span('Component List: '),
             dcc.Input(id='filter-component-input')
        ], style=dict(display='none')),
        html.Div(id='eccentricity-params-div', children=[
            html.Span('Method: '),
            make_dropdown(cid='eccentricity-method-dropdown', options=[
                {'label': 'Mean', 'value': 'mean'},
                {'label': 'Medoid', 'value': 'medoid'}]),
        ], style=dict(display='none'))
    ])

def make_upload_div(name='mapper-upload', style={}):
    """
    Make the Div for uploading a CSV dataset

    Parameters
    ----------
    name : str
        Base name for the Upload component and prefix for any children
    style : dict
        Dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
        html.H3('Data'),
        dcc.Upload(id=name,
          children=html.Div([
            html.Div(id=name+'-div', children='Drop a file here or click to select file.'),
            html.Button('Select file...', id=name+'-button', n_clicks=0)
        ])),
        dcc.Checklist(
            id=name+'-check',
            options=[{'label': 'No index column', 'value': 'no_index'},
                     {'label': 'No header', 'value': 'no_header'}],
            value=[]),
        html.Span('Remove columns:'),
        make_dropdown(cid=name+'-remove-dropdown', clearable=True, multi=True, searchable=True),
        html.Button(
            'Upload',
            id=name+'-button',
            n_clicks=0)
    ])

def make_columns_div(name='mapper-columns', style={}):
    """
    Make div with dropdown for selecting columns to use in MAPPER
    """
    return html.Div(style=style, children=[
        html.H3('Columns'),
        dcc.Dropdown(id=name+'-dropdown',
                     multi=True,
                     placeholder='Select columns')
    ])

def make_misc_settings_div(name='misc', style={}):
    return html.Div(style=style, children=[
        html.H4('Data Normalization'),
        make_dropdown(cid=name+'-normalize',
            options=[
                {'label': 'No normalization', 'value': 'none'},
                {'label': 'Min-max normalization', 'value': 'minmax'},
                {'label': 'Max-abs normalization', 'value': 'maxabs'},
                {'label': 'Standard scaling', 'value': 'standard'},
                {'label': 'Robust scaling', 'value': 'robust'}],
            value='none')
    ])

def make_filter_div(name='filter', style={}):
    """
    Make the div for filter selection and parameters

    Parameters
    ----------
    name : str
        Base name for the dropdown and any children
    style : dict
        Dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
            html.H4('Filter'),
            make_dropdown(cid=name + '-dropdown',
                value='pca',
                options=[
                    {'label': 'tSNE', 'value': 'tsne'},
                    {'label': 'UMAP', 'value': 'umap'},
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 'Identity', 'value': 'identity'},
                    {'label': 'Component', 'value': 'component'},
                    {'label': 'Eccentricity', 'value': 'eccentricity'}]),
            make_filter_params(),
    ])

def make_cover_div(name='cover', style={}):
    """
    Make the div for selecting the MAPPER cover parameters

    Parameters
    ----------
    name : str
        Base name used as a prefix
    style : dict
        Dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
            html.H4('Cover'),
            html.Span('Num. Intervals: '),
            dcc.Input(id=name + '-interval-input',
                      debounce=True,
                      inputMode='numeric',
                      min=1,
                      value=5),
            html.Br(),
            html.Span('Percent Overlap: '),
            dcc.Input(id=name + '-overlap-input',
                      debounce=True,
                      inputMode='numeric',
                      min=0.0,
                      max=100.0,
                      value=15.0)])

def make_clustering_div(name='cluster', style={}):
    """
    Make the div for cluster and metric selection

    Parameters
    ----------
    name : str
        Base name used as a prefix
    style : dict
        Dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
            html.H4('Clustering'),
            html.Span('Method: '),
            make_dropdown(cid=name + '-method-dropdown',
                        options=[
                            {'label': 'Single Linkage', 'value': 'single'},
                            {'label': 'Complete Linkage', 'value': 'complete'},
                            {'label': 'Average Linkage', 'value': 'average'},
                            {'label': 'Weighted', 'value': 'weighted'},
                            {'label': 'Centroid', 'value': 'centroid'},
                            {'label': 'Median', 'value': 'median'},
                            {'label': 'Ward', 'value': 'ward'}]),
            html.Span('Metric: '),
            make_dropdown(cid=name + '-metric-dropdown',
                options=[
                    {'label': 'Euclidean', 'value': 'euclidean'},
                    {'label': 'Manhattan', 'value': 'manhattan'},
                    {'label': 'Standardized Euclidean', 'value': 'seuclidean'},
                    {'label': 'Cosine', 'value': 'cosine'},
                    {'label': 'Correlation', 'value': 'correlation'},
                    {'label': 'Chebyshev', 'value': 'chebyshev'}])])

def make_network_settings_div(name='network', style={}):
    """
    Make the div for network layout algorithm and coloring

    Parameters
    ----------
    name : str
        Prefix to use for ids
    style : dict
        The dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
            html.H4('Network View'),
            html.Span('Layout Algorithm: '),
            make_dropdown(cid=name + '-layout-dropdown',
                options=[
                    {'label': 'COSE', 'value': 'cose'}]),
            html.Span('Node Coloring: '),
            make_dropdown(cid=name + '-coloring-dropdown',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': 'Point Density', 'value': 'density'},
                    {'label': 'Column', 'value': 'column'}]),
            html.Div(id=name + '-coloring-params-div', style=dict(display='none'), children=[
                html.Span('Column: '),
                dcc.Dropdown(id=name + '-coloring-column-dropdown', options=[], clearable=False)
            ])
    ])

def make_network_view_div(name='mapper', style={}):
    """
    Make the div containing the Cytoscape network view for MAPPER

    Parameters
    ----------
    name : str
        Prefix to use for ids
    style : dict
        The dash style dict for the div

    Returns
    -------
    dash_html_components.Div
    """
    return html.Div(style=style, children=[
        html.Div(style=dict(borderBottomStyle='solid'), children=[
            html.Span(id=name + '-data-info-span', children='No file loaded.'),
            html.Span(id=name + '-network-info-span', style=dict(float='right'), children=[]),
            html.Span(id=name + '-color-info-span', style=dict(float='right'), children=[])
        ]),
        html.Div(style=dict(display='flex', flexDirection='column', height='100%', width='100%',
                            paddingBottom='2pt'), 
                 children=[
                    cyto.Cytoscape(id=name + '-graph',
                        layout=dict(name='cose'),
                        responsive=True,
                        style=dict(width='100%', height='98%', zIndex=999),
                        stylesheet=[colorings['density']],
                        elements=[])
        ])
    ])

def make_node_info_div(name='mapper', style={}):
    return html.Div(style=style, children=[
        html.H3('Node Selection'),
        html.Div(style=dict(display='grid', gridTemplateColumns='50% 50%'), children=[
            html.Div(style=dict(gridColumn='1 / 2'), children=[
                html.H4('Summary'),
                DataTable(id=name+'-node-summary',
                          page_size=10,
                          columns=[],
                          data=[],
                          **DATATABLE_STYLE)
            ]),
            html.Div(style=dict(gridColumn='2 / 3'), children=[
                html.H4('Data'),
                DataTable(id=name+'-node-data',
                          page_size=10,
                          columns=[],
                          data=[],
                          **DATATABLE_STYLE)
    ])])])

