import json
import dash
import os 
from dash import html, dcc, callback, Input, Output, State
import dash_table
import plotly.graph_objs as go
import requests

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set the directory path
# directory = '/Users/petrosskoufis/Desktop/Everything_Else/Petros/Αθηνά/Stelar/Code/'
directory = config['result_folder']

# Get a list of file names in the directory
file_names = os.listdir(directory)

layout = html.Div(children=[
    html.H1(children='Comparison View'),
    html.H2(children='Schemata Selection'),
    html.Div(children=[
        html.H3(children='Schema 1 Selection'),
        dcc.Dropdown(
            id='file-1-dropdown',
            options=[{'label': name, 'value': name} for name in file_names],
            value=None
        ),

    ], style={'display': 'inline-block', 'width': '50%'}),
    html.Div(children=[
        html.H3(children='Schema 2 Selection'),
        dcc.Dropdown(
            id='file-2-dropdown',
            options=[{'label': name, 'value': name} for name in file_names],
            value=None
        ),
    ], style={'display': 'inline-block', 'width': '50%'}),
    html.Div(children=[
        html.Button("Compare", id="compare-button")
    ]),
    html.Div(children=[
        html.H1(children='Schema 1 View'),
        html.H2(children='Schema Summary'),
        # create a table with the contents of 'general_info' key from json file
        dash_table.DataTable(
            id='comp-table-1-dynamic',
            columns=[
                {'name': 'Key', 'id': 'key'},
                {'name': 'Value', 'id': 'value'}
            ]
    ),],style={'display': 'inline-block', 'width': '50%'}),
    html.Div(children=[
        html.H1(children='Schema 2 View'),
        html.H2(children='Schema Summary'),
        # create a table with the contents of 'general_info' key from json file
        dash_table.DataTable(
            id='comp-table-2-dynamic',
            columns=[
                {'name': 'Key', 'id': 'key'},
                {'name': 'Value', 'id': 'value'}
            ]
    ),],style={'display': 'inline-block', 'width': '50%'}),
    html.H2(children='Clustering Quality Scores'),
    dash_table.DataTable(
        id='scores-table-dynamic',
        columns=[
            {'name': 'Metric', 'id': 'metric'},
            {'name': 'Score', 'id': 'score'}
        ]
    ),
    html.H2(children='Class Distribution Diagram'),
    dcc.Graph(id='sankey-graph-dynamic'),
    html.Div(id='selected-rows', children=[
        html.H2('Selected Rows'),
        dash_table.DataTable(
            id='selected-rows-table',
            columns=[
                {'name': 'Schema', 'id': 'schema'},
                {'name': 'Class', 'id': 'class'},
                {'name': 'Total Instances', 'id': 'total_instances'},
                {'name': 'Dominant Type', 'id': 'dominant_type'},
                {'name': 'Most Common Properties', 'id': 'most_common_properties'},
                {'name': 'Depth', 'id': 'depth'}, 
                {'name': 'Missingness Ratio', 'id': 'missingness_ratio'}
            ],
            data=[]
        )
    ]),
    html.H2(children='Class 2D View'),
    # create a scatter plot with the contents of 'coordinates' key from json file
    dcc.Graph(id='scatterplot-comp-dynamic'),
    html.Div(id='selected-dot-comp', style={'display': 'none'})

])


def update_general_table(selected_file): 
    if selected_file is not None:
        tmp_file = directory + selected_file
        with open(tmp_file, 'r') as f:
            tmp_data = json.load(f)
        table_data =[
            {'key': key, 'value': tmp_data['general_info'][key]} for key in tmp_data['general_info'].keys()
        ]
    return table_data

@callback(
    Output('comp-table-1-dynamic', 'data'),
    Input('file-1-dropdown', 'value')
)

def update_comp_table_1(selected_file):
    return update_general_table(selected_file) 

@callback(
    Output('comp-table-2-dynamic', 'data'),
    Input('file-2-dropdown', 'value')
)

def update_comp_table_2(selected_file):
    return update_general_table(selected_file) 


@callback(
    Output('selected-dot-comp', 'children'),
    Input('scatterplot-comp', 'clickData')
)
def update_selected_dot(clickData):
    if clickData is not None:
        selected_dot = clickData['points'][0]['text']
        print(clickData)
        print(f"Selected dot: {selected_dot}")
        return selected_dot
    else:
        return None

@callback(
    Output('selected-rows-table', 'data'),
    Input('scatterplot-comp-dynamic', 'clickData'), 
    State('file-1-dropdown','value'),
    State('file-2-dropdown','value'),
    State('selected-rows-table', 'data')
)
def update_selected_rows_table(clickData, selected_file_1, selected_file_2, data):
    tmp_file_1 = directory + selected_file_1
    with open(tmp_file_1, 'r') as f:
        data_1 = json.load(f)
    tmp_file_2 = directory + selected_file_2
    with open(tmp_file_2, 'r') as f:
        data_2 = json.load(f)
    if clickData is not None:
        if clickData['points'][0]['curveNumber'] == 0:
            key = 'Cluster_' + str(clickData['points'][0]['pointIndex'])
            selected_schema = 'Schema 1'
            new_row = [{'schema': 'Schema 1', 
                        'class': key,
                        'total_instances': data_1[key]['total_instances'],
                        'dominant_type': data_1[key]['dominant_type'],
                        'most_common_properties': ', '.join(data_1[key]['most_common_properties']),
                        'depth': data_1[key]['depth'],
                        # round missingness_ratio_retrieve to 2 decimal places
                        'missingness_ratio': round(data_1[key]['missingness_ratio_retrieve'], 2)
                        }
            ]
            # check if new row is already in data
            for datum in data:
                if (datum['class'] == key) and (datum['schema'] == 'Schema 1'):
                    return data
            data = data + new_row
            return data
        elif clickData['points'][0]['curveNumber'] == 1:
            key = 'Cluster_' + str(clickData['points'][0]['pointIndex'])
            selected_schema = 'Schema 2'
            new_row = [{'schema': 'Schema 2', 
                        'class': key,
                        'total_instances': data_2[key]['total_instances'],
                        'dominant_type': data_2[key]['dominant_type'],
                        'most_common_properties': ', '.join(data_2[key]['most_common_properties']),
                        'depth': data_2[key]['depth'],
                        # round missingness_ratio_retrieve to 2 decimal places
                        'missingness_ratio': round(data_2[key]['missingness_ratio_retrieve'], 2)
                        }
            ]
            for datum in data:
                if (datum['class'] == key) and (datum['schema'] == 'Schema 2'):
                    return data
            data = data + new_row
            return data
        return data
    else:
        return []

@callback(
    Output('scatterplot-comp-dynamic', 'figure'),
    Input('file-1-dropdown', 'value'),
    Input('file-2-dropdown', 'value')
)

def update_scatterplot_comp(selected_file_1, selected_file_2):
    if selected_file_1 is not None:
        tmp_file_1 = directory + selected_file_1
        with open(tmp_file_1, 'r') as f:
            tmp_data_1 = json.load(f)
    if selected_file_2 is not None:
        tmp_file_2 = directory + selected_file_2
        with open(tmp_file_2, 'r') as f:
            tmp_data_2 = json.load(f)
    figure={
        'data': [
            go.Scatter(
                x=[key for key in tmp_data_1['coordinates']['x']],
                y=[key for key in tmp_data_1['coordinates']['y']],
                text=[key for key in tmp_data_1['coordinates']['labels']],
                mode='markers',
                marker=dict(
                    size=16,
                    color='blue',
                    opacity=0.7
                ), 
                name = 'Schema 1'
            ), 
            go.Scatter(
                x=[key for key in tmp_data_2['coordinates']['x']],
                y=[key for key in tmp_data_2['coordinates']['y']],
                text=[key for key in tmp_data_2['coordinates']['labels']],
                mode='markers',
                marker=dict(
                    size=16,
                    color='red',
                    opacity=0.7
                ), 
                name = 'Schema 2'
            ), 
        ],
        'layout': go.Layout(
            xaxis={'title': 'X Axis'},
            yaxis={'title': 'Y Axis'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            clickmode='event+select'
        )
    }
    return figure

# Callback to update data when the page is loaded or refreshed
@callback(
    Output('file-dropdown-1', 'options'),
    Output('file-dropdown-2', 'options'),
    Input('url', 'pathname')
)
def update_data(_):
    global file_names  # Use a global variable to store the data
    file_names = os.listdir(directory)
    options=[{'label': name, 'value': name} for name in file_names]
    return options

@callback(
    Output('scores-table-dynamic', 'data'),
    Output('sankey-graph-dynamic', 'figure'),
    Input("compare-button", "n_clicks"),  
    State("file-1-dropdown", "value"),
    State("file-2-dropdown", "value")
)

def generate_comparison(n_clicks, file_1, file_2):
    if n_clicks is None or file_1 is None or file_2 is None:
        return dash.no_update
    fields = {}
    fields['file_1'] = file_1
    fields['file_2'] = file_2
    response = requests.post('http://127.0.0.1:5000/compare', json=fields)
    # convert response data to dict 
    response_dict = response.json()
    comp_location = response_dict['comp_location']

    with open(comp_location, 'r') as f:
        comparison_data = json.load(f)
    data=[
        {'metric': 'AMI', 'score': comparison_data['ami']},
        {'metric': 'ARI', 'score': comparison_data['ari']}
    ]

    figure={
        'data': [
            go.Sankey(
                node = {
                    'label': comparison_data['sankey']['label']
                },
                link = {
                    'source': comparison_data['sankey']['source'],
                    'target': comparison_data['sankey']['target'],
                    'value': comparison_data['sankey']['value'],

                }
            )
        ],
        'layout': go.Layout(
            title='Sankey Diagram',
        )
    }

    return data, figure