import json
import dash
import os 
from dash import html, dcc, callback, Input, Output, State
import dash_table
import plotly.graph_objs as go

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set the directory path
directory = config['result_folder']
# Get a list of file names in the directory
file_names = os.listdir(directory)

layout = html.Div(children=[
    html.H1(children='Schema View'),
    html.H2(children='Schema Summary'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': name, 'value': name} for name in file_names],
        value=None
    ),
    html.Div(id='drop-output'),
    dash_table.DataTable(
        id='table-1-dynamic',
        columns=[
            {'name': 'Key', 'id': 'key'},
            {'name': 'Value', 'id': 'value'}
        ]
    ),
    html.Div(children=[
        html.Button("Generate Ontospy Hierarchy", id="ontospy-button")
    ]),
    html.H2(children='Class Details'),
    dash_table.DataTable(
        id='table-2-dynamic',
        columns=[
            {'name': 'Class', 'id': 'class'},
            {'name': 'Total Instances', 'id': 'total_instances'},
            {'name': 'Dominant Type', 'id': 'dominant_type'},
            {'name': 'Most Common Properties', 'id': 'most_common_properties'},
            {'name': 'Depth', 'id': 'depth'}, 
            {'name': 'Missingness Ratio', 'id': 'missingness_ratio'}
        ],
        style_data_conditional=[
        {
            'if': {'row_index': None},
            'backgroundColor': 'lightgrey'
        },
        ]
    ),
    html.H2(children='Class 2D View'),
    # create a scatter plot with the contents of 'coordinates' key from json file
    dcc.Graph(id='scatterplot-dynamic'), 
    html.Div(id='selected-dot', style={'display': 'none'})

])

@callback(
    Output('drop-output', 'children'),
    Input('file-dropdown', 'value')
)
def print_selection(selected_option):
    if selected_option is not None:
        return f'You selected "{selected_option}"'
    else:
        return 'Please select an option'

@callback(
    Output('table-1-dynamic', 'data'),
    Input('file-dropdown', 'value')
)

def update_table_1_dynamic(selected_file):
    if selected_file is not None:
        tmp_file = directory + selected_file
        with open(tmp_file, 'r') as f:
            tmp_data = json.load(f)
        table_data =[
            {'key': key, 'value': tmp_data['general_info'][key]} for key in tmp_data['general_info'].keys()
        ]
    return table_data

@callback(
    Output('table-2-dynamic', 'data'),
    Input('file-dropdown', 'value')
)
def update_table_2_dynamic(selected_file):
    if selected_file is not None:
        tmp_file = directory + selected_file
        with open(tmp_file, 'r') as f:
            tmp_data = json.load(f)
        table_data =[
            {
                'class': key,
                'total_instances': tmp_data[key]['total_instances'],
                'dominant_type': tmp_data[key]['dominant_type'],
                'most_common_properties': ', '.join(tmp_data[key]['most_common_properties']),
                'depth': tmp_data[key]['depth'],
                'missingness_ratio': round(tmp_data[key]['missingness_ratio_retrieve'], 2)
            }
            for key in tmp_data.keys() if key != 'coordinates' and key != 'general_info'
        ]
    return table_data

@callback(
    Output('scatterplot-dynamic', 'figure'),
    Input('file-dropdown', 'value')
)

def update_scatterplot_dynamic(selected_file):
    if selected_file is not None:
        tmp_file = directory + selected_file
        with open(tmp_file, 'r') as f:
            tmp_data = json.load(f)
        scatter_data =[
            go.Scatter(
                x=[key for key in tmp_data['coordinates']['x']],
                y=[key for key in tmp_data['coordinates']['y']],
                text=[key for key in tmp_data['coordinates']['labels']],
                mode='markers',
                marker=dict(
                    size=16,
                    color=tmp_data['coordinates']['labels'],
                    colorscale='Viridis',
                    opacity=0.7
                )
            )
        ]
        scatter_fig={
            'data': scatter_data,
            'layout': go.Layout(
                xaxis={'title': 'X Axis'},
                yaxis={'title': 'Y Axis'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest',
                clickmode='event+select'
            )
        }
    return scatter_fig



@callback(
    Output('selected-dot', 'children'),
    Input('scatterplot', 'clickData')
)
def update_selected_dot(clickData):
    if clickData is not None:
        print(clickData)
        return clickData['points'][0]['pointIndex']
        #return clickData

    else:
        return None

@callback(
    Output('table-2-dynamic', 'style_data_conditional'),
    Input('scatterplot-dynamic', 'clickData')
)
def update_style_data_conditional(clickData):
    if clickData is not None:
        return [
            {
                'if': {'row_index': clickData['points'][0]['pointIndex']},
                'backgroundColor': 'yellow'
            }
        ]
    else:
        return [
            {
                'if': {'row_index': None},
                'backgroundColor': 'lightgrey'
            },
        ]

# Callback to update data when the page is loaded or refreshed
@callback(
    Output('file-dropdown', 'options'),
    Input('url', 'pathname')
)
def update_data(_):
    global file_names  # Use a global variable to store the data
    file_names = os.listdir(directory)
    options=[{'label': name, 'value': name} for name in file_names]
    return options
