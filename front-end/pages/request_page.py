import dash
from dash import html, dcc, callback, Input, Output, State
import requests

layout = html.Div(
    className="container",
    children=[
        html.H1(children="Schema Extraction Request Page"),
        html.Div(
            className="form",
            children=[
                html.H2(children="Knowledge Graph - Dataset"),
                dcc.Dropdown(
                    id="kg-graph", 
                    options=[
                        {"label": "Nobel Prize Dataset", "value": "nobel"}
                        ],
                    value="nobel",
                ),
                html.H2(children="Chracteristic Set Extraction Method"),
                dcc.Dropdown(
                    id="cs-method", 
                    options=[
                        {"label": "Undirected-Properties", "value": "undirected-properties"},
                        {"label": "Directed-Properties", "value": "directed-properties"},
                        {"label": "Directed-Properties + Types", "value": "directed-properties-types"},
                    ],
                    value="directed-properties",
                ),
                html.H2(children="Extraction Method"),
                dcc.Dropdown(
                    id="extraction-method", 
                    options=[
                        {"label": "DBSCAN", "value": "dbscan"},
                        {"label": "HDBSCAN", "value": "hdbscan"},
                        {"label": "Simple Frequent Itemset", "value": "frequent-itemset"},
                    ],
                    value="dbscan",
                ),
                dcc.Store(id="form-data"),
                html.Div(id="form-fields"),
                html.H2(children="Similarity Function"),
                dcc.Dropdown(
                    id="similarity-function",
                    options=[
                        {"label": "Cosine", "value": "cosine"},
                        {"label": "Euclidean", "value": "euclidean"},
                        {"label": "Jaccard", "value": "jaccard"},
                    ],
                    value="cosine",
                ),
                html.Button("Submit", id="submit-button"),
            ]
        ),
        html.Div(id="output-div"),
    ]
)

@callback(
    Output("form-data", "data"),
    Input("extraction-method", "value"),
)
def update_form_data(extraction_method):
    if extraction_method == "dbscan":
        return {"fields": ["Epsilon", "Min Samples"]}
    elif extraction_method == "hdbscan":
        return {"fields": ["Min Cluster Size"]}
    elif extraction_method == "frequent-itemset":
        return {"fields": ["Min Global Support", "Max Interclass Similarity"]}
    else:
        return {"fields": []}


@callback(
    Output("form-fields", "children"),
    Input("form-data", "data"),
)
def update_form_fields(form_data):
    if form_data is None:
        return []
    fields = form_data["fields"]
    input_fields = [
        html.Div(
            className="form-field",
            children=[
                html.Label(children=field),
                dcc.Input(id=f"input-{i}", type="text", placeholder=f"Enter {field}", className="form-input")
            ]
        ) for i, field in enumerate(fields)
    ]
    if len(input_fields) > 0:
        input_fields.insert(0, html.H2(children="Additional Parameters"))
    return input_fields

@callback(
    dash.dependencies.Output("output-div", "children"),
    dash.dependencies.Input("submit-button", "n_clicks"),
    dash.dependencies.State("cs-method", "value"),
    dash.dependencies.State("extraction-method", "value"),
    dash.dependencies.State("form-fields", "children")
)
def store_form_data(n_clicks, cs_method, extraction_method, extra_params):
    if n_clicks is None:
        return dash.no_update
    extra_param_dict = {}
    for i in extra_params:
        if type(i["props"]["children"][1]) != str:
            temporary_key = i["props"]["children"][1]['props']['id']
            temporary_value = i["props"]["children"][1]['props']['value']
            extra_param_dict[temporary_key] = temporary_value
    fields = {"cs-method": cs_method, "extraction-method": extraction_method}
    fields.update(extra_param_dict)
    response = requests.post('http://127.0.0.1:5000/', json=fields)
    if response.status_code == 200:
        return f"Form submitted and processed with: {cs_method}, and {extraction_method}"
    else:
        return "Error submitting form"
