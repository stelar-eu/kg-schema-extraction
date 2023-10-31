import dash
import dash_bootstrap_components as dbc

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from pages.view_clustering import layout as view_clustering_layout
from pages.view_comparison import layout as view_comparison_layout
from pages.request_page import layout as request_page_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Location(id="url", refresh=False),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Submit Request", href="/request-page")),
                dbc.NavItem(dbc.NavLink("View Clustering", href="/view-clustering")),
                dbc.NavItem(dbc.NavLink("View Comparison", href="/view-comparison")),
            ],
            brand="KGSEC Dashboard",
            color="primary",
            dark=True,
        ),
        dbc.Container(id="page-content", className="mt-4"),
    ],
)

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/view-clustering":
        return view_clustering_layout
    elif pathname == "/view-comparison":
        return view_comparison_layout
    elif pathname == "/request-page":
        return request_page_layout
    else:
        return "404 Page Not Found"


if __name__ == "__main__":
    app.run_server(debug=False)