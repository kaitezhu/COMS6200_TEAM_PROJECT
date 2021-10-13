import dash
import json
import plotly.express as px
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from callbacks.callbacks import register_callbacks
import numpy as np

NAV_LOGO = "https://aaas.asn.au/wp-content/uploads/2020/03/UQ-Logo.png"

SIDEBAR_STYLE = {
    "position": "sticky",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    # "margin-left": "18rem",
    "margin-top": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    # "border": "1px solid #000",
    "box-shadow": "0 3px 10px rgb(0 0 0 / 0.2)",
}

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "COMS6200 Project"
server = app.server
app.config["suppress_callback_exceptions"] = True

# with open('../data/result.json') as f:
#     data = json.load(f)
# keys = ['Model','Train_time','Testing_time','Trained_accuracy','Testing_accuracy','TN','FP','FN','TP','Precision','recall','fpr','fnr','tnr','npv','fdr','for','f1',]
# data_full = {}
# for key in keys:
#     data_full[key] = []
# for i in data['81173']:
#     data_full['Model'].append(i)
# for i in data_full['Model']:
#     for j in data['81173'][i]:
#         data_full[j].append(data['81173'][i][j])
# # print(data)
# df = pd.DataFrame(data_full)
# print(df)





nav_bar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=NAV_LOGO, height="50px")),
                    dbc.Col(dbc.NavbarBrand("COMS6200 Project", className="ml-2")),

                    dbc.ButtonGroup(
                        [
                            dbc.Button("Home", id="home-btn"),
                            dbc.Button("Result 1", id="result1-btn"),
                            dbc.Button("Result 2", id="result2-btn"),
                        ],
                        style={
                            "position": "absolute",
                            "right": "2em",
                        },
                        id="page-id",
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
    ],
    color="dark",
    dark=True,
    # style={"margin-left": "20rem"}
)

side_p1 = html.Div(
    [
        html.H1("Project", className="display-4"),
        html.Hr(),
        html.P(
            "Page Content", className="lead",
        ),
        dbc.Nav(
            [
                dbc.Button(html.A('Project Definition', href="",
                                  style={"text-decoration": "none", "color": "#fff"}), color="primary"),
                html.Br(),
                dbc.Button(html.A('ML Models', href="#",
                                  style={"text-decoration": "none", "color": "#fff"}), color="primary"),
                html.Br(),
                dbc.Button(html.A('Datasets', href="",
                                  style={"text-decoration": "none", "color": "#fff"}), color="primary"),
                html.Br(),
                dbc.Button(html.A('link to bottom', href="",
                                  style={"text-decoration": "none", "color": "#fff"}), color="primary"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

side_p2 = html.Div(
    [
        html.H1("Result #1", className="display-4"),
        html.Hr(className="mb-4"),
        # html.Label("Select Your Models", className="mb-4"),
        dbc.Alert("Select Your Models", color="primary", className="mb-4"),
        dcc.Dropdown(
            options=[
                {'label': 'Decision Tree', 'value': '0'},
                {'label': 'Random Forest', 'value': '1'},
                {'label': 'Gradient Boosting', 'value': '2'},
                {'label': 'XGBoost', 'value': '3'},
            ],
            id="model-dropdown",
            value=[],
            multi=True,
            className="mb-4",
        ),
        dbc.Button("Confirm", color="primary", block=True, id="model-dropdown-btn"),
    ],
    style=SIDEBAR_STYLE,
)

side_p3 = html.Div(
    [
        html.H1("Result #2", className="display-4"),
        html.Hr(className="mb-4"),
        dbc.Alert("Select Your Columns", color="primary", className="mb-4"),
        dcc.Dropdown(
            options=[
                {'label': 'Training Time', 'value': '0'},
                {'label': 'Testing Time', 'value': '1'},
                {'label': 'Training Accuracy', 'value': '2'},
                {'label': 'Testing Accuracy', 'value': '3'},
                {'label': 'True Positive Rate', 'value': '4'},
                {'label': 'True Negative Rate', 'value': '5'},
                {'label': 'False Positive Rate', 'value': '6'},
                {'label': 'False Negative Rate', 'value': '7'},
                {'label': 'Precision', 'value': '8'},
                {'label': 'Recall', 'value': '9'},
                {'label': 'Negative Predictive Value', 'value': '10'},
                {'label': 'False Discovery Rate', 'value': '11'},
                {'label': 'False Omission Rate', 'value': '12'},
                {'label': 'F1', 'value': '13'},
            ],
            id="col-dropdown",
            value=['1', '3'],
            multi=True,
            className="mb-4",
        ),
        dbc.Button("Confirm", color="primary", block=True, id="col-dropdown-btn"),
    ],
    style=SIDEBAR_STYLE,
)

intro = [
    dbc.CardHeader("Machine Learning Based IDS"),
    dbc.CardBody(
        [
            # html.H5("Machine Learning Based IDS", className="card-title"),
            dbc.Row(
                [
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                    dbc.Col(
                        html.Img(src="https://static8.depositphotos.com/1026550/i/600/depositphotos_9546122-stock"
                                     "-photo-close-op-of-fiber-optics.jpg", height="100px"),
                        # width={"size": 2}
                    ),
                ]
            ),
        ]
    )
]

motivation = [
    dbc.CardHeader("Project Motivation"),
    dbc.CardBody(
        [
            # html.H5("Machine Learning Based IDS", className="card-title"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="https://static8.depositphotos.com/1026550/i/600/depositphotos_9546122-stock"
                                     "-photo-close-op-of-fiber-optics.jpg", height="100px"),
                        # width={"size": 2}
                    ),
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                ]
            ),
        ]
    ),
]

dt = [
    dbc.CardHeader("Decision Tree"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="https://i.ytimg.com/vi/ZVR2Way4nwQ/maxresdefault.jpg", height="100px"),
                        width={"size": 2}
                    ),
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                ]
            ),
        ]
    ),
]

rf = [
    dbc.CardHeader("Random Forest"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            src="https://lh4.googleusercontent.com/ANxp1CRZ8m4UohG8XJpvJusOpP-v-_7JLVwXXCpOAWwxdH2cEq0LRUym6WZgkdJNukBhUjCdBwnVYGm-gqXAn6YIEYM96CUbOHr0JhEAWawyygy5mShAiny3IRcdYmA63sxz26pT",
                            height="100px"),
                        width={"size": 2}
                    ),
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis. Pellentesque habitant morbi tristique senectus et netus et "
                            "malesuada fames ac turpis egestas. Etiam lacinia vestibulum dictum. Vivamus facilisis nibh ac eros "
                            "porta vulputate. Suspendisse potenti.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                ]
            ),
        ]
    ),
]

gb = [
    dbc.CardHeader("Gradient Boosting"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            src="https://gblobscdn.gitbook.com/assets%2F-LvBP1svpACTB1R1x_U4%2F-Lw6zezdliKWkGknCJ6R%2F-Lw70EB_T-Y3OCO-L_4o%2Fimage.png?alt=media&token=a3edaf4a-d3d2-4c84-9a10-3d870c21d641",
                            height="100px"),
                        width={"size": 2}
                    ),
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis. Pellentesque habitant morbi tristique senectus et netus et "
                            "malesuada fames ac turpis egestas. Etiam lacinia vestibulum dictum. Vivamus facilisis nibh ac eros "
                            "porta vulputate. Suspendisse potenti.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                ]
            ),
        ]
    ),
]

xgb = [
    dbc.CardHeader("XGBoost"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="https://miro.medium.com/max/1400/1*FLshv-wVDfu-i54OqvZdHg.png", height="100px"),
                        width={"size": 2}
                    ),
                    dbc.Col(
                        html.P(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque eget dui magna. Curabitur eget "
                            "mauris eget ex euismod auctor. Donec in massa non massa accumsan ornare. Mauris a luctus tortor. In "
                            "ornare ante nec mauris convallis mattis. Pellentesque habitant morbi tristique senectus et netus et "
                            "malesuada fames ac turpis egestas. Etiam lacinia vestibulum dictum. Vivamus facilisis nibh ac eros "
                            "porta vulputate. Suspendisse potenti.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                    ),
                ]
            ),
        ]
    ),
]

data_label_graph = go.Figure()
data_label_graph.add_trace(
    go.Pie(labels=['Normal Data', 'Intrusion Data'], values=[56000, 119341], hole=.3)
)

data_type_graph = go.Figure()
data_type_graph.add_trace(
    go.Pie(labels=['type 1', 'type 2', 'type 3', 'type 4'], values=[450, 790, 1000, 730], hole=.3)
)

data_label_card = [
    dbc.CardHeader("Dataset Label Distribution"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dcc.Graph(
                        id='data-label-graph',
                        figure=data_label_graph),
                ]
            ),
        ]
    )
]

data_type_card = [
    dbc.CardHeader("Data Types in Dataset"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dcc.Graph(
                        id='data_type_graph',
                        figure=data_type_graph),
                ]
            ),
        ]
    )
]

home_content = html.Div(
    [
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(intro, color="dark", inverse=True)),
                dbc.Col(dbc.Card(motivation, color="light"))
            ],
            className="mb-5",
        ),
        dbc.Row(
            [
                dbc.Button("See What's Inside Our Dataset", color="primary", block=True, id="data-card-toggle",
                           n_clicks=0),
            ],
            className="mb-5",
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(data_label_card)
                    ),
                    id="data-label-card-div",
                    style={"display": "none"}
                ),
                dbc.Col(
                    html.Div(
                        dbc.Card(data_type_card)
                    ),
                    id="data-type-card-div",
                    style={"display": "none"}
                ),
                # dbc.Col(dbc.Card(data_type_card)),
            ],
            id="data-card-row",
            className="mb-5",
            # style={"display": "none"}
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dt, color="light"))
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(rf, color="light"))
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(gb, color="light"))
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(xgb, color="light"))
            ],
            id="anchor-point"
        ),
    ]
)

result1_content = html.Div(
    [
        html.H1("Model Comparison"),
        html.Hr(),
        dcc.Graph(
            id='train-test-graph',
            # figure=train_test_fig,
        ),

        html.Br(),
        html.Hr(),
        dcc.Graph(
            id='acc-graph',
            # figure=acc_fig,
        ),
        html.Br(),
        html.Hr(),
        dcc.Graph(
            id='pre_rec-graph',
            # figure=pre_rec_fig,
        ),
        html.Br(),
        html.Hr(),
        dcc.Graph(
            id='dt-heatmap',
        ),
        dcc.Graph(
            id='rf-heatmap',
        ),
        dcc.Graph(
            id='gb-heatmap',
        ),
        dcc.Graph(
            id='xgb-heatmap',
        )

    ],
    style=CONTENT_STYLE,
)

result2_content = html.Div(
    [
        html.H1("Model Vulnerability Test with Different Data Volume"),
        html.Hr(className='mb-5'),
        dcc.Slider(
            min=100,
            max=82332,
            step=None,
            marks={
                100: '100',
                1000: '1000',
                5000: '5000',
                10000: '10000',
                30000: '30000',
                50000: '50000',
                82332: '82332'
            },
            value=5,
            className="mb-4",
        ),
        dcc.Graph(
            id='',
        )
    ],
    style=CONTENT_STYLE,
)

app.layout = html.Div(
    [
        dbc.Row(dbc.Col(
            nav_bar
        )),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(side_p1, id="side-div-p1"),
                        html.Div(
                            side_p2, id="side-div-p2", style={"display": "none"}
                        ),
                        html.Div(
                            side_p3, id="side-div-p3", style={"display": "none"}
                        ),
                    ],
                    width={"size": 2},
                ),
                dbc.Col(
                    [
                        html.Div(home_content, style={"display": "block"}, id="page-1"),
                        html.Div(result1_content, style={"display": "block"}, id="page-2"),
                        html.Div(result2_content, style={"display": "block"}, id="page-3"),
                    ],
                    width={"size": 9},
                ),
            ]
        ),
    ]
)

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
