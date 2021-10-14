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

CARD_HEADER = {
     "background-color": "#1D3557",
     "color": "#fff"
}

SUB_CARD_HEADER = {
    "background-color": "#24292d",
    "color": "#fff"
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
                    dbc.Col(dbc.NavbarBrand("COMS6200 IDS Project", style={"color": "#fff"}, className="ml-2")),
                    dbc.ButtonGroup(
                        [
                            dbc.Button("Home", id="home-btn", color="light"),
                            dbc.Button("Result 1", id="result1-btn", color="light"),
                            dbc.Button("Result 2", id="result2-btn", color="light"),
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
    color="#1D3557",
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
                dbc.Button(html.A('Project Definition', href="", style={"text-decoration": "none", "color": "#fff"}),
                           color="dark", className='mb-4'),
                dbc.Button(html.A('ML Models', href="", style={"text-decoration": "none", "color": "#fff"}),
                           color="dark", className='mb-4'),
                dbc.Button(html.A('Datasets', href="", style={"text-decoration": "none", "color": "#fff"}),
                           color="dark", className='mb-4'),
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
        dbc.Alert("Select Your Models", color="primary", className="mb-4"),
        dbc.Checklist(
            options=[
                {'label': 'Decision Tree', 'value': '0'},
                {'label': 'Random Forest', 'value': '1'},
                {'label': 'Gradient Boosting', 'value': '2'},
                {'label': 'XGBoost', 'value': '3'},
            ],
            id="model-checklist",
            value=['0', '3'],
            switch=True,
            className="mb-4",
        ),
        # dbc.Button("Confirm", color="primary", block=True, id="model-dropdown-btn"),
    ],
    style=SIDEBAR_STYLE,
)

side_p3 = html.Div(
    [
        html.H1("Result #2", className="display-4"),
        html.Hr(className="mb-4"),
        dbc.Alert("Select Your Columns", color="primary", className="mb-4"),
        dbc.Checklist(
            options=[
                {'label': 'F1 Score', 'value': '0'},
                {'label': 'Precision', 'value': '1'},
                {'label': 'Recall', 'value': '2'},
                {'label': 'Testing ACC', 'value': '3'},
                {'label': 'Balanced ACC', 'value': '4'},
            ],
            id="col-checklist",
            value=['0', '4'],
            switch=True,
            className="mb-4",
        ),
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
                            "This project's overall aim will be to produce the best possible machine learning model to "
                            "detect intrusion that is an improvement from existing models. Sub-goals will be involved "
                            "around the following parameters, focusing on the accuracy, recall, F1, construction time, "
                            "prediction time, and precision",
                            className="card-text ml-3 mr-5",
                            ),
                        # width={"size": 8}
                    ),
                    dbc.Col(
                        html.Img(src="https://miro.medium.com/max/1400/1*cRIbVMjOa7jEDS6yLd0lNg.jpeg", height="150px"),
                        # width={"size": 2}
                        className="ml-5",
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
                        html.P(
                            "Machine learning methods will essentially turn intrusion detection into classification"
                            " problems through modelling with different attributes(independent factors) of the associated "
                            "networking packets and identify intrusion among audit data. ",
                            className="card-text ml-3",
                        ),
                    ),
                    dbc.Col(
                        html.Img(src="https://img.deusm.com/informationweek/June21/machinelearning-WrightStudio-adobe.jpg", height="150px"),
                        # width={"size": 2}
                        className="ml-4",
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
                            "The decision tree is a tree structure used to reveal structured information in data. By "
                            "using this structure, a large record set can be divided into interconnected small "
                            "recordsets. Through each successive segmentation, the members in the result set become "
                            "more and more similar. Hyperparameters such as max, depth, splitter, minimum weight "
                            "fraction would be adjusted and tested to obtain the best prediction accuracy.The decision "
                            "tree algorithm can be used to visualize the data rules.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                        className="ml-3",
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
                            "Gradient Boosting is a kind of implementing Boosting. The core principle is that the "
                            "gradient descent direction of the loss function is set before researchers establish a "
                            "model. The loss function represents the varying degree of this model. If the value of "
                            "the loss function is larger, the model may generate more errors. Therefore, if the loss "
                            "function continues to decline, it indicates that the model's performance is steadily "
                            "improving.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                        className="ml-3",
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
                            "Gradient Boosting is a kind of implementing Boosting. The core principle is that the "
                            "gradient descent direction of the loss function is set before researchers establish a "
                            "model. The loss function represents the varying degree of this model. If the value of "
                            "the loss function is larger, the model may generate more errors. Therefore, if the loss "
                            "function continues to decline, it indicates that the model's performance is steadily "
                            "improving.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                        className="ml-3",
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
                            "XGBoost is based on decision trees, and it is an integrated machine learning algorithm. "
                            "This model applies a gradient boosting algorithm on a known data set and then classifies "
                            "the data accordingly. The idea of the Boosting algorithm is to integrate many weak "
                            "classifiers to form a more robust classifier.",
                            className="card-text",
                        ),
                        # width={"size": 8}
                        className="ml-3",
                    ),
                ]
            ),
        ]
    ),
]

data_label_graph = go.Figure()
data_label_graph.add_trace(
    go.Pie(labels=['Normal Data', 'Malicious Data'], values=[56000, 119341], hole=.3,
           marker=dict(colors=['#99ccff', '#cc99ff']))
)

data_type_graph = go.Figure()
data_type_graph.add_trace(
    go.Pie(labels=['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor',
                   'Shellcode', 'Worms'], values=[56000, 40000, 33393, 18184, 12264, 10491, 2000, 1746, 1133, 130],
           hole=.3,
           marker=dict(
               colors=['#e6f2ff', '#99ccff', '#ccccff', '#cc99ff', '#ff99ff', '#ff6699', '#ff9966', '#ff6600',
                       '#ff5050', '#ff0000']
           ))
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
                dbc.Col(dbc.Card(intro)),
                dbc.Col(dbc.Card(motivation))
            ],
            className="mb-5",
        ),
        dbc.Row(
            [
                dbc.Button("See What's Inside Our Dataset", color="danger", block=True, outline=True, id="data-card-toggle",
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
        html.H1("Model Comparison using Full Dataset"),
        html.Hr(),
        dcc.Graph(id='f1-graph', className='mb-4'),
        html.Hr(),
        dcc.Graph(id='roc-graph', className='mb-4'),
        html.Hr(),
        dcc.Graph(id='acc-graph', className='mb-4'),
        html.Hr(),
        dcc.Graph(id='pre_rec-graph', className='mb-4'),
        html.Hr(),
        dcc.Graph(id='train-test-graph', className='mb-4'),
        html.Hr(),
        dcc.Graph(id='dt-heatmap'),
        dcc.Graph(id='rf-heatmap'),
        dcc.Graph(id='gb-heatmap'),
        dcc.Graph(id='xgb-heatmap')

    ],
    style=CONTENT_STYLE,
)

result2_content = html.Div(
    [
        html.H1("Model Performance vs. Data Volume"),
        html.Hr(className='mb-5'),
        html.Br(),
        dcc.Slider(
            min=100,
            max=81173,
            step=None,
            marks={
                100: '100',
                1000: '1000',
                5000: '5000',
                10000: '10000',
                30000: '30000',
                50000: '50000',
                81173: '81173'
            },
            id="volume-slider",
            value=100,
            className="mb-4",
        ),
        dcc.Graph(
            id='slider-graph',
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
                        html.Div(result1_content, style={"display": "none"}, id="page-2"),
                        html.Div(result2_content, style={"display": "none"}, id="page-3"),
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
