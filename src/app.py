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
#
# df_full = pd.DataFrame.from_records(data['82332'])
# dtc_full = df_full['DTC']
# rfc_full = df_full['RFC']
# gb_full = df_full['GB']
# xgb_full = df_full['XGB']
#
# model = ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
# train_time = []
# test_time = []
# train_acc = []
# test_acc = []
# precision = []
# recall = []
#
# # train_time = dtc_full['Train_time'].__add__(rfc_full['Train_time'])
# train_time = np.append(dtc_full['Train_time'], rfc_full['Train_time'], gb_full['Train_time'], xgb_full['Train_time'])
# # gb_full['Train_time'] + xgb_full['Train_time']
# print(train_time)
#
# # df_full.reset_index(level=0, inplace=True)


df = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"],
    "Training Time": [810, 1120, 18290, 290],
    "Testing Time": [40, 150, 1080, 40],
    "Training ACC": [99.21, 99.61, 98.52, 97.94],
    "Testing ACC": [90.61, 90.83, 91.23, 91.61],
    "Precision": [96.78, 98.21, 97.47, 97.11],
    "Recall": [89.16, 88.13, 89.43, 90.37],
    "FPR": [6.32, 3.42, 4.95, 5.74],
    "FNR": [10.84, 11.87, 10.57, 9.63],
    "TPR": [66.98, 66.04, 66.72, 67.14],
    "TNR": [6.32, 3.42, 4.95, 5.74],
})

train_test_fig = go.Figure()
train_test_fig.add_trace(go.Bar(
    y=df["Training Time"],
    x=df["Model"],
    name="Training Time",
    marker_color='#778899',
    opacity=0.95,
    text=df["Training Time"],
    textposition='outside',
))

train_test_fig.add_trace(go.Bar(
    y=df["Testing Time"],
    x=df["Model"],
    name="Testing Time",
    marker_color='#dc143c',
    opacity=0.95,
    text=df["Testing Time"],
    textposition='outside',
))

train_test_fig.update_layout(
    # xaxis_title_text="Model Name",
    yaxis_title_text="Time (ms)",
    bargap=0.2,
    bargroupgap=0.1,
    title={
        'text': 'Training & Testing Time',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    uniformtext_minsize=8,
    uniformtext_mode='hide',
)

acc_fig = go.Figure()

acc_fig.add_trace(go.Bar(
    x=df["Training ACC"],
    y=df["Model"],
    name="Training ACC",
    marker_color="#119dff",
    # marker_color='#EB89B5',
    opacity=0.95,
    orientation='h',
    text=df["Training ACC"].apply(lambda x: round(x, 2)),
    textposition='outside',
))

acc_fig.add_trace(go.Bar(
    x=df["Testing ACC"],
    y=df["Model"],
    name="Testing ACC",
    marker_color="#66c2a5",
    # marker_color='#330C73',
    opacity=0.95,
    orientation='h',
    text=df["Testing ACC"].apply(lambda x: round(x, 2)),
    textposition='outside',
))

acc_fig.update_layout(
    xaxis_title_text="Percentage",
    # yaxis_title_text="Percentage",
    bargap=0.5,
    bargroupgap=0.1,
    yaxis_tickangle=-45,
    title={
        'text': 'Training & Testing Accuracy',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    uniformtext_minsize=8,
    uniformtext_mode='hide',
)

pre_rec_fig = go.Figure()

pre_rec_fig.add_trace(go.Bar(
    x=df["Precision"],
    y=df["Model"],
    name="Precision",
    marker_color='#EB89B5',
    opacity=0.95,
    orientation='h',
    text=df["Precision"].apply(lambda x: round(x, 2)),
    textposition='outside',
))

pre_rec_fig.add_trace(go.Bar(
    x=df["Recall"],
    y=df["Model"],
    name="Recall",
    marker_color='#330C73',
    opacity=0.95,
    orientation='h',
    text=df["Recall"].apply(lambda x: round(x, 2)),
    textposition='outside',
))

pre_rec_fig.update_layout(
    xaxis_title_text="Percentage",
    bargap=0.5,
    bargroupgap=0.1,
    yaxis_tickangle=-45,
    title={
        'text': 'Precision & Recall',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    uniformtext_minsize=8,
    uniformtext_mode='hide',
)

# [FNR + TPR] + [TNR + FPR]
dt = df[df['Model'] == 'Decision Tree']
dt_fnr = df[df['Model'] == 'Decision Tree']['FNR'].values
dt_tpr = df[df['Model'] == 'Decision Tree']['TPR'].values
dt_tnr = df[df['Model'] == 'Decision Tree']['TNR'].values
dt_fpr = df[df['Model'] == 'Decision Tree']['FPR'].values
# print(dt_fnr, dt_tpr, dt_tnr, dt_fpr)


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

side_bar = html.Div(
    [
        html.H6("Project", className="display-4"),
        html.Hr(),
        html.P(
            "Page Content", className="lead",
        ),
        dbc.Nav(
            [
                dbc.Button(html.A('Project Definition', href="#anchor-point",
                                  style={"text-decoration": "none", "color": "#555"}), color="info", outline=True),
                html.Br(),
                dbc.Button(html.A('ML Models', href="#anchor-point",
                                  style={"text-decoration": "none", "color": "#555"}), color="info", outline=True),
                html.Br(),
                dbc.Button(html.A('Datasets', href="#anchor-point",
                                  style={"text-decoration": "none", "color": "#555"}), color="info", outline=True),
                html.Br(),
                dbc.Button(html.A('link to bottom', href="#anchor-point",
                                  style={"text-decoration": "none", "color": "#555"}), color="info", outline=True),

            ],
            vertical=True,
            pills=True,
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
            figure=train_test_fig,
        ),

        html.Br(),
        html.Hr(),
        dcc.Graph(
            id='acc-graph',
            figure=acc_fig,
        ),
        html.Br(),
        html.Hr(),
        dcc.Graph(
            id='pre_rec-graph',
            figure=pre_rec_fig,
        ),
        html.Br(),
        html.Hr(),

    ],
    id="page-content",
    style=CONTENT_STYLE,
)

result2_content = html.Div(
    [
        "RESULT 2 CONTENT"
    ]
)

app.layout = html.Div(
    [
        dbc.Row(dbc.Col(
            nav_bar
        )),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(side_bar),
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
