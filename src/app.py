import dash
import plotly.express as px
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from callbacks.callbacks import register_callbacks

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
    "FPR": [6.32, 3.42, 4.95, 5.74],
})

train_test_fig = go.Figure()
train_test_fig.add_trace(go.Bar(
    y=df["Training Time"],
    x=df["Model"],
    name="Training Time",
    # marker_color='#EB89B5',
    opacity=0.95,
))

train_test_fig.add_trace(go.Bar(
    y=df["Testing Time"],
    x=df["Model"],
    name="Testing Time",
    # marker_color='#330C73',
    opacity=0.95,
))

train_test_fig.update_layout(
    title_text="Training & Testing Time",
    xaxis_title_text="Model Name",
    yaxis_title_text="Time (ms)",
    bargap=0.2,
    bargroupgap=0.1
)

acc_fig = go.Figure()
acc_fig.add_trace(go.Bar(
    y=df["Training ACC"],
    x=df["Model"],
    name="Training ACC",
    marker_color='#EB89B5',
    opacity=0.95,
))

acc_fig.add_trace(go.Bar(
    y=df["Testing ACC"],
    x=df["Model"],
    name="Testing ACC",
    marker_color='#330C73',
    opacity=0.95,
))

acc_fig.update_layout(
    title_text="Training & Testing Accuracy",
    xaxis_title_text="Model Name",
    yaxis_title_text="Percentage",
    bargap=0.2,
    bargroupgap=0.1
)


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
        html.H6("Results", className="display-4"),
        html.Hr(),
        html.P(
            "Choose a page to begin", className="lead",
        ),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink("Model Performance", href="/page-1", active="exact"),
                dbc.NavLink("Model Performance 2", href="/page-2", active="exact"),
                dbc.NavLink("Model Performance 3", href="/page-3", active="exact"),
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
                        html.Img(src="https://static8.depositphotos.com/1026550/i/600/depositphotos_9546122-stock"
                                     "-photo-close-op-of-fiber-optics.jpg", height="100px"),
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

rf = [
    dbc.CardHeader("Random Forest"),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="https://lh4.googleusercontent.com/ANxp1CRZ8m4UohG8XJpvJusOpP-v-_7JLVwXXCpOAWwxdH2cEq0LRUym6WZgkdJNukBhUjCdBwnVYGm-gqXAn6YIEYM96CUbOHr0JhEAWawyygy5mShAiny3IRcdYmA63sxz26pT", height="100px"),
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
                        html.Img(src="https://gblobscdn.gitbook.com/assets%2F-LvBP1svpACTB1R1x_U4%2F-Lw6zezdliKWkGknCJ6R%2F-Lw70EB_T-Y3OCO-L_4o%2Fimage.png?alt=media&token=a3edaf4a-d3d2-4c84-9a10-3d870c21d641", height="100px"),
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

home_content = html.Div(
    [
        html.Br(),
        html.H1("Our Project"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(intro, color="light"))
            ]
        ),
        html.Br(),
        # dbc.Carousel(
        #     items=[
        #         {"key": "1", "src": "https://3.bp.blogspot.com/-RBu_zCzBmZc/WtZbzRVHFuI/AAAAAAAABzE/_CWY7woKdnApBTbzx-latDjP3TiTCbUPQCLcBGAs/s640/Decision%2BTree%2BStage%2B1.jpg"},
        #         {"key": "2", "src": "https://3.bp.blogspot.com/-AwUteKA-yXw/WtjYPllqShI/AAAAAAAAB0c/fYwB4Q8-nmk_3u9x8r5X7smfWmuMsu_DwCLcBGAs/s640/Split%2Bby%2BRainy.jpg"},
        #         {"key": "3", "src": "https://1.bp.blogspot.com/-vEy0tVpBuQ4/Wte39ZkiXpI/AAAAAAAABzk/8n-CF4cmYnEylEKUKf0-yiJtWmmYy2pSgCLcBGAs/s640/Decision-Tree-Final.jpg"},
        #     ],
        #     className="carousel-fade"
        # ),
        # html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dt, color="light"))
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(rf, color="light"))
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(gb, color="light"))
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(xgb, color="light"))
            ]
        ),
    ]
)

result1_content = html.Div(
    [
        html.H1("Model Comparison"),
        html.Div('''
            Training Time of Models
        '''),
        dcc.Graph(
            id='train-test-graph',
            figure=train_test_fig,
        ),

        html.Br(),
        html.Div('''
            Accuracy of Models
        '''),
        dcc.Graph(
            id='acc-graph',
            figure=acc_fig,
        ),
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
                        html.Div(home_content, style={"display": "none"}, id="page-1"),
                        html.Div(result1_content, style={"display": "block"}, id="page-2"),
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
