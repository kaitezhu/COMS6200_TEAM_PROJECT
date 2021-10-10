from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate


def register_callbacks(app):
    @app.callback(
        [
            Output("page-1", "style"),
            Output("page-2", "style"),
            Output("page-3", "style"),
        ],
        [
            Input("home-btn", "n_clicks"),
            Input("result1-btn", "n_clicks"),
            Input("result2-btn", "n_clicks"),
        ]
    )
    def change_page(home, res1, res2):
        if home is None and res1 is None and res2 is None:
            raise PreventUpdate
        if res1:
            return [
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            ]
        if res2:
            return [
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            ]
        if home:
            return [
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            ]
