from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.figure_factory as ff


def register_callbacks(app):
    with open('../data/result.json') as f:
        data = json.load(f)
    keys = ['Model', 'Train_time', 'Testing_time', 'Trained_accuracy', 'Testing_accuracy', 'fpr_list', 'tpr_list',
            'thresholds', 'AUC', 'Balance', 'TN', 'FP', 'FN', 'TP', 'Precision', 'recall', 'fpr', 'fnr', 'tnr', 'npv',
            'fdr', 'for', 'f1']
    data_full = {}
    for key in keys:
        data_full[key] = []

    @app.callback(
        [
            Output("page-1", "style"),
            Output("page-2", "style"),
            Output("page-3", "style"),
            Output("side-div-p1", "style"),
            Output("side-div-p2", "style"),
            Output("side-div-p3", "style"),
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
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            ]
        if res2:
            return [
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            ]
        if home:
            return [
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            ]

    @app.callback(
        [
            Output("data-label-card-div", "style"),
            Output("data-type-card-div", "style"),
        ],
        [
            Input("data-card-toggle", "n_clicks"),
        ]
    )
    def show_data_card(n):
        if n:
            return [{"display": "block"}, {"display": "block"}]
        else:
            raise PreventUpdate

    @app.callback(
        [
            Output("f1-graph", "figure"),
            Output("roc-graph", "figure"),
            Output("acc-graph", "figure"),
            Output("pre_rec-graph", "figure"),
            Output("train-test-graph", "figure"),
            Output("dt-heatmap", "figure"),
            Output("rf-heatmap", "figure"),
            Output("gb-heatmap", "figure"),
            Output("xgb-heatmap", "figure"),
        ],
        [
            Input("model-checklist", "value")
        ],
    )
    def generate_graph(model_check_val):

        # with open('../data/result.json') as f:
        #     data = json.load(f)

        # data_full = {}
        # for key in keys:
        #     data_full[key] = []
        for i in data['81173']:
            data_full['Model'].append(i)
        for i in data_full['Model']:
            for j in data['81173'][i]:
                data_full[j].append(data['81173'][i][j])

        model_name = []
        for val in model_check_val:
            model_name.append(data_full["Model"][int(val)])

        f1 = []
        for val in model_check_val:
            f1.append(data_full["f1"][int(val)])
        f1_fig = go.Figure()
        f1_fig.add_trace(go.Bar(
            x=f1,
            y=model_name,
            name="F1 Score",
            marker_color=['#0b84a5', '#f6c85f'],
            opacity=0.95,
            orientation='h',
            # text=data_full["Trained_accuracy"].apply(lambda x: round(x, 2)),
            text=f1,
            textposition='outside',
        ))
        f1_fig.update_layout(
            xaxis_title_text="Percentage",
            bargap=0.5,
            bargroupgap=0.1,
            yaxis_tickangle=-45,
            title={
                'text': 'F1 Score',
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

        train_time = []
        for val in model_check_val:
            train_time.append(data_full["Train_time"][int(val)])
        train_test_fig = go.Figure()

        train_test_fig.add_trace(go.Bar(
            x=train_time,
            y=model_name,
            name="Training Time",
            marker_color='#0b84a5',
            opacity=0.95,
            text=train_time,
            textposition='outside',
            orientation='h',
        ))

        fpr_list = []
        tpr_list = []
        thresholds = []
        AUC = []
        for val in model_check_val:
            fpr_list.append(data_full["fpr_list"][int(val)])
            tpr_list.append(data_full["tpr_list"][int(val)])
            thresholds.append(data_full["thresholds"][int(val)])
            AUC.append(data_full["AUC"][int(val)])
        roc_fig = go.Figure()
        roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        for i in range(len(model_check_val)):
            name = f"{model_name[i]} (AUC={AUC[i]:.2f})"
            roc_fig.add_trace(go.Scatter(x=fpr_list[i], y=tpr_list[i], name=name, mode='lines'))

        roc_fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            # height=500,
            title={
                'text': 'ROC Curve',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            uniformtext_minsize=8,
            uniformtext_mode='hide',
        )

        test_time = []
        for val in model_check_val:
            test_time.append(data_full["Testing_time"][int(val)])
        train_test_fig.add_trace(go.Bar(
            x=test_time,
            y=model_name,
            name="Testing Time",
            marker_color='#f6c85f',
            opacity=0.95,
            text=test_time,
            textposition='outside',
            orientation='h',
        ))

        train_test_fig.update_layout(
            xaxis_title_text="Time (s)",
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

        train_acc = []
        for val in model_check_val:
            train_acc.append(data_full["Trained_accuracy"][int(val)])
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Bar(
            x=train_acc,
            y=model_name,
            name="Training ACC",
            marker_color="#119dff",
            # marker_color='#EB89B5',
            opacity=0.95,
            orientation='h',
            # text=data_full["Trained_accuracy"].apply(lambda x: round(x, 2)),
            text=train_acc,
            textposition='outside',
        ))

        test_acc = []
        for val in model_check_val:
            test_acc.append(data_full["Testing_accuracy"][int(val)])
        acc_fig.add_trace(go.Bar(
            x=test_acc,
            y=model_name,
            name="Testing ACC",
            marker_color="#66c2a5",
            # marker_color='#330C73',
            opacity=0.95,
            orientation='h',
            # text=data_full["Testing_accuracy"].apply(lambda x: round(x, 2)),
            text=test_acc,
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

        pre = []
        for val in model_check_val:
            pre.append(data_full["Precision"][int(val)])
        pre_rec_fig = go.Figure()
        pre_rec_fig.add_trace(go.Bar(
            x=pre,
            y=model_name,
            name="Precision",
            marker_color='#EB89B5',
            opacity=0.95,
            orientation='h',
            text=pre,
            # text=data_full["Precision"].apply(lambda x: round(x, 2)),
            textposition='outside',
        ))

        rec = []
        for val in model_check_val:
            rec.append(data_full["recall"][int(val)])
        pre_rec_fig.add_trace(go.Bar(
            x=data_full["recall"],
            y=model_name,
            name="Recall",
            marker_color='#330C73',
            opacity=0.95,
            orientation='h',
            text=rec,
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

        # FN,TP,TN,FP
        dt_heat_data = []
        rf_heat_data = []
        gb_heat_data = []
        xgb_heat_data = []

        for x in ['FN', 'TP', 'TN', 'FP']:
            dt_heat_data.append(data_full[x][0])
            rf_heat_data.append(data_full[x][1])
            gb_heat_data.append(data_full[x][2])
            xgb_heat_data.append(data_full[x][3])

        dt_heat_data = np.around(np.reshape(dt_heat_data, (-1, 2)) / 35179 * 100, decimals=2)
        rf_heat_data = np.around(np.reshape(rf_heat_data, (-1, 2)) / 35179 * 100, decimals=2)
        gb_heat_data = np.around(np.reshape(gb_heat_data, (-1, 2)) / 35179 * 100, decimals=2)
        xgb_heat_data = np.around(np.reshape(xgb_heat_data, (-1, 2)) / 35179 * 100, decimals=2)

        colorscale = [[0, 'navy'], [1, 'plum']]
        font_colors = ['white', 'black']
        x_text = ['Predicted 0', 'Predicted 1']
        y_text = ['Actual 1', 'Actual 0']
        dt_heat = ff.create_annotated_heatmap(dt_heat_data, x=x_text, y=y_text, colorscale=colorscale,
                                              font_colors=font_colors)
        rf_heat = ff.create_annotated_heatmap(rf_heat_data, x=x_text, y=y_text, colorscale=colorscale,
                                              font_colors=font_colors)
        gb_heat = ff.create_annotated_heatmap(gb_heat_data, x=x_text, y=y_text, colorscale=colorscale,
                                              font_colors=font_colors)
        xgb_heat = ff.create_annotated_heatmap(xgb_heat_data, x=x_text, y=y_text, colorscale=colorscale,
                                               font_colors=font_colors)
        dt_heat.update_layout(
            title={
                'text': 'Decision Tree Heatmap',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        )
        rf_heat.update_layout(
            title={
                'text': 'Random Forest Heatmap',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        )
        gb_heat.update_layout(
            title={
                'text': 'Gradient Boosting Heatmap',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        )
        xgb_heat.update_layout(
            title={
                'text': 'XGBoost Heatmap',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        )
        return [
            f1_fig,
            roc_fig,
            acc_fig,
            pre_rec_fig,
            train_test_fig,
            dt_heat,
            rf_heat,
            gb_heat,
            xgb_heat
        ]

    @app.callback(
        [
            Output("slider-graph", "figure"),
        ],
        [
            Input("volume-slider", "value"),
            Input("col-checklist", "value"),
        ]
    )
    def generate_graph_2(selected_volume, check_val):
        with open('../data/result.json') as f:
            data = json.load(f)
        col_list = ['DTC', 'RFC', 'GB', 'XGB']
        data_full = {}
        for key in data:
            data_full[key] = {}
        for key in data:
            for i in col_list:
                for j in data[key][i]:
                    data_full[key][j] = []
        for key in data:
            for i in col_list:
                for j in data[key][i]:
                    data_full[key][j].append(data[key][i][j])

        # [data_full[selected_volume][k] for k in ['Train_time', 'Testing_accuracy']]

        fig = make_subplots(5, 1, horizontal_spacing=0.05)
        col_names = ['f1', 'Precision', 'recall', 'Testing_accuracy', 'Balance']
        selected_cols = []
        for val in check_val:
            selected_cols.append(col_names[int(val)])
        print(selected_cols)
        for k in range(len(selected_cols)):
            filtered_data = data_full[str(selected_volume)]
            # filtered = filtered_data[0][col_names[k]]
            filtered = filtered_data[selected_cols[k]]
            if selected_cols[k] in ['AUC', 'Balance']:
                filtered = [num * 100 for num in filtered]
            fig.add_trace(go.Scatter(
                x=col_list,
                y=filtered,
                mode='lines+markers',
                name=selected_cols[k],
                marker=dict(
                    size=[num / 4 for num in filtered],
                ),
            ), k + 1, 1
            )

        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            height=900,
            template='ggplot2',
        )

        return [fig]
