import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import numpy as np
from plotly import tools
import pandas as pd
import joblib


TRAIN_MEAN = [8.137795, 73.759843]
TRAIN_STD = [7.223710, 14.157503]


def normalize(item, mean, std):
    return np.asarray((item-mean) / std).reshape(1, -1)


def one_hot_mrs(mrs):
    z = np.zeros(6)
    z[mrs] = 1
    return z


def call_model(model, age, mrs, nihss):
    example = np.concatenate([normalize(nihss, TRAIN_MEAN[0], TRAIN_STD[0]),
                              normalize(age, TRAIN_MEAN[1], TRAIN_STD[1]),
                              one_hot_mrs(mrs).reshape(1, -1)
                              ],
                             axis=1)
    print('Calling model')
    probs = model.predict_proba(example)
    print(f'Probabilities are {probs}')
    return probs[0]


def load_model(path):
    model = joblib.load(path)
    return model


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_pickle('data/data.df')
age_max = df['Age'].max()
nihss_max = df['NIHSS'].max()
outcome_dict = {0: 'Death', 1: 'Inpatient', 2: 'CH', 3: 'Home'}
inverse_outcome_dict = {v:k for k,v in outcome_dict.items()}

outcome_colorscale = [[0, 'green'], [0.24 ,'green'],
                      [.25, 'red'], [.49, 'red'],
                      [.5, 'orange'], [.74, 'orange'],
                      [.75, 'blue'], [1., 'blue'] ]

model = load_model('code/logistic_regression.model')


app.layout = html.Div(className='container',
                      style={'height': '100%'},
                      children=[
                          html.H1(children='''
        Predicting discharge locations after stroke
    '''),
                          html.Div(className='row',
                                   children=[
                                       html.Div(className='six columns',
                                                children=[
                                                    html.Div(
                                                        'In this demo, you can change the patient data to see how the predictions of the'
                                                        ' model change. Below, we highlight the values you select in plots of the original'
                                                        ' data. Each level of the MRS score is associated with a different subplot.',
                                                        style={'padding-bottom': '20px'}),

                                                    html.Div(id='mrs-slider-output-container'),
                                                    html.Div(style={'height': '40px', 'padding-left': '20px'},
                                                             children=[dcc.Slider(
                                                                 id='mrs-slider',
                                                                 marks={i: 'Level {}'.format(i) for i in range(0, 6)},
                                                                 min=0,
                                                                 max=5,
                                                                 value=4,
                                                             )]),
                                                    html.Div(id='age-slider-output-container'),
                                                    html.Div(style={'height': '40px', 'padding-left': '20px'},
                                                             children=[dcc.Slider(
                                                                 id='age-slider',
                                                                 min=10,
                                                                 max=100,
                                                                 marks={10: 10, 100: 100},
                                                                 value=50)]),
                                                    html.Div(id='nihss-slider-output-container'),
                                                    html.Div(style={'height': '40px', 'padding-left': '20px'},
                                                             children=[dcc.Slider(
                                                                 id='nihss-slider',
                                                                 min=0,
                                                                 max=40,
                                                                 marks={0: 0, 40: 40},
                                                                 value=5)]),
                                                ]),

                                       html.Div(className='six columns',
                                                children=[dcc.Graph(id='graph-with-slider')]
                                                )]),
                          html.Div(className='row',
                                   children=[html.Div(dcc.Graph(id='mrs-graph'), className='twelve columns')])])


def _draw_crosshairs(age, nihss, mrs):
    xref = 'x' + str(mrs + 1)
    yref = 'y'

    return [{'type': 'line',
             'xref': xref,
             'yref': yref,
             'x0': age,
             'y0': 0,
             'x1': age,
             'y1': nihss_max,
             'line': {
                 'color': 'rgba(200, 50, 50, .7)',
                 'width': 2},
             },
            {
                'type': 'line',
                'xref': xref,
                'yref': yref,
                'x0': 0,
                'y0': nihss,
                'x1': age_max,
                'y1': nihss,
                'line': {
                    'color': 'rgba(200, 50, 50, .7)',
                    'width': 2},
            }
            ]


rand_data = [np.random.randint(0, 100, (50, 2)) for _ in range(6)]


@app.callback(
    Output('mrs-graph', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def update_graph(age, mrs, nihss):
    def _plot_mrs_level(mrs, pos):
        data = df.loc[df['MRS'] == pos - 1, ['Age', 'NIHSS']].values
        outcome = df.loc[df['MRS'] == pos - 1, 'Outcome'].values
        return go.Scatter(x=data[:, 0],
                          y=data[:, 1],
                          mode='markers',
                          marker=dict(
                              size=5,
                              color=[inverse_outcome_dict[o] for o in outcome],
                              colorscale=outcome_colorscale,
                              opacity=0.6 if mrs == pos - 1 else 0.2,
                              colorbar=dict(
                                  titleside='top',
                                  ticks='outside',
                                  tickvals=[0, 1, 1.9, 2.7],
                                  ticktext=list(outcome_dict.values()),
                                  thickness=20) if pos == 1 else None
                          ))

    fig = tools.make_subplots(rows=1, cols=6,
                              shared_xaxes=False,
                              shared_yaxes=True,
                              vertical_spacing=0.001)

    for pos in range(1, 7):
        fig.append_trace(_plot_mrs_level(mrs, pos), 1, pos)

    fig['layout'].update(shapes=_draw_crosshairs(age, nihss, mrs))
    fig['layout'].update(margin=dict(t=5, l=50, r=0))
    fig['layout'].update(height=300)
    fig['layout'].update(xaxis=dict(range=[0, age_max + 5]),
                         yaxis=dict(range=[0, nihss_max + 5]),
                         xaxis1=dict(range=[0, age_max + 5]),
                         xaxis2=dict(range=[0, age_max + 5]),
                         xaxis3=dict(range=[0, age_max + 5]),
                         xaxis4=dict(range=[0, age_max + 5]),
                         xaxis5=dict(range=[0, age_max + 5]),
                         xaxis6=dict(range=[0, age_max + 5]))

    fig['layout'].update(showlegend=False)
    fig['layout'].update(xaxis=dict(
        title='Age'))
    fig['layout'].update(yaxis=dict(
        title='NIHSS'))

    return fig


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def plot_model_predictions(age, mrs, nihss):
    probs = call_model(model, age, mrs, nihss)
    return {
        'data': [
            {'x': list(outcome_dict.values()), 'y': probs, 'type': 'bar', 'name': 'values'},
        ],
        'layout': go.Layout(
            margin={'t': 10},
            height=350,
            yaxis=dict(range=[0, 1], title='probability'))
    }


@app.callback(
    Output('age-slider-output-container', 'children'),
    [Input('age-slider', 'value')])
def update_output(value):
    return 'Age: {}'.format(value)


@app.callback(
    Output('nihss-slider-output-container', 'children'),
    [Input('nihss-slider', 'value')])
def update_output(value):
    return 'NIHSS: {}'.format(value)


@app.callback(
    Output('mrs-slider-output-container', 'children'),
    [Input('mrs-slider', 'value')])
def update_output(value):
    return 'MRS: {}'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)
