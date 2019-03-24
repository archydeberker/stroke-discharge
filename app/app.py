import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='''
        Predicting discharge locations after stroke
    '''),
    html.Div(className='row',
             children=[
                 html.Div(className='six columns',
                          style={'height':'50%'},
                          children=[
                            html.Div('In this demo, you can change the patient data to see how the predictions of the'
                                     ' model change. Below, we highlight the values you select in plots of the original'
                                     ' data. Each level of the MRS score is associated with a different subplot.',
                                     style={'padding-bottom': '50px'}),

                              html.Div(id='mrs-slider-output-container'),
                              html.Div(style={'height': '50px', 'padding-left': '20px'},
                                       children=[dcc.Slider(
                                           id='mrs-slider',
                                           marks={i: 'Level {}'.format(i) for i in range(0, 6)},
                                           min=0,
                                           max=5,
                                           value=0,
                                       )]),
                              html.Div(id='age-slider-output-container'),
                              html.Div(style={'height': '50px', 'padding-left': '20px'},
                                       children=[dcc.Slider(
                                           id='age-slider',
                                           min=10,
                                           max=100,
                                           marks={10: 10, 100: 100},
                                           value=50)]),
                              html.Div(id='nihss-slider-output-container'),
                              html.Div(style={'height': '50px', 'padding-left': '20px'},
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
             children=[html.Div(dcc.Graph(id='mrs-0-graph'), className='two columns'),
                       html.Div(dcc.Graph(id='mrs-1-graph'), className='two columns'),
                       html.Div(dcc.Graph(id='mrs-2-graph'), className='two columns')])
                       # html.Div(dcc.Graph(id='mrs-3-graph'), className='two columns'),
                       # html.Div(dcc.Graph(id='mrs-4-graph'), className='two columns'),
                       # html.Div(dcc.Graph(id='mrs-5-graph'), className='two columns')]),
])

def _draw_crosshairs(age, nihss):
    return {'shapes': [
            # Line reference to the axes
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'x0': age,
                'y0': 0,
                'x1': age,
                'y1': max(rand_data[0][:, 1]),
                'line': {
                    'color': 'rgba(200, 50, 50, .3)',
                    'width': 2},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'x0': 0,
                'y0': nihss,
                'x1': max(rand_data[0][:, 0]),
                'y1': nihss,
                'line': {
                    'color': 'rgba(200, 50, 50, .3)',
                    'width': 2},
            }
        ],
        'margin': dict(t=0)}


rand_data = [np.random.randint(0, 100, (50, 2)) for _ in range(6)]

@app.callback(
    Output('mrs-0-graph', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def mrs_0(age, mrs, nihss):
    mrs_num=0
    return {
        'data': [
            {'x': rand_data[0][:, 0], 'y': rand_data[0][:, 1],
             'type': 'scatter',
             'mode': 'markers',
             'name': 'values',
             'opacity': 0.5 if mrs == mrs_num else 0.2},
        ],
        'layout': _draw_crosshairs(age, nihss) if mrs == mrs_num else {'margin': dict(t=0)}}

@app.callback(
    Output('mrs-1-graph', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def mrs_1(age, mrs, nihss):
    mrs_num=1
    return {
        'data': [
            {'x': rand_data[mrs_num][:, 0], 'y': rand_data[mrs_num][:, 1],
             'type': 'scatter',
             'mode': 'markers',
             'name': 'values',
             'opacity': 0.5 if mrs == mrs_num else 0.2},
        ],
        'layout': _draw_crosshairs(age, nihss) if mrs == mrs_num else {'margin': dict(t=0)}}

@app.callback(
    Output('mrs-2-graph', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def mrs_2(age, mrs, nihss):
    mrs_num=2
    return {
        'data': [
            {'x': rand_data[mrs_num][:, 0], 'y': rand_data[mrs_num][:, 1],
             'type': 'scatter',
             'mode': 'markers',
             'name': 'values',
             'opacity': 0.5 if mrs == mrs_num else 0.2},
        ],
        'layout': _draw_crosshairs(age, nihss) if mrs == mrs_num else {'autosize': False, 'margin': dict(t=0)}}



@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('age-slider', 'value'), Input('mrs-slider', 'value'), Input('nihss-slider', 'value'), ])
def plot_variables(age, mrs, nihss):
    return {
        'data': [
            {'x': [1, 2, 3], 'y': [age, mrs, nihss], 'type': 'bar', 'name': 'values'},
        ],
   'layout': go.Layout(
       margin={'t': 0},
        height=400)
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
