# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:40:30 2020

@author: Jason
"""

import pandas as pd
import numpy as np
import plotly.express as px  # (version 4.7.0)
import plotly
import plotly.graph_objs as go
import requests
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import dash_table
from datetime import datetime
from datetime import date
from datetime import timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import warnings
from arima import *
warnings.filterwarnings("ignore")



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# ------------------------------------------------------------------------------
        
# Calls
if __name__ =='__main__':    
    
    model = Model(links)
    df1 = model.df_transformations()
    a_dict = model.plot_graphs()
    fig, fig1, fig2 = model.plot_graphs2()
    arima, df_forecast = model.arima_series()
    nov_arima = model.nov_forecast()

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Ontario Covid-19 Tracking", style={'text-align': 'center'}),
    
    html.H3('The number of cases today ({}) is {}'.format(df1['Reported Date'].iloc[-1],int(df1['Daily Cases'].iloc[-1]))),
    html.H3('The total number of tests completed in the last day was {}'.format(int(df1['Total tests completed in the last day'].iloc[-1]))),
    
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df_forecast.columns],
    data=df_forecast.to_dict('records'),
    ),
    
    html.Div([
            html.Div([
                dcc.Graph(id='g1',figure=nov_arima)
        
        ], className='six columns'),
            html.Div([
                dcc.Graph(id='g2',figure=arima)
                ], className='six columns'),

        ], className='row'),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=fig)
    
    ], className='six columns'),
        html.Div([
            dcc.Graph(figure=a_dict['fig4'])
            ], className='six columns'),

    ], className='row'),
           
            
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=a_dict['fig0']),
            dcc.Graph(figure=a_dict['fig1']),
            dcc.Graph(figure=a_dict['fig2']),
            dcc.Graph(figure=a_dict['fig3']),
            dcc.Graph(figure=fig2)
])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
'''
@app.callback(
    [Output(component_id='display-value', component_property='children')],
     [Input(component_id='my_bee_map', component_property='figure')]
)

def update_graph(df):
    container = 'The number of cases today is: {}'.format(df1['Daily Cases'].iloc[-1:].values)

    return container
'''

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
    
    