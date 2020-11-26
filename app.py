# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:40:30 2020

@author: User
"""

import pandas as pd
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
warnings.filterwarnings("ignore")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)
links = ['https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv',
         'https://data.ontario.ca/dataset/f4f86e54-872d-43f8-8a86-3892fd3cb5e6/resource/ed270bb8-340b-41f9-a7c6-e8ef587e6d11/download/covidtesting.csv',
         'https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv']



# Download CSV
def get_csv(links):
    for link in links:
        r = requests.get(link)
        with open((str(link).split('/')[8]), 'wb') as f:
            f.write(r.content)
            
def read_in_csv(links):
    csv = get_csv(links)
    files = []
    for link in links:
        files.append(str(link).split('/')[8])
    df = pd.read_csv(files[0])
    df1 = pd.read_csv(files[1])
    
    return df_transformations(df,df1)

def daily_cases(df1):
    df1['Daily Cases'] = df1['Total Cases'].diff()
    
    return df1

def df_transformations(df,df1):
    df1 = daily_cases(df1)
    df_cases = df1[['Reported Date','Daily Cases']]
    df_cases.set_index('Reported Date',inplace=True)
    df_cases.dropna(inplace=True)
    
    return (df,df_cases,df1)

def plot_graphs(df):
    ax = {} 
    lst = ['Age_Group', 'Case_AcquisitionInfo', 'Outcome1', 'Client_Gender','Reporting_PHU_City']

    for i in lst:
        ax["%s" %i] = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                                columns=i, aggfunc='count')
    a_dict = {}
    for i in range(0, len(ax)):
        a_dict["fig%s" %i] = px.bar(ax[list(ax.keys())[i]])
        
    return a_dict

def plot_graphs2(df_cases, df1):
    fig = px.bar(df_cases,y='Daily Cases',title='Daily Cases in Ontario')
    fig1 = px.bar(df1,x='Reported Date', y=['Number of patients hospitalized with COVID-19',
                                           'Number of patients in ICU with COVID-19',
                                           'Number of patients in ICU on a ventilator with COVID-19'],
                                          title='Patients in Hospital')
    fig7=px.line(df1, x='Reported Date', y=['Resolved','Deaths','Total Cases'],title='Resolved Vs. Deaths Vs. Total Cases')
    
    return fig, fig1, fig7

def arima_series(df_cases):
    global date
    dates = []
    date = date.today()
    for i in range(30):
        date += timedelta(days=1)
        dates.append(date)
    
    date_series = pd.Series(dates)
      
    model = ARIMA(df_cases['Daily Cases'], order=(3, 1, 2))  
    fitted = model.fit(disp=-1)  
    
    # Forecast
    fc, se, conf = fitted.forecast(30, alpha=0.05)  # 95% conf
    
    # Make as pandas series
    fc_series = pd.Series(fc, index=date_series)
    lower_series = pd.Series(conf[:, 0], index=date_series)
    upper_series = pd.Series(conf[:, 1], index=date_series)
    
    trace_train = go.Scatter(
    x = df_cases.index,
    y = df_cases['Daily Cases'],
    mode = 'lines',
    line = {'color':'blue'},
    name="Number of Cases"
    )
    
    trace_forecast = go.Scatter(
        x = fc_series.index,
        y= fc_series.astype(int),
        mode = 'lines',
        line = {'color':'red'},
        name = 'Forecast'
    )
    
    trace_upperbound = go.Scatter(
        x = upper_series.index,
        y= upper_series.astype(int),
        mode = 'lines',
        fill = 'tonexty',
        line = {'color':'gray'},
        name = 'upperbound'
    )
    
    trace_lowerbound = go.Scatter(
        x = lower_series.index,
        y= lower_series.astype(int),
        mode = 'lines',
        fill = 'tonexty',
        line = {'color':'gray'},
        name = 'lowerbound'
    )

    data = [trace_train, trace_forecast, trace_upperbound, trace_lowerbound]
    layout = go.Layout(title="Forecasted ARIMA model",xaxis_rangeslider_visible=True)
    arima = go.Figure(data=data,layout=layout)
    
    return arima, yday_forecast(df_cases)       

def yday_forecast(df_cases):
    global date
    dates = []
    date = date.today() - timedelta(1)
    for i in range(30):
        date += timedelta(days=1)
        dates.append(date)
    
    date_series = pd.Series(dates)
    
    model = ARIMA(df_cases['Daily Cases'], order=(3, 1, 2))  
    fitted = model.fit(disp=-1)  
    
    # Forecast
    fc, se, conf = fitted.forecast(30, alpha=0.05)  # 95% conf
    
    # Make as pandas series
    fc_series = pd.Series(fc, index=date_series)
    lower_series = pd.Series(conf[:, 0], index=date_series)
    upper_series = pd.Series(conf[:, 1], index=date_series)
    
    
    df_yday = pd.DataFrame([fc_series.astype(int),upper_series.astype(int),lower_series.astype(int)]).transpose()
    df_yday.columns = ['Forecast','upper bound','lower bound']
    
    return forecast_df(df_cases, df_yday)

def forecast_df(df_cases, df_yday):
    df_forecast = pd.concat([df_cases,df_yday])
    df_forecast[['Forecast','upper bound','lower bound']] = df_forecast[['Forecast','upper bound','lower bound']].shift(-1)
    df_forecast = df_forecast.shift(1)
    df_forecast.reset_index(inplace=True)
    df_forecast.columns = ['Date','Daily Cases','Model Forecast','Upper Bound','Lower Bound']
    df_forecast = df_forecast.iloc[-30:-28,:]
    return df_forecast
    
def main():
    df, df_cases, df1 = read_in_csv(links)
    a_dict = plot_graphs(df)
    arima, df_forecast = arima_series(df_cases)
    fig, fig1, fig2 = plot_graphs2(df_cases, df1)
    
    return df1, a_dict, fig, fig1, fig2, arima, df_forecast
    
if __name__ == '__main__':
    df1, a_dict, fig, fig1, fig2, arima, df_forecast = main()

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
    
    dcc.Graph(figure=arima),
    dcc.Graph(figure=fig),
    dcc.Graph(figure=a_dict['fig4']),
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
    
    