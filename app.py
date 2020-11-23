# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:40:30 2020

@author: User
"""

import pandas as pd
import plotly.express as px  # (version 4.7.0)
import requests
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt


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

def df_transformations(df,df1):
    df1 = daily_cases(df1)
    df_cases = df1[['Reported Date','Daily Cases']]
    df_cases.set_index('Reported Date',inplace=True)
    return (df,df_cases,df1)
    
def daily_cases(df1):
    df1['Daily Cases'] = df1['Total Cases'].diff()
    return df1



df, df_cases, df1 = read_in_csv(links) 

fig = px.bar(df_cases,y='Daily Cases',title='Daily Cases in Ontario')
fig1 = px.bar(df1,x='Reported Date', y=['Number of patients hospitalized with COVID-19',
                                       'Number of patients in ICU with COVID-19',
                                       'Number of patients in ICU on a ventilator with COVID-19'],
                                      title='Patients in Hospital')

lst = ['Age_Group', 'Case_AcquisitionInfo', 'Outcome1', 'Client_Gender','Reporting_PHU_City']

ax = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                    columns=lst[0], aggfunc='count')

ax1 = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                    columns=lst[1], aggfunc='count')

ax2 = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                    columns=lst[2], aggfunc='count')

ax3 = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                    columns=lst[3], aggfunc='count')

ax4 = pd.pivot_table(df, index='Accurate_Episode_Date', values='Row_ID',
                    columns=lst[4], aggfunc='count')


fig2=px.bar(ax,color=lst[0],title='Cases by Age Group')
fig3=px.bar(ax1,color=lst[1], title='Cases by Acquisition')
fig4=px.bar(ax2,color=lst[2], title='Outcome of Cases')
fig5=px.bar(ax3,color=lst[3], title='Cases by Gender')
fig6=px.bar(ax4,color=lst[4], title='Cases by City')
fig7=px.line(df1, x='Reported Date', y=['Resolved','Deaths','Total Cases'],title='Resolved Vs. Deaths Vs. Total Cases')
#fig8=px.bar(df1,x='Reported Date',y=['Total tests completed in last day','Daily Cases'],title='Total tests Vs. Daily Cases')
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Ontario Covid-19 Tracking", style={'text-align': 'center'}),
    html.H3('The number of cases today ({}) is {}'.format(df1['Reported Date'].iloc[-1],int(df1['Daily Cases'].iloc[-1]))),
    
    dcc.Graph(figure=fig),
    dcc.Graph(figure=fig6),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig7)
    #dcc.Graph(figure=fig8)

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
    
    