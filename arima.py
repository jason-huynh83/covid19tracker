# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:57:55 2020

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
warnings.filterwarnings("ignore")


# Import and clean data (importing csv into pandas)
links = ['https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv',
         'https://data.ontario.ca/dataset/f4f86e54-872d-43f8-8a86-3892fd3cb5e6/resource/ed270bb8-340b-41f9-a7c6-e8ef587e6d11/download/covidtesting.csv',
         'https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv']

class Model:
    def __init__(self, links):
        self.read_in_csv(links)
        
    # Download CSV
    def get_csv(self, links):
        '''
        download csv files from links
        '''
        for link in links:
            r = requests.get(link)
            with open((str(link).split('/')[8]), 'wb') as f:
                f.write(r.content)

    def read_in_csv(self, links):
        '''
        read in csv files into variables df and df1
        '''
        csv = self.get_csv(links)
        files = []
        for link in links:
            files.append(str(link).split('/')[8])
        self.df = pd.read_csv(files[0])
        self.df1 = pd.read_csv(files[1])
        
        return self.df_transformations()

    def daily_cases(self):
        '''
        differencing df1 in order to get current active cases
        '''
        self.df1['Daily Cases'] = self.df1['Total Cases'].diff()
        return self.df1
    
    def df_transformations(self):
        '''
        dataframe transformations
        '''
        self.df1 = self.daily_cases()
        self.df_cases = self.df1[['Reported Date','Daily Cases']]
        self.df_cases.set_index('Reported Date', inplace=True)
        self.df_cases.dropna(inplace=True)
        
        return self.df1 
    
    def plot_graphs(self):
        ax = {} 
        lst = ['Age_Group', 'Case_AcquisitionInfo', 'Outcome1', 'Client_Gender','Reporting_PHU_City']

        for i in lst:
            ax["%s" %i] = pd.pivot_table(self.df, index='Accurate_Episode_Date', values='Row_ID',
                                    columns=i, aggfunc='count')
        a_dict = {}
        for i in range(0, len(ax)):
            a_dict["fig%s" %i] = px.bar(ax[list(ax.keys())[i]])

        return a_dict

    def plot_graphs2(self):
        fig = px.bar(self.df_cases,y='Daily Cases',title='Daily Cases in Ontario')
        fig1 = px.bar(self.df1,x='Reported Date', y=['Number of patients hospitalized with COVID-19',
                                               'Number of patients in ICU with COVID-19',
                                               'Number of patients in ICU on a ventilator with COVID-19'],
                                              title='Patients in Hospital')
        fig7=px.line(self.df1, x='Reported Date', y=['Resolved','Deaths','Total Cases'],title='Resolved Vs. Deaths Vs. Total Cases')

        return fig, fig1, fig7

    def arima_series(self):
        global date
        dates = []
        date = date.today()
        for i in range(30):
            date += timedelta(days=1)
            dates.append(date)

        date_series = pd.Series(dates)

        model = ARIMA(self.df_cases['Daily Cases'], order=(3, 1, 2))  
        fitted = model.fit(disp=-1)  

        # Forecast
        fc, se, conf = fitted.forecast(30, alpha=0.05)  # 95% conf

        # Make as pandas series
        fc_series = pd.Series(fc, index=date_series)
        lower_series = pd.Series(conf[:, 0], index=date_series)
        upper_series = pd.Series(conf[:, 1], index=date_series)

        trace_train = go.Scatter(
        x = self.df_cases.index,
        y = self.df_cases['Daily Cases'],
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
        layout = go.Layout(title="Forecasted ARIMA model for next 30 days",xaxis_rangeslider_visible=True)
        arima = go.Figure(data=data,layout=layout)

        return arima, self.yday_forecast()       

    def yday_forecast(self):
        global date
        dates = []
        date = date.today() - timedelta(1)
        for i in range(30):
            date += timedelta(days=1)
            dates.append(date)

        date_series = pd.Series(dates)

        model = ARIMA(self.df_cases['Daily Cases'], order=(3, 1, 2))  
        fitted = model.fit(disp=-1)  

        # Forecast
        fc, se, conf = fitted.forecast(30, alpha=0.05)  # 95% conf

        # Make as pandas series
        fc_series = pd.Series(fc, index=date_series)
        lower_series = pd.Series(conf[:, 0], index=date_series)
        upper_series = pd.Series(conf[:, 1], index=date_series)


        df_yday = pd.DataFrame([fc_series.astype(int),upper_series.astype(int),lower_series.astype(int)]).transpose()
        df_yday.columns = ['Forecast','upper bound','lower bound']

        return self.forecast_df(df_yday)

    def forecast_df(self, df_yday):
        df_forecast = pd.concat([self.df_cases,df_yday])
        df_forecast[['Forecast','upper bound','lower bound']] = df_forecast[['Forecast','upper bound','lower bound']].shift(-1)
        df_forecast = df_forecast.shift(1)
        df_forecast.reset_index(inplace=True)
        df_forecast.columns = ['Date','Daily Cases','Model Forecast','Upper Bound','Lower Bound']
        df_forecast = df_forecast.iloc[-30:-28,:]
        return df_forecast
    
    def nov_forecast(self):
        global date
        dates = []
        date = date.today() - timedelta(23)
        for i in range(30):
            date += timedelta(days=1)
            dates.append(date)
    
        date_series = pd.Series(dates)
    
        model = ARIMA(self.df_cases['Daily Cases'][:-23], order=(3, 1, 2))  
        fitted = model.fit(disp=-1)  
    
        # Forecast
        fc, se, conf = fitted.forecast(30, alpha=0.05)  # 95% conf
    
        # Make as pandas series
        fc_series = pd.Series(fc, index=date_series)
        lower_series = pd.Series(conf[:, 0], index=date_series)
        upper_series = pd.Series(conf[:, 1], index=date_series)
    
        trace_train = go.Scatter(
        x = self.df_cases.index,
        y = self.df_cases['Daily Cases'],
        mode = 'lines',
        line = {'color':'blue'},
        name="Number of Cases"
        )
    
        trace_track = go.Scatter(
        x = self.df_cases.index[-23:],
        y = self.df_cases['Daily Cases'].iloc[-23:],
        mode = 'lines',
        line = {'color':'orange'},
        name="Number of Cases Since Forecast"
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
    
        data = [trace_train, trace_track, trace_forecast, trace_upperbound, trace_lowerbound]
        layout = go.Layout(title="Forecasted ARIMA model",xaxis_rangeslider_visible=True)
        arima = go.Figure(data=data,layout=layout)
    
        return arima