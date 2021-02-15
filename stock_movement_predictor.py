#####################################################################################################Program Required Libraries#########################################################################################

import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
# import finnhub
import re
from datetime import timedelta, datetime, date
import time
from time import sleep
import pandas_datareader as pdr
import math
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from pytrends.request import TrendReq
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from credentials import token

pytrends = TrendReq(hl='en-US', tz=360)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

from urllib.request import urlopen, Request

print("Done reading requirements")

################################################################################################Program Functions#######################################################################################################

def get_Stock_News_Data(stock:str, token):

    stock_news_df = pd.DataFrame()

    base = datetime.today()
    num_days = 120
    day_range_interval = 10

    for _ in range(int(num_days / day_range_interval)):
        
        date_list = [base - timedelta(days=x) for x in range(day_range_interval)]
        # str_date_list = [date.strftime("%Y-%m-%d") for date in date_list]
        
        min_date = min(date_list).strftime("%Y-%m-%d")
        max_date = max(date_list).strftime("%Y-%m-%d")
        
        sleep(3)
        
        r = requests.get(f'https://finnhub.io/api/v1/company-news?symbol={stock}&from={min_date}&to={max_date}&token={token}')

        df = pd.DataFrame(r.json())
        
        try:
            
            try:

                df['datetime'] = df['datetime'].apply(lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))

            except:

                df['datetime'] = df['datetime'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'))
                
        except:
            
            continue
        
        
        stock_news_df = stock_news_df.append(df)
        
        
        base = base - timedelta(days = day_range_interval)

    return stock_news_df


def df_Data_Enhancements(stock_news_df):

    analyzer = SentimentIntensityAnalyzer()

    stock_news_df['headline_sentiment'] = stock_news_df['headline'].apply(lambda x: analyzer.polarity_scores(x).get('compound'))
    stock_news_df['summary_sentiment'] = stock_news_df['summary'].apply(lambda x: analyzer.polarity_scores(x).get('compound'))
    stock_news_df['comb_sentiment'] = (stock_news_df['headline_sentiment'] + stock_news_df['summary_sentiment']) / 2

    stock_news_df['date'] = stock_news_df['datetime'].apply(lambda x: x.split(' ')[0])
    stock_news_df['time'] = stock_news_df['datetime'].apply(lambda x: x.split(' ')[1])

    stock_news_df['date'] = pd.to_datetime(stock_news_df['date'])
    stock_news_df['weekday'] = stock_news_df['date'].apply(lambda x: x.weekday())
    stock_news_df['hour'] = stock_news_df['time'].apply(lambda x: int(x[:2]))
    stock_news_df['minutes'] = stock_news_df['time'].apply(lambda x: int(x[3:5]))
    stock_news_df['time_of_day'] = stock_news_df['time'].apply(lambda x: x[5:])

    stock_news_df = stock_news_df.reset_index(drop = True)

    return stock_news_df


def df_Date_Adjustments(stock_news_df, stock:str):

    dates = []

    for i in range(min(stock_news_df.index), max(stock_news_df.index) + 1):
        
        if stock_news_df.iloc[i]['weekday'] == 5:
            
            day = stock_news_df.iloc[i]['date'] + timedelta(2)
            dates.append(day)
            
        elif stock_news_df.iloc[i]['weekday'] == 6:
        
            day = stock_news_df.iloc[i]['date'] + timedelta(2)
            dates.append(day)
        
        elif stock_news_df.iloc[i]['weekday'] < 4 and stock_news_df.iloc[i]['time_of_day'] == 'PM' and stock_news_df.iloc[i]['hour'] >= 4:
            
            day = stock_news_df.iloc[i]['date'] + timedelta(1)
            dates.append(day)
            
        elif stock_news_df.iloc[i]['weekday'] == 4 and stock_news_df.iloc[i]['time_of_day'] == 'PM' and stock_news_df.iloc[i]['hour'] >= 4:
            
            day = stock_news_df.iloc[i]['date'] + timedelta(3)
            dates.append(day)
            
        else:
            
            dates.append(stock_news_df.iloc[i]['date'])
        
    stock_news_df['date'] = dates

    stock_news_df = pd.DataFrame(stock_news_df.groupby('date').comb_sentiment.mean()).reset_index(drop = False)
    stock_news_df = stock_news_df.rename(columns = {'comb_sentiment':'sentiment'})
    stock_news_df['ticker'] = stock

    return stock_news_df


def get_Stock_Data(stock_news_df, stock):

    date_min = stock_news_df.iloc[min(stock_news_df.index)]['date']
    date_max = stock_news_df.iloc[max(stock_news_df.index)]['date']

    stock_df = pdr.DataReader(stock, 'yahoo', date_min, date_max).reset_index(drop = False)
    stock_df = stock_df.rename(columns = {'Date':'date'})

    merged_df = stock_df.merge(stock_news_df, on = 'date')

    return merged_df


def create_MA_Indicators(merged_df):

    # List where we will keep track of long and short average points
    indicators = pd.DataFrame(index=merged_df.index)

    # Exponential moving averages using the closing data (5-day & 20-day)
    indicators['5_short_avg'] = merged_df['Close'].ewm(span=5, adjust=False).mean()
    indicators['20_long_avg'] = merged_df['Close'].ewm(span=20, adjust=False).mean()

    # Exponential moving averages using the closing data (13-day & 49-day)
    indicators['13_short_avg'] = merged_df['Close'].ewm(span=13, adjust=False).mean()
    indicators['49_long_avg'] = merged_df['Close'].ewm(span=49, adjust=False).mean()

    # Exponential moving averages using the closing data (50-day & 200-day)
    indicators['50_short_avg'] = merged_df['Close'].ewm(span=50, adjust=False).mean()
    indicators['200_long_avg'] = merged_df['Close'].ewm(span=200, adjust=False).mean()

    indicators['date'] = merged_df['date']

    merged_df = merged_df.merge(indicators, on = 'date')

    # Exponential moving averages using the closing data (5-day & 20-day)
    merged_df['diff_5_20'] = merged_df['5_short_avg'] - merged_df['20_long_avg']

    merged_df['ma_indicator_5_20'] = np.where(
        abs(merged_df['diff_5_20']) < 0.02,
        1.0,
        0.0
    )

    # Exponential moving averages using the closing data (13-day & 49-day)
    merged_df['diff_13_49'] = merged_df['13_short_avg'] - merged_df['49_long_avg']

    merged_df['ma_indicator_13_49'] = np.where(
        abs(merged_df['diff_13_49']) < 0.02,
        1.0,
        0.0
    )

    # Exponential moving averages using the closing data (50-day & 200-day)
    merged_df['diff_50_200'] = merged_df['50_short_avg'] - merged_df['200_long_avg']

    merged_df['ma_indicator_50_200'] = np.where(
        abs(merged_df['diff_50_200']) < 0.02,
        1.0,
        0.0
    )


    merged_df['daily_change'] = merged_df['Close'] - merged_df['Open']

    merged_df['up_down'] = np.where(
        merged_df['daily_change'] < 0,
        0.0,
        1.0
    )

    merged_df = merged_df.rename(columns = {'key_0':'Date'})

    return merged_df


def calculate_TrueRange(merged_df):

    index_max = max(merged_df.index)

    start = 0
    end = 2

    true_ranges = []

    while end < index_max + 2:
        
        df = merged_df.iloc[start:end].copy()
        
        first_calc = df.iloc[1]['High'] - df.iloc[1]['Low']
        second_calc = abs(df.iloc[1]['High'] - df.iloc[0]['Close'])
        third_calc = abs(df.iloc[1]['Low'] - df.iloc[0]['Close'])

        true_range = max(first_calc, second_calc, third_calc)
        
        true_ranges.append(true_range)
        
        start += 1
        end += 1

    true_ranges.insert(0, 0.0)

    merged_df['true_range'] = true_ranges

    return merged_df


def calculate_ATR(merged_df):

    index_max = max(merged_df.index)

    start = 1
    end = 15

    atrs = []

    while end < index_max + 2:
        
        df = merged_df.iloc[start:end].copy()
        
        atr = sum(df['true_range']) / len(df['true_range'])
        
        atrs.append(atr)
        
        start += 1
        end += 1
        
    atrs_values = ([0.0] * 14) + atrs

    # merged_df['ATR'] = atrs_values

    yesterday_atr = round(atrs_values[len(atrs_values) - 1], 3)

    return yesterday_atr


def get_INTL_Markets_Data(merged_df):

    final_stock_df = merged_df.copy()

    date_max = final_stock_df.iloc[max(final_stock_df.index)]['date'].strftime('%Y-%m-%d')
    date_min = final_stock_df.iloc[min(final_stock_df.index)]['date'].strftime('%Y-%m-%d')

    jp_df = pdr.DataReader('^N225', 'yahoo', date_min, date_max)
    eu_df = pdr.DataReader('^N100', 'yahoo', date_min, date_max)
    cn_df = pdr.DataReader('000001.SS', 'yahoo', date_min, date_max)

    #JP Market
    jp_df_adj = jp_df.reset_index()
    jp_df_adj['daily_change'] = jp_df_adj['Adj Close'] - jp_df_adj['Open']
    jp_df_adj['up_down'] = np.where(
        jp_df_adj['daily_change'] < 0,
        0.0,
        1.0
    )
    jp_df_adj = jp_df_adj.rename(columns = {'up_down':'JP', 'Date': 'date'})[['JP', 'date']]
    final_stock_df = final_stock_df.merge(jp_df_adj, on = 'date', how = 'left')

    #EU Market
    eu_df = eu_df.reset_index()
    eu_df['daily_change'] = eu_df['Adj Close'] - eu_df['Open']
    eu_df['up_down'] = np.where(
        eu_df['daily_change'] < 0,
        0.0,
        1.0
    )
    eu_df = eu_df.rename(columns = {'up_down':'EU', 'Date': 'date'})[['EU', 'date']]
    final_stock_df = final_stock_df.merge(eu_df, on = 'date', how = 'left')

    #CN Market
    cn_df = cn_df.reset_index()
    cn_df['daily_change'] = cn_df['Adj Close'] - cn_df['Open']
    cn_df['up_down'] = np.where(
        cn_df['daily_change'] < 0,
        0.0,
        1.0
    )
    cn_df = cn_df.rename(columns = {'up_down':'CN', 'Date': 'date'})[['CN', 'date']]
    final_stock_df = final_stock_df.merge(cn_df, on = 'date', how = 'left')

    final_stock_df = final_stock_df.fillna(0.0)

    return final_stock_df


def scale_Data(final_stock_df):

    df_filtered = final_stock_df.drop(['date', 'ticker', '5_short_avg', '20_long_avg', '13_short_avg', '49_long_avg', '50_short_avg', '200_long_avg', 'diff_5_20', 'diff_13_49', 'diff_50_200', 'true_range'], axis = 1)

    scaler = MinMaxScaler()

    scaled_df = df_filtered.copy()

    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df))
    scaled_df.columns = list(df_filtered.columns)

    return scaled_df


def adjust_Response_Var(scaled_df):

    shift_list = list(scaled_df['up_down'][1:]) 
    shift_list = shift_list + [0.0]
    scaled_df['up_down'] = shift_list

    yesterday_df = scaled_df.iloc[max(scaled_df.index)]
    scaled_df = scaled_df.iloc[:max(scaled_df.index)]

    return scaled_df, yesterday_df


def logit_Model(scaled_df):

    df_clean = scaled_df.copy()

    adj_X = df_clean.drop('up_down', axis = 1)
    adj_y = df_clean["up_down"]

    x_train, x_test, y_train, y_test = train_test_split(adj_X, adj_y, test_size=0.35, random_state=42)

    clf = LR().fit(x_train,y_train)

    model_score = clf.score(x_test,y_test)

    predictions = clf.predict(x_test)
    pred_probs = clf.predict_proba(x_test)

    probabilities = []

    for x,i in enumerate(predictions):
        
        if i == 0:

            probabilities.append(pred_probs[x][0])
            
        else:

            probabilities.append(pred_probs[x][1])

    x_test['preds'] = predictions
    x_test['reals'] = y_test
    x_test['prob'] = probabilities

    cm = metrics.confusion_matrix(y_test, predictions)

    return clf, model_score, cm

######################################################################################################################################################################################################################

st.title("Stock Direction Predictor")

stock = st.text_input('Input Stock Ticker Here:')

stock_news_df = get_Stock_News_Data(stock = stock, token = token)

stock_news_df_en = df_Data_Enhancements(stock_news_df = stock_news_df)

stock_news_df_date_adj = df_Date_Adjustments(stock_news_df = stock_news_df_en, stock = stock)

merged_df = get_Stock_Data(stock_news_df = stock_news_df_date_adj, stock = stock)

fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['Close'], name='Close',
                         line=dict(color='royalblue', width=2.5)))

fig.update_layout(title=f'{stock} Price Trend',
                   xaxis_title='Date',
                   yaxis_title='Price (USD)')

st.plotly_chart(fig)

merged_df_ind = create_MA_Indicators(merged_df = merged_df)

merged_df_tr = calculate_TrueRange(merged_df = merged_df_ind)

latest_atr = calculate_ATR(merged_df = merged_df_tr)

final_stock_df = get_INTL_Markets_Data(merged_df = merged_df_tr)

scaled_df = scale_Data(final_stock_df = final_stock_df)

scaled_df_exc_yest, yesterday_df = adjust_Response_Var(scaled_df)

clf, model_score, cm = logit_Model(scaled_df = scaled_df_exc_yest)

##############################################################################################Return Values############################################################################################################

predictor_df = yesterday_df.drop('up_down')

predictor_df = predictor_df.values.reshape(1,-1)

prediction = clf.predict(predictor_df)[0]


if prediction == 0:

    probability = round(clf.predict_proba(predictor_df)[0][0],4)

else:

    probability = round(clf.predict_proba(predictor_df)[0][1],4)


if prediction == 0:

    direction = f"{stock} is going down"

else:

    direction = f"{stock} is going up"


st.write("Stock:", stock)
st.write(f"Stock Predictor Model's Accuracy: {round(model_score * 100, 2)}%")
st.write("Prediction for tomorrow:", direction)
st.write(f"Prediction's probability: {round(probability * 100, 2)}%")

print("Done!")
