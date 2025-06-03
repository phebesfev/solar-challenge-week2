import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import talib
import numpy as np
import re
import pynance as pn

class NewsEDA:
    def __init__(self, df):
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'],format="%Y-%m-%d %H:%M:%S")

    def headline_length_stats(self):
        self.df['headline_length'] = self.df['headline'].apply(len)
        print(self.df['headline_length'].describe())

    def publisher_article_count(self, top_n=20):
        publisher_counts = self.df['publisher'].value_counts().head(top_n)
        print(publisher_counts)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=publisher_counts.index, y=publisher_counts.values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Publishers by Article Count')
        plt.ylabel('Article Count')
        plt.xlabel('Publisher')
        plt.tight_layout()
        plt.show()


    def publication_trend(self):
        self.df['day'] = self.df['date'].dt.date
        trend = self.df.groupby('day').size()
        trend.plot(figsize=(12, 5), title='Articles over Time')
        plt.ylabel('Count')
        plt.show()

    def topic_modeling(self):
        all_text = ' '.join(self.df['headline'].dropna().values)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        common_words = word_freq.most_common(20)
        words_df = pd.DataFrame(common_words, columns=['word', 'freq'])
        sns.barplot(x='freq', y='word', data=words_df)
        plt.title('Top Keywords in Headlines')
        plt.show()

    def publication_frequency_by_time(self):
        freq = self.df['date'].dt.to_period('D').value_counts().sort_index()
        freq.plot(figsize=(14, 5), title='Daily Publication Frequency')
        plt.ylabel('Articles')
        plt.show()

    def peak_publishing_times(self):
        self.df['hour'] = self.df['date'].dt.hour
        sns.countplot(x='hour', data=self.df)
        plt.title('Publishing Hour Distribution')
        plt.show()

    def analyze_publishers(self):
        self.df['publisher_domain'] = self.df['publisher'].apply(
            lambda x: x.split('@')[-1] if '@' in x else x
        )
        domain_counts = self.df['publisher_domain'].value_counts()
        print(domain_counts.head())
        domain_counts.head(10).plot(kind='bar', title='Top Publisher Domains')
        plt.show()
        
class TechnicalAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.prices = self.df['Close'].to_numpy(dtype='float64')

    def sma(self, window):
        return talib.SMA(self.prices, timeperiod=window)

    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        self.df['RSI'] = talib.RSI(self.prices, timeperiod=period)
        return self.df

    def add_macd(self,
                 fastperiod: int = 12,
                 slowperiod: int = 26,
                 signalperiod: int = 9) -> pd.DataFrame:
        macd, macd_signal, macd_hist = talib.MACD(
            self.prices,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macd_signal
        self.df['MACD_Hist'] = macd_hist
        return self.df

    def get_data(self) -> pd.DataFrame:
        return self.df
    
class FinancialAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def cumulative_return(self):
        self.df['Cumulative Return'] = self.df['Adj Close'] / self.df['Adj Close'].iloc[0] - 1
        return self.df

    def daily_volatility(self):
        self.df['Daily Return'] = self.df['Adj Close'].pct_change()
        self.df['Daily Volatility'] = self.df['Daily Return'].rolling(window=20).std()
        return self.df
    def plot_cumulative_return(self):
        if 'Cumulative Return' not in self.df.columns:
            self.cumulative_return()
        plt.figure(figsize=(12, 5))
        plt.plot(self.df['Date'], self.df['Cumulative Return'], label='Cumulative Return')
        plt.title('Cumulative Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_daily_volatility(self):
        if 'Daily Volatility' not in self.df.columns:
            self.daily_volatility()
        plt.figure(figsize=(12, 5))
        plt.plot(self.df['Date'], self.df['Daily Volatility'], label='20-Day Rolling Volatility', color='orange')
        plt.title('Daily Volatility Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)
        plt.legend()
        plt.show()

