# Useful Libraries
import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt

import yfinance as yf

# your answer here
df = yf.download("KO PEP", start="2014-11-01", end="2016-06-01")['Adj Close']

plt.plot(pd.to_datetime(df.index).year, df['KO'])
#plt.plot(pd.to_datetime(df.index).year, df['PEP'])
