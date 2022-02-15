# Useful Libraries
import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt


#%%
import yfinance as yf

# your answer here
df = yf.download("KO PEP", start="2014-11-01", end="2016-06-01")['Adj Close']

plt.plot(df.index, df['KO'])
plt.plot(df.index, df['PEP'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% 
# use coint() to check for cointegration...

score, pvalue, _ = coint(df['KO'], df['PEP']) # there seems to be cointegration... we can reject null
                                              # almost at the 5 perfect level


#%% 
# Construct a linear regression to find a coefficient for the linear 
# combination of PEP and KO that makes their spread stationary

# k 

pep_t = np.array(df['PEP'])[1:]
ko_t = np.array(df['KO'])[1:]

vY = pep_t
mX = np.ones((396, 2))
mX[:,1] = ko_t

betas = np.linalg.inv(mX.T@mX)@mX.T@vY

spread = vY - mX@betas
plt.plot(spread)

#%% 
# construct a algo based on spread

#normalize spread

# construct long or short signals based on norm_spread

# track purchases and returns based on purchases