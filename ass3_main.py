# load in appropriate packages
import pandas as pd
import numpy as np

# load in and prepare data
data_train = pd.read_csv('Bond_MidModelTraining.csv')
data_test  = pd.read_csv('Bond_QuoteLive.csv')

data_train = data_train.ffill()
data_train = data_train.bfill() # because the first entry in some columns is a nan too

data_test = data_test.ffill()
#data_test = data_test.bfill() # same for the test set

# check if both sets are ordered properly by epochhours
#if data_train.iloc[:,0] == data_train.sort_values(0)

if not np.all(data_train.sort_values('epochhours').index == data_train.index):
    data_train = data_train.sort_values('epochhours', ignore_index=True)

if not np.all(data_test.sort_values('epochhours').index == data_test.index):
    data_test = data_test.sort_values('epochhours', ignore_index = True)

np.all(data_train.index == data_train.sort_values('epochhours').index)
np.all(data_test.index == data_test.sort_values('epochhours').index)

data_train['midDealerQuotes'] = (data_train.firm_executable_bid + data_train.firm_executable_ask) / 2
data_train['midMarketEstimate'] = (data_train.market_estimate_bid + data_train.market_estimate_ask) / 2