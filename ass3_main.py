# load in appropriate packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

class Assignment3():

    def __init__(self, train_data_file, test_data_file):

        self.train_data = pd.read_csv(train_data_file)
        self.test_data = pd.read_csv(test_data_file)

    def fill_na(self):
        
        self.train_data = self.train_data.ffill()
        self.train_data = self.train_data.bfill() # some first entries are nans
        
        self.test_data = self.test_data.ffill()

    def check_order(self, data):
        if not np.all(data.sort_values('epochhours').index == data.index):
            data = data.sort_values('epochhours')
    
    def calculate_returns(self, data):
        # LastPrice, midDealerQuotes, midMarketEstimate 
        print((data.last_price / data.last_price.shift(1) - 1) * 10000)
        data['LastPrice_rets'] = (data.last_price / data.last_price.shift(1) -1) * 10000
        #data.midDealerQuotesRets = (data.midDealerQuotes / data.midDealerQuotes.shift(1)) -1 * 10000
        #data.midMarketEstimate = (data.midMarketEstimate / data.midMarketEstimate.shift(1)) -1 * 10000
    
    def make_matrix(self):
        return
    
    def est_OLS(self):
        return
    
    def get_yhat(self, Betas, X):
        return
    
    def get_r2(self, yhat, Y):
        return 
    
    def get_MAE(self, yhat, Y):
        return
        
    
    
# HERE WE RUN THE CODE

# load in and prepare data
# Get current working directory.
cwd = os.curdir

# Get data file paths.
train_data_file = os.path.join(cwd, 'Bond_MidModelTraining.csv')
test_data_file = os.path.join(cwd, 'Bond_QuoteLive.csv')

# Instantiate Assignment3

assignment = Assignment3(train_data_file, test_data_file)

# Fill nans.
assignment.fill_na()
assignment.fill_na()

# check if both sets are ordered properly by epochhours
assignment.check_order(assignment.train_data)
assignment.check_order(assignment.test_data)

# calculate returns
assignment.calculate_returns(assignment.train_data)
assignment.calculate_returns(assignment.test_data)
