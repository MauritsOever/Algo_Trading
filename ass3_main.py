# load in appropriate packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
class Assignment3():

    def __init__(self, train_data_file, test_data_file):

        self.train_data = pd.read_csv(train_data_file)
        self.test_data = pd.read_csv(test_data_file)

    def fill_na(self, data):

        data.ffill()
        data.bfill()

    def check_order(self, data):
        if not np.all(data.sort_values('epochhours').index == data.index):
            data = data.sort_values('epochhours')

# HERE WE RUN THE CODE

# load in and prepare data
train_data_file = r'C:\Users\gebruiker\Documents\GitHub\Algo_Trading\Bond_MidModelTraining.csv'
test_data_file  = r'C:\Users\gebruiker\Documents\GitHub\Algo_Trading\Bond_QuoteLive.csv'

# Instantiate Assignment3

assignment = Assignment3(train_data_file, test_data_file)

# Fill nans.
assignment.fill_na(assignment.train_data)
assignment.fill_na(assignment.test_data)

# check if both sets are ordered properly by epochhours
assignment.check_order(assignment.train_data)
assignment.check_order(assignment.test_data)



# np.all(data_train.index == data_train.sort_values('epochhours').index)
# np.all(data_test.index == data_test.sort_values('epochhours').index)

# data_train['midDealerQuotes'] = (data_train.firm_executable_bid + data_train.firm_executable_ask) / 2
# data_train['midMarketEstimate'] = (data_train.market_estimate_bid + data_train.market_estimate_ask) / 2

# plt.plot(data_train.midDealerQuotes)