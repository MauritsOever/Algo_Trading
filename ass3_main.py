# load in appropriate packages
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
class Assignment3():

    def __init__(self, train_data_file, test_data_file):

        self.train_data = pd.read_csv(train_data_file)
        self.test_data = pd.read_csv(test_data_file)

    def fill_na(self):

        self.train_data = self.train_data.ffill()
        self.test_data = self.train_data.bfill()

        self.test_data.ffill()
        self.test_data.bfill()

    def check_order(self, data):

        if not np.all(data.sort_values('epochhours').index == data.index):

            data = data.sort_values('epochhours')

    def calculate_mids(self):

        self.train_data['midDealerQuotes'] = (self.train_data['firm_executable_bid']
                                              + self.train_data['firm_executable_ask']) / 2

        self.train_data['midMarketEstimate'] = (self.train_data['market_estimate_bid']
                                              + self.train_data['market_estimate_ask']) / 2

        self.test_data['midDealerQuotes'] = (self.test_data['firm_executable_bid']
                                              + self.test_data['firm_executable_ask']) / 2

        self.test_data['midMarketEstimate'] = (self.test_data['market_estimate_bid']
                                              + self.test_data['market_estimate_ask']) / 2

    def generate_plots(self):

        # Create figure.
        fig, axs = plt.subplots(2, 2, sharex = True)

        fig.suptitle('Epoch Hours vs LastPrice, midDealerQuotes and midMarketEstimate')

        axs[0, 0].plot(
            self.train_data['epochhours'],
            self.train_data['last_price'],
            color = 'orange'
        )
        axs[0, 0].set_title("Last Price vs Epoch Hours")

        axs[0, 1].plot(
            self.train_data['epochhours'],
            self.train_data['midDealerQuotes'],
            color = 'green'
        )
        axs[0, 1].set_title("midDealerQuotes vs Epoch Hours")

        axs[1, 0].plot(
            self.train_data['epochhours'],
            self.train_data['midMarketEstimate'],
            color = 'blue'
        )
        axs[1, 0].set_title("midMarketEstimate vs Epoch Hours")

        axs[1, 1].set_title("All vs Epoch Hours")

        axs[1, 1].plot(
            self.train_data['epochhours'],
            self.train_data['last_price'],
            label = 'Last Price',
            color = 'orange',
            alpha = 0.5
        )

        axs[1, 1].plot(
            self.train_data['epochhours'],
            self.train_data['midDealerQuotes'],
            label = 'midDealerQuotes',
            color = 'green',
            alpha = 0.3
        )

        axs[1, 1].plot(
            self.train_data['epochhours'],
            self.train_data['midMarketEstimate'],
            label = 'midMarketEstimate',
            color = 'blue',
            alpha = 0.3
        )

        plt.tight_layout()
        plt.legend()
        plt.show()


# HERE WE RUN THE CODE

# Setup filepaths.

# Get current working directory.
cwd = os.curdir

# Get data file paths.
train_data_file = os.path.join(cwd, 'Bond_MidModelTraining.csv')
test_data_file = os.path.join(cwd, 'Bond_QuoteLive.csv')

# Instantiate Assignment3

assignment = Assignment3(train_data_file, test_data_file)

# Fill nans.
assignment.fill_na()

# check if both sets are ordered properly by epochhours
assignment.check_order(assignment.train_data)
assignment.check_order(assignment.test_data)

assignment.calculate_mids()

assignment.generate_plots()
# np.all(data_train.index == data_train.sort_values('epochhours').index)
# np.all(data_test.index == data_test.sort_values('epochhours').index)

# data_train['midDealerQuotes'] = (data_train.firm_executable_bid + data_train.firm_executable_ask) / 2
# data_train['midMarketEstimate'] = (data_train.market_estimate_bid + data_train.market_estimate_ask) / 2

# plt.plot(data_train.midDealerQuotes)