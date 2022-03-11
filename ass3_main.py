# load in appropriate packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

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
            
    def calculate_mids(self):

        self.train_data['midDealerQuotes'] = (self.train_data['firm_executable_bid']
                                              + self.train_data['firm_executable_ask']) / 2

        self.train_data['midMarketEstimate'] = (self.train_data['market_estimate_bid']
                                              + self.train_data['market_estimate_ask']) / 2

        self.test_data['midDealerQuotes'] = (self.test_data['firm_executable_bid']
                                              + self.test_data['firm_executable_ask']) / 2

        self.test_data['midMarketEstimate'] = (self.test_data['market_estimate_bid']
                                              + self.test_data['market_estimate_ask']) / 2
    
    def calculate_returns(self, data):
        # LastPrice, midDealerQuotes, midMarketEstimate 
        data['LastPrice_rets'] = (data['last_price'] / data['last_price'].shift(1) -1) * 10000
        data['midDealerQuotes_rets'] = (data['midDealerQuotes'] / data['midDealerQuotes'].shift(1) -1) * 10000
        data['midMarketEstimate_rets'] = (data['midMarketEstimate'] / data['midMarketEstimate'].shift(1) -1) * 10000
    
    def make_matrices(self):
        self.Y = np.array(self.train_data['LastPrice_rets'][1:])
        self.X = np.ones((len(self.train_data)-1 , 3))
        self.X[:,1] = self.train_data['midDealerQuotes_rets'][1:]
        self.X[:,2] = self.train_data['midMarketEstimate_rets'][1:] 
        return
    
    def est_OLS(self):
        self.betas = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        print('estimated betas are equal to')
        print(f'constant                              = {round(self.betas[0], 4)}')
        print(f'midDealerQuotes returns coefficient   = {round(self.betas[1], 4)}')
        print(f'midMarketEstimate returns coefficient = {round(self.betas[2], 4)}')
        
        return 
    
    def get_yhat(self):
        self.yhat = self.X @ self.betas
        return
    
    def get_r2(self):
        self.r2 = 1 - np.sum( (self.Y - self.yhat)**2) / np.sum( (self.Y - np.mean(self.Y))**2)
        print(f'R squared = {self.r2}')
        return 
    
    def get_MAE(self):
        self.MAE = np.mean(abs(self.Y - self.yhat))
        
        print(f'MAE       = {self.MAE}')
        return 
        
    def gboost(self):
        
        x_train = self.train_data[[
                'epochhours',
                'firm_executable_bid',
                'firm_executable_ask',
                'market_estimate_bid',
                'market_estimate_ask',
                'midDealerQuotes',
                'midMarketEstimate'
            ]].iloc[1:,:]

        y_train = self.train_data['last_price'].iloc[:-1]
        
        print(len(x_train))
        print(len(y_train))
        
        reg = GradientBoostingRegressor(n_estimators = 500, 
                                        max_depth=4, 
                                        min_samples_split=5, 
                                        learning_rate=0.01,
                                        loss = 'lad')
        reg.fit(x_train, y_train)
        
        x_test = self.test_data[[
                'epochhours',
                'firm_executable_bid',
                'firm_executable_ask',
                'market_estimate_bid',
                'market_estimate_ask',
                'midDealerQuotes',
                'midMarketEstimate'
            ]]
        
        self.test_data.iloc[1:, 6] = reg.predict(x_test.iloc[:-1,:])
        
        # still need to get MAE and stuff
        errors = np.array(self.test_data['last_price'] - self.test_data['FlowTradersMidpoint '])[1:]
        MAE    = np.mean(abs(errors))
        print(MAE)
        
        
    def flow_bid_and_ask(self):
        self.test_data.iloc[:, 7] = self.test_data.iloc[:, 6] * 1.001
        self.test_data.iloc[:, 8] = self.test_data.iloc[:, 6] * 0.999
        
        
    def gboost2(self):

        X = self.train_data[[
                'epochhours',
                'firm_executable_bid',
                'firm_executable_ask',
                'market_estimate_bid',
                'market_estimate_ask',
                'midDealerQuotes',
                'midMarketEstimate'
            ]]

        y = self.train_data['last_price']

        model = GradientBoostingRegressor()

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv = cv, n_jobs = -1, error_score = 'raise')

        print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        
        return 
    
# HERE WE RUN THE CODE

# load in and prepare data
# Get current working directory
cwd = os.getcwd()

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

# calculate mid prices
assignment.calculate_mids()

# calculate returns
assignment.calculate_returns(assignment.train_data)
assignment.calculate_returns(assignment.test_data)

# OLS here
# assignment.make_matrices()
# assignment.est_OLS() # estimate OLS using analytical solution
# assignment.get_yhat() # calculate the estimated Y
# assignment.get_r2() # calculate r2
# assignment.get_MAE() # calculate mean absolute error

# ML here
assignment.gboost()

# fill in flow bid and ask
assignment.flow_bid_and_ask()

