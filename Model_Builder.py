import yfinance as yf
import pandas as pd
import Macro_Data as md
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

'''
Class used to create the prediction models for individual stocks

Parameters:
ticker (string) --> represents the ticker for which a model will be created
num_trees (int) --> represents the number of trees that the Random Forest Classifier model will create for the given ticker. Default set to 300
num_leaves (int) --> represents the minimum number of leaves a decision tree node can have. Default set to 50
horizon1, horizon2, horizon3, horizon4, horizon5 (int) --> time horizons used in the creation of the rolling averages for price and rolling trends of increases. Defaults set to 2, 5, 60, 250, 1000
'''

class Model():
    
    '''
    Class variables:
    macro_data (Pandas DataFrame) --> holds the values for rolling averages and trends in CPI, unemployment, interest rates, and home sales
    macro_predictors (list of string values) --> holds the column headers for the different predictors created
    '''

    macro_data = md.macro_data
    macro_predictors = macro_data.columns
    

    '''
    Instance variables:
    data (Pandas DataFrame) --> holds the instance data of rolling averages and trends based on the input ticker's recent price data
    predictors (list of string values) --> holds the column names for the data variable as well as the class variable, macro_predictors
    full_data (Pandas DataFrame) --> holds the instance data along with the class variable, macro_data, in one DataFrame
    both_sets (list of Pandas DataFrame values) --> holds the full_data instance variable sliced into two groups, one for training and the other for testing
    training_set (Pandas DataFrame) --> holds the training set
    testing_set (Pandas DataFrame) --> holds the testing set
    model (RandomForestClassifier) --> pointer to a variable of instance RandomForestClassifier which holds the final machine-learning model
    predictions (list of float values) --> holds the model's percentage predictions of backtesting on the ten most recent years of data for the ticker
    precision_score (float) --> holds the model's precision score, calculated using the instance variable predictions
    '''

    def __init__(self, ticker, num_trees = 300, num_leaves = 50, horizon1 = 2, horizon2 = 5, horizon3 = 60, horizon4 = 250, horizon5 = 1000):
        self.data = self.prepare_data(ticker)
        
        #as part of derive_features, macro_predictors are added to the predictors list for the model to consider
        self.predictors = self.derive_features(horizon1, horizon2, horizon3, horizon4, horizon5)

        #merges the instance variable's DataFrame with the class' macro_data; merge from left necessary to adjust macro_data to only include trading days
        self.full_data = pd.merge(self.data, Model.macro_data, left_index = True, right_index = True, how = 'left')
        
        #training set is stored in index zero of the list, testing set is stored in index one
        self.both_sets = self.split_sets()
        self.training_set = self.both_sets[0]
        self.testing_set = self.both_sets[1]

        #RandomForestClassifier models are instantiated based on default parameters of 300 trees and a minimum of 50 leaves per node. 
        #random_state is set to 1 to ensure the initial seed used to create the trees stays consistent; this is done for repeatability of results
        self.model = RandomForestClassifier(n_estimators = num_trees, min_samples_split = num_leaves, random_state = 1)

        self.predictions = self.backtest(self.full_data, self.model, self.predictors)
        
        #precision_score is used instead of accuracy as the models generated are meant to trade only on price upswings,
        #as the necessary percentage value for the model to consider its prediction to be an increase is set to 60% instead of the standard 50%
        self.precision_score = precision_score(self.predictions['Target'], self.predictions['Predictions'])


    '''
    Downloads and preprocesses ticker data so the machine learning model can use it.

    Parameters:
    ticker (string) --> the yahoo finance ticker of the stock for which a model should be built. Needs to be accessible on yahoo finance, else a ValueError is raised

    Return type: Pandas DataFrame, which is stored in the instance variable data
    '''

    def prepare_data(self, ticker):
        
        #ensures the input value is of type string
        if not isinstance(ticker, str):
            raise ValueError('Ticker must be a string')
        
        #tries to download the data to ensure that it is accessible on yahoo finance
        try:
            data = yf.Ticker(ticker).history(period = 'max')
            if data.empty:
                raise ValueError('No data for the given ticker')
        except Exception:
            raise ValueError('Could not retrieve data for the given ticker')
        
        #deletes unnecessary columns from downloaded data
        if 'Dividends' in data:
            del data['Dividends']
        
        if 'Stock Splits' in data:
            del data['Stock Splits']
        
        #creates a new column, 'Tomorrow', which allows for another new column, 'Target', to determine if the price increased from day to day.
        #'Target' will be fed into the machine learning model as the value we are trying to predict
        data['Tomorrow'] = data['Close'].shift(-1)
        data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

        #removes the timezone of the data so that it can be combined with the class variable macro_data
        data.index = data.index.tz_localize(None)

        return data
    

    '''
    Derives the instance variable features for the data, which are moving averages and price trends across different time horizons

    Parameters:
    horizon1, horizon2, horizon3, horizon4, horizon5 (int) --> the time horizons for which moving averages and price trends will be calculated for.
    These features are used to train the machine learning model.

    Return type: list of strings containing the new column names of the derived features. These predictors will indicate which columns to consider when training the model
    '''

    def derive_features(self, horizon1, horizon2, horizon3, horizon4, horizon5):                
        time_horizons = [horizon1, horizon2, horizon3, horizon4, horizon5]

        #check if any of the time horizons are not integers or are negative, raises a ValueError if they are
        for horizon in time_horizons:
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError('Time horizons must be positive integers')
        
        #initial list to hold the predictors
        predictors = []

        #loops through each of the input horizons, and creates both the moving average and trend data columns for the ticker
        for horizon in time_horizons:

            #uses a rolling window calculation of the mean to derive the moving average relative to the current day's closing price
            rolling_average = self.data.rolling(horizon).mean()
            ratio_column = 'Close_Ratio_' + str(horizon)
            self.data[ratio_column] = self.data['Close'] / rolling_average['Close']

            #uses a rolling window calculation of the sum of the 'Target' column, which represents price increases, to derive the price trend
            trend = self.data.shift(1).rolling(horizon).sum()['Target']
            trend_column = 'Last_' + str(horizon) + '_Trend'
            self.data[trend_column] = trend

            #adds both the moving average as well as the trend columns to the list of predictors for the machine learning model to consider
            predictors.append(ratio_column)
            predictors.append(trend_column)

            #removes rows which would not help the machine learning model due to some features having 'NaN';
            #'NaN' occurs because some days do not have enough prior data to calculate moving averages or trends for
            self.data = self.data.dropna()

        #appends the macro_predictors to the list of predictors for the machine learning model to consider
        for predictor in Model.macro_predictors:
            predictors.append(predictor)

        #returns the predictors, as the instance variable data is modified in place and thus does not need to be returned
        return predictors


    '''
    Takes the instance variable full_data and splits it into training and testing sets.

    Return type: list of Pandas DataFrames, where index zero represents the training set, and index one represents the testing set
    '''

    def split_sets(self):
        
        #gets a string value of the year of the first row of the data
        first = str(self.full_data.index[0])
        year = int(first[:4])

        #checks if the first year of the data is prior to 1995; if it is not, raises a ValueError that there will not be enough data to adequately train a model on
        if year >= 1995:
            raise ValueError('Not enough data for this ticker to train a model')

        #returns a copy of the data from January 1st, 1990 to April 30th, 2024 which will be the data that the model is used to train on
        self.full_data = self.full_data.loc['1990-01-01':'2024-04-30'].copy()
        
        #finds where to split the data for the training and testing sets; 75% of the data is given to the training set, and 25% is given to the testing set
        total_days = int(self.full_data.shape[0])
        split_index = int(total_days * 0.75)
        
        #splits the data into the appropriate sets
        training_set = self.full_data[:split_index]
        testing_set = self.full_data[split_index:]
        data_sets = [training_set, testing_set]

        return data_sets
    

    '''
    Fits the model based on the input training data, and then stores and returns predictions on the input testing set as binary values of price increases or decreases.

    Parameters:
    train (Pandas DataFrame) --> the data which will be used to fit the model based on the input predictors
    test (Pandas DataFrame) --> the remaining data which will be used to test the model based on the input predictors. The 'Target' column is the model's target
    predictors (list of strings) --> the instance variable 'predictors'
    model (RandomForest Classifier) --> the instance variable 'model'

    Return type: Pandas DataFrame, which contains the test set that was originally input with an additional column of binary values indicating a price increase/decrease
    '''
    
    def predict(self, train, test, predictors, model):
        #fits the model to determine 'Target' based on the predictors
        model.fit(train[predictors], train['Target'])

        #finds the percentage probabilities of an increase in price for the days in the test set; the slice at the end ensures only the likelihood of an increase is returned
        prediction_percentages = self.model.predict_proba(test[predictors])[:, 1]

        #merges the prediction percentages into a series so it can be merged into a Pandas DataFrame with the index values
        prediction_percentages = pd.Series(prediction_percentages, index = test.index, name = 'Prediction_Percentages')

        #creates a new Pandas DataFrame that includes all the test set's data values as well as the predictions
        combined = pd.concat([test['Target'], prediction_percentages], axis = 1)

        #adjusts the 'Predictions' column to only suggest price increases if the probability is greater than 60% as opposed to the default 50%;
        #purpose is to increase the model's precision (higher proportion of true positives) at the expense of accuracy (due to more false negatives)
        combined['Predictions'] = (combined['Prediction_Percentages'] > 0.6).astype(int)

        return combined
    

    '''
    Function to test the model's performance by backtesting on the ten most recent years of data. The model repeatedly backtests at a step of 250,
    which is equivalent to one full year worth of trading days.

    Parameters:
    data (Pandas DataFrame) --> the instance variable full_data
    model (RandomForestClassifier) --> the instance variable model
    predictors (list of strings) --> the instance variable predictors
    start (int) --> represents the first year that the model should begin backtesting, with the default set to 2500 (ten full trading years)
    step (int) --> represents the increase each time the model should backtest again, with the default set to 250 (one full trading year)

    Return type: Pandas DataFrame holding all the backtested predictions
    '''
    
    def backtest(self, data, model, predictors, start = 2500, step = 250):
        #initializing a list to store the DataFrames of predictions for each iteration of the backtesting
        all_predictions = []

        #loop through all the backtesting periods
        for i in range(start, data.shape[0], step):
            
            #makes new training and testing data sets depending on what iteration of backtesting the model is on
            train = data.iloc[0 : i].copy()
            test = data.iloc[i : (i + step)].copy()

            #feeds the data into the predict function to generate predictions for the current iteration, and adds these results to a list
            prediction = self.predict(train, test, predictors, model)
            all_predictions.append(prediction)

        #combines all the DataFrames of results into one DataFrame so they can be returned
        combined = pd.concat(all_predictions)

        return combined
    

    '''
    Function to be accessed outside of the class to use the instance variable model to generate predictions based on new data

    Parameters:
    latest_data (Pandas DataFrame) --> should be four trading year's worth of new data for a given ticker, including the latest macroeconomic and derived price features

    Return type: list of percentages representing the model's predictions of price increases for the latest input data
    '''
    
    def future_predictions(self, latest_data):
        return self.model.predict_proba(latest_data[self.predictors])[:, 1]