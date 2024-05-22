import yfinance as yf
import pandas as pd
import Macro_Data as md
import Serialization as all_models

'''
Downloads the latest data of the input ticker and preprocesses it so that the model associated with the ticker can create the latest prediction.
The preprocessing step involves deriving the price features, as well as concatenating the latest macroeconomic data with the returned Pandas DataFrame.
The data is also cleaned to remove unnecessary rows and columns.

Parameters:
ticker (string) --> should be a ticker saved from the 'Usable_Stocks' module for which a new prediction is desired

Return type: Pandas DataFrame containing a singular row representing the stock's most recent full trading day.
All derived features and macroeconomic data are included with this singular row so that they can be fed into the model
'''

def preprocess_latest_data(ticker):
    #downloads the last 1000 trading days worth of data for the ticker so that features can be derived for it
    latest = yf.Ticker(ticker)
    latest = latest.history(period = '1000d')

    #deletes unnecessary columns from downloaded data
    if 'Dividends' in latest:
        del latest['Dividends']
        
    if 'Stock Splits' in latest:
        del latest['Stock Splits']

    #creates a temporary copy of the data so that the derived features can be created
    temp = latest.copy()

    #derives the DTD price increase of the stock, similar to the 'Target' column of the instance variables 'data' for the models.
    #this is needed in order to derive trend features
    temp['Yesterday'] = temp['Close'].shift(1)
    temp['DTD_Increase'] = (temp['Close'] > temp['Yesterday']).astype(int)

    #adds this newly derived column to the actual dataset
    latest['DTD_Increase'] = temp['DTD_Increase']

    #sets the time horizons for which the features should be derived for; these should match what was fed to the model when it was initially created
    time_horizons = [2, 5, 60, 250, 1000]

    #loops through the time horizons and creates the moving averages and trend features for the ticker
    for horizon in time_horizons:
        
        #rolling window calculation for the moving average ratios, similar to 'derive_features' in the 'Model_Builder' module
        #adds this new column to the DataFrame
        rolling_averages = latest.rolling(horizon).mean()
        ratio_column = 'Close_Ratio_' + str(horizon)
        latest[ratio_column] = latest['Close'] / rolling_averages['Close']

        #rolling window calculation for the trends, similar to 'derive_features' in the 'Model_Builder' module except 'DTD_Increase' is used instead of 'Target' (same output)
        #adds this new column to the DataFrame
        trends = latest.rolling(horizon).sum()['DTD_Increase']
        trend_column = 'Last_' + str(horizon) + '_Trend'
        latest[trend_column] = trends

    #adjusts the timezone format of the index so it can be merged with the macroeconomic data from the 'Macro_Data' module; merges these two DataFrames into one
    latest.index = latest.index.tz_localize(None)
    latest = pd.merge(latest, md.macro_data, left_index = True, right_index = True, how = 'left')
    
    #forward fills any missing values for the macroeconomic data due to it being reported monthly, not daily, and drops the unnecessary rows
    latest.ffill(inplace = True)
    latest = latest.dropna()

    #deletes the unnecessary rows
    del latest['Close']
    del latest['Open']
    del latest['High']
    del latest['Low']
    del latest['Volume']
    del latest['DTD_Increase']

    return latest


'''
Feeds the ticker and its data into the model previously saved in 'Serialization' so that a predicition for tomorrow's price can be generated

Parameters:
ticker (string) --> the ticker for which a new price prediction is desired
model (model class from 'Model_Builder') --> the model previously saved in 'Serialization' and corresponding to the input ticker

Return type: a list containing a single percentage representing the likelihood of the stock's price increasing during the next trading day
'''

def generate_predictions(ticker, model):
    #preprocesses the data for the input ticker by sending it to the preprocess_latest_data function
    latest_data = preprocess_latest_data(ticker)

    #sends the latest data to the input model's instance method 'future_predictions' to generate the latest prediction
    prediction = model.future_predictions(latest_data)

    return prediction


'''
Generates predictions for all the models saved in 'Serialization' by using the 'generate_predictions' function

Return type: dictionary which uses the tickers as keys and their price increase predictions for the next trading day
'''

def generate_all_predictions():
    #calls 'Serialization' to load all the models into a dictionary that includes the ticker as its key and the associated model as its value
    models = all_models.load_all_models()
    
    #initializes a dictionary variable to hold the tickers as keys and their price increase predictions for the next day
    predictions = dict()

    #loops through all the tickers and models to generate a prediction for each;
    #formats these predictions as a neat string with the values rounded to two decimal places
    for ticker, model in models.items():
        prediction = generate_predictions(ticker, model)
        prediction_percentage = round(prediction[0] * 100, 2)
        predictions[ticker] = str(prediction_percentage) + '%'

    return predictions

#uncomment the below line and run the file to see the list of predictions in the terminal
#important note: the Fred API key in 'Macro_Data' and the paths in 'Serialization' all need to be adjusted in order to run the file on your own

#print(generate_all_predictions())