from fredapi import Fred
import pandas as pd

#IMPORTANT note: to run this file, an API key is needed to be input into line 8 from https://fred.stlouisfed.org/

#storing all Fred data using an api key from the St. Louis Federal Reserve's website
#important note: to run this file, need to add your own api key as a string within the brackets
fred = Fred(api_key = '')

#listing off the column names for the equivalent data keys from the Fred api
series_ids = {
    '3-Month Treasury Yield': 'TB3MS',  
    'CPI': 'CPIAUCSL',  
    'Home Sales': 'HSN1F',  
    'Unemployment Rate': 'UNRATE'  
}

#storing the data for each individual key in a new dictionary
data_frames = {}
for name, series_id in series_ids.items():
    data = fred.get_series(series_id)
    data_frames[name] = data

#creating a joined Pandas DataFrame for all four macroeconomic factors' datasets
macro_data = pd.DataFrame(data_frames)

#slicing the DataFrame to get rid of the unnecessary years worth of data
start_date = '1985-01-01'
macro_data = macro_data[macro_data.index >= start_date]

#initializing a temporary DataFrame so that features can be derived
temp = macro_data.copy()

#initializing some time horizons for the macroeconomic features to be derived; note that Fred's data is reported monthly,
#so the time horizons represent months, not days
horizons = [2, 6, 12, 24, 48]
metrics = ['3-Month Treasury Yield', 'CPI', 'Home Sales', 'Unemployment Rate']

#looping through each metric and determining if there was an increase from the previous month, represented by a binary value
#a column called 'DTD_Increase' is added for each metric to signify this increase
for metric in metrics:
    temp['Yesterday_' + metric] = temp[metric].shift(1)
    temp[metric + '_DTD_Increase'] = (temp[metric] > temp['Yesterday_' + metric]).astype(int)
    macro_data[metric + '_DTD_Increase'] = temp[metric + '_DTD_Increase']

#nested for loop to create moving averages and trends for each time horizon and each macroeconomic feature;
#process is similar to that of the function 'derive_features' in 'Model_Builder'
#each time a feature is derived in the temporary DataFrame, it is copied over to the actual DataFrame, macro_data
for horizon in horizons:
    for metric in metrics:
        rolling_average = macro_data.rolling(horizon).mean()
        ratio_column = metric + '_Ratio_Last_' + str(horizon) + '_Months'
        macro_data[ratio_column] = macro_data[metric] / rolling_average[metric]

        trend = macro_data.shift(1).rolling(horizon).sum()[metric + '_DTD_Increase']
        trend_column = metric + '_Last_' + str(horizon) + '_Months_Trend'
        macro_data[trend_column] = trend

#adjusting the macro_data to have values for each day. This is necessary in order to properly combine it with the instance variable datasets for each model,
#as their values are reported for each trading day. Since Fred data is adjusted only monthly instead of daily, each day's increase or decrease is relative to
#the most recent report given by the Fred, not necessarily the DTD increase. ffill() accomplishes this purpose
macro_data = macro_data.resample('D').ffill()
macro_data.ffill(inplace = True)

#adjusts the timezone format of the DataFrame's index so it is compatible to be merged with the instance variable's DataFrames
macro_data.index = macro_data.index.tz_localize(None)

#removes the unnecessary columns from the DataFrame
del macro_data['3-Month Treasury Yield']
del macro_data['CPI']
del macro_data['Home Sales']
del macro_data['Unemployment Rate']
del macro_data['3-Month Treasury Yield_DTD_Increase']
del macro_data['CPI_DTD_Increase']
del macro_data['Home Sales_DTD_Increase']
del macro_data['Unemployment Rate_DTD_Increase']

#removes the unnecessary rows from the DataFrame
macro_data = macro_data.loc['1990-01-01':'2024-04-30']

#macro_data is what is stored as the class variable 'macro_data' in 'Model_Builder'