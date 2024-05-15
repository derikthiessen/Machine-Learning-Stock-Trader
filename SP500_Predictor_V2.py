import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class Model():
    def __init__(self, ticker, num_trees = 200, num_leaves = 50, horizon1 = 2, horizon2 = 5, horizon3 = 60, horizon4 = 250, horizon5 = 1000):
        self.data = self.prepare_data(ticker)
        self.predictors = self.derive_features(horizon1, horizon2, horizon3, horizon4, horizon5)
        self.both_sets = self.split_sets()
        self.training_set = self.both_sets[0]
        self.testing_set = self.both_sets[1]
        self.model = RandomForestClassifier(n_estimators = num_trees, min_samples_split = num_leaves, random_state = 1)
        self.predictions = self.backtest(self.data, self.model, self.predictors)
        self.precision_score = precision_score(self.predictions['Target'], self.predictions['Predictions'])

    def prepare_data(self, ticker):
        if not isinstance(ticker, str):
            raise ValueError('Ticker must be a string')
        
        try:
            data = yf.Ticker(ticker).history(period = 'max')
            if data.empty:
                raise ValueError('No data for the given ticker')
        except Exception:
            raise ValueError('Could not retrieve data for the given ticker')

        data.index = data.index.tz_localize(None)
        
        if 'Dividends' in data:
            del data['Dividends']
        
        if 'Stock Splits' in data:
            del data['Stock Splits']
        
        data['Tomorrow'] = data['Close'].shift(-1)
        data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

        return data
    
    def derive_features(self, horizon1, horizon2, horizon3, horizon4, horizon5):                
        time_horizons = [horizon1, horizon2, horizon3, horizon4, horizon5]

        for horizon in time_horizons:
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError('Time horizons must be positive integers')
        
        predictors = []

        for horizon in time_horizons:
            rolling_average = self.data.rolling(horizon).mean()
            ratio_column = 'Close_Ratio_' + str(horizon)
            self.data[ratio_column] = self.data['Close'] / rolling_average['Close']

            trend = self.data.shift(1).rolling(horizon).sum()['Target']
            trend_column = 'Last_' + str(horizon) + '_Trend'
            self.data[trend_column] = trend

            predictors.append(ratio_column)
            predictors.append(trend_column)

            self.data = self.data.dropna()

        return predictors
    
    def split_sets(self):
        first = str(self.data.index[0])
        year = int(first[:4])

        if year >= 1990:
            raise ValueError('Not enough data for this ticker to train a model')

        self.data = self.data.loc['1990-01-01':'2024-04-30'].copy()
        total_days = int(self.data.shape[0])
        split_index = int(total_days * 0.75)
        
        training_set = self.data[:split_index]
        testing_set = self.data[split_index:]
        data_sets = [training_set, testing_set]

        return data_sets
    
    def predict(self, train, test, predictors, model):
        model.fit(train[predictors], train['Target'])
        prediction_percentages = self.model.predict_proba(test[predictors])[:, 1]
        prediction_percentages = pd.Series(prediction_percentages, index = test.index, name = 'Prediction_Percentages')
        combined = pd.concat([test['Target'], prediction_percentages], axis = 1)
        combined['Predictions'] = (combined['Prediction_Percentages'] > 0.6).astype(int)

        return combined
    
    def backtest(self, data, model, predictors, start = 2500, step = 250):
        all_predictions = []

        for i in range(start, data.shape[0], step):

            train = data.iloc[0 : i].copy()
            test = data.iloc[i : (i + step)].copy()
        
            prediction = self.predict(train, test, predictors, model)
            all_predictions.append(prediction)

        combined = pd.concat(all_predictions)

        return combined

test = Model('^GSPC')
print(test.precision_score)
