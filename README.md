This repository contains all the files for a machine-learning stock predictor. 

NOTE: to run these files on your local machine, some edits need to be made some of the variables. These include:

* In ‘Macro_Data.py’, an API key from the St. Louis Federal Reserve (https://fred.stlouisfed.org/) is needed so that the latest macroeconomic data can be accessed.
    o A note at the top of the file specifies where this API key is needed to be input.

* In ‘Serialization.py’, several default values for directory paths are needed to be updated so that the models and precision score values can be saved locally. Saving the models created using ‘Model_Builder.py’ is preferred to rerunning ‘Model_Builder.py’ several times as the models take a significant amount of time to create.
    o A note at the top of the file specifies all the lines where the directory variable should be updated.

* In ‘Emailer.py’, a credentials.json file is needed to be downloaded from Google Cloud’s Gmail API service. The path to this file should then be updated within the document. Additionally, the intended recipient of the email needs to have their email address input (preferably, but not necessarily, a Gmail address).
    o A note at the top of the file specifies the two lines where these values need to be input.

The files within the repository are described as follows:
* Model_Builder.py: this file contains the code for the ‘Model’ class that creates machine learning models for different stock tickers.
        o A Random Forest Classifier model was employed for creating all the machine learning models. This is because stock market data tends to be non-linearly correlated, which Random Forest Classifier             models can capture well. Additionally, these models allow for easy fine-tuning by pruning the number of leaves or adjusting the number of trees made. Since stock market models are easy to                     overfit, this ease of fine-tuning was preferred.
o Models created by this file all record their precision scores as instance variables. Precision scores are preferred to accuracy in the case of these models as the intended use of the models is only to trade price upswings, not price downswings. Therefore, precision scores, which capture the true positive rate instead of the overall correctness of the model, are preferred. Additionally, the threshold that the models use to determine price increases is a 60% probability, not the default 50% probability, meaning more emphasis is placed on the true positive rate as opposed to the overall correctness.
o 31 models were trained and saved in my testing, and the average precision score of the models is around 54%.
o To determine the precision scores of the models, iterative backtesting based on 10 years of recent data and a step increase of 1 year was conducted. Initial data fed to train the models included 30 years of data, where 75% represented the training sets, and 25% represented the testing sets. The most recent data included in training models goes to April 30th, 2024.
o Models built using this file are trained on the following features: 5 moving average ratios of the price close over different time horizons (default values set to 2, 5, 60, 250, and 1000 trading days), 5 price trends of the number of days in the past ‘x’ days where the price increased (default values again set to 2, 5, 60, 250, and 1000 trading days), and the macroeconomic data generated from Macro_Data.py (see below). This means each model is trained on a total of 30 parameters.
o To have enough data to train a model, stocks needed at least 30 years’ worth of price data. Stocks that are eligible to be fed to the class are determined in Usable_Stocks.py.

* Macro_Data.py: this file downloads the latest macroeconomic data as reported by the Federal Reserve Bank of St. Louis. It then preprocesses this data and derives some features that are stored as a class variable in Model_Builder.py so that all models can use this data as inputs.
o The macroeconomic data used as inputs includes moving average ratios and trends over 5 separate time horizons (default values set to 2, 6, 12, 24, and 48 months) for 4 separate variables: CPI, interest rates, home sales, and the unemployment rate.

* Usable_Stocks.py: this file uses BeautifulSoup to scrape the top 100 largest tickers in terms of market capitalization off Yahoo Finance. These tickers then have their price data downloaded, and if enough price data is available to train models on, are appended to a list of usable tickers. These tickers are later saved in Serialization.py and represent the 31 tickers that I trained my models on.
o The threshold for having enough data was set to be ~30 years where all the model’s features were available. This meant that stocks needed to have data dating back to at least 1990, as some of the models’ features involved the use of trailing data up to 1000 trading days (or 4 full years).

* Serialization.py: this file is used to save the outputs of Model_Builder.py and Usable_Stocks.py. Outputs can be saved to the user’s local desktop by adjusting the default path variables, as mentioned at the start of this README. Both these files (especially Model_Builder.py) take significant amounts of time and computing power to run, so saving previously created models and tickers scraped off Yahoo Finance are beneficial.
o This file employs Python’s built-in libraries, pickle and os, to create directories on the local computer as well as to dump and retrieve outputs from pickle files.

* New_Predictions.py: this file takes the saved models and tickers from Serialization.py and generates predictions for tomorrow’s price changes. To generate these predictions, the latest data for each saved ticker is downloaded, and the model features are derived from this data. Similarly, the latest macroeconomic data is downloaded from the Fred API so that the necessary features using that data can be included in the models’ inputs.
o This file returns tomorrow’s price increase predictions as a dictionary of ticker-prediction pairs that can be printed to the terminal. Alternatively, the model can send these predictions in email format to an intended recipient using Emailer.py

* Emailer.py: this file runs a script that collects all the saved models’ price increase predictions for the next trading day and emails them to an intended recipient. This email also reports the precision scores of the models that are used to make predictions.
o This file makes use of Google’s cloud computing platform and its Gmail API to send emails to an intended recipient.

* Finally, Tickers_List.pkl includes the saved tickers from Usable_Stocks.py, and the folder Saved_Models includes a bunch of .pkl files containing models I created for the tickers in Tickers_List.pkl. In Saved_Models, there is also a pickle file, Precision_Scores.pkl, which has the precision scores of all the saved models.

If you made it this far in my README, thank you! I would love some feedback on how I can improve this project, or some other ideas I can build next. You can reach me at derikt03@live.com
