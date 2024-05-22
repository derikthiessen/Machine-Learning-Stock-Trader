import requests
from bs4 import BeautifulSoup
import yfinance as yf

'''
Webscraping function using BeautifulSoup to return the tickers of the current top 100 largest market cap stocks, as listed on yahoo finance
Return type: list of strings, where the strings represent the scraped tickers
'''

def get_top_100_tickers():
    #sets the necessary variables to allow BeautifulSoup to scrape yahoo finance
    url = 'https://finance.yahoo.com/screener/unsaved/89c2964c-4625-49c0-9d59-8b0b090a86e6?offset=0&count=100'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    #initializes a list to hold the tickers, and stores the rows of data scraped from yahoo finance in the 'rows' variable
    tickers = []
    table = soup.find('table', {'class': 'W(100%)'})
    rows = table.find_all('tr')[1:101]  
    
    #parses each row to return just the ticker, and appends these tickers to the list of all tickers
    for row in rows:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)
    
    return tickers

#calls the webscraping function so that the top 100 tickers on yahoo finance can be stored
top_100_tickers = get_top_100_tickers()

#initializes a list to hold the tickers that will be usable in the sense that models can be built for them;
#for a model to be able to be built, sufficient data is needed. The current threshold is stocks that have data that can be trained from 1995 onwards.
#note that the threshold in the for loop is set to 1990. This is to ensure that the derived features of lagging moving averages and trends can be calculated
#for the 1995 data and onwards, as these features require several years of past price data to be derived
#the ticker '^GSPC', which represents the S&P500 index, is not scraped from yahoo finance's page but is included in the list so a model can be built for it anyway
usable = ['^GSPC']

#loops through all of the top 100 tickers and checks if the ticker meets the necessary data quantity requirements.
#if the ticker does, it is appended to 'usable'
for ticker in top_100_tickers:
    data = yf.Ticker(ticker).history(period = 'max')

    first = data.index[0]
    year = first.year

    if year < 1990:
        usable.append(ticker)

#the variable 'usable' contains the list of tickers that will be serialized in the 'Serialization' module