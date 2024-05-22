import pickle
import os
import Model_Builder
import Usable_Stocks

#IMPORTANT note: when calling any of these functions, the directory parameters must be adjusted to the desired paths of the user
#it is recommended to create a specific directory for all the pickle files of the models and the precision scores
#the lines where the directory parameters need to be adjusted are 20, 44, 63, 87, 147, and 178

'''
Saves all the tickers deemed eligible from the 'Usable' module into a file so that webscraping does not need to happen each time a model should be made

Parameters:
tickers (list of strings) --> represents the list of usable tickers. Default is set to the variable 'usable' from the module 'Usable_Stocks'
filename (string) --> the filename of the file where the tickers will be saved. Default is set to 'Tickers_List.pkl' as a pickle file
directory (string) --> where the file will be saved on the individual's computer
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def save_tickers(tickers = Usable_Stocks.usable, filename = 'Tickers_List.pkl', directory = ''):
    #ensures that the path specified by the directory exists; if not, the directory is made
    if not os.path.exists(directory):
        os.makedirs(directory)

    #creates the full path from the input filename and the directory
    filepath = os.path.join(directory, filename)

    #uses pickle to dump the data into a file
    with open(filepath, 'wb') as file:
        pickle.dump(tickers, file)


'''
Loads all the previously saved tickers from the file into a list of strings.

Parameters:
filename (string) --> should be the same as what was used when calling 'save_tickers'. Default set to 'Tickers_List.pkl', which is the same as the default from 'save_tickers'
directory (string) --> where the file should be loaded from the individual's computer. Should be the same as what was passed to 'save_tickers'

Return type: list of strings containing all the usable tickers
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def load_tickers(filename = 'Tickers_List.pkl', directory = ''):
    #creates the filepath for the file so the data can be loaded
    filepath = os.path.join(directory, filename)

    #uses pickle to open the file and return the list of tickers
    with open(filepath, 'rb') as file:
        return pickle.load(file)


'''
Saves individual models so that the 'Model' class in 'Model_Builder' does not need to be reinstantiated each time a model is needed

Parameters:
model (Model class from 'Model_Builder' module) --> the desired model to save
filename (string) --> the name of the pickle file where the model can be saved. Ensure that when calling the function. '.pkl' is at the end of this argument
directory (string) --> the name of the path where the model should be saved. Recommended to create a specific folder for the models
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def save_model(model, filename, directory = ''):
    #ensures that the path specified by the directory exists; if not, the directory is made
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #creates the full path from the input filename and the directory
    filepath = os.path.join(directory, filename)
    
    #uses pickle to dump the data into a file
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


'''
Loads an individual model previously saved via the 'save_model' function.

Parameters:
filename (string) --> the name of the pickle file that the desired model is saved in. Should be the same as the string previously passed to 'save_model'
directory (string) --> where the model should be loaded from the individual's computer. Should be the same as what was passed to 'save_model'

Return type: Model class of module 'Model_Builder'
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def load_model(filename, directory = ''):
    #creates the filepath for the file so the data can be loaded
    filepath = os.path.join(directory, filename)
    
    #uses pickle to open the file and return the model
    with open(filepath, 'rb') as file:
        return pickle.load(file)


'''
Saves all the models based on the tickers previously saved from 'save_tickers'
'''

def save_all_models():
    #determines which tickers should models be saved for based on what was previously passed to 'save_tickers'
    tickers = load_tickers()
    
    #loops through all the tickers
    for ticker in tickers:
        print('Starting saving file for ' + str(ticker))
        
        #creates a model variable using the 'Model_Builder' to be saved
        current_model = Model_Builder.Model(ticker)
        
        #creates the filename for the individual model
        filename = str(ticker) + '.pkl'

        #passes the model and the filename to the 'save_model' function
        save_model(current_model, filename)
        print('Finished saving file for ' + str(ticker))


'''
Loads all the models that were previously saved using 'save_all_models' into a dictionary with their tickers as the keys

Return type: list of models
'''

def load_all_models():
    #collects the necessary tickers for which models should be loaded for, and initializes a dictionary to hold the models
    tickers = load_tickers()
    models = dict()

    #loops through all the tickers to load each individual model and save it to the dictionary
    for ticker in tickers:
        current_model = load_model(str(ticker) + '.pkl')
        models[ticker] = current_model
    
    return models


'''
Saves all the precision scores of the individual models into a dictionary so they can be accessed as part of the results shared

Parameters:
filename (string) --> the name of the file where the precision scores will be saved. Should be a pickle file. Default set to 'Precision_Scores.pkl'
directory (string) --> the path where the precision scores should be saved. Should be a pickle file. Recommended to save these in the same directory as where all the models are saved
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def save_all_precision_scores(filename = 'Precision_Scores.pkl', directory = ''):
    #initialize a dictionary to hold the precision scores, with the tickers as the keys, and call 'load_tickers' to get the list of tickers previously saved
    model_scores = dict()
    tickers = load_tickers()

    #loops through the tickers and calls 'load_model' to load each individual model; then accesses the precision_score of these models to add them to the dictionary 
    for ticker in tickers:
        print('Loading model of ' + str(ticker))
        current_model = load_model(str(ticker) + '.pkl')
        model_scores[ticker] = current_model.precision_score
        print('Precision score for model of ' + str(ticker) + ': ' + str(current_model.precision_score)) 

    #creates the full filepath where the precision scores will be saved
    filepath = os.path.join(directory, filename)

    #uses pickle to dump the precision scores into a file
    with open(filepath, 'wb') as file:
        pickle.dump(model_scores, file)


'''
Loads all the precision scores from the previouslt saved file and returns them as a dictionary.

Parameters:
filename (string) --> should be the same as what was used when calling 'save_all_precision_scores'
directory (string) --> should be the same as what was used when calling 'save_all_precision_scores'

Return type: dictionary, where the keys are tickers and the values are floats representing the precision scores of the model associated with the ticker
'''

#important note: change the default value of directory to the actual path where you want this file to be saved
def load_all_precision_scores(filename = 'Precision_Scores.pkl', directory = ''):
    #creates the full filepath of where the precision scores are saved
    filepath = os.path.join(directory, filename)

    #uses pickle to open the file and return the dictionary of the precision scores
    with open(filepath, 'rb') as file:
        return pickle.load(file)