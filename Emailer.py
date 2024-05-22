import base64
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
import New_Predictions
import Serialization as model_scores

#IMPORTANT note: to run this file, two lines must be adjusted: line 35 with the path of the credentials.json file installed from google,
#and line 43 with the email address (preferrably gmail) of the intended recipient

#creates a list to hold the entire body message to be included in the email sent out to recipients
body_message = ['Tomorrow\'s predictions for price increases are:', '',]

#calls the 'New_Predictions' module to generate predictions for all the saved tickers for the next trading day
predictions = New_Predictions.generate_all_predictions()

#calls the 'Serialization' module to load the precision scores of the model to include in the email output to recipients
scores = model_scores.load_all_precision_scores()

#loops through the tickers and their associated predictions to include them in the body message
for ticker, prediction in predictions.items():
    body_message.append(ticker + ': ' + prediction + ', model has precision score of ' + str(round(scores[ticker] * 100, 2)) + '%')

#joins together the list to form one neat string as the body message sent to recipients
complete_message = '\n'.join(body_message)

#sets the scope variable to access the gmail API
SCOPES = [
        'https://www.googleapis.com/auth/gmail.send'
    ]

#accesses the json file containing the google credentials necessary to use the gmail API
#important note: to run this file, the first input for the function 'from_client_secrets_file' needs to be the path to the credentials.json file installed on your local machine
flow = InstalledAppFlow.from_client_secrets_file('', SCOPES)
creds = flow.run_local_server(port = 0)

#finalizes the email's subject, message, and recipients
service = build('gmail', 'v1', credentials = creds)
message = MIMEText(complete_message)

#important note: the below line needs to be adjusted to include the email address (preferrably gmail) of the intended recipient
message['to'] = ''
message['subject'] = 'Derik\'s Stock Picks'
create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

#attempts to send the email to the recipient
try:
    message = (service.users().messages().send(userId = 'me', body = create_message).execute())
    print(F'sent message to {message} Message Id: {message["id"]}')
except HTTPError as error:
    print(F'An error occurred: {error}')
    message = None