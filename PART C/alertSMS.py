import os
from twilio.rest import TwilioRestClient

def SendMessage():
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = 'AC6c6edca702b834b2b53338d4203c1f89'
    auth_token = '533565b9c8652a9b5caa83423efa43f3'
    client = TwilioRestClient(account_sid, auth_token)

    message = client.messages.create(
                              body='Suspecious Activity Detected',
                              from_='+14582305533',
                              to='+918777675892'
                          )

