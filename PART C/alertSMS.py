import os
from twilio.rest import TwilioRestClient

def SendMessage():
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = 'XXXXXXXXXXXXXXX'
    auth_token = 'XXXXXXXXXXXXXXX'
    client = TwilioRestClient(account_sid, auth_token)

    message = client.messages.create(
                              body='Suspecious Activity Detected',
                              from_='XXXXXXXXXXX',
                              to='+XXXXXXXXXXXXX'
                          )

