

get_ipython().system('pip install pandas')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install vaderSentiment')



import tweepy
import pandas as pd
from IPython.display import display 
from nltk.sentiment.vader import SentimentIntensityAnalyzer



#My Twitter API Authentication Variables

consumer_key = 'lxrmC2IN4GYb1ZN4NfD3Tf8Gz'
consumer_secret = 'emTlpALk2a9R0ffoyfyj5A73aFt9nE2lU4MnQrELr7VWMJGHLr'
access_token = '3020825006-yA4l1d35uGwZmnjYpZkDy8ciMnm0vVGkVvwTMtO'
access_token_secret = 'vdFLOF7b02cnEOnagb1N6bCexmFXDlDpJ91dwV5NmkYg9'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('Modi', count = 200)

data = pd.DataFrame(data = [tweet.text for tweet in tweets], columns = ['Tweets'])

display(data.head(10))

print(tweets[0].created_at)



import nltk

nltk.download('vader_lexicon')



sid = SentimentIntensityAnalyzer()

list = []

for index, row in data.iterrows():
    ss = sid.polarity_scores(row['Tweets'])
    list.append(ss)
    
se = pd.Series(list)

data['Polarity'] = se.values

display(data.head(100))



display(data)




