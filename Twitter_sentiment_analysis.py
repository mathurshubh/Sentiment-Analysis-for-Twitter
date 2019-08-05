#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install vaderSentiment')


# In[2]:


import tweepy
import pandas as pd
import numpy as np
#from IPython.display import display 
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[3]:


#My Twitter API Authentication Variables

consumer_key = 'lxrmC2IN4GYb1ZN4NfD3Tf8Gz'
consumer_secret = 'emTlpALk2a9R0ffoyfyj5A73aFt9nE2lU4MnQrELr7VWMJGHLr'
access_token = '3020825006-yA4l1d35uGwZmnjYpZkDy8ciMnm0vVGkVvwTMtO'
access_token_secret = 'vdFLOF7b02cnEOnagb1N6bCexmFXDlDpJ91dwV5NmkYg9'


# In[5]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('Machine Learning', count = 200)

data = pd.DataFrame(data = [tweet.text for tweet in tweets], columns = ['Tweets'])

display(data.head(10))

print(tweets[0].created_at)


# In[6]:


import nltk

nltk.download('vader_lexicon')


# In[7]:


sid = SentimentIntensityAnalyzer()

list = []

for index, row in data.iterrows():
    ss = sid.polarity_scores(row['Tweets'])
    list.append(ss)
    
se = pd.Series(list)

data['Polarity'] = se.values

display(data.head(10))


# In[8]:


display(data)


# In[9]:


data.head()
type(data.Tweets)


# In[10]:


type(data.Polarity)
data.info()


# In[11]:


data[data.isnull().any(axis=1)].head()
data.isnull().any(axis=0)


# In[12]:


#we print the 5 most recent tweets
print('5 recent tweets: \n')
for tweet in tweets[:5]:
    
    print(tweet.text)
    print()

    


# In[13]:


for index, row in data.iterrows():
    ss = sid.polarity_scores(row['Tweets'])
    list.append(ss)


# In[14]:


#join tweets to a single string
words = ' '.join(data['Tweets'])
print(words)


# In[15]:


get_ipython().system('pip install wordcloud')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# In[16]:


#remove URLs, RTs, and twitter handles 
no_urls_no_tags = ' '.join([word for word in words.split()
                           if 'http' not in word
                            and not word.startswith('@')
                            and word != 'RT'
                           ])


# In[17]:


wordcloud = WordCloud(
                font_path = '/Users/shubmath/anaconda3/lib/python3.7/site-packages/wordcloud/CabinSketch-Bold.ttf',
                stopwords = STOPWORDS,
                background_color = 'black',
                width = 1800,
                height = 1400
                ).generate(no_urls_no_tags)


# In[20]:


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[21]:


from scipy.misc import imread

#use a twitter logo as a mask
twitter_mask = imread('/Users/shubmath/twitter_mask.png', flatten = True)

wordcloud = WordCloud(
                font_path = '/Users/shubmath/anaconda3/lib/python3.7/site-packages/wordcloud/CabinSketch-Bold.ttf',
                stopwords = STOPWORDS,
                background_color = 'white',
                width = 1000,
                height = 1000,
                mask = twitter_mask, 
                ).generate(no_urls_no_tags)


# In[22]:


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




