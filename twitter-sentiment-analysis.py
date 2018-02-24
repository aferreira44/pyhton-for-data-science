import os
import csv
import tweepy
from textblob import TextBlob
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

CONSUMER_KEY = os.environ.get("CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET")

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself

csvfile = open('twitter_sentiment.csv', 'w') #open file for operation
writer = csv.writer(csvfile)

for tweet in public_tweets:
    foo = tweet.text.encode('utf-8').strip()  #this encoding formatting was required to make file writing of tweets  troublefree, as some   characters in the tweets faced format problems

    analysis = TextBlob(tweet.text).sentiment
    emotion = analysis.polarity
    if emotion > 0:
       writer.writerow([foo,"positive",analysis]) 
    else : 
       writer.writerow([foo,"negative",analysis])         
    
csvfile.close()
print("Entire process completed successfully ! Open your CSV file and look at the results.")
