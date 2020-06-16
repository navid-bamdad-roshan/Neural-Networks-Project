import tweepy
from tweepy import OAuthHandler
import pandas as pd
import json
import os.path as path_
import time
import math

phrases_to_search = 'Covid19 OR Covid-19 OR covid19 OR covid-19 OR Coronavirus OR coronavirus OR Corona OR corona'

consumer_key_one = ""
consumer_secret_one = ""
access_token_one = ""
access_secret_one = ""

consumer_key_two = ""
consumer_secret_two = ""
access_token_two = ""
access_secret_two = ""


auth_one = OAuthHandler(consumer_key_one,consumer_secret_one)
auth_one.set_access_token(access_token_one,access_secret_one)

auth_two = OAuthHandler(consumer_key_two,consumer_secret_two)
auth_two.set_access_token(access_token_two,access_secret_two)



api_one = tweepy.API(auth_one) 
api_two = tweepy.API(auth_two) 


apis = [api_one, api_two]
api_iterator = -1


dates = ['05-01', '05-02', '05-03', '05-04', '05-05', '05-06']
for date_iteratir,date in enumerate(dates):
    for fifteen_minutes_iteration in range(10):  # x 17000 tweets
        tweets = pd.DataFrame(columns=["id_", "text", "date"])
        print("fifteen_minutes_iteration: "+str(fifteen_minutes_iteration))
        if ((fifteen_minutes_iteration > 0) or (date_iteratir > 0)):
            print("Halt for "+str(math.ceil(((910-(60*len(apis)))/len(apis)) / 60))+" minutes")
            time.sleep(math.ceil((910-(60*len(apis)))/len(apis)))    # try not to pass the limits of twitter
        api_iterator +=1
        if api_iterator >= len(apis):
            api_iterator = 0
        api = apis[api_iterator]
        for iteration in range(170):
            try:
                print(iteration)
                #                                                                                               #United States
                result = api.search(q=phrases_to_search, count=100, lang='en', tweet_mode='extended', geocode='40.352622,-99.473923,1800mi', until='2020-'+date)
            
                for i in range(len(result)):
                    try:
                        if (not result[i].full_text.endswith("…")):
                            temp_str = result[i].full_text
                            if temp_str.startswith("RT"):
                                temp_index = temp_str.find(":")
                                temp_str = temp_str[temp_index+1:]
                            tweets = tweets.append({"id_":'t_'+result[i].id_str, "text":temp_str, "date":result[i].created_at}, ignore_index = True)
                            
                        else:
                            tweets = tweets.append({"id_":'t_'+result[i].retweeted_status.id_str, "text":result[i].retweeted_status.full_text, "date":result[i].retweeted_status.created_at}, ignore_index = True)
                    except Exception as e:
                        print("There was a problem in iteration: "+str(iteration) + ",  i: "+str(i))
                        print(e)
            except Exception as e:
                print("There was a problem in iteration: "+str(iteration))
                print(e)

        if path_.isfile('tweets-'+date+'.csv'):
            tweets.to_csv('tweets/tweets-'+date+'.csv', index=False, mode='a', header=False, encoding="utf-8")
        else:
            tweets.to_csv('tweets/tweets-'+date+'.csv', index=False, mode='a', header=True, encoding="utf-8")
