import pandas as pd
import re
import math

file_name = "tweets-"

files = ['05-01', '05-02', '05-03', '05-04', '05-05', '05-06']


for file_index in files:
    tweets = pd.read_csv('tweets/'+file_name+file_index+".csv", encoding="utf-8")
    table_size = tweets.shape[0]
    #table_size = 1000
    percentage=0.00
    for index, row in tweets.iterrows():
        if(percentage<math.floor((index/table_size)*100)):
            percentage = math.floor((index/table_size)*100)
            print('File '+file_index+':  '+str(percentage) + " %")
        tweet_text = tweets.loc[index].text
        
        
        # removing URLs
        while(tweet_text.find("http:/")>=0):
            temp_index_start = tweet_text.find("http:/")
            temp_index_end = tweet_text[temp_index_start:].find(" ")
            if(temp_index_end == -1):
                temp_index_end = len(tweet_text)
            else:
                temp_index_end += temp_index_start
            tweet_text = tweet_text[0:temp_index_start] + tweet_text[temp_index_end:]
        
        while(tweet_text.find("https:/")>=0):
            temp_index_start = tweet_text.find("https:/")
            temp_index_end = tweet_text[temp_index_start:].find(" ")
            if(temp_index_end == -1):
                temp_index_end = len(tweet_text)
            else:
                temp_index_end += temp_index_start
            tweet_text = tweet_text[0:temp_index_start] + tweet_text[temp_index_end:]
        
        
        # removing @Name
        while(tweet_text.find('@')>=0):
            temp_index_start = tweet_text.find("@")
            temp_index_end = tweet_text[temp_index_start:].find(" ")
            if(temp_index_end == -1):
                temp_index_end = len(tweet_text)
            else:
                temp_index_end += temp_index_start
            tweet_text = tweet_text[0:temp_index_start] + tweet_text[temp_index_end:]
        
        
        # Do not remove emojis ATTENTION
        tweet_text = re.sub(r"[^a-zA-Z0-9:,'/\.&!?#]+", ' ', tweet_text)
        
        
        # lowercase
        tweet_text = tweet_text.lower()
                               
        
        if(len(tweet_text) < 30):
            tweets.drop([index], inplace=True)
        else:
            tweets.at[index, 'text'] = tweet_text
            #tweets.set_value(index, 'text', tweet_text)
            
    tweets.to_csv("tweets/"+file_name+file_index+"_cleaned.csv", index=False, header=True, encoding="utf-8")
