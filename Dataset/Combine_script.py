import pandas as pd
import math
from sklearn.utils import shuffle


dataset = pd.read_csv("text_emotion.csv", encoding='utf-8')

f = open("Jan9-2012-tweets-clean.txt", "r")

i = 0
percentage = 0
print('Combining: 0 %')


emoji_to_change = [':-)', '=)', ':-(', '=(', ':-O', ':-o', '=O', '=o', ';-)', ';-(', '=D', ':-D', ':,(', ':-|', ':-/', ':-p', ':-P', '=p', '=P']
#emoji =           [':)',  ':)', ':(' , ':(', ':o' , ':o' , ':o', ':o', ';)' , ';(' , ':D', ':D' , ":'(", ':|' , ':/' , ':p' , ':p' , ':p', ':p']


# combinin two data set
for line in f:
    if(math.floor((i*100)/21051) > percentage):
        percentage = math.floor((i*100)/21051)
        print('Combining: ' + str(percentage) + ' %')
    id_end_index = line.find(':')
    id_ = line[:id_end_index]
    temp_str = line[id_end_index+2:-1]
    label_start_index = len(temp_str) - temp_str[::-1].find('::') + 1
    label = temp_str[label_start_index:]
    message = temp_str[:label_start_index-4]
    dataset = dataset.append({'tweet_id':id_, 'sentiment':label, 'author':'', 'content':message}, ignore_index=True)
    i += 1
    
f.close()
#dataset.reset_index(drop=True, inplace = True) 
    
percentage = 0
print('Cleaning: 0 %')
table_size = dataset.shape[0]
for index, row in dataset.iterrows():
    if(percentage<math.floor((index/table_size)*100)):
        percentage = math.floor((index/table_size)*100)
        print('Cleaning: ' + str(percentage) + " %")
    
    tweet_text = dataset.loc[index].content
    
    
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
        name_tag_index_start = tweet_text.find("@")
        name_tag_index_end = tweet_text[name_tag_index_start:].find(" ")
        if(name_tag_index_end == -1):
            name_tag_index_end = len(tweet_text)
        else:
            name_tag_index_end += name_tag_index_start
        tweet_text = tweet_text[0:name_tag_index_start] + tweet_text[name_tag_index_end:]
    
    
    # modifying emojies
    #for i,j in enumerate(emoji_to_change):
    #    tweet_text = tweet_text.replace(j, emoji[i])
    
    
    # removing emojies
    for emoji_ in emoji_to_change:
        tweet_text.replace(emoji_, '')
    
    
    # lowercase
    tweet_text = tweet_text.lower()
    
    
    dataset.at[index, 'content'] = tweet_text
    
    # label modification
    label = dataset.loc[index].sentiment
    if(label == 'fun' or label == 'joy' or label == 'love'):
        dataset.at[index, 'sentiment'] = 'happiness'
    
    if(label == 'none' or label == 'empty'):
        dataset.at[index, 'sentiment'] = 'neutral'
        
    if(label == 'disgust'):
        dataset.at[index, 'sentiment'] = 'hate'
        
    if(label == 'fear'):
        dataset.at[index, 'sentiment'] = 'worry'
        
    if(label == 'enthusiasm'):
        dataset.at[index, 'sentiment'] = 'surprise'
        
    if(label == 'boredom'):
        dataset.at[index, 'sentiment'] = 'sadness'
        
    #if(label == 'hate'):
    #    dataset.at[index, 'sentiment'] = 'anger'



# Balancing dataset

labels = dataset['sentiment'].unique()

label_datasets = {}

max_lable_size = 9000
min_lable_size = 7000

for label in labels:
    label_datasets[label] = dataset[dataset.sentiment==label]

for label in labels:
    if(label_datasets[label].shape[0] > max_lable_size):
        label_datasets[label] = label_datasets[label].sample(n=max_lable_size, replace=False, random_state=52)
    if(label_datasets[label].shape[0] < min_lable_size):
        label_datasets[label] = label_datasets[label].sample(n=min_lable_size, replace=True, random_state=52)
        

balanced_dataset = pd.concat(list(label_datasets.values()), sort=False)
balanced_dataset = shuffle(balanced_dataset)
    
balanced_dataset.to_csv('final_dataset.csv', index=False, header=True, encoding="utf-8")


print(balanced_dataset['sentiment'].value_counts())
