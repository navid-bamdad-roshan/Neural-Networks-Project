# Predict Sentiments of Coronavirus Tweets from May 1st 2020 till June 12th 2020

### The project aims to train an ensemble classification model by fine-tuning three different models to predict the sentiments of tweets related to coronavirus. The models that are being used are the BERT embedding model, ELMo embedding model, and XLNet embedding model.

### Authors
Navid Bamdad Roshan     navid.bamdadroshan@gmail.com<br>
Hasan Mohammed Tanvir   hasantanvir79@gmail.com<br>
Behrad Moeini           behrad@ut.ee<br>


### Training dataset
The dataset for training the model is composed of two different datasets. Those two datasets got merged together to have more data to train the model. However, before training the model using the dataset, the dataset has been modified. One of the modifications was to delete links, mentions, emoticons, and etc. In addition, the distribution of the tweets over different sentiments is not uniform, also there are too many sentiments in the dataset as it can be seen in figure 1. This unbalanced dataset hurts accuracy, so there should be a modification in the dataset to overcome. The issue is solved by two steps. The first step was to merge some of the sentiments which are related together. For instance, “worry” and “fear” or “disgust” and “hate” can be merged together due to their close meaning. By combining some of the labels, it got better but not good enough, the sentiment distribution of this stage is presented in figure 2. Therefore, in the next step, the dataset is balanced by over/under-sampling instances of each class to approximate the uniform distribution. The result is shown in figure 3. To conclude, the training dataset has 64,000 instances and 8 classes namely neutral, sadness, surprise, worry, happiness, hate, relief, and anger.

<img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig%20merged%201-2-3.png">


### Dataset
The data that is used for training consists of two different datasets that got combined and cleaned.
For cleaning the data, the url addresses and name tags (@Name) are removed.
Also, some of the labels are merged together. Merged class lables are as follow. {fun, joy, happiness} as happiness - {empty, neutral} as neutral - {fear, worry} as worry - {disgust, hate} as hate.
