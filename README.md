# Prediction the sentiments of Coronavirus tweets from May 1st 2020 till June 12th 2020 using neural networks

Fist there is presentation of the project and then Description about implementation.
<br>
<br>
<br>
<br>
## Presentation of the project

#### The project aims to train an ensemble classification model by fine-tuning three different models to predict the sentiments of tweets related to coronavirus. The models that are being used are the BERT embedding model, ELMo embedding model, and XLNet embedding model.

### Authors
Navid Bamdad Roshan     navid.bamdadroshan@gmail.com<br>
Hasan Mohammed Tanvir   hasantanvir79@gmail.com<br>
Behrad Moeini           behrad@ut.ee<br>


### Training dataset
The dataset for training the model is composed of two different datasets. Those two datasets got merged together to have more data to train the model. However, before training the model using the dataset, the dataset has been modified. One of the modifications was to delete links, mentions, emoticons, and etc. In addition, the distribution of the tweets over different sentiments is not uniform, also there are too many sentiments in the dataset as it can be seen in figure 1. This unbalanced dataset hurts accuracy, so there should be a modification in the dataset to overcome. The issue is solved by two steps. The first step was to merge some of the sentiments which are related together. For instance, “worry” and “fear” or “disgust” and “hate” can be merged together due to their close meaning. By combining some of the labels, it got better but not good enough, the sentiment distribution of this stage is presented in figure 2. Therefore, in the next step, the dataset is balanced by over/under-sampling instances of each class to approximate the uniform distribution. The result is shown in figure 3. To conclude, the training dataset has 64,000 instances and 8 classes namely neutral, sadness, surprise, worry, happiness, hate, relief, and anger.
<br><br>
<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig%20merged%201-2-3.png">
  <figcaption>Fig.1,2,3 - Sentimen distribution of train dataset.</figcaption>
</figure>
<br><br>

### Collecting tweets about coronavirus
The tweets related to coronavirus are gathered using tweeter API. The tweets are from the United States of America and between May 1, 2020, and June 12, 2020. 750,000 tweets have been collected from tweeter to be predicted by the proposed models.
<br><br>

### Pre-processing the collected tweets
Pre-processing the data can be considered as the most important part of utilizing prediction models. Accordingly, the gathered tweets must be cleaned before being predicted by the models. As a result, the data is cleaned by removing links, mentions, and emoticons that are not understandable by embedding models. Also, all tweets have been lower-cased in the pre-processing stage.
<br><br>

### XLNet embedder
XLNET is a bidirectional generalized autoregressive model. In a simple word, it means that the next token is dependent on all previous and next tokens. XLNET captures bi-directional context by “permutation language modeling.” It integrates the idea of auto-regressive models and bi-directional context modeling. To implement XLNET, the transformer is tweaked to look only at the hidden representation of tokens preceding the token to be predicted. XLNET tried to cover some of the problems that the BERT model has in some NLP tasks. <br>
Fast-bert package is used to implement the XLNet network. The description of the fast-bert says “Fast-Bert is the deep learning library that allows developers and data scientists to train and deploy BERT and XLNet based models for natural language processing tasks beginning with Text Classification.”
<br>
The hyperparameters of the model are as follows.
<br>
Number of epochs = 20
Batch size = 128
Dropout rate = 0.1
Optimizer type= lambLearning rate = 0.0001
batch_size_per_gpu = 128
max_seq_length= 16Learning rate = 0.0001
pretrained model= xlnet-base-cased
Optimizer = lamb
Learning rate = 1e-4
<br>
The result of the model can be seen in Figure 3.3.


<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig4.png">
  <figcaption>Fig.4</figcaption>
</figure>
<br><br>

### ELMo embedder
The next embedding model that has been used is the ELMo model. This model gets a sentence as input and gives 1024 features as output. So, fine-tuning can be done by adding some layers after ELMo embedding layer to construct the model. Accordingly, the constructed neural network structure is presented in figures 5.1 and 5.2.
<br><br>

<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig5-1.png">
  <figcaption>Fig.5.1</figcaption>
</figure>
<br><br>

<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig5-2.png">
  <figcaption>Fig.5.2</figcaption>
</figure>
<br><br>


The hyperparameters of the model are as follows.
<br><br>

Number of epochs = 25
Batch size = 32
Dropout rate = 0.4
Optimizer = Adam
Learning rate = 1e-4
Regularization = L2
Regularization rate = 1e-3
<br><br>

And the training history of the model is presented in figure 6.1.
<br><br>

<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig6-1.png">
  <figcaption>Figure.6.1 Training history of ELMo model fine-tuning</figcaption>
</figure>
<br><br>

<figure>
  <img src="https://github.com/navid-bamdad-roshan/Neural-Networks-Project/blob/master/Presentation/fig6-2.png">
  <figcaption>Figure.6.2 ELMo model accuracy</figcaption>
</figure>
<br><br>






### Dataset
The data that is used for training consists of two different datasets that got combined and cleaned.
For cleaning the data, the url addresses and name tags (@Name) are removed.
Also, some of the labels are merged together. Merged class lables are as follow. {fun, joy, happiness} as happiness - {empty, neutral} as neutral - {fear, worry} as worry - {disgust, hate} as hate.
