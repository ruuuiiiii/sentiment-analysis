import torch
import numpy as np
import pandas as pd
import transformers
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
Implementation of Bidirectional Encoder Representations from Transformers (BERT)
for sentiment analysis of IMDB reviews (positive reviews vs. negative reviews)

The implementation is based on: 
https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
"""



"""Step#1: Download IMDB dataset"""

#### Download IMDB review text into a pandas frame ###
imdb_reviews = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
#### Using the first 2K reviews in this experiments ###
imdb_reviews = imdb_reviews[:2000]


"""Step#2: Load the Pre-trained BERT model"""

model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer  = tokenizer_class.from_pretrained(pretrained_weights)
model      = model_class.from_pretrained(pretrained_weights)

"""Step#3: Conudct tokenization using BERT, padding, and masking """
tokenized_reviews = imdb_reviews[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized_reviews.values:
    if len(i) > max_len:
        max_len = len(i)

padded_reviews = np.array([i + [0]*(max_len-len(i)) for i in tokenized_reviews.values])
attention_mask = np.where(padded_reviews != 0, 1, 0)


"""Step#4: Run the pre-trained BERT model to get embeddings for the review sentences"""
input_ids = torch.tensor(padded_reviews)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)


data_X = last_hidden_states[0][:,0,:].numpy()
data_Y = imdb_reviews[1]

"""Step#5: Using logistic regression to learn from the embedding repreenations for sentiment analysis"""
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y)

lr_clf = LogisticRegression()
lr_clf.fit(train_X, train_Y)
test_predict_Y = lr_clf.predict(test_X)

""" Step#6: Report the performane of sentiment analysis using BERT + Logistic regression"""
target_names = ['class 0', 'class 1']
print(classification_report(test_Y, test_predict_Y, target_names=target_names))






