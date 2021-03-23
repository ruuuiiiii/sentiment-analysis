# Sentiment Analysis for IMDB Reviews

## Problem formulation
This project aims to develop a sentiment analysis model using deep learning. Given review from IMDB, the model is expected to automatically conduct sentiment analysis to classfiy if the review is positive or not.

In developing the model, one of the main goal is to leverage the Bidirectional Encoder Representations from Transformers (BERT) represent of the sentences/reviews and learn from the representation to improve the sentiment analysis performance.

## Dataset
The IMDB dataset is used in this project, which is available from https://github.com/clairett/pytorch-sentiment-classification/tree/master/data/SST2. A sample of IMDB reviews is shown below:
![Image of IMDB](https://github.com/ruuuiiiii/sentiment-analysis/blob/main/IMDB.png)

## Performance
For the performance of the sentiment analysis using BERT + Logistic regression, the classification report has been generated and is shown below.

|               | Precision     | Recall | F-1     | Support  |
| ------------- | ------------- |--------|---------|--------- |
| Class 0       | 0.80          | 0.80   | 0.80     | 232     |
| Class 1       | 0.83          | 0.83   | 0.83     | 268     |
| Accuracy      |               |        | 0.82     | 500     |
| Macro Average | 0.82          |  0.82  | 0.82     | 500     |
| Weighted Average | 0.82       |  0.82 | 0.82      | 500     |


