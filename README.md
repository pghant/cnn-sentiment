## Deep Learning for Sentiment Analysis

Creates and runs a convolutional neural network to classify Amazon reviews as either positive or negative. The network
is based on Yoon Kim's Convlutional Networks for Sentence Classification (https://arxiv.org/abs/1408.5882). The
TensorFlow implementation was largely taken from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.

A few modifications were made to that model to allow for pre-training the model with Google News word2vec embeddings, and a larger dataset.

Data for the network can be downloaded from http://jmcauley.ucsd.edu/data/amazon/.
Word2vec embeddings used to train the network can be downloaded from https://code.google.com/archive/p/word2vec/

Training the network:

1. Run the parse_data.py script to extract and analyze the compressed JSON data. See the various sections in the script to see how to parse and clean the data in order to create a training file

2. Run the train_cnn.py script to train and evaluate the model. Adjust the variables at the beginning to the appropriate values for the positive and negative training files. Various hyperparameters can also be adjusted to see the effects.
