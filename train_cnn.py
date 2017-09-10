'''
Creates and runs a convolutional neural network to classify Amazon reviews as either positive or negative. The network
is based on Yoon Kim's Convlutional Networks for Sentence Classification (https://arxiv.org/abs/1408.5882). The
TensorFlow implementation was largely taken from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.
A few modifications were made to that model to allow for pre-training the model with Google News word2vec embeddings, and a larger
dataset. The word2vec embeddings were gotten from https://code.google.com/archive/p/word2vec/.
'''

import tensorflow as tf
import numpy as np
import datetime
import time
from tensorflow.contrib import learn

# Initialize variables
positiveFile = "./pos_train.txt"
negativeFile = "./neg_train.txt"
word2vecFile = "./GoogleNews-vectors-negative300.bin"
filterSizes = [3, 4, 5]
numFilters = 100
batchSize = 200
numEpochs = 15
useWord2Vec = True

# Load data to train on (already been cleaned and parsed)
# Each line in the data file is a single review with each word/punctuation split by space
print("Loading data...")
with open(positiveFile, "r") as f:
    positiveReviews = list(f.readlines())
with open(negativeFile, "r") as f:
    negativeReviews = list(f.readlines())

xText = positiveReviews + negativeReviews

# Create labels and concatenate data
positiveLabels = [[0, 1] for __ in positiveReviews]
negativeLabels = [[1, 0] for __ in negativeReviews]
y = np.concatenate([positiveLabels, negativeLabels])

# Build vocabulary and transform review text into arrays of integers representing words in vocabulary
maxDocLength = max([len(x.split(" ")) for x in xText])
vocabProcessor = learn.preprocessing.VocabularyProcessor(maxDocLength)
x = np.array(list(vocabProcessor.fit_transform(xText)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

# Split training data into train and validation sets with 10% being used for validation
dev_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

# Load word2vec pre-trained embeddings to use in the CNN
if useWord2Vec:
    print("Loading embeddings...")
    with open(word2vecFile, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabProcessor.vocabulary_), vector_size)).astype('float32')
        binary_len = np.dtype('float32').itemsize * vector_size
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = str(b''.join(word), encoding='utf-8', errors='strict')
            idx = vocabProcessor.vocabulary_.get(word)
            if idx != 0:
                embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.seek(binary_len, 1)

# Create CNN model - has embedding layer, convolutional layer, max-pooling, and softmax
cnn_inputX = tf.placeholder(tf.int32, [None, x_train.shape[1]])
cnn_inputY = tf.placeholder(tf.float32, [None, 2])
cnn_dropoutProb = tf.placeholder(tf.float32)

# Embedding layer - converts reviews to lists of word vector, optionally uses word2vec pre-trained embeddings
with tf.device("/cpu:0"), tf.name_scope("embedding"):
    if useWord2Vec:
        cnn_W = tf.Variable(embedding_vectors, trainable=False, name="W")
    else:
        cnn_W = tf.Variable(tf.random_uniform([len(vocabProcessor.vocabulary_), 300], -1.0, 1.0), name="W")
    cnn_embeddedReviews = tf.expand_dims(tf.nn.embedding_lookup(cnn_W, cnn_inputX), -1)

# Convolution and max-pool layers for each filter size
pooledOutputs = []
for i, filterSize in enumerate(filterSizes):
    with tf.name_scope("conv-maxpool-%s" % filterSize):
        filterShape = [filterSize, 300, 1, numFilters]
        # Initialize filter weights and biases for convolution layer
        W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[numFilters]), name="b")
        # Do the convolution
        conv = tf.nn.conv2d(cnn_embeddedReviews, W, strides=[1, 1, 1, 1], padding="VALID")
        # ReLU nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Max-pooling
        pooled = tf.nn.max_pool(h, ksize=[1, x_train.shape[1] - filterSize + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        pooledOutputs.append(pooled)

# Combined pooled outputs
totalFilters = numFilters * len(filterSizes)
cnn_hPool = tf.concat(pooledOutputs, 3)
cnn_hPool = tf.reshape(cnn_hPool, [-1, totalFilters])

# Dropout
cnn_hDrop = tf.nn.dropout(cnn_hPool, 0.5)

# Final scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable("W", shape=[totalFilters, 2], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
    cnn_scores = tf.nn.xw_plus_b(cnn_hDrop, W, b, name="scores")
    cnn_predictions = tf.argmax(cnn_scores, 1, name="predictions")

# Calculate loss and accuracy
with tf.name_scope("loss"):
    cnn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_scores, labels=cnn_inputY))
with tf.name_scope("accuracy"):
    cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(cnn_predictions, tf.argmax(cnn_inputY, 1)), "float"), name="accuracy")


print("Start training...")
trainRunFile = "./runs/run-{}-train.csv".format(int(time.time()))
devRunFile = "./runs/run-{}-dev.csv".format(int(time.time()))
sess = tf.Session()
with sess.as_default():
    # Declare variables and train the model
    globalStep = tf.Variable(0, name="globalStep", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    gradsAndVars = optimizer.compute_gradients(cnn_loss)
    trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

    sess.run(tf.global_variables_initializer())

    def trainStep(xBatch, yBatch, outputFile):
        feedDict = {cnn_inputX: xBatch, cnn_inputY: yBatch, cnn_dropoutProb: 0.5}
        __, step, loss, accuracy = sess.run([trainOp, globalStep, cnn_loss, cnn_accuracy], feedDict)
        timeStr = datetime.datetime.now().isoformat()
        outputFile.write("{},{},{:g},{:g}\n".format(timeStr, step, loss, accuracy))

    def devStep(devBatches, numBatches, outputFile):
        losses = []
        accuracies = []
        step = 1
        for batch in devBatches:
            xBatch, yBatch = zip(*batch)
            feedDict = {cnn_inputX: xBatch, cnn_inputY: yBatch, cnn_dropoutProb: 1}
            step, loss, accuracy = sess.run([globalStep, cnn_loss, cnn_accuracy], feedDict)
            losses.append(loss)
            accuracies.append(accuracy)
        accuracy = sum(accuracies) / len(accuracies)
        loss = sum(losses) / len(losses)
        timeStr = datetime.datetime.now().isoformat()
        outputFile.write("{},{},{:g},{:g}\n".format(timeStr, step, loss, accuracy))
        

    # Split dev set into batches
    zippedDevData = np.array(list(zip(x_dev, y_dev)))
    numDevBatches = len(zippedDevData) / batchSize
    devBatches = np.split(zippedDevData, numDevBatches)
    # Generate training batches
    batches = []
    zippedData = np.array(list(zip(x_train, y_train)))
    dataSize = len(zippedData)
    batchesPerEpoch = int((dataSize - 1) / batchSize) + 1
    for epoch in range(numEpochs):
        # Shuffle the data in each epoch
        shuffle_indices = np.random.permutation(np.arange(dataSize))
        zippedData = zippedData[shuffle_indices]
        for n in range(batchesPerEpoch):
            start = n * batchSize
            end = min((n + 1 ) * batchSize, dataSize)
            batches.append(zippedData[start:end])

    print("Doing " + str(len(batches)) + " batches\n")
    # Training loop
    with open(trainRunFile, 'w') as trf, open(devRunFile, 'w') as drf:
        for batch in batches:
            xBatch, yBatch = zip(*batch)
            trainStep(xBatch, yBatch, trf)
            currentStep = tf.train.global_step(sess, globalStep)
            if currentStep % 200 == 0:
                devStep(devBatches, numDevBatches, drf)
        
sess.close()
