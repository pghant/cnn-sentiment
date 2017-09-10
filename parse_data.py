'''
Parse Amazon data from JSON into text file to input. The JSON data is gotten from http://jmcauley.ucsd.edu/data/amazon/.
Contains helper functions to parse, clean, and graph data files.
'''

import gzip
import json
import re
import numpy as np
import matplotlib.pyplot as plt


'''
Extracts JSON Amazon review data from a compressed .json.gz file and creates a new compressed .json.gz file with the essential data
including: review text, the review score, and the length of the review in characters. It has an optional parameter to filter the data
by review length, including only non-neutral (3 star) reviews between 400 and 2000 characters.
'''
def extractEssentialData(filePath, newFileName, useFilter=False):
    i = 0
    with gzip.open(newFileName, 'w') as newGzipped:
        with gzip.open(filePath, 'r') as originalGzipped:
            for line in originalGzipped:
                review = eval(line)
                reviewLength = len(review['reviewText'])
                if useFilter and not (reviewLength >= 400 and reviewLength <= 2000 and review['overall'] != 3.0):
                    continue
                reviewData = {'reviewText': review['reviewText'], 'score': review['overall'], 'reviewLength': reviewLength}
                newGzipped.write((json.dumps(reviewData) + '\n').encode('utf-8'))
                if i % 100000 == 0:
                    print(i / 100000, end=' ')
                i = i + 1

'''
Extracts data from a compressed file and writes it to an uncompressed file.
'''                
def uncompressData(filePath, newFileName):
    with open(newFileName, 'wb') as outputFile:
        with gzip.open(filePath, 'r') as compressed:
            for line in compressed:
                outputFile.write(line)

'''
Splits the essential compressed JSON data into two JSON files, one with positive reviews, the other with negative reviews.
'''
def splitEssentialCompressedData(filePath, newFileNameBase):
    i = 0
    with gzip.open(filePath, 'r') as compressed:
        with open(newFileNameBase + "_positive.json", 'wb') as positiveFile:
            with open(newFileNameBase + "_negative.json", 'wb') as negativeFile:
                for line in compressed:
                    d = eval(line)
                    if d['score'] > 3:
                        positiveFile.write(line)
                    elif d['score'] < 3:
                        negativeFile.write(line)
                    if i % 100000 == 0:
                        print(i / 100000, end=' ')
                    i = i + 1

'''
Reads the essential compressed JSON data into a CSV file with review length, score and sentiment classification. This
CSV file can then be analyzed using analyzeCsvSummaryData.
'''
def readEssentialCompressedDataIntoCsv(filePath, newFileName):
    i = 0
    with open(newFileName, 'w') as outputFile:
        with gzip.open(filePath, 'r') as compressed:
            for line in compressed:
                d = eval(line)
                if d['score'] > 3:
                    sentiment = '1'
                elif d['score'] < 3:
                    sentiment = '-1'
                else:
                    sentiment = '0'
                outputFile.write(str(d['reviewLength']) + "," + str(d['score']) + "," + sentiment + '\n')
                if i % 100000 == 0:
                    print(i / 100000, end=' ')
                i = i + 1

'''
Counts the number of files in a line.
'''
def fileLength(fileName):
    with open(fileName) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

'''
Analyzes the CSV output from readEssentialCompressedDataIntoCsv to calculate the range of length, the number of positive and negative
reviews, and create a historgram to visualize the distribution of review length.
'''
def analyzeCsvSummaryData(filePath):
    data = np.loadtxt(filePath, dtype='float32', delimiter=',')
    print("Loaded data")
    positive = data[data[:,2] > 0]
    positive = positive[:,0]
    print("Got positive")
    negative = data[data[:,2] < 0]
    negative = negative[:,0]
    print("Got negative")
    wordCounts = data[:,0]
    print("\nTotal Number Reviews")
    print(wordCounts.size)
    print(positive.size)
    print(negative.size)
    print("\nMax length")
    maxLen = np.max(wordCounts)
    print(maxLen)
    print(np.max(positive))
    print(np.max(negative))
    print("\nMin length")
    minLen = np.min(wordCounts)
    print(minLen)
    print(np.min(positive))
    print(np.min(negative))
    print("\nMeans")
    print(np.mean(wordCounts))
    print(np.mean(positive))
    print(np.mean(negative))
    print("\nStd Dev")
    print(np.std(wordCounts))
    print(np.std(positive))
    print(np.std(negative))
    bins = np.arange(400, 8100, 100)
    counts, bins = np.histogram(wordCounts, bins)
    counts = np.append(counts, 0) # to make the sizes equal
    print(np.column_stack((counts, bins)))
    print(np.histogram(wordCounts, bins))
    plt.hist(wordCounts, bins)
    plt.title("Word Counts")
    plt.show()

'''
Select a random set of lines from a file and write it to a new file.
'''
def selectRandomLines(fileName, selectCount, selectedFileName):
    lineCount = fileLength(fileName)
    selector = np.ones(selectCount)
    selector.resize(lineCount)
    np.random.shuffle(selector)
    i = 0
    with open(selectedFileName, 'w') as selectedFile:
        with open(fileName, 'r') as originalFile:
            for line in originalFile:
                if selector[i] == 1:
                    selectedFile.write(line)
                i = i + 1

'''
Create a training file from the essential JSON file which writes each review on a separate line after cleaning it.
'''
def createTrainFileFromJson(jsonFileName, trainFileName):
    with open(jsonFileName, 'r') as jsonFile:
        with open(trainFileName, 'w') as trainFile:
            for line in jsonFile:
                d = json.loads(line)
                trainFile.write(cleanStr(d['reviewText']) + "\n")

'''
Clean the review as a string.
Modified from https://github.com/dennybritz/cnn-text-classification-tf
'''
def cleanStr(string):
    string = re.sub(r"&#\d+;", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

'''
Given the name of the run, plot the loss and accuracy of the training and dev output.
'''
def plotAccuracyAndLoss(runName):
    devFile = runName + "-dev.csv"
    trainFile = runName + "-train.csv"
    devData = np.genfromtxt(devFile, delimiter=',', usecols=(1, 2, 3), dtype=None)
    trainData = np.genfromtxt(trainFile, delimiter=',', usecols=(1, 2, 3), dtype=None)
    if len(trainData) > 100000:
        trainData = trainData[0::1000]
    elif len(trainData) > 10000:
        trainData = trainData[0::100]
    devData = list(zip(*devData))
    trainData = list(zip(*trainData))
    
    plt.figure(figsize=(10, 7))
    # Plot accuracy of train output
    plt.subplot(2, 2, 1)
    plt.title("Train Accuracy")
    plt.plot(trainData[0], trainData[2], 'r')
    # Plot accuracy of dev output
    plt.subplot(2, 2, 2)
    plt.title("Dev Accuracy")
    plt.plot(devData[0], devData[2], 'b')
    # Plot loss of train output
    plt.subplot(2, 2, 3)
    plt.title("Train Loss")
    plt.plot(trainData[0], trainData[1], 'r')
    # Plot loss of dev output
    plt.subplot(2, 2, 4)
    plt.title("Dev Loss")
    plt.plot(devData[0], devData[1], 'b')
    
    plt.tight_layout()
    plt.show()


##################################################################################################################
# Run the following to extract and analyze the compressed Amazon JSON data
# extractEssentialData('reviews_Books_5.json.gz', 'reviews_parsed_filtered_min400_max2000.json.gz', True)
# readEssentialCompressedDataIntoCsv('reviews_parsed_filtered_min400_max2000.json.gz', 'review_data_filtered_min400_max2000.csv')
# analyzeCsvSummaryData('review_data_filtered_min400_max2000.csv')
##################################################################################################################

##################################################################################################################
# Run the following to split and create training files for the model
# splitEssentialCompressedData('reviews_parsed_filtered_min400_max2000.json.gz', 'review_data_filtered_min400_max2000_split')
# selectRandomLines('review_data_filtered_min400_max2000_split_negative.json', 300000, 'selected_negative_300000.json')
# selectRandomLines('review_data_filtered_min400_max2000_split_positive.json', 300000, 'selected_positive_300000.json')
# createTrainFileFromJson('selected_negative_300000.json', 'neg_train.txt')
# createTrainFileFromJson('selected_positive_300000.json', 'pos_train.txt')
##################################################################################################################

##################################################################################################################
# Run the following to generate plots of the accuracy and loss from the output files after running the model
plotAccuracyAndLoss('runs/run-1492886908-filters345-batch200-epochs15')
##################################################################################################################