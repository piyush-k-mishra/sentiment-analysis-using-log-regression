import numpy as np
import csv
import re
import random
import math
import io

def countPronouns(searchStr):
    pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]
    count = 0
    searchStr = re.sub(r'[\.,!]', '', searchStr)
    for word in searchStr.split(' '):
        if word in pronouns and word != '':
            count = count + 1
    return count

def countSentiment(sentimentList, searchStr):
    searchStr = re.sub(r'[\.,!]', '', searchStr)
    count = 0
    searchStr = searchStr.split(' ')
    for word in searchStr:
        if word in sentimentList and word != '':
            count = count + 1
    return count

def negationCount(searchStr):
    return len(re.findall(r' not? ', searchStr)
               + re.findall(r'n\'t ', searchStr)
               + re.findall(r' nothing ', searchStr)
               + re.findall(r' never ', searchStr))

def genFeature(reviewString):
    f = open('positive-words.txt', 'r')
    posWords = f.read().split('\n')
    f.close()
    f = open('negative-words.txt', 'r')
    negWords = f.read().split('\n')
    f.close()
    
    feat1 = countSentiment(posWords, reviewString)
    feat2 = countSentiment(negWords, reviewString)
    feat3 = negationCount(reviewString)
    feat4 = countPronouns(reviewString)
    feat5 = len(re.findall(r'!+', reviewString))
    feat6 = np.around(np.log(len(reviewString)), 2)
    return [feat1, feat2, feat3, feat4, feat5, feat6]

def featureMatrix(reviewList, label):
    scoreList = []
    for review in reviewList:
        if review is not '':
            score = []
            review = review.split('\t')
            score.append(review[0])
            score.extend(genFeature(review[1].lower()))
            if label != None:
                score.append(label)
            scoreList.append(score)
    return scoreList

def sigmoidFunc(z):
    return 1 / (1 + np.exp(-z))
    
def predictSentiment(testFeatures, weights):
    sentimentList = []
    for feature in testFeatures:
        X = []
        for i in range(1,7):
            X.append(feature[i])
        X.append(1)
        unboundedY = np.dot(weights, X)
        boundedY = sigmoidFunc(unboundedY)
        
        if boundedY > 0.5:
            sentimentList.append([feature[0], 'POS'])
        else:
            sentimentList.append([feature[0], 'NEG'])
    return sentimentList

trainingData = []
f = open('hotelPosT-train.txt', 'r')
posReviews = f.read().split('\n')
f.close()
f = open('hotelNegT-train.txt', 'r')
negReviews = f.read().split('\n')
f.close()

trainingData = featureMatrix(posReviews, 1) + featureMatrix(negReviews, 0)

with open('feature_set.csv', 'w') as myfile:
    for score in trainingData:
        wr = csv.writer(myfile)
        wr.writerow(score)

random.shuffle(trainingData)

W = [0, 0, 0, 0, 0, 0, 1]
learnRate = 0.1
weightMatrix = []
prev_loss = math.inf

for vec in trainingData:
    y = vec[-1]
    X = vec[1:7]
    X.append(1)
    for i in range(1000):
        h = sigmoidFunc(np.dot(W, X))
        loss = -(y*np.log(h) + (1-y)*np.log(1-h))
        if loss < prev_loss:
            gradient = np.dot((h-y), X)
            W = W - (learnRate * gradient)
    weightMatrix.append(W)
wt = np.mean(weightMatrix, axis=0)

f = open('testset.txt', 'r')
testReviews = f.read().split('\n')
f.close()

testFeatures = featureMatrix(testReviews, label=None)

with open('result.txt', 'w') as myfile:
    for sentiment in predictSentiment(testFeatures, wt):
        myfile.write(' '.join(sentiment))
        myfile.write('\n')
