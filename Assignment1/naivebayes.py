#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:20:27 2018

@author: jithin
"""
import csv
import random
import math
def loadInput(fname): #read the input dataset from the input file
	line = csv.reader(open(fname, "r"))
	inputData = list(line)
	for i in range(len(inputData)):
		inputData[i] = [float(j) for j in inputData[i]]
	return inputData


def convertTrain_Test(inputData, splitRatio): #function to split the input dataset to train and test
    trainSize = int(len(inputData) * splitRatio)
    trainData = []
    test = list(inputData)
    while len(trainData) < trainSize:
        random_pos=random.randrange(len(test))
        trainData.append(test.pop(random_pos))
        #print(trainData)
    return [trainData, test]

def classifyonClass(inputData): #classify the training dataset according to the class label
	classDict = {}
	for i in range(len(inputData)):
		data_row = inputData[i]
		if (data_row[-1] not in classDict):
			classDict[data_row[-1]] = []
		classDict[data_row[-1]].append(data_row)
	return classDict

def featureCalculation(inputData): #calculate mean and variance for each attribute value of the input dataset
    feature=[]
    for attribute in zip(*inputData):
        mean=sum(attribute)/float(len(attribute))
        variance = sum([pow(x-mean,2) for x in attribute])/float(len(attribute)-1)
        sd=math.sqrt(variance)
        feature.append([mean,sd])
    del feature[-1]
    return feature

def featurebyClass(inputData): #calculate the mean and variance of each attribute value and for the corresponding class
    classDict = classifyonClass(inputData)
    features = {}
    for classkey, attrValues in classDict.items():	
            features[classkey] = featureCalculation(attrValues)
    return features

def priorprobabilityofClass(trainDict,trainset):
    classprobDict={}
    #flag=0
    for key,value in trainDict.items():
        count=len(value)
        classprobDict[key]=count/len(trainset)
        #print("count=",count,"len(trainset)=",len(trainset))
    return classprobDict
        

def pdfCalculation(xi, mean, sd): #calculate the gaussian pdf
    temp = math.exp(-(math.pow(xi-mean,2)/(2*math.pow(sd,2))))
    probability= (1 / (math.sqrt(2*math.pi) * sd)) * temp
    return probability

def pdfbyClass(features, testset,prior):
    probabilities = {}
    #prior=probabilityofClass(trainset)
    for classkey, attrvalues in features.items():
        probabilities[classkey] = 1*prior[classkey]
        for i in range(len(attrvalues)):
            mean, sd = attrvalues[i]
            xi = testset[i]
            probabilities[classkey] *= pdfCalculation(xi, mean, sd)
            #print(probabilities)
    return probabilities

def classPredict(features, testset_row,prior):
	probabilities = pdfbyClass(features, testset_row,prior)
	classLabel, classProb = None, -1
	for classkey, probability in probabilities.items():
		if classLabel is None or probability > classProb:
			classProb = probability
			classLabel = classkey
	return classLabel
 
def classifyTestset(features, testSet,prior):
	classPredictions = []
	for i in range(len(testSet)):
		result = classPredict(features, testSet[i],prior)
		classPredictions.append(result)
	return classPredictions

def getAccuracy(testSet, classes):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == classes[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
    
def main():
    filename = '/media/jithin/New Volume/Sem2/SML/data_banknote_authentication.csv'
    dataset = loadInput(filename)
    splitRatio = 0.625
    trainset, testset = convertTrain_Test(dataset, splitRatio)
    trainDict = classifyonClass(trainset)
    #print(trainDict)
    
    features=featurebyClass(trainset)
    #print(features)
    #print(testset[0])
    prior=priorprobabilityofClass(trainDict,trainset)
    #print(prior)
    classes=classifyTestset(features, dataset,prior)
    #print(classes)
    accuracy = getAccuracy(dataset, classes)
    print(accuracy)


main()

