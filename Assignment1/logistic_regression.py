#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 04:55:14 2018

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

 
def minMax(inputData): #Find the minimum and maximum values for each feature
	minmax = list()
	for i in range(len(inputData[0])):
		col = [row[i] for row in inputData]
		minElem = min(col)
		maxElem = max(col)
		minmax.append([minElem, maxElem])
	return minmax
 
def rescale_dataset(inputData): #change the dataset to the range 0-1
    minmax = minMax(inputData)
    for row in inputData:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def split_folds(inputData, n_folds):  #split the input dataset to n folds, here n=5
	split = list()
	copy = list(inputData)
	fold_len = int(len(inputData) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_len:
			random_pos = random.randrange(len(copy))
			fold.append(copy.pop(random_pos))
		split.append(fold)
	return split
 
def getAccuracy(testSet, classes):
	count = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == classes[i]:
			count += 1
	return (count/float(len(testSet))) * 100.0
 
def evaluate_algorithm(dataset, n_folds, *args):
	folds = split_folds(dataset, n_folds)
	scores = list()
	for fold in folds:
		trainset = list(folds)
		trainset.remove(fold)
		trainset = sum(trainset, [])
		predicted = logistic_regression(trainset, dataset, *args) #passing training set for training, and complete dataset for testing
		accuracy = getAccuracy(dataset, predicted) #passing complete dataset along woith the predicted values to calculate the accuracy
		scores.append(accuracy)
	return scores
 
def predictwithCoefficients(row, coef): #Calculating the general form with the input calculation
	temp = coef[0]
	for i in range(len(row)-1):
		temp += coef[i + 1] * row[i]
	return 1.0 / (1.0 + math.exp(-temp)) # general form (1/1+e^(-b))
 

def coefficientsCalc(trainset, l_rate, epoch_num): # This function uses stochastic gradient descent to Estimate logistic regression coefficients 
	coef = [0.0 for i in range(len(trainset[0]))]
	for epoch in range(epoch_num):
		for row in trainset:
			y = predictwithCoefficients(row, coef)
			error = row[-1] - y
			coef[0] = coef[0] + l_rate * error * y * (1.0 - y)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * y * (1.0 - y) * row[i]
	return coef
 

def logistic_regression(trainset, testset, l_rate, epoch_num): #Linear regression algorithm with Gradient descent method
	predictions = list()
	coef = coefficientsCalc(trainset, l_rate, epoch_num)
	for row in testset:
		y = predictwithCoefficients(row, coef)
		y = round(y)
		predictions.append(y)
	return(predictions)
 
def main():
    filename = '/media/jithin/New Volume/Sem2/SML/data_banknote_authentication.csv'
    n_folds = 3
    l_rate = .625
    epoch_num = 100
    dataset = loadInput(filename)
    rescale_dataset(dataset) 
    scores = evaluate_algorithm(dataset, n_folds, l_rate, epoch_num)
    print('Scores: %s' % scores)
    print(' Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
    
main()