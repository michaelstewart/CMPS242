#!/usr/bin/python

import random
import numpy as np
import math

NUM_SAMPLES = 500;
DIM_SAMPLES = 11;
eta = 1

def trainOverData(data, labels, w, wList):
	errorCount = 0
	for i, x in enumerate(data):
		if np.dot(w, x)*labels[i] <= 0:
			w = w + eta*labels[i]*x
			errorCount += 1
		wList.append(w)
	return w, errorCount

def testOverData(data, labels, w):
	errorCount = 0
	for i, x in enumerate(data):
		if np.dot(w, x)*labels[i] <= 0:
			errorCount += 1
	return errorCount


def randomSample(length):
	return [int(random.getrandbits(1)*2-1) for i in xrange(length)]

def labelList(data):
	labels = []
	for x in data:
		labels.append(x[0])
	return labels

def labelList2(data):
	labels = []
	for x in data:
		label = 1 if sum(x) > 0 else -1
		labels.append(label)
	return labels


def labelList3(data):
	labels = []
	for x in data:
		r = random.randint(-3, 3)
		label = 1 if sum(x)+r > 0 else -1
		labels.append(label)
	return labels

def testOverDataVoting(data, labels, wList):
	errorCount = 0
	for i, x in enumerate(data):
		vote = 0
		for w in wList:
			vote += math.copysign(1, np.dot(w, x))
		if vote*labels[i] <= 0:
			errorCount += 1
	return errorCount

def avgW(wList):
	avg = np.zeros(11)
	for x in wList:
		avg += x/1000.0
	return avg


def partCTest(wList):

	w_1000 = wList[-1]
	w_avg = avgW(wList)
	
	train_data = [np.array(randomSample(DIM_SAMPLES)) for i in xrange(NUM_SAMPLES)]
	labels =  labelList3(train_data)

	accuracies = []
	accuracies.append(testOverData(train_data, labels, w_1000))
	accuracies.append(testOverData(train_data, labels, w_avg))
	accuracies.append(testOverDataVoting(train_data, labels, wList))

	return accuracies

def sim(labelType):
	# Init data
	data = [np.array(randomSample(DIM_SAMPLES)) for i in xrange(NUM_SAMPLES)]
	# Create labels
	if labelType == 2:
		labels =  labelList2(data)
	else:
		labels =  labelList(data)
	# Init w
	w = np.array([0 for i in xrange(DIM_SAMPLES)])
	wList = []

	cumError = 0
	errorCount = int(True)
	epochCount = 0
	while errorCount:
		w, errorCount = trainOverData(data, labels, w, wList)
		cumError += errorCount
		# print 'w: ', str(w)
		
		epochCount += 1


	print 'In %d epochs with %d errors' % (epochCount, cumError)

	return 

def sim3():
	# Init data
	data = [np.array(randomSample(DIM_SAMPLES)) for i in xrange(NUM_SAMPLES)]
	# Create labels
	labels =  labelList3(data)
	# Init w
	w = np.array([0 for i in xrange(DIM_SAMPLES)])
	wList = []

	errorCount = int(True)
	epochCount = 0
	while errorCount:
		w, errorCount = trainOverData(data, labels, w, wList)
		# print 'w: ', str(w)
		# print '%d errors' % errorCount
		epochCount += 1
		if epochCount >= 2: errorCount = 0


	# print 'In %d epochs' % epochCount

	return partCTest(wList)

def run3():
	results = []
	for i in xrange(10):
		results.append(sim3())

	avg = map(np.mean, zip(*results))

	print avg
def main():
	sim(1)
	sim(2)
	run3()


if __name__ == '__main__':
	main()