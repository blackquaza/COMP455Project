# Imports. Only needs to be run once.
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.special import expit

# There are a lot of rows in this file. It may take a few seconds to load.
raw = pd.read_csv('data.csv')

# Positives are sparse in this file, as it contains anonymised real-world data.
# Out of 284,807 transactions in a two-day period, only 492 are fraud.
# Because we're not going to train on a file this large, we're instead going
# to separate out the positives, and combine a number of positives and negatives
# to make both a training set and a test set.
raw = raw.drop(['Time'], axis=1) # Time is an irrelevent variable.
positives = raw[raw['Class']==1]
negatives = raw[raw['Class']==0]

# Quick creation of training and test data sets.
# Run this section at least once, and run it multiple times to get different sets.

# The training set shall consist of 800 negatives and 100 positives.
trainingSet = negatives.sample(n=800).append(positives.sample(n=100)).sample(frac=1)
trainingSet.reset_index(drop=True, inplace=True)
trainingSetIn = trainingSet.drop(['Class'], axis=1).reset_index(drop=True)
trainingSetIn.columns = range(trainingSetIn.shape[1])
trainingSetOut = trainingSet['Class'].to_frame().reset_index(drop=True)
trainingSetOut.columns = range(trainingSetOut.shape[1])

# The test set shall be 3200 negatives and 400 positives, totalling 3600 entries.
testSet = negatives.sample(n=3200).append(positives.sample(n=400)).sample(frac=1)
testSet.reset_index(drop=True, inplace=True)
testSetIn = testSet.drop(['Class'], axis=1).reset_index(drop=True)
testSetIn.columns = range(testSetIn.shape[1])
testSetOut = testSet['Class'].to_frame().reset_index(drop=True)
testSetOut.columns = range(testSetOut.shape[1])

#Scale the data to make it usable by the network.
t = scale(trainingSetIn, copy=False)
t = scale(testSetIn, copy=False)

# Run this cell to see the neural network in action.
# Since a random element is involved with initialization,
# multiple runs on the same sets may have slightly
# different results.

# The activation function. For this, we're using sigmoid.
# This is using the scipy library, since manually doing
# the function resulted in several overflow warnings.
# The library version handles these automatically.
def act(x):
    return expit(x)

# The derivitive of the activation function.
def act_der(x):
    x = act(x)
    return x * (1 - x)

# The training function with 1 hidden layer with 16 nodes.
def train(trainingIn, trainingRes, loops=1000):
    # Random weights are needed because all the nodes in the hidden layer
    # have the same inputs. If they were all the same, all the weights
    # would update to be the same, and the neural network would get nowhere.
    weights1 = np.random.random_sample((29, 16))
    weights2 = np.random.random_sample((16, 1))
    
    #Split the output from the input. Remove the indices to allow easier calculations.
    
    for i in range(loops):
        
        # Forward pass of inputs through the layers.
        layer1 = act(trainingIn.dot(weights1))
        output = act(layer1.dot(weights2))
        
        # Backpropigate. Using MSE as it's easily derived.
        temp = (2 / output.shape[0]) * (trainingRes - output) * act_der(output)
        deltaW2 = layer1.T.dot(temp)
        temp = temp.dot(weights2.T) * act_der(layer1)
        deltaW1 = trainingIn.T.dot(temp)
        
        weights2 += deltaW2
        weights1 += deltaW1
        
    return weights1, weights2

def test(testIn, testRes, weights1, weights2):
    
    # Forward pass of inputs through the layers.
    layer1 = act(testIn.dot(weights1))
    output = act(layer1.dot(weights2))
    
    testRes.set_axis(['Provided'], axis='columns', inplace=True)
    output.set_axis(['Guess'], axis='columns', inplace=True)
    
    results = [0, 0, 0, 0]
    for i in range(testIn.shape[0]):
        if (testRes['Provided'][i] == 0 and output['Guess'][i] < 0.5):
            results[0] += 1
        elif (testRes['Provided'][i] == 0 and output['Guess'][i] >= 0.5):
            results[1] += 1
        elif (testRes['Provided'][i] == 1 and output['Guess'][i] < 0.5):
            results[2] += 1
        elif (testRes['Provided'][i] == 1 and output['Guess'][i] >= 0.5):
            results[3] += 1
    
    print(str(results[0]) + " correct negatives.")
    print(str(results[3]) + " correct positives.")
    print(str(results[2]) + " false negatives.")
    print(str(results[1]) + " false positives.")
    percent = int(1000 * (results[0] + results[3]) / testIn.shape[0]) / 10.0
    print(str(percent) + "% correct.")
    
    return testRes.join(output).round(decimals=5)

w1, w2 = train(trainingSetIn, trainingSetOut, 1500)
test(testSetIn, testSetOut, w1, w2)