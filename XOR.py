import numpy as np
from net import Net

# This file contains a simple XOR model

# Here we create a super simple XOR dataset
trainingData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
trainingLabels = np.array([[0], [1], [1], [0]])

brain = Net(trainingData, trainingLabels, learningRate=0.75)
brain.createNeuronLayer(2)
brain.createNeuronLayer(1)

d1 = np.array([trainingData[0]])
d2 = np.array([trainingData[1]])
d3 = np.array([trainingData[2]])
d4 = np.array([trainingData[3]])

print(str(d1) + ' : ' + str(brain.feedForward(d1)[0]))
print(str(d2) + ' : ' + str(brain.feedForward(d2)[0]))
print(str(d3) + ' : ' + str(brain.feedForward(d3)[0]))
print(str(d4) + ' : ' + str(brain.feedForward(d4)[0]))

brain.trainNetwork(rounds=1000)

print(str(d1) + ' : ' + str(brain.feedForward(d1)[0]))
print(str(d2) + ' : ' + str(brain.feedForward(d2)[0]))
print(str(d3) + ' : ' + str(brain.feedForward(d3)[0]))
print(str(d4) + ' : ' + str(brain.feedForward(d4)[0]))

exit(0)
