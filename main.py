import numpy as np
from net import Net

flowerData = np.loadtxt('data/iris-flower-data', dtype=str, delimiter=',')

labels = []
for name in flowerData[:, 4]:
    if name == 'Iris-setosa':
        labels.append(np.array([[1], [0], [0]]))
    if name == 'Iris-versicolor':
        labels.append(np.array([[0], [1], [0]]))
    if name == 'Iris-virginica':
        labels.append(np.array([[0], [0], [1]]))

labels = np.array(labels)
dataVectors = flowerData[:, 0:3].astype(float)

brain = Net(dataVectors, labels)
brain.createNeuronLayer(3)

exit()
