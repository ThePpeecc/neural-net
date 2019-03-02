import numpy as np
import matplotlib.pyplot as plt
from net import Net

flowerData = np.loadtxt('data/iris-flower-data', dtype=str, delimiter=',')
np.set_printoptions(suppress=True)
labels = []
for name in flowerData[:, 4]:
    if name == 'Iris-setosa':
        labels.append(np.array([[1], [0], [0]]))
    if name == 'Iris-versicolor':
        labels.append(np.array([[0], [1], [0]]))
    if name == 'Iris-virginica':
        labels.append(np.array([[0], [0], [1]]))


def preprocess(ind: np.ndarray) -> np.ndarray:
    row = ind.std(axis=1, keepdims=True)
    out = ind / row[:, ]
    mean = out.mean(axis=1, keepdims=True)
    return out-mean[:, ]


labels = np.array(labels)
dataVectors = preprocess(flowerData[:, 0:4].astype(float))

nn = Net(dataVectors, labels, learningRate=0.01)
nn.createNeuronLayer(10)
nn.createNeuronLayer(3)

nn.trainNetwork(5000)

res = 0
for i in range(150):
    d1 = np.array([dataVectors[i]])
    r = nn.feedForward(d1)
    res += np.sum(np.power(labels[i].reshape(r.shape)-r, 2))

print(res)

exit(0)
