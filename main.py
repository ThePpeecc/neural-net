import numpy as np
from net import Net, LayerFunctions

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

permutation = np.random.permutation(len(labels))[0:30]

testLabels = labels[permutation]
testVectors = dataVectors[permutation]
trainingLabels = np.delete(labels, permutation, axis=0)
trainingVectors = np.delete(dataVectors, permutation, axis=0)


nn = Net(trainingVectors, trainingLabels, learningRate=0.001, lam=0.00001)

# nn.setSeed()

nn.createNeuronLayer(6, LayerFunctions.relu)
nn.createNeuronLayer(3, LayerFunctions.relu)
nn.trainNetwork(1000, sampleCost=130, batchSize=20)

print('Accuracy 🎯: ' + str(nn.accuracy(testVectors, testLabels)))
print('Total Error: ' + str(nn.costRate(testVectors, testLabels)))

exit(0)
