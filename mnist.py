import struct
import numpy as np
import matplotlib.pyplot as plt
from net import Net


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def displayMnistImage(data):
    plt.imshow(data.reshape((28, 28)), cmap='gist_gray')
    plt.show()


def createLabel(num):
    label = np.zeros(10)
    label[num] = 1
    return label.reshape((10, 1))


def fixLabels(labels):
    updateLabels = []
    for l in labels:
        updateLabels.append(createLabel(l))
    return np.array(updateLabels)


def preprocess(ind: np.ndarray) -> np.ndarray:
    row = ind.std(axis=1, keepdims=True)
    out = ind / row[:, ]
    mean = out.mean(axis=1, keepdims=True)
    return out-mean[:, ]


def fixData(data):
    updatedData = []
    for d in data:
        updatedData.append(np.ravel(d))
    return preprocess(np.array(updatedData))


# testData = fixData(read_idx('data/mnist-dataset/t10k-images-idx3-ubyte').astype(float))
# testLabels = fixLabels(read_idx('data/mnist-dataset/t10k-labels-idx1-ubyte'))
#
# trainingData = fixData(read_idx('data/mnist-dataset/train-images-idx3-ubyte').astype(float))
# trainingLabels = fixLabels(read_idx('data/mnist-dataset/train-labels-idx1-ubyte'))

nn = Net(inputs=np.ndarray, labels=np.ndarray, learningRate=0.1)

# Here we load in the final mnist model, with an accuracy of around 0.94
nn.loadNetwork('./models/mnist')

garbage = np.random.rand(28*28)
displayMnistImage(garbage)
print(nn.predict(garbage, getPercentage=True))

# test = np.random.permutation(testData)[0:1000]
# for i in range(len(test)):
#     res = nn.predict(testData[i])-testLabels[i]
#     if np.isin(res.ravel(), 1).sum() == 1:
#         displayMnistImage(testData[i])

# x = 0
# y = 0
# figure, axarr = plt.subplots(6, 6, figsize=(24, 24))
#
# for w in nn.hiddenLayers[0]:
#     ax = axarr[x, y]
#     im = ax.imshow(w.reshape((28, 28)))
#     figure.colorbar(im, ax=ax)
#     x += 1
#     if x > 5:
#         x = 0
#         y += 1
# plt.show()
# figure.savefig('neurons.png')
# print('Test set Accuracy 🎯: ' + str(nn.accuracy(testData, testLabels)))
# print('Test set Total Error: ' + str(nn.costRate(testData, testLabels)))
# print('Training set Accuracy 🎯: ' + str(nn.accuracy(trainingData, trainingLabels)))
# print('Training set Total Error: ' + str(nn.costRate(trainingData, trainingLabels)))
