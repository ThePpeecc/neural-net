import numpy as np
from typing import Union, Tuple, List

np.random.seed(0)
np.set_printoptions(suppress=True)


def sigmoid(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the sigmoid of a given input
    :param ind: The given input to be performed sigmoid on
    :return: The resulting data after having been run through the sigmoid function
    """
    return 1 / (1 + np.exp(-ind))


class Net:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, learningRate: float = 0.1) -> None:
        """
        Simply the initializer for our network
        :param inputs: This is the inputs that our network needs to train itself on
        :param labels: This is the labels we use to perform back propagation and gradient decent when learning
        :param learningRate: This is the learning rate of our gradient decent
        """
        self.trainingSet = inputs
        self.trainingLabels = labels
        self.hiddenLayers = []
        self.hiddenBiases = []
        self.learningRate = learningRate

    def addHiddenLayer(self, layer: np.ndarray, bias: np.ndarray) -> None:
        """
        # Add a 2D matrix layer of neurons to the hidden layer of the network with their biases
        :param layer: A matrix of our layers weights
        :param bias: A vector of our layers biases
        """

        self.hiddenLayers.append(layer)
        self.hiddenBiases.append(bias)

    def createNeuronLayer(self, layerSize: int = 1) -> (np.ndarray, np.ndarray):
        """
        This function creates a 2D numpy matrix that represents on layer of the hidden layers inside the network
        It takes one parameter layerSize, that simply determines how many neurons there are in the layer
        If returns the created layer, after having added it to the network
        :param layerSize: Explains how many neurons we have in this new layer.
        :rtype: np.ndarray it returns the layer, which is in our case the weights and biases for each neuron (w, b)
        """

        # We check to see how many weights we need to add to each neuron.
        # If we don't have any previous layers we then need to make weights depending on the size of the training data.
        if len(self.hiddenLayers) < 1:
            numWeights = self.trainingSet.shape[1]
        else:
            numWeights = len(self.hiddenLayers[-1])

        # We add the first neuron to the layer
        layer = np.matrix(np.random.rand(numWeights))
        # The biases are only one vector
        bias = np.array([np.full(layerSize, 0.1)])
        for i in range(layerSize-1):
            layer = np.vstack([layer, np.random.rand(numWeights)])
        # We add the layer to the network
        self.addHiddenLayer(layer, bias)
        return layer, bias

    def feedForward(self, ind: np.ndarray, training=False) -> Union[Tuple[np.ndarray, List[List[np.ndarray]]], np.ndarray]:
        """
        Simple function that runs a set of data through the network and returns the result
        :param ind: The data to be run through our network, make sure that it matches the trained data's dimensions
        :param training: If we are training we wan't the calculations performed in the layers as well,
                         by default we don't want them
        :return: We return out a vector with the networks predictions
        """
        finalResult = []
        layerResults = []
        for x in ind:
            # We fix up our input and save it in our results
            results = [np.matrix(x).T]
            for i in range(len(self.hiddenLayers)):
                layer = self.hiddenLayers[i]
                bias = self.hiddenBiases[i].T

                # Here we calculate our current layers activation functions and save them
                out = sigmoid(np.matmul(layer, results[i])+bias)
                results.append(out)

            # We here save our results if we had more than one data element given
            layerResults.append(results)
            finalResult.append(results[-1])
        if training:
            return np.array(finalResult), layerResults
        else:
            return np.array(finalResult)

    def backPropagation(self, trainSets: np.ndarray, targets: np.ndarray) -> None:
        """
        Here lies the heart of the neural network.
        This performs the learning of the network and is explained carefully in many places but one good source is
        3Blue1Brown's video on the subject or section 6.5 in the book called Deep Learning by Ian Goodfellow,
        Yoshua Bengio and Aaron Courville.
        :param trainSets: This is the data we want to feed the network with so we can train
        :param targets: This is the expected output of the network from the inputs it had received
        """

        # First we feed our network the given data and harvest the output
        outPuts, layerOutPuts = self.feedForward(trainSets, training=True)
        # We then calculate our output errors
        outPutErrors = targets.reshape(outPuts.shape)-outPuts

        # Here we copy our weights in the network
        weights = []
        for layer in self.hiddenLayers:
            weights.append(layer.copy())

        # Now we are running through all of our errors that we have from our inputs, and perform back propagation on
        # all of them
        for j in range(len(outPutErrors)):
            error = outPutErrors[j]
            layerOutPut = layerOutPuts[j]

            # Initiate the hidden error, since the first layer we run on is the output layer, the hidden error starts
            # as the output error
            hiddenErrors = error

            # We now recursively calculate all the layers gradients
            for i in reversed(range(len(weights))):
                currentLayerOut = layerOutPut[i+1]
                pastLayerOut = layerOutPut[i]

                # Here we calculate the derivative of the sigmoid function regarding our current layers output
                derivativeSigmoid = np.multiply(currentLayerOut, (1-currentLayerOut))

                # We now calculate the gradients, that we then can use later again
                gradient = np.multiply(np.multiply(derivativeSigmoid, hiddenErrors), self.learningRate)
                weightGrad = np.matmul(gradient, pastLayerOut.T)

                # We calculate the errors for the next layer, which depends on the current error and the current
                # weights influence on said error
                hiddenErrors = np.matmul(weights[i].T, hiddenErrors)

                # We finally perform gradient decent
                self.hiddenLayers[i] += weightGrad
                self.hiddenBiases[i] += gradient.T

    def trainNetwork(self, rounds: int = 100, numberOfSets: int = 20) -> None:
        """
        This function trains the network via Stochastic Gradient Descent, a specified number of rounds
        :param numberOfSets: This is the number of training sets that the network trains on each round.
                             The sets are picked randomly from the given training set.
        :param rounds: The number of rounds we wan't to train the network
        """

        # X is the counter we use to that we have an idea of how far we are in the training
        x = 1

        # We take a copy of our data
        trainCopy = self.trainingSet.copy()
        trainCopyLabel = self.trainingLabels.copy()

        for i in range(rounds):
            if i >= 1000*x:
                x += 1
                print('Iteration ' + str(i) + ' have been reached')

            if len(trainCopy) < numberOfSets:
                # We are running low on our training set, so we run back prop on the last remaining data
                self.backPropagation(trainCopy, trainCopyLabel)

                # And we get a fresh copy of our training data
                trainCopy = self.trainingSet.copy()
                trainCopyLabel = self.trainingLabels.copy()
            else:
                # We pull some random indices from our training set
                permutation = np.random.permutation(len(trainCopy))[0:numberOfSets]
                # We run our back prop on the subsets
                self.backPropagation(trainCopy[permutation], trainCopyLabel[permutation])

                # We now remove the subset from our training set
                trainCopy = np.delete(trainCopy, permutation, axis=0)
                trainCopyLabel = np.delete(trainCopyLabel, permutation, axis=0)

