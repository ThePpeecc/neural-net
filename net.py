import numpy as np

np.random.seed(0)


def sigmoid(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the sigmoid of a given input
    :param ind: The given input to be performed sigmoid on
    :return: The resulting data after having been run through the sigmoid function
    """
    return 1 / (1 + np.exp(-ind))


class Net:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, learningRate: int = 0.1):
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
        self.resultLayer = []
        self.learningRate = learningRate

    def addHiddenLayer(self, layer: np.ndarray, bias: np.ndarray):
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
        bias = np.array([np.random.rand(layerSize)])
        for i in range(layerSize-1):
            layer = np.vstack([layer, np.random.rand(numWeights)])
        # We add the layer to the network
        self.addHiddenLayer(layer, bias)
        return layer, bias

    def feedForward(self, ind: np.ndarray) -> np.ndarray:
        """
        Simple function that runs a set of data through the network and returns the result
        :param ind: The data to be run through our network, make sure that it matches the trained data's dimensions
        :return: We return out a vector with the networks predictions
        """

        self.resultLayer = [ind]
        for i in range(len(self.hiddenLayers)):
            layer = self.hiddenLayers[i]
            bias = self.hiddenBiases[i].T

            # Here we calculate our current layers activation functions
            out = sigmoid(np.matmul(layer, self.resultLayer[i])+bias)
            self.resultLayer.append(out)

        return self.resultLayer[-1]

    def backPropagation(self, out: np.ndarray, expect: np.ndarray):
        """
        Here lies the heart of the neural network.
        This performs the learning of the network and is explained carefully in many places but one good source is
        3Blue1Brown's video on the subject or section 6.5 in the book called Deep Learning by Ian Goodfellow,
        Yoshua Bengio and Aaron Courville.
        :param out: This is the output of the layer after having run feedForward on some data
        :param expect: This is the expected output of the network from the inputs it had received
        """

        # First we calculate the cost function differentiated
        # This also works as the carry of the C/a_{L-1} which is the previous layer differentiated with the cost
        layerCostDifference = np.multiply(2, out - expect)

        for i in range(len(self.hiddenLayers)-1, -1, -1):
            # First we calculate the differentiated sigmoid function
            sigDif = np.matmul(self.resultLayer[i+1].T, 1-self.resultLayer[i+1])

            # Gradients are calculated here
            # C/W_{L} & # C/b_{L} respectively
            deltaW = np.matmul(np.matmul(self.resultLayer[i], sigDif), layerCostDifference.T).T
            deltaB = np.matmul(sigDif, layerCostDifference.T)

            # Here we calculate the gradient for our last layer compared with our current layers cost
            layerCostDifference = np.matmul(np.multiply(self.hiddenLayers[i], sigDif).T, layerCostDifference)
            
            # We finally perform gradient descent with our gradients and our learning rate
            self.hiddenLayers[i] += -np.multiply(self.learningRate, deltaW)
            self.hiddenBiases[i] += -np.multiply(self.learningRate, deltaB)


