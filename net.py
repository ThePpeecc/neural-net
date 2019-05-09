import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Dict, Callable
from progress.bar import Bar
from pync import Notifier
from enum import Enum
# np.random.seed(0)
np.set_printoptions(suppress=True)


# ------ ACTIVATION FUNCTIONS START ------
def sigmoid(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the sigmoid of a given input
    :param ind: The given input to be performed sigmoid on
    :return: The resulting data after having been run through the sigmoid function
    """
    return 1 / (1 + np.exp(-ind))


def sigmoidDerived(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the derivative of a sigmoid output
    :param ind: The given input to be performed the derivative on
    :return: The resulting data after having calculated the derivative
    """
    return np.multiply(ind, (1-ind))


def relu(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the relu of a given input
    :param ind: The given input to be performed relu on
    :return: The resulting data after having been run through the relu function
    """
    return np.maximum(ind, 0.0)


def reluDerived(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the derivative of a relu output
    :param ind: The given input to be performed the derivative on
    :return: The resulting data after having calculated the derivative
    """
    ind[ind < 0] = 0.0
    ind[ind > 0] = 1.0
    return ind


def softmax(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the softmax of a given input.
    Note that the softmax function should only be used on the output layer,
    and is best used when having to deal with a classification problem.
    :param ind: The given input to be performed softmax on
    :return: The resulting data after having been run through the softmax function
    """
    ex = np.exp(ind)
    return ex/np.sum(ex)


def softmaxDerived(ind: np.ndarray) -> np.ndarray:
    """
    Simply calculates the derivative of a softmax output
    :param ind: The given input to be performed the derivative on
    :return: The resulting data after having calculated the derivative
    """

    # Here we setup the jacobian matrix and calculate the derivative
    jacobian = np.diag(ind)
    for i in range(len(jacobian)):
        for j in range(len(jacobian)):
            if i == j:
                jacobian[i][j] = ind[i] * (1 - ind[i])
            else:
                jacobian[i][j] = -ind[i] * ind[j]
    return jacobian
# ------ ACTIVATION FUNCTIONS END ------


class LayerMethods(Enum):
    # This enum class contains the standard strings for all layer activation functions.
    # Activation is for the activation step, and the derivative is for back propagation
    activation = 'activation'
    derivative = 'derivative'


class LayerFunctions(Enum):
    # This enum class contains the different functions that have been implemented.
    sigmoid = 'sigmoid'
    relu = 'relu'
    softmax = 'softmax'


# noinspection PyMethodParameters
def functionDecoder(funcName: LayerFunctions) -> Union[Dict[LayerMethods, Union[Callable[[np.ndarray], np.ndarray]]]]:
    """
    This functions takes a LayerFunctions string type, and returns the related functions for said string.
    Eks: for sigmoid, it returns a dictionary of layer methods for activation and the derivative of the sigmoid
    :param funcName: LayerFunctions class string
    :return Dictionary of layer methods
    """
    switcher = {
        LayerFunctions.sigmoid: {
            LayerMethods.activation: sigmoid,
            LayerMethods.derivative: sigmoidDerived
        },
        LayerFunctions.relu: {
            LayerMethods.activation: relu,
            LayerMethods.derivative: reluDerived
        },
        LayerFunctions.softmax: {
            LayerMethods.activation: sigmoid,
            LayerMethods.derivative: sigmoid
        }
    }
    return switcher.get(funcName, lambda: print('Invalid function'))


class Net:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, learningRate: float = 0.1, lam: float = 0.0001) -> None:
        """
        Simply the initializer for our network
        :param inputs: This is the inputs that our network needs to train itself on
        :param labels: This is the labels we use to perform back propagation and gradient decent when learning
        :param learningRate: This is the learning rate of our gradient decent
        :param lam: This is the lambda value used in L2 regularisation
        """
        self.trainingSet = inputs
        self.trainingLabels = labels
        self.hiddenLayers = []
        self.hiddenBiases = []
        self.activationFunctions = []
        self.learningRate = learningRate
        self.lam = lam
        self.seed = None

        # Here we save our errors each round so that we can plot our convergence rate,
        # and x axis is where we took our error measurements
        self.errorRates = []
        self.xAxis = []

    def setSeed(self, seed: int = 0) -> None:
        """
        Simply sets the seed for the network
        :param seed: The seed to be set
        """
        self.seed = seed
        np.random.seed(seed)

    def addHiddenLayer(self, layer: np.ndarray, bias: np.ndarray, activationFunction: LayerFunctions) -> None:
        """
        # Add a 2D matrix layer of neurons to the hidden layer of the network with their biases
        :param layer: A matrix of our layers weights
        :param bias: A vector of our layers biases
        :param activationFunction: An activation function found in the LayerFunctions class
        """

        self.hiddenLayers.append(layer)
        self.hiddenBiases.append(bias)
        self.activationFunctions.append(activationFunction)

    def createNeuronLayer(self, layerSize: int = 1, activationFunction: LayerFunctions = LayerFunctions.sigmoid) -> (np.ndarray, np.ndarray, LayerFunctions):
        """
        This function creates a 2D numpy matrix that represents on layer of the hidden layers inside the network
        It takes one parameter layerSize, that simply determines how many neurons there are in the layer
        If returns the created layer, after having added it to the network
        :param activationFunction: An activation function found in the LayerFunctions class
        :param layerSize: Explains how many neurons we have in this new layer.
        :rtype: np.ndarray it returns the layer,
                which is in our case the weights and biases
                for each neuron (w, b), and the activation function
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
        self.addHiddenLayer(layer, bias, activationFunction)
        return layer, bias, activationFunction

    def feedForward(self, ind: np.ndarray, training=False) -> Union[Tuple[np.ndarray, List[List[np.ndarray]]], np.ndarray]:
        """
        Simple function that runs a set of data through the network and returns the result
        :param ind: The data to be run through our network, make sure that it matches the trained data's dimensions
        :param training: If we are training we wan't the calculations performed in the layers as well,
                         by default we don't want them
        :return: We return out a vector with the networks predictions
        """

        # Here we update our input if its only one data input
        if len(ind.shape) < 2:
            ind = np.array([ind])

        finalResult = []
        layerResults = []
        for x in ind:
            # We fix up our input and save it in our results
            results = [np.matrix(x).T]
            for i in range(len(self.hiddenLayers)):
                layer = self.hiddenLayers[i]
                bias = self.hiddenBiases[i].T
                ActivationFunction = functionDecoder(self.activationFunctions[i]).get(LayerMethods.activation)
                # Here we calculate our current layers activation functions and save them
                out = ActivationFunction(np.matmul(layer, results[i])+bias)
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

        lam = self.lam
        wSum = 0
        wSumDev = 0
        weights = []
        numOut = len(outPuts) > 0 if len(outPuts) else 1
        m = 1/numOut

        # Here we copy our weights in the network
        for layer in self.hiddenLayers:
            weights.append(layer.copy())
            wSum += np.sum(np.square(layer))
            wSumDev += np.sum(layer)

        # Here we calculate the L2 regularization values
        L2reg = m*0.5*lam*wSumDev
        L2regDev = m*lam*wSumDev
        # print(L2reg)
        # print(L2regDev)
        # We then calculate our output errors
        # Update with new error functions, like cross entropy
        outPutErrors = np.add(targets.reshape(outPuts.shape)-outPuts, L2reg)
        # print(outPutErrors)
        # exit(0)
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
                derivedFunction = functionDecoder(self.activationFunctions[i]).get(LayerMethods.derivative)
                # Here we calculate the derivative of the sigmoid function regarding our current layers output
                derivative = derivedFunction(currentLayerOut)

                # We now calculate the gradients, that we then can use later again
                gradient = np.multiply(np.multiply(derivative, hiddenErrors), self.learningRate)
                weightGrad = np.matmul(gradient, pastLayerOut.T)-L2regDev
                # We calculate the errors for the next layer, which depends on the current error and the current
                # weights influence on said error
                hiddenErrors = np.matmul(weights[i].T, hiddenErrors)

                # We finally perform gradient decent
                self.hiddenLayers[i] += weightGrad
                self.hiddenBiases[i] += gradient.T

    def accuracy(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Simply calculates the accuracy of the network on new data
        :param data: The new data that we want to see how accurate we are on
        :param labels: The labels for said data
        :return: We finally return the accuracy percentage
        """

        # First we predict what our data is
        predictions = self.predict(data)

        errors = np.absolute(predictions-labels)

        cleanedErrors = []
        for error in errors:
            if error.sum() != 0:
                cleanedErrors.append(1)
            else:
                cleanedErrors.append(0)
        cleanedErrors = np.array(cleanedErrors)
        # The we calculate the error sum
        errorSum = cleanedErrors.sum()
        # And finally we calculate how accurate we are
        accuracy = round(1-errorSum/len(data), 3)

        return accuracy

    def costRate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        This method calculates the total error cost of some data given to the network
        :param data: Data we want to calculate the error cost on for the network
        :param labels: The labels for the data, used to calculate the error
        :return: The error cost
        """

        # Start cost
        cost = 0

        # We loop over the data and sum up the errors
        for i in range(len(data)):
            results = self.predict(data[i], getPercentage=True)
            cost += np.sum(np.power(labels[i].reshape(results.shape) - results, 2))

        # Finally we return the cost
        return round(cost, 3)

    def predict(self, data: np.ndarray, getPercentage: bool = False) -> np.ndarray:
        """
        This method tries to predict the label of some data that it is given, by running it through the network.
        It will pick the result it is most confident with, and then round the numbers.
        :param data: The data to try to predict the labels for
        :param getPercentage: If we want to have the percentage of the confidence of the network on all
        :return:
        """

        # We run the data through our network
        result = self.feedForward(data)

        if getPercentage:
            # If we want to have the percentages, we just return our result
            return result
        else:
            # We run through our results and figure out what our most confident guess is, and round that up, and
            # remove the rest
            for i in range(len(result)):
                # First we fold out our current result
                pred = result[i].ravel()

                # We find our most confident guess
                maxIndex = np.argmax(pred)

                # We reset our prediction and insert a 1 where we are most confident,
                # and the reinsert it back into our results array
                pred = np.multiply(pred, 0)
                pred[maxIndex] = 1
                result[i] = pred.reshape(result[i].shape)
            return result

    def saveNetwork(self, fileName: str) -> None:
        """
        This method saves the network in two .npy files.
        It will add w to one file for the weights, and a b for the bias file.
        :param fileName: The path to the file
        """
        np.save(fileName + 'w', self.hiddenLayers)
        bias = []
        for i in range(len(self.hiddenBiases)):
            bias.append(self.hiddenBiases[i].ravel())
        np.save(fileName + 'b', bias)
        np.save(fileName + 'f', self.activationFunctions)

    def loadNetwork(self, fileName: str) -> None:
        """
        This method loads a saved network.
        It will automatically load in the w and b files, so don't add w.npy or b.npy to the ned of the file name
        :param fileName: The path to the file
        """
        self.hiddenLayers = []
        self.hiddenBiases = []
        for l in np.load(fileName + 'w.npy'):
            self.hiddenLayers.append(l)
        for b in np.load(fileName + 'b.npy'):
            self.hiddenBiases.append(np.array([b]))
        for f in np.load(fileName + 'f.npy'):
            self.activationFunctions.append(f)

    def trainNetwork(self, rounds: int = 100, batchSize: int = 20, sampleCost: int = 20, verbose: bool = True, saveRate: int = 20) -> None:
        """
        This function trains the network via Stochastic Gradient Descent, a specified number of rounds
        :param sampleCost: This is the number of samples we pull when we calculate the cost rate
        :param saveRate: This determines how often we save our error rate
        :param verbose: This boolean turns on our off our print and plotting part
        :param batchSize: This is the number of training sets that the network trains on each round.
                          The batches are picked randomly from the given training set.
        :param rounds: The number of rounds we wan't to train the network
        """

        # y is our save rate counter
        y = 1

        # Setting up progress bar
        bar = Bar('Training 🤖:', fill='🐝️', max=rounds)

        # We take a copy of our data
        trainCopy = self.trainingSet.copy()
        trainCopyLabel = self.trainingLabels.copy()

        for i in range(rounds):

            if verbose:
                bar.next()
                if i >= saveRate*y:
                    y += 1
                    self.xAxis.append(i)
                    permutation = np.random.permutation(len(self.trainingSet))[0:sampleCost]
                    self.errorRates.append(self.costRate(self.trainingSet[permutation], self.trainingLabels[permutation])/batchSize)

            if len(trainCopy) < batchSize:
                # We are running low on our training set, so we run back prop on the last remaining data
                self.backPropagation(trainCopy, trainCopyLabel)

                # And we get a fresh copy of our training data
                trainCopy = self.trainingSet.copy()
                trainCopyLabel = self.trainingLabels.copy()
            else:
                # We pull some random indices from our training set
                permutation = np.random.permutation(len(trainCopy))[0:batchSize]
                # We run our back prop on the subsets
                self.backPropagation(trainCopy[permutation], trainCopyLabel[permutation])

                # We now remove the subset from our training set
                trainCopy = np.delete(trainCopy, permutation, axis=0)
                trainCopyLabel = np.delete(trainCopyLabel, permutation, axis=0)

        if verbose:
            Notifier.notify('The network have finished training', title='Finished training 🎓')
            bar.finish()
            print('Finished training 🎓')
            plt.plot(self.xAxis, self.errorRates, linewidth=2.0)
            plt.ylabel('Our error rates each round')
            plt.xlabel('Measurement points')
            plt.title('Error rate convergence, with sample sizes of: ' + str(sampleCost) + '\n Learning Rate: ' + str(self.learningRate))
            plt.show()
