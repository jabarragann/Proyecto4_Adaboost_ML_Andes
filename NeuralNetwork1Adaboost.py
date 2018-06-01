import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime
from scipy.optimize import minimize
from PreprocessingDataset import splitIntoFeatureMatrixAndLabels


class NeuralNetworkArchi:
    def __init__(self, inputNeurons, outputNeurons, hiddenNeurons, innerActivation):
        self.inputNeurons = inputNeurons
        self.outputNeurons = outputNeurons
        self.hiddenNeurons = hiddenNeurons

        self.w1Size = hiddenNeurons * (inputNeurons + 1)
        self.w2Size = outputNeurons * (hiddenNeurons + 1)
        self.innerActivation = innerActivation

    def __str__(self):
        str1 = "{:35} \n".format("Neural Network Architecture")
        str2 = "{:35} {:4d}\n".format("Layers:", 1)
        str3 = "{:35} {:04d}\n".format("Input Neurons:", self.inputNeurons)
        str3 += "{:35} {:04d}\n".format("Hidden Neurons:", self.hiddenNeurons)
        str3 += "{:35} {:04d}\n".format("Output Neurons:", self.outputNeurons)
        str3 += "{:35} {:04d}\n".format("Total Number of Parameters:", self.w1Size + self.w2Size)
        str3 += "{:35} {:4s}\n".format("Output activation:", 'sigmoid')
        str3 += "{:35} {:4s}\n".format("Inner activation:", self.innerActivation)

        return str1 + str2 + str3


def featureScaling(X):
    meanVec = np.mean(X, axis=0)
    maxVec = X.max(axis=0)
    minVec = X.min(axis=0)

    return meanVec, maxVec, minVec


def activationFunct(z, act):
    if act == 'sigmoid':
        return sigmoid(z)
    elif act == 'tanh':
        return hyperTan(z)
    else:
        return sigmoid(z)


def activationDerivative(z, act):
    # where z is the output of the specified activation
    # z=tanh(a) or
    # z=1/(1-exp(-a))

    if act == 'sigmoid':
        return z * (1 - z)
    elif act == 'tanh':
        return (1 - z ** 2)
    else:
        return z * (1 - z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hyperTan(z):
    return np.tanh(z)


def computeCost(theta, X, y, reg, nnArchi):
    # Recover parameter matrices
    m = y.shape[0]
    n = X.shape[1] - 1
    w1 = theta[:nnArchi.w1Size].reshape(nnArchi.hiddenNeurons, nnArchi.inputNeurons + 1)
    w2 = theta[nnArchi.w1Size:].reshape(nnArchi.outputNeurons, nnArchi.hiddenNeurons + 1)

    # Estimate-Forward Propagation
    z1 = X.transpose()
    a2 = w1 @ z1
    z2 = activationFunct(a2, nnArchi.innerActivation)
    z2 = np.insert(z2, 0, 1, axis=0)
    a3 = w2 @ z2

    # Ouput layer
    z3 = sigmoid(a3)
    safetyFactor = 0.999999999999999
    y_estimate = z3.T

    # Cost Function
    m = y.shape[0]
    cost = -y * np.log(y_estimate + (1 - safetyFactor)) - (1 - y) * np.log(1 - y_estimate * safetyFactor)
    cost = (1 / m) * sum(cost)

    # Add Regularization Term
    regularization_term = np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2)
    regularization_term = (regularization_term * reg) / (2 * m)

    return cost[0] + regularization_term


def computeGradient(theta, X, y, reg, nnArchi):
    # Recover parameter matrices
    m = y.shape[0]
    n = X.shape[1] - 1
    w1 = theta[:nnArchi.w1Size].reshape(nnArchi.hiddenNeurons, nnArchi.inputNeurons + 1)
    w2 = theta[nnArchi.w1Size:].reshape(nnArchi.outputNeurons, nnArchi.hiddenNeurons + 1)

    # Initilize Gradients
    w2Grad = np.zeros_like(w2)
    w1Grad = np.zeros_like(w1)

    # Forward-Propagation
    z1 = X.T
    a2 = w1 @ z1
    z2 = activationFunct(a2, nnArchi.innerActivation)
    z2 = np.insert(z2, 0, 1, axis=0)
    a3 = w2 @ z2
    # Output layer
    z3 = sigmoid(a3)
    y_estimate = z3

    # Back-Propagation
    d3 = y_estimate - y.T
    d2 = w2.T @ d3 * activationDerivative(z2, nnArchi.innerActivation)
    d2 = d2[1:, :]

    for i in range(m):
        w1Grad = w1Grad + d2[:, i].reshape(-1, 1) @ z1[:, i].reshape(-1, 1).T
        w2Grad = w2Grad + d3[:, i].reshape(-1, 1) @ z2[:, i].reshape(-1, 1).T

    w1Grad = w1Grad / m
    w2Grad = w2Grad / m

    # Add regularization Term
    w1Grad[:, 1:] = w1Grad[:, 1:] + (w1[:, 1:] * reg) / m
    w2Grad[:, 1:] = w2Grad[:, 1:] + (w2[:, 1:] * reg) / m

    # Flat matrices into one single parameter vector
    grad = np.append(w1Grad.reshape(-1, 1), w2Grad.reshape(-1, 1))

    return grad


def debugInitializeWeights(fan_out, fan_in):
    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.sin([i for i in range(1, W.size + 1)]).reshape(W.shape[0], W.shape[1]) / 10;

    return W


def gradientCheck(reg, epsilon=10e-5):
    # Initilize a smaller Network to test Backpropagation
    n = 3
    hiddenNeurons = 5
    outputNeurons = 1
    m = 5
    testNN = NeuralNetworkArchi(n, outputNeurons, hiddenNeurons, 'tanh')

    # Generate 'random' test data and weights
    theta1 = debugInitializeWeights(hiddenNeurons, n);
    theta2 = debugInitializeWeights(outputNeurons, hiddenNeurons);
    theta = np.append(theta1.reshape(-1, 1), theta2.reshape(-1, 1))
    X = debugInitializeWeights(m, n - 1);
    X = np.insert(X, 0, 1, axis=1)
    y = np.array(1 + np.arange(m) % (outputNeurons)).reshape(m, 1);

    # Calculate Exact Gradient
    exactGradient = computeGradient(theta, X, y, reg, testNN)

    # Aproximate Gradient
    aproxGradient = np.zeros_like(theta)
    epsilonVector = np.zeros_like(theta)

    for i in range(theta.size):
        epsilonVector[i] = epsilon
        up = computeCost(theta + epsilonVector, X, y, reg, testNN)
        down = computeCost(theta - epsilonVector, X, y, reg, testNN)
        aproxGradient[i] = (up - down) / (2 * epsilon)
        epsilonVector[i] = 0

    exactGradient = exactGradient.reshape(-1, 1)
    aproxGradient = aproxGradient.reshape(-1, 1)

    # Calculate the norm of the Error vector
    result = (sum((exactGradient - aproxGradient) ** 2)) ** 0.5

    return result[0]


def gradientDescent(theta, X, y, alpha, numIters):
    m = y.shape[0]
    costHistory = np.zeros((numIters, 1))

    for i in range(numIters):
        # Compute Cost and Gradient
        cost = computeCost(theta, X, y)
        gradient = computeGradient(theta, X, y)

        # Update parameter vector
        theta = theta - (alpha / m) * gradient

        costHistory[i] = cost

    return theta, costHistory


def minibatchGradientDescent(theta, X, y, alpha, reg, numIters, nnArchi,P, batchSize=45):
    takeHistory = numIters
    costHistory = np.zeros(int(numIters / takeHistory) + 1)
    m = y.shape[0]

    for i in range(numIters):
        # Get minibatch - random Rows from X matrix
        randIdx = np.random.choice(X.shape[0], batchSize, replace=False ,p=P)
        X_batch = X[randIdx, :]
        y_batch = y[randIdx, :]

        # Update parameter vector
        gradient = computeGradient(theta, X_batch, y_batch, reg, nnArchi)
        theta = theta - (alpha / m) * gradient

        # Compute Cost and Gradient
        if i % takeHistory == 0:
            if True:
                costHistory[int(i / takeHistory)] = computeCost(theta, X, y, reg, nnArchi)

    return theta, costHistory


def evaluateModel(theta, X, y,labels, nnArchi):
    # Recover parameter matrices
    m = y.shape[0]
    n = X.shape[1] - 1
    w1 = theta[:nnArchi.w1Size].reshape(nnArchi.hiddenNeurons, nnArchi.inputNeurons + 1)
    w2 = theta[nnArchi.w1Size:].reshape(nnArchi.outputNeurons, nnArchi.hiddenNeurons + 1)

    # Forward Propagation
    z1 = X.transpose()
    a2 = w1 @ z1
    z2 = activationFunct(a2, nnArchi.innerActivation)
    z2 = np.insert(z2, 0, 1, axis=0)
    a3 = w2 @ z2
    z3 = sigmoid(a3)
    y_estimate = z3.T

    predictions = np.argmax(y_estimate,axis=1)


    acc = sum(np.equal(predictions, labels))
    return acc / m


def printToLogAndConsole(strToPrint):
    print(strToPrint)
    logging.debug(strToPrint)

def initAdaboostWeights(xTrain,labelsTrain):
    B = xTrain.shape[0] * 3
    D = np.array([])

    xTrainSize = xTrain.shape[0]
    # D-Weights
    for i in range(xTrainSize):
        D = np.append(D, {})
        for j in range(4):
            if j != labelsTrain[i]:
                D[i][j] = 1 / B

    return D.reshape(-1, 1)

def getAdaboostPWeights(D):
    P = np.array(list(map(lambda t: sum(t.values()), D.reshape(-1, ) ) ) )
    P = P / sum(P)

    return P / sum(P)

def predict(theta, X, y,labels, nnArchi):
    # Recover parameter matrices
    m = y.shape[0]
    n = X.shape[1] - 1
    w1 = theta[:nnArchi.w1Size].reshape(nnArchi.hiddenNeurons, nnArchi.inputNeurons + 1)
    w2 = theta[nnArchi.w1Size:].reshape(nnArchi.outputNeurons, nnArchi.hiddenNeurons + 1)

    # Forward Propagation
    z1 = X.transpose()
    a2 = w1 @ z1
    z2 = activationFunct(a2, nnArchi.innerActivation)
    z2 = np.insert(z2, 0, 1, axis=0)
    a3 = w2 @ z2
    z3 = sigmoid(a3)
    y_estimate = z3.T

    return y_estimate


# COMMAND LINE ARGUMENT PARSER
# GLOBAL CONSTANTS
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reg", required=False, default="0.01", help="Neural Network Regularization Parameter")
ap.add_argument("-l", "--load", required=False, default="0", help="Load a trained neural network")
ap.add_argument("-s", "--save", required=False, default="0", help="Save the neural network")
loadDirectory = "06-01_00-42-30_ACC_TR-0.72643_ACC_TE-0.72393_FC-0.34179_REG_0.010_HN1-020_I-400000_para.csv"
ap.add_argument("-d", "--directory", required=False, default=loadDirectory,
                help="Directory where the neural network is found")
ap.add_argument("-a", "--alpha", required=False, default="3000", help="Alpha")
ap.add_argument("-i", "--iterations", required=False, default="1000", help="Number of interations of backpropagation")

args = vars(ap.parse_args())

NUM_ITERS = int(args['iterations'])
REG = float(args['reg'])
SAVE = bool(int(args['save']))
LOAD = bool(int(args['load']))
FILE_TO_LOAD = args['directory']
ALPHA = int(args['alpha'])

if __name__ == '__main__':

    # Log file
    timeStamp = datetime.now().strftime('%m-%d_%H-%M-%S_')
    logging.basicConfig(
        filename="savedData/" + timeStamp + "training.log",
        level=logging.DEBUG,
        format="%(asctime)s:%(message)s"
    )

    # Read DataSet and Split X and Y data
    testSet = pd.read_csv('exerciseData/testSetGenre.csv')
    trainingSet = pd.read_csv('exerciseData/trainingSetGenre.csv')
    xTrain, yTrain, labelsTrain = splitIntoFeatureMatrixAndLabels(trainingSet)
    xTest, yTest, labelsTest    = splitIntoFeatureMatrixAndLabels(testSet)

    # FEATURE SCALING
    meanVec, maxVec, minVec = featureScaling(xTrain)
    xTrain = (xTrain - meanVec) / (maxVec - minVec)
    xTest = (xTest - meanVec) / (maxVec - minVec)

    # ADD BIAS COLUMN
    xTrain = np.insert(xTrain, 0, 1, axis=1)
    xTest = np.insert(xTest, 0, 1, axis=1)

    # NEURAL NETWORK PARAMETERS
    hiddenNeurons = 20
    outputNeurons = 4
    inputNeurons = xTrain.shape[1] - 1
    innerActivation = 'sigmoid'
    nn1 = NeuralNetworkArchi(inputNeurons, outputNeurons, hiddenNeurons, innerActivation)

    # PRINT NEURAL NETWORK ARCHITECTURE
    printToLogAndConsole("\n" + str(nn1))

    if LOAD:
        # Load saved parameters
        theta = []
        with open("./savedData/"+FILE_TO_LOAD) as f1:
            for i in f1:
                theta = theta + [float(i.strip('\n'))]
            theta = np.array(theta)
    else:
        # Initialize Weights randomly between (-ep,ep)
        ep = 0.7
        w1 = np.random.rand(nn1.hiddenNeurons, nn1.inputNeurons + 1) * ep * 2 - ep
        w2 = np.random.rand(nn1.outputNeurons, nn1.hiddenNeurons + 1) * ep * 2 - ep

        # Flat matrices into one single parameter vector
        theta = np.append(w1, w2)

    # Cheking Backpropagation implementation
    printToLogAndConsole(
        "\nCheck Backpropagation(Difference between Numerical Gradient and Analytical Gradient): {:.4E}\n".format(
            gradientCheck(REG)))

    # Print Global Constants
    printToLogAndConsole("{:25}: {:06.4f}".format("Regularization", REG))
    printToLogAndConsole("{:25}: {:06d}".format("Maxiters", NUM_ITERS))
    printToLogAndConsole("{:25}: {:6s}".format("Load", str(LOAD)))
    printToLogAndConsole("{:25}: {:6s}".format("Save", str(SAVE)))
    printToLogAndConsole("{:25}: {:6s}\n".format("File", FILE_TO_LOAD))

    # Initial Cost Calculation
    printToLogAndConsole("Training Neural Network...\n")
    initialCost = computeCost(theta, xTrain, yTrain, REG, nn1)
    printToLogAndConsole("Initial Cost: {:.5f}\n".format(initialCost))

    ###########################################################
    # Init Adaboost
    D=initAdaboostWeights(xTrain,labelsTrain)
    # Loop
    P=getAdaboostPWeights(D)
    # Stochastic Gradient Descent
    startTime1 = time.time()
    theta, costHistory = minibatchGradientDescent(theta, xTrain, yTrain, ALPHA, REG, NUM_ITERS, nn1,P.reshape(-1,))
    endTime1 = time.time()
    # Pseudoloss calculation
    yEstimates = predict(theta, xTrain, yTrain, labelsTrain, nn1)
    print(3)
    loss = 0
    for i in range(xTrain.shape[0]):
        res = np.array( list(map(lambda t: list(t), D[i,0].items())))
        const = 1 - yEstimates[i,labelsTrain[i]]
        loss += sum( res[:,1] * (const + yEstimates[i,res[:,0].astype(np.int)]) )
    loss = 0.5 * loss
    beta = loss / (1 - loss)

    ############################################################

    finalCost = computeCost(theta, xTrain, yTrain, REG, nn1)

    # Logging training information
    printToLogAndConsole("Finish Training...\n")
    printToLogAndConsole("Gradient Descent Execution time: {:4.3f}\n".format(endTime1 - startTime1))
    printToLogAndConsole("Regularization: {:.4f}".format(REG))
    printToLogAndConsole("Iterations: {:03d}".format(NUM_ITERS))
    printToLogAndConsole("Initial Cost: {:.5f}".format(initialCost))
    printToLogAndConsole("Final Cost: {:.5f}".format(finalCost))

    # EVALUATE MODEL IN TEST SET
    trainingAccuracy = evaluateModel(theta, xTrain, yTrain,labelsTrain, nn1)
    testAccuracy = evaluateModel(theta, xTest, yTest,labelsTest, nn1)

    # LOGGING EVALUATIONS
    printToLogAndConsole("Model evaluation\n")
    printToLogAndConsole("Accuracy in training Set: {:.5f}".format(trainingAccuracy))
    printToLogAndConsole("Accuracy in test Set (Exact Gradient): {:.5f}".format(testAccuracy))

    # SAVE DATA
    if SAVE:
        from datetime import datetime

        timeStamp = datetime.now().strftime('%m-%d_%H-%M-%S_')
        name = 'ACC_TR-{:.5f}_ACC_TE-{:.5f}_FC-{:.5f}_REG_{:.3f}_HN1-{:03d}_I-{:05d}_' \
            .format(trainingAccuracy, testAccuracy, finalCost, REG, nn1.hiddenNeurons, NUM_ITERS)

        with open('savedData/' + timeStamp + name + 'para.csv', 'w') as parametersFile:
            [parametersFile.write("{:+015.10f}\n".format(i)) for i in theta.squeeze()]

    # GENERATE PLOTS
    # fig, axes = plt.subplots(1)
    # plt.plot(costHistory[1:-3])
    #
    # plt.show()
