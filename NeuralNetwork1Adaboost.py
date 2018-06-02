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


def computeCost(theta, X, y, reg, nnArchi,D,labels):
    # Recover parameter matrices
    v = getAdaboostVWeights(D, X, y, labels)
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
    cost =  v * ( -y * np.log(y_estimate + (1 - safetyFactor)) - (1 - y) * np.log(1 - y_estimate * safetyFactor) )
    cost = (1 / m) * sum(np.sum(cost,axis=1))

    # Add Regularization Term
    regularization_term = np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2)
    regularization_term = (regularization_term * reg) / (2 * m)

    return cost + regularization_term


def computeGradient(theta, X, y, reg, nnArchi,D,labels):
    # Recover parameter matrices
    v = getAdaboostVWeights(D, X, y, labels)
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
    d3 =  v.T * (y_estimate - y.T)
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
    outputNeurons = 3
    m = 5
    testNN = NeuralNetworkArchi(n, outputNeurons, hiddenNeurons, 'tanh')

    # Generate 'random' test data and weights
    theta1 = debugInitializeWeights(hiddenNeurons, n);
    theta2 = debugInitializeWeights(outputNeurons, hiddenNeurons);
    theta = np.append(theta1.reshape(-1, 1), theta2.reshape(-1, 1))
    X = debugInitializeWeights(m, n - 1);
    X = np.insert(X, 0, 1, axis=1)
    labels = np.array(0 + np.arange(m) % (outputNeurons)).reshape(m, 1);
    y = np.zeros((m,outputNeurons))
    for i in range(m):
        y[i,labels[i]]=1

    D = initAdaboostWeights(X,y, labels)
    D[0,0][1] =D[0,0][1]+0.05
    D[0,0][2] =D[0,0][2]-0.05
    D[1,0][0] = D[1, 0][0] + 0.07
    D[1,0][2] = D[1, 0][2] - 0.07

    # Calculate Exact Gradient
    exactGradient = computeGradient(theta, X, y, reg, testNN,D,labels)

    # Aproximate Gradient
    aproxGradient = np.zeros_like(theta)
    epsilonVector = np.zeros_like(theta)

    for i in range(theta.size):
        epsilonVector[i] = epsilon
        up = computeCost(theta + epsilonVector, X, y, reg, testNN,D,labels)
        down = computeCost(theta - epsilonVector, X, y, reg, testNN,D,labels)
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


def minibatchGradientDescent(theta, X, y, alpha, reg, numIters, nnArchi,P,D,labels, batchSize=45):
    takeHistory = numIters
    costHistory = np.zeros(int(numIters / takeHistory) + 1)
    m = y.shape[0]

    for i in range(numIters):
        # Get minibatch - random Rows from X matrix
        randIdx = np.random.choice(X.shape[0], batchSize, replace=False ,p=P)
        X_batch = X[randIdx, :]
        y_batch = y[randIdx, :]

        # Update parameter vector
        gradient = computeGradient(theta, X_batch, y_batch, reg, nnArchi,D,labels)
        theta = theta - (alpha / m) * gradient

        # Compute Cost and Gradient
        if i % takeHistory == 0:
            if True:
                costHistory[int(i / takeHistory)] = computeCost(theta, X, y, reg, nnArchi,D,labelsTrain)

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

def initAdaboostWeights(xTrain,y,labelsTrain):
    numberOfClasses = y.shape[1]
    B = xTrain.shape[0] * (numberOfClasses -1)
    D = np.array([])

    xTrainSize = xTrain.shape[0]
    # D-Weights
    for i in range(xTrainSize):
        D = np.append(D, {})
        for j in range(numberOfClasses):
            if j != labelsTrain[i]:
                D[i][j] = 1 / B

    return D.reshape(-1, 1)

def getAdaboostPWeights(D):
    P = np.array(list(map(lambda t: sum(t.values()), D.reshape(-1, ) ) ) )
    P = P / sum(P)
    return P

def getAdaboostVWeights(D,xTrain,y,labelsTrain):
    classes = y.shape[1]
    m = xTrain.shape[0]
    v = np.zeros((m,classes))

    for i in range(m):
        v[i,labelsTrain[i]] = 1
        for key,values in D[i,0].items():
            v[i,key] = D[i,0][key] / max(D[i,0].values())

    return v

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

def initNeuralNetworksParameters(nn1):
    ep = 0.7
    w1 = np.random.rand(nn1.hiddenNeurons, nn1.inputNeurons + 1) * ep * 2 - ep
    w2 = np.random.rand(nn1.outputNeurons, nn1.hiddenNeurons + 1) * ep * 2 - ep

    # Flat matrices into one single parameter vector
    return np.append(w1, w2)

def predictionAdaboost(bankOfNeuralNetworks,bankOfBeta, X, y,labels, nnArchi):
    results = np.zeros((y.shape[0],4))
    m = y.shape[0]
    errorHistory = np.zeros(bankOfNeuralNetworks.shape[0])
    for i in range(bankOfNeuralNetworks.shape[0]):
        theta = bankOfNeuralNetworks[i,:]
        yEstimates = np.log(1/bankOfBeta[i])*predict(theta,X,y,labels,nnArchi)
        results = results + yEstimates

        tempPredictions = np.argmax(results,axis = 1)
        errorHistory[i] =  1 - (sum(np.equal(tempPredictions, labels)))/m

    predictions = np.argmax(results,axis = 1)

    acc = sum(np.equal(predictions, labels))
    return acc / m , errorHistory

# COMMAND LINE ARGUMENT PARSER
# GLOBAL CONSTANTS
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reg", required=False, default="0.001", help="Neural Network Regularization Parameter")
ap.add_argument("-l", "--load", required=False, default="0", help="Load a trained neural network")
ap.add_argument("-s", "--save", required=False, default="1", help="Save the neural network")
loadDirectory = "06-01_00-42-30_ACC_TR-0.72643_ACC_TE-0.72393_FC-0.34179_REG_0.010_HN1-020_I-400000_para.csv"
ap.add_argument("-d", "--directory", required=False, default=loadDirectory,
                help="Directory where the neural network is found")
ap.add_argument("-a", "--alpha", required=False, default="3000", help="Alpha")
ap.add_argument("-i", "--iterations", required=False, default="80000", help="Number of interations of backpropagation")
ap.add_argument("-b", "--adaboost_iter", required=False, default="3", help="Number of interations of Adaboost")

args = vars(ap.parse_args())

NUM_ITERS = int(args['iterations'])
REG = float(args['reg'])
SAVE = bool(int(args['save']))
LOAD = bool(int(args['load']))
FILE_TO_LOAD = args['directory']
ALPHA = int(args['alpha'])
ADABOOST_ITERATIONS = int(args['adaboost_iter'])

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
    hiddenNeurons = 90
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
    check = gradientCheck(REG)
    printToLogAndConsole("\nCheck Backpropagation(Difference between Numerical" \
                         "Gradient and Analytical Gradient): {:.4E}\n".format(check))

    # Print Global Constants
    printToLogAndConsole("{:25}: {:06.4f}".format("Regularization", REG))
    printToLogAndConsole("{:25}: {:06d}".format("Backpropagation iterations", NUM_ITERS))
    printToLogAndConsole("{:25}: {:06d}".format("Adaboost iterations", ADABOOST_ITERATIONS))
    printToLogAndConsole("{:25}: {:6s}".format("Load", str(LOAD)))
    printToLogAndConsole("{:25}: {:6s}".format("Save", str(SAVE)))
    printToLogAndConsole("{:25}: {:6s}\n".format("File", FILE_TO_LOAD))

    # Initial Cost Calculation
    D=initAdaboostWeights(xTrain,yTrain,labelsTrain)
    printToLogAndConsole("Training Neural Network...\n")
    initialCost = computeCost(theta, xTrain, yTrain, REG, nn1,D,labelsTrain)
    printToLogAndConsole("Initial Cost: {:.5f}\n".format(initialCost))

    adaboostIterations = ADABOOST_ITERATIONS
    bankOfNeuralNetworks = np.zeros((adaboostIterations,theta.shape[0]))
    bankOfPseudoLoss = np.zeros(adaboostIterations)
    bankOfBeta = np.zeros(adaboostIterations)

    ###########################################################
    # Init Adaboost
    D=initAdaboostWeights(xTrain,yTrain,labelsTrain)

    # Adaboost Loop
    startTime1 = time.time()
    for t in range(adaboostIterations):
        theta = initNeuralNetworksParameters(nn1)
        P=getAdaboostPWeights(D)
        # Stochastic Gradient Descent
        theta, costHistory = minibatchGradientDescent(theta, xTrain, yTrain, ALPHA, REG, NUM_ITERS, nn1,P.reshape(-1,),D,labelsTrain)
        bankOfNeuralNetworks[t,:] = theta.reshape(1,-1)

        # Pseudoloss calculation
        yEstimates = predict(theta, xTrain, yTrain, labelsTrain, nn1)

        loss = 0
        for i in range(xTrain.shape[0]):
            res = np.array( list(map(lambda t: list(t), D[i,0].items())))
            const = 1 - yEstimates[i,labelsTrain[i]]
            loss += sum( res[:,1] * (const + yEstimates[i,res[:,0].astype(np.int)]) )
        loss = 0.5 * loss
        beta = loss / (1 - loss)

        bankOfPseudoLoss[t] = loss
        bankOfBeta[t] = beta

        # Update Weights
        for i in range(xTrain.shape[0]):
            for key,value in D.reshape(-1,)[i].items():
                D[i,0][key]= D.reshape(-1,)[i][key] * beta ** (0.5*(1 + yEstimates[i,labelsTrain[i]] + yEstimates[i,key]))

        normalization = sum(map(lambda dict1: sum(dict1.values()), D.reshape(-1, )))
        for i in range(xTrain.shape[0]):
            for key,value in D[i,0].items():
                D[i,0][key]= D[i,0][key]/normalization

        #tmp = sum(map(lambda dict1: sum(dict1.values()), D.reshape(-1, )))
        print("number of weak classifiers:",t)
        print("adaboost pseudo loss:", loss,"\n")
        #print("temp:",tmp)
        ############################################################

    endTime1 = time.time()

    # Logging training information
    adaboostTrainAccuracy,trainErrorHistory = predictionAdaboost(bankOfNeuralNetworks, bankOfBeta, xTrain, yTrain, labelsTrain, nn1)
    adaboostTestAccuracy,testErrorHistory = predictionAdaboost(bankOfNeuralNetworks, bankOfBeta, xTest, yTest, labelsTest, nn1)
    finalCost = computeCost(theta, xTrain, yTrain, REG, nn1,D,labelsTrain)
    formatedTime = time.strftime('%H:%M:%S', time.gmtime(endTime1 - startTime1))

    printToLogAndConsole("Finish Training...\n")
    printToLogAndConsole("Adaboost Execution time: {:4.3f}".format(endTime1 - startTime1))
    printToLogAndConsole("Adaboost Execution time: {:s}".format(formatedTime))
    printToLogAndConsole("Regularization: {:.6f}".format(REG))
    printToLogAndConsole("Iterations: {:03d}".format(NUM_ITERS))
    printToLogAndConsole("Initial Cost: {:.6f}".format(initialCost))
    printToLogAndConsole("Final Cost: {:.6f}".format(finalCost))

    # # EVALUATE MODEL IN TEST SET
    # trainingAccuracy = evaluateModel(theta, xTrain, yTrain,labelsTrain, nn1)
    # testAccuracy = evaluateModel(theta, xTest, yTest,labelsTest, nn1)

    # LOGGING EVALUATIONS
    printToLogAndConsole("Model evaluation\n")
    printToLogAndConsole("Accuracy in training Set: {:.6f}".format(adaboostTrainAccuracy))
    printToLogAndConsole("Accuracy in test Set (Exact Gradient): {:.6f}".format(adaboostTestAccuracy))

    # SAVE DATA
    if SAVE:
        from datetime import datetime

        timeStamp = datetime.now().strftime('%m-%d_%H-%M-%S_')
        name = 'ACC_TR-{:.6f}_ACC_TE-{:.6f}_FC-{:.6f}_REG_{:.6f}_HN1-{:03d}_I-{:05d}_ADA_BOOST_{:04d}' \
            .format(adaboostTrainAccuracy, adaboostTestAccuracy, finalCost, REG, nn1.hiddenNeurons, NUM_ITERS,adaboostIterations)

        with open('savedData/' + timeStamp + name + 'para.csv', 'w') as parametersFile:
            for i in range(bankOfNeuralNetworks.shape[0]):
                for j in range(bankOfNeuralNetworks.shape[1]):
                    parametersFile.write( "{:+015.10f},".format(bankOfNeuralNetworks[i,j]))
                parametersFile.write("\n")

    # GENERATE PLOTS
    fig, axes = plt.subplots(1,figsize = (10, 8), dpi = 100)
    axes.plot(trainErrorHistory,label="Error en el set de Entrenamiento")
    axes.plot(testErrorHistory,label="Error en el set de Prueba")
    #axes.set_ylim((min(trainErrorHistory)-2,max(trainErrorHistory)+2))
    axes.set_title(name)
    axes.set_xlabel("Número de iteraciones de Adaboost")
    axes.set_ylabel("Porcentaje de error (%)")
    axes.grid()
    axes.legend()

    fig.savefig('savedData/' + timeStamp + name+".jpg")

    plt.show()
