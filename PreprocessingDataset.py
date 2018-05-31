import pandas as pd
import numpy as np

def splitDataIntoClasses(numOfSamples=3500, randomSeed=42):
    data = pd.read_csv('exerciseData/msd_genre_dataset.csv')

    metalPunkData = data[(data.genre == 'metal') | (data.genre == 'punk')]
    danceElectronicaData = data[(data.genre == 'dance and electronica')]
    jazzBluesClassicalData = data[(data.genre == 'jazz and blues') | (data.genre == 'classical')]
    folkData = data[(data.genre == 'folk')]

    print("Dance data size:", danceElectronicaData.shape[0])
    print("Metal & Punk data size:", metalPunkData.shape[0])
    print("Jazz Blues & Classical size:", jazzBluesClassicalData.shape[0])
    print("folk data size:", folkData.shape[0])

    # Remove Track_id, artist_name, title
    headers = list(danceElectronicaData.columns.values)
    danceElectronicaData = danceElectronicaData.drop(labels=[headers[1], headers[2], headers[3]], axis=1)
    metalPunkData = metalPunkData.drop(labels=[headers[1], headers[2], headers[3]], axis=1)
    jazzBluesClassicalData = jazzBluesClassicalData.drop(labels=[headers[1], headers[2], headers[3]], axis=1)
    folkData = folkData.drop(labels=[headers[1], headers[2], headers[3]], axis=1)

    # Sample equal number of samples for each class
    frac1 = numOfSamples / danceElectronicaData.shape[0]
    danceElectronicaData = danceElectronicaData.sample(frac=frac1, replace=False, random_state=randomSeed)

    frac1 = numOfSamples / metalPunkData.shape[0]
    metalPunkData = metalPunkData.sample(frac=frac1, replace=False, random_state=randomSeed + 2)

    frac1 = numOfSamples / jazzBluesClassicalData.shape[0]
    jazzBluesClassicalData = jazzBluesClassicalData.sample(frac=frac1, replace=False, random_state=randomSeed + 3)

    frac1 = numOfSamples / folkData.shape[0]
    folkData = folkData.sample(frac=frac1, replace=False, random_state=randomSeed + 4)

    metalPunkData.index.name = 'data_base_index'
    danceElectronicaData.index.name = 'data_base_index'
    jazzBluesClassicalData.index.name = 'data_base_index'
    folkData.index.name = 'data_base_index'

    metalPunkData.to_csv("exerciseData/metalPunk.csv")
    danceElectronicaData.to_csv("exerciseData/danceElectronica.csv")
    jazzBluesClassicalData.to_csv(("exerciseData/jazzBluesClassical.csv"))
    folkData.to_csv(("exerciseData/folk.csv"))

    print("\nNumber of samples in each class: " ,numOfSamples,"\n")


def createTrainAndTestSets(randomSeed=34, testFrac=0.20):
    #load Data
    metalPunkData = pd.read_csv('exerciseData/metalPunk.csv')
    danceElectronicaData = pd.read_csv('exerciseData/danceElectronica.csv')
    jazzBluesClassicalData = pd.read_csv('exerciseData/jazzBluesClassical.csv')
    folkData = pd.read_csv('exerciseData/folk.csv')

    folkTest = folkData.sample(frac=testFrac, replace=False, random_state=randomSeed + 12)
    folkTrain = folkData.drop(folkTest.index.values)

    metalPunkTest = metalPunkData.sample(frac=testFrac, replace=False, random_state=randomSeed + 9)
    metalPunkTrain = metalPunkData.drop(metalPunkTest.index.values)

    danceElectronicaTest = danceElectronicaData.sample(frac=testFrac, replace=False, random_state=randomSeed + 6)
    danceElectronicaTrain = danceElectronicaData.drop(danceElectronicaTest.index.values)

    jazzBluesClassicalTest = jazzBluesClassicalData.sample(frac=testFrac, replace=False, random_state=randomSeed + 3)
    jazzBluesClassicalTrain = jazzBluesClassicalData.drop(jazzBluesClassicalTest.index.values)

    testSet = pd.concat([metalPunkTest, folkTest, danceElectronicaTest, jazzBluesClassicalTest])
    trainingSet = pd.concat([metalPunkTrain, folkTrain, danceElectronicaTrain, jazzBluesClassicalTrain])

    testSet = testSet.sample(frac=1, replace=False)
    testSet.index.name = 'data_base_index_2'
    trainingSet = trainingSet.sample(frac=1, replace=False)
    trainingSet.index.name = "data_base_index_2"

    print("Training set size:", trainingSet.shape[0])
    print("Test set size:", testSet.shape[0])

    # Save Training Set And Test Set
    trainingSet.to_csv("exerciseData/trainingSetGenre.csv")
    testSet.to_csv("exerciseData/testSetGenre.csv")


def splitIntoFeatureMatrixAndLabels(dataSet):

    labels = dataSet['genre'].values
    X = dataSet.drop(labels=['genre', "data_base_index","data_base_index_2"], axis=1).values
    y = np.zeros((labels.shape[0],4))
    
    for i in range(labels.shape[0]):
        if labels[i] == "dance and electronica":
            y[i,:] = np.array([1,0,0,0]).reshape(1,4)
        elif labels[i] == "metal" or labels[i] == "punk":
            y[i,:] = np.array([0,1,0,0]).reshape(1,4)
        elif labels[i] == "folk":
            y[i,:] = np.array([0,0,1,0]).reshape(1,4)
        elif labels[i]=='jazz and blues' or labels[i] == 'classical':
            y[i,:] = np.array([0,0,0,1]).reshape(1,4)
        else:
            print("ERROR")
        
    y = y.astype(np.int)
    X = X.astype(np.float)
    return X,y


if __name__ == '__main__':
    # splitDataIntoClasses(randomSeed=25,numOfSamples=3500)
    # createTrainAndTestSets(randomSeed=34,testFrac=0.20)

    #Load Train and Test Set
    trainingSet = pd.read_csv("exerciseData/trainingSetGenre.csv")
    testSet = pd.read_csv("exerciseData/testSetGenre.csv")

    xTrain, yTrain = splitIntoFeatureMatrixAndLabels(trainingSet)
    xTest, yTest = splitIntoFeatureMatrixAndLabels(testSet)