from sklearn.datasets import load_boston
import numpy, math, heapq
import itertools

import matplotlib.pyplot as plt


def getPearson(A, B):
    A = numpy.array(A)
    B = numpy.array(B)
    AB = numpy.array(A * B)

    return (numpy.mean(AB) - (numpy.mean(A) * numpy.mean(B))) / (numpy.std(A) * numpy.std(B))


def getSplit(target, data, start, end):
    end = start + end

    test_target = []
    test_data = []
    training_target = []
    training_data = []
    train_index = []
    test_index = []

    l = len(data)
    for i in range(0, l):
        if i >= start and i < end:
            test_data.append(data[i])
            test_target.append(target[i])
            test_index.append(i)
        else:
            training_data.append(data[i])
            training_target.append(target[i])
            train_index.append(i)

    return [training_data, training_target, test_data, test_target, test_index, train_index]


def getTopNPearson(features_training_data, training_target, k, excluding):
    pear = []
    n = len(features_training_data)
    for i in range(0, n):
        if i == excluding:
            continue
        x = getPearson(features_training_data[i], training_target)
        y = []
        y.append(x)
        y.append(i)
        pear.append(y)

    top4 = heapq.nsmallest(k, pear, key=lambda x: -abs(x[0]))
    return top4


def getSelectedFeatures(normalized_features_training_data, normalized_features_test_data, top4):
    selective_normalized_features_training_data = []
    selective_normalized_features_test_data = []

    for i in top4:
        if i == -1:
            continue

        index = i
        selective_normalized_features_test_data.append(normalized_features_test_data[index])
        selective_normalized_features_training_data.append(normalized_features_training_data[index])
    return [selective_normalized_features_training_data, selective_normalized_features_test_data]


def getFeaturesMeannStd(features_training_data):
    features_mean = []
    features_std = []
    n = len(features_training_data)
    for i in range(0, n):
        features_mean.append(numpy.mean(features_training_data[i]))
        features_std.append(numpy.std(features_training_data[i]))

    # print 'mean'
    # print features_mean
    # print features_std
    return [features_mean, features_std]


def getNormalized(features_training_data, features_test_data, features_mean, features_std):
    normalized_features_training_data = []
    normalized_features_test_data = []
    n = len(features_training_data)
    for i in range(0, n):
        list = [(x - features_mean[i]) / features_std[i] for x in features_training_data[i]]
        normalized_features_training_data.append(list)

        ist = [(x - features_mean[i]) / features_std[i] for x in features_test_data[i]]
        normalized_features_test_data.append(ist)

    return [normalized_features_training_data, normalized_features_test_data]


def getFeatures(training_data, test_data):
    features_training_data = []
    features_test_data = []
    n = len(training_data[0])

    for i in range(0, n):
        list = [item[i] for item in training_data]
        features_training_data.append(list)

        ist = [item[i] for item in test_data]
        features_test_data.append(ist)

    return [features_training_data, features_test_data]


def plotHistograms(features_training_data, training_target):
    n = len(features_training_data)
    for i in range(0, n):
        plt.hist(features_training_data[i], bins=10)  # plt.hist passes it's arguments to np.histogram

        pear = getPearson(features_training_data[i], training_target)
        print "Pearson for attribute : ", AttributeName.get(str(i)), " = ", pear
        plt.title("Histogram with attribute " + AttributeName.get(str(i)))
        # plt.xlabel("Examples")
        # plt.ylabel("Values")
        plt.show()


def seperateTrainNTest(target, data, start, step):
    test_target = []
    test_data = []
    training_target = []
    training_data = []
    l = len(data)
    train_index = []
    test_index = []

    for i in range(0, l):
        if i == start:
            test_target.append(target[i])
            test_data.append(data[i])
            start += step
            test_index.append(i)
        else:
            training_target.append(target[i])
            training_data.append(data[i])
            train_index.append(i)
    return [training_data, training_target, test_data, test_target, test_index, train_index]


def returnMSE(normalized_features_training_data, normalized_features_test_data, lambd, training_target, test_target):
    # Setting up X,X' for train n test
    X_transpose = numpy.array(normalized_features_training_data)
    m = len(X_transpose[0])
    ones = numpy.array([[1] * m])
    X_transpose = numpy.concatenate((ones, X_transpose))
    X = X_transpose.transpose()

    X_transpose_test = numpy.array(normalized_features_test_data)
    m_test = len(X_transpose_test[0])
    ones = numpy.array([[1] * m_test])
    X_transpose_test = numpy.concatenate((ones, X_transpose_test))
    X_test = X_transpose_test.transpose()

    # Calculate w for linear regression

    theta = calculateW(X, X_transpose, training_target, lambd)

    # predict
    x = calculateMSE(theta, X, training_target, m)
    y = calculateMSE(theta, X_test, test_target, m_test)
    return [x[0], y[0], x[1]]

    pass


def calculateW(X, X_transpose, training_target, alpha):
    m = len(X_transpose)
    n = len(X_transpose[0])
    temp = float(n) * alpha * numpy.eye(m, dtype=int)  # n+1 - size of matrix
    temp[0][0] = 0
    return numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(X_transpose, X) + temp), X_transpose),
                     numpy.array(training_target))


def calculateMSE(theta, X, training_target, m):
    y_training_predict = []
    for i in range(0, m):
        y = numpy.dot(X[i], theta)
        y_training_predict.append(y)

    MSE = 0
    residual = []

    for i in range(0, m):
        t = -(y_training_predict[i] - training_target[i])

        residual.append(-t)
        MSE += math.pow(t, 2)
    MSE /= m;

    return [MSE, residual]


boston = load_boston()

# Data Analysis

x = seperateTrainNTest(boston.target, boston.data, 0, 7)
test_target = x[3]
test_data = x[2]
training_target = x[1]
training_data = x[0]

global AttributeName
AttributeName = {'0': 'CRIM', '1': 'ZN', '2': 'INDUS', '3': 'CHAS', '4': 'NOX', '5': 'RM', '6': 'AGE', '7': 'DIS',
                 '8': 'RAD', '9': 'TAX', '10': 'PTRATIO', '11': 'B', '12': 'LSTAT'}

x = getFeatures(training_data, test_data)
features_training_data = x[0]
features_test_data = x[1]

# plot histogram
plotHistograms(features_training_data, training_target)

# Data Preprocessing
x = getFeaturesMeannStd(features_training_data)
features_mean = x[0]
features_std = x[1]

x = getNormalized(features_training_data, features_test_data, features_mean, features_std)
normalized_features_training_data = x[0]
normalized_features_test_data = x[1]

print '\n'
# predict
lambd = 0
x = returnMSE(normalized_features_training_data, normalized_features_test_data, lambd, training_target, test_target)
print 'Linear Regression'
print 'MSE for train data ' + str(x[0])
print 'MSE for test  data ' + str(x[1])
print '\n'


# Making the dataSet









