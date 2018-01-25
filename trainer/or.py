import math
import numpy as np
import h5py
import scipy.io as sio
import argparse
from tensorflow.python.lib.io import file_io
np.random.seed(1)
def initialize_parameters():
    W1=np.random.randn(60,100)
    b1=np.zeros((60,1))
    W2=np.random.randn(30,60)
    b2=np.zeros((30,1))
    W3=np.random.randn(1,30)
    b3=np.zeros((1,1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3, A2) + b3

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             }
    return Z3, cache
def compute_cost(Z3, Y):
    cost=np.sqrt(((Z3- Y) ** 2).mean())
    return cost
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z3 = cache["Z3"]
    dZ3=Z3-Y
    dW3=1.0 / m * np.dot(dZ3, A2.T)
    db3=1.0 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.dot(W3.T, dZ3) * (1 - np.power(A2, 2))
    dW2 = 1.0 / m * np.dot(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)
    ### END CODE HERE ###

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3,
             }

    return grads
def update_parameters(parameters, grads, learning_rate = 0.000001):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    W3 = W3 - learning_rate*dW3
    b3 = b3 - learning_rate*db3
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  }
    return parameters
def nn_model(train_file='5000test.mat', job_dir='./tmp/crop-challenge', **args):
    file_stream = file_io.FileIO(train_file, mode='r')
    num_iterations=100
    data= sio.loadmat (file_stream)
    X = data['X_train'][0:100,0:100]
    Y = data['Y_train'][0:100]
    X=X.T
    Y=Y.T
    np.random.seed(3)
    parameters = initialize_parameters()
    # W1 = parameters["W1"]
    # b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]
    # W3 = parameters["W3"]
    # b3 = parameters["b3"]
    for i in range(0, num_iterations):
        Z3, cache = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 0.000001)
        if i % 1== 0:
            print ("Cost after iteration %i: %f %f" %(i, cost, np.corrcoef (Z3, Y)[0, 1]))
    return parameters
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)

