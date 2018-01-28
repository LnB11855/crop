import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import KFold
import scipy.io as sio
import argparse
from tensorflow.python.lib.io import file_io
from datetime import datetime
import time
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = [n_x, None])
    Y = tf.placeholder(tf.float32, shape = [n_y, None])
    return X, Y
def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [60,19465], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [60,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [30,60], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [30,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,30], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

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
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.tanh(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.tanh(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             }
    return Z3
def compute_cost(Z3, Y):
    cost=tf.sqrt(tf.reduce_mean(tf.squared_difference(Z3, Y)))
    return cost
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
def update_parameters(parameters, grads, learning_rate = 1.2):
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
def nn_model(log, X_train, Y_train, XX_val,YY_val,num_epochs = 10000, learning_rate = 0.0012,minibatch_size = 32,print_cost=False):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    logpath = log
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    n_x = X_train.shape[0]
    m=X_train.shape[1]# (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                           # n_y : output size
    costs = []                                        # To keep track of the cost
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3= forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    train_cost_summary = tf.summary.scalar("train_cost", cost)
    writer = tf.summary.FileWriter(logpath, graph=tf.get_default_graph())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      for epoch in range(num_epochs):
        num_minibatches = int(m/ minibatch_size)
        epoch_cost = 0
        seed = seed + 1
        # _,minicost = sess.run([optimizer, cost])
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            epoch_cost += minibatch_cost / num_minibatches
        
        if print_cost and epoch % 1 == 0:
            coff=np.corrcoef(sess.run(Z3, feed_dict={X: X_train, Y: Y_train}), Y_train)[0, 1]
            print ("Cost after epoch %i: %f correlation coefficient: %f" %(epoch, epoch_cost,coff)
            writer.add_summary(coff, epoch)
        costs.append(epoch_cost)
        
      val_cost=0
      val_cost= sess.run(cost, feed_dict={X: XX_val, Y: YY_val})
      val_pre=sess.run(Z3, feed_dict={X: XX_val, Y: YY_val})
      val_corr= np.corrcoef(val_pre, YY_val)[0, 1]
      print("Cost-validation data : %f correlation coefficient-validation data: %f" % (val_cost,val_corr))
    writer.flush()
    return parameters,val_cost,val_corr
def train_model(train_file='5000test.mat', job_dir='./tmp/crop-challenge', **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()

    file_stream = file_io.FileIO(train_file, mode='r')
    data= sio.loadmat (file_stream)
    n_fold=10
    cv=KFold(n_splits=n_fold, shuffle=True, random_state=None)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_train=np.float32(X_train)
    Y_train=np.float32(Y_train)
    np.random.seed(1)
    record=np.zeros((n_fold,2))
    for k,(train_index, val_index) in enumerate(cv.split(X_train)):
        # print("TRAIN:", train_index, "TEST:", val_index)
        XX_train, XX_val = X_train[train_index].T, X_train[val_index].T
        YY_train, YY_val = Y_train[train_index].T, Y_train[val_index].T

        parameters, val_cost, val_corr = nn_model(logs_path,XX_train, YY_train,XX_val,YY_val, 10000, learning_rate=0.00012, minibatch_size=500, print_cost=True)
        # print("W1 = " + str(parameters["W1"]))
        # print("b1 = " + str(parameters["b1"]))
        # print("W2 = " + str(parameters["W2"]))
        # print("b2 = " + str(parameters["b2"]))
        # print("W3 = " + str(parameters["W3"]))
        # print("b3 = " + str(parameters["b3"]))

        record[k,0]=val_cost
        record[k,1]=val_corr
    print("mean cost:%f mean corr: %f" % (record.mean(axis=0)[0],record.mean(axis=0)[1]))


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
