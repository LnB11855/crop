import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import KFold
import scipy.io as sio
import argparse
from tensorflow.python.lib.io import file_io
from datetime import datetime
import pickle
import time

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    A1= tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(A1, weights['h2']), biases['b2'])
    A2 = tf.nn.tanh(layer_2)
    layer_3 = tf.add(tf.matmul(A2, weights['h3']), biases['b3'])
    A3 = tf.nn.tanh(layer_3)
    layer_4 = tf.add(tf.matmul(A3, weights['h4']), biases['b4'])
    A4 = tf.nn.tanh(layer_4)
    layer_5 = tf.add(tf.matmul(A4, weights['h5']), biases['b5'])
    A5 = tf.nn.tanh(layer_5)
    layer_6 = tf.add(tf.matmul(A5, weights['h6']), biases['b6'])
    A6 = tf.nn.tanh(layer_6)
    layer_7 = tf.add(tf.matmul(A6, weights['h7']), biases['b7'])
    A7 = tf.nn.tanh(layer_7)
    layer_8 = tf.add(tf.matmul(A7, weights['h8']), biases['b8'])
    A8 = tf.nn.tanh(layer_8)
    layer_9 = tf.add(tf.matmul(A8, weights['h9']), biases['b9'])
    A9 = tf.nn.tanh(layer_9)
    out_layer = tf.matmul(A9, weights['out']) + biases['out']
    return out_layer
def compute_cost(Z3, Y):
    cost=tf.sqrt(tf.reduce_mean(tf.squared_difference(Z3, Y)))
    return cost
def train_model(train_fileA='5000test.mat',train_fileB='5000test.mat', job_dir='./tmp/crop-challenge', training_epochs=100,batch_size = 100,learning_rate = 0.001,opt=1,**args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    ops.reset_default_graph()
    file_stream = file_io.FileIO(train_fileA, mode='r')
    X_train,Y_train=pickle.load(file_stream,encoding='bytes')
    X_train=np.float64(X_train[:,1:])
    Y_train=np.float64(Y_train).reshape((X_train.shape[0],1))
    file_stream = file_io.FileIO(train_fileB, mode='r')
    X_trainB,Y_trainB=pickle.load(file_stream,encoding='bytes')
    X_trainB=np.float64(X_trainB[:,1:])
    Y_trainB=np.float64(Y_trainB).reshape((X_trainB.shape[0],1))
    X_train=np.concatenate((X_trainB,X_train),axis=1)
    Y_train=np.concatenate((Y_trainB,Y_train),axis=1)
    learning_rate=np.float64(learning_rate)
    batch_size=int(batch_size)
    training_epochs=int(training_epochs)
    print(X_train.shape,Y_train.shape)
    np.random.seed(1)
    n_input =X_train.shape[1]
    print('number of features',n_input)
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    n_hidden_3 = 128 # 2nd layer number of neurons
    n_hidden_4 = 64 # 2nd layer number of neurons
    n_hidden_5 = 32 # 2nd layer number of neurons
    n_hidden_6 = 16 # 2nd layer number of neurons
    n_hidden_7 = 8 # 2nd layer number of neurons
    n_hidden_8 = 4 # 2nd layer number of neurons
    n_hidden_9 = 2 # 2nd layer number of neurons
    n_output=1
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input,    n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
        'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
        'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
        'h9': tf.Variable(tf.random_normal([n_hidden_8, n_hidden_9])),
        'out':tf.Variable(tf.random_normal([n_hidden_9, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'b6': tf.Variable(tf.random_normal([n_hidden_6])),
        'b7': tf.Variable(tf.random_normal([n_hidden_7])),
        'b8': tf.Variable(tf.random_normal([n_hidden_8])),
        'b9': tf.Variable(tf.random_normal([n_hidden_9])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }
    outlayer=multilayer_perceptron(X,weights,biases)
    cost=compute_cost(outlayer, Y)
    train_cost_summary = tf.summary.scalar("train_cost", cost)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    if opt==1:
      optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    if opt==2:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
    if opt==3:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)  
    init = tf.global_variables_initializer()
    m=X_train.shape[0]
    costs = []
    seed=0
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        for epoch in range(training_epochs):
            
            num_minibatches = int(m / batch_size)
            epoch_cost = 0
            seed = seed + 1
            # _,minicost = sess.run([optimizer, cost])
            minibatches = random_mini_batches(X_train, Y_train, batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if epoch % 1 == 0:
                coff = 0
                print("Cost after epoch %i: %f correlation coefficient: %f" % (
                epoch, epoch_cost, coff ))
                _train_cost_summary=sess.run(train_cost_summary,feed_dict={X: X_train, Y: Y_train})
                writer.add_summary(_train_cost_summary, epoch)
            costs.append(epoch_cost)
    writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-fileA',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--train-fileB',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--training-epochs',
        help='number of epochs',
        required=True
    )
    parser.add_argument(
        '--batch-size',
        help='batch size',
        required=True
    )
    parser.add_argument(
        '--learning-rate',
        help='learning rate',
        required=True
       
    )
    parser.add_argument(
        '--opt',
        help='choice of optimizer',
        required=True
       
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)
