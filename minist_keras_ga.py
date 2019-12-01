from __future__ import print_function
import argparse
import copy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import math
import numpy as np
import pandas as pd
def random_mini_batches(X, Y, mini_batch_size=100, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]
    mini_batch_size=int(mini_batch_size)
    num_complete_minibatches =int( math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
def train_model(batch_size=1000,epochs=100,var_base_req=0.99,var_shrin=0.015):
    num_classes = 10
    print('var1:',var_base_req)
    print('var2:',var_shrin)
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train=x_train[0:10000]
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)[0:10000]
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28,1)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.sgd(),
                  metrics=['accuracy'])
    model.summary()
    n_pop=10
    n_para=84170
    fit=np.zeros((n_pop,1))
    pop = np.random.normal(0,0.01,(n_pop, n_para))
    count_update=0

    flag_add=0
    flag_stop=0
    minibatches = random_mini_batches(x_train, y_train, batch_size)
    len_batch=len(minibatches)
    record=np.zeros((epochs,4))


    for i in range(epochs):
        print(i,'th iteration------------------------------------------------')
        count_batches =0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            model.train_on_batch(minibatch_X, minibatch_Y)

            for n in range (10):
                a = np.array(model.get_weights())
                score_old = model.evaluate(minibatch_X, minibatch_Y, verbose=0)[0]
                b = np.array(model.get_weights())
                if flag_add==0:
                    j=np.random.randint(0, len(b))
                    adding_random=np.random.normal(0,0.01,b[j].shape)
                b[j]=b[j]+adding_random
                model.set_weights(b)
                score_new=model.evaluate(minibatch_X, minibatch_Y, verbose=0)[0]
                if score_new<score_old*(var_base_req+i*var_shrin/epochs):
                    a=copy.deepcopy(b)
                    score_old=score_new
                    count_update=count_update+1
                    flag_add=1
                    break
                else:
                    model.set_weights(a)
                    flag_add = 0
            count_batches = count_batches +1
            score = model.evaluate(x_train, y_train, verbose=0)
            # with K.Session() as sess:
            #
            #     outputTensor = model.output  # Or model.layers[index].output
            #     listOfVariableTensors = model.trainable_weights
            #     gradients =K.gradients(outputTensor, listOfVariableTensors)

        record[i , 0], record[i , 1] = model.evaluate(x_train,y_train,verbose=0)
        record[i, 2], record[i , 3] = model.evaluate(x_test,   y_test,verbose=0)
        print('Train loss:', record[i,0])
        print('Train accuracy:', record[i,1])
	print('Test loss:',record[i,2])
	print('Test accuracy',record[i,3])
        print('count_update:', count_update)
    df = pd.DataFrame(record, columns= ['train_loss', 'train_acc','test_loss', 'test_acc'])
    df.to_csv ('minist_ga_10000'+str(var_base_req)+str(var_shrin)+'.csv', index = None, header=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--batch-size',
        help='batchsize',
        type=int,
        required=True
    )
    parser.add_argument(
        '--epochs',
        help='epochs',
        type=int,
        required=True
    )
    parser.add_argument(
        '--var-base-req',
        help='var-base-req',
        type=float,
        required=True
    )
    parser.add_argument(
        '--var-shrin',
        help='var_shrin',
        type=float,
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)
