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
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    return weights
def fitness(pop,ind,model,X,Y):
    weights=[]
    for i in range(len(ind)):
        weights.append(pop[ind[i]][i])
    model.set_weights(weights)
    return model.evaluate(X,Y,verbose=0)[0]
def sor(A,B):
    index_A=np.argsort(A,0)
    A=A[index_A,:].reshape(A.shape)
    B=B[index_A,:].reshape(B.shape)
    return A, B
def train_model(batch_size=1000,epochs=1005,num_samples=10000,var1=0.99,var2=0.01):
    num_classes = 10
    print('var1:',var1)
    print('var2:',var2)
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
    x_train=x_train[0:num_samples]
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)[0:num_samples]
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
    n_pop=100
    count_effective=0
    flag_add=0
    flag_stop=0
    minibatches = random_mini_batches(x_train, y_train, batch_size)
    len_batch=len(minibatches)
    record=np.zeros((epochs,4))
    count_non_update_global=0
    count_non_update=0
    loss_min=1000000
    a = model.get_weights()
    n_weights= len(a)
    pop=[]
    pop.append(a)
    for i in range(n_pop-1):
        pop.append(shuffle_weights(model, weights=a))
    fit=np.zeros((n_pop,1))
    combi=np.random.randint(0,n_pop, size=[n_pop,n_weights])
    combi_best=combi[0,:]
    fit_best=10000
    for i in range(n_pop):
        fit[i,0]=fitness(pop,combi[i,:],model,x_train,y_train)
    fit, combi = sor(fit, combi)
    epochs_global=100
    n_elite=n_pop*0.1
    n_worst=n_pop*0.2
    count_update_global=np.zeros((n_pop,2))
    for iter in range(epochs_global):
        for j in range(n_pop):
            if j<n_elite:
                newcombi= copy.deepcopy(combi[j, :])
                newcombi[np.random.randint(n_weights)]=np.random.randint(n_pop)
                newcombi[np.random.randint(n_weights)] = np.random.randint(n_pop)
                newfit=fitness(pop,newcombi,model,x_train,y_train)
                if newfit+0.01<fit[j,0]:
                    count_update_global[j,0]=count_update_global[j,0]+1
                    count_update_global[j,1] = count_update_global[j,1] + fit[j,0]-newfit
                    combi[j, :]=newcombi
                    fit[j,0]=newfit
            elif j>n_pop-n_elite and j<n_pop*(1-n_worst):
                point_cut=np.random.randint(n_weights-8,size=2)
                parents=np.random.randint(n_pop*0.1,size=2)
                newcombi=copy.deepcopy(combi[j,:])
                newcombi[point_cut[0]:point_cut[0]+4] = copy.deepcopy(combi[parents[0],point_cut[0]:point_cut[0]+4])
                if np.random.random()<0.1:
                    newcombi[np.random.randint(n_weights)] = np.random.randint(n_pop)
                    newcombi[np.random.randint(n_weights)] = np.random.randint(n_pop)
                newfit = fitness(pop, newcombi, model,x_train,y_train)
                if newfit+0.03<fit[j,0]:
                    count_update_global[j, 0] = count_update_global[j, 0] + 1
                    count_update_global[j, 1] = count_update_global[j, 1] + fit[j, 0] - newfit
                    combi[j, :]=newcombi
                    fit[j,0]=newfit
            else:
                point_cut=np.random.randint(n_weights,size=4)
                parents=np.random.randint(n_elite,size=4)
                newcombi=copy.deepcopy(combi[j,:])
                newcombi[point_cut[0]]=copy.deepcopy(combi[parents[0],point_cut[0]])
                #newcombi[point_cut[1]] =copy.deepcopy( combi[parents[1], point_cut[1]])

                if np.random.random()<0.1:
                    newcombi[np.random.randint(n_weights)] = np.random.randint(n_pop)
                    newcombi[np.random.randint(n_weights)] = np.random.randint(n_pop)
                newfit = fitness(pop, newcombi,model,x_train,y_train)
                if newfit+0.02<fit[j,0]:
                    count_update_global[j,0]=count_update_global[j,0]+1
                    count_update_global[j,1] = count_update_global[j,1] + fit[j,0]-newfit
                    combi[j, :]=newcombi
                    fit[j,0]=newfit
        fit,combi = sor(fit, combi)
        if iter%10==9:
            print(iter)
        if fit[0,0]<fit_best:
            count_non_update_global = 0
            fit_best=fit[0,0]
            combi_best=combi[0,:]
            print((fit_best,iter))

        else:
            count_non_update_global=count_non_update_global+1
        if count_non_update_global==20:
            print(iter,"no update for 20 iterations")
            break
#--------------------------local search----------------------------------------
    for i in range(epochs):
        print(i,'th iteration------------------------------------------------')
        count_batches =0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            model.train_on_batch(minibatch_X, minibatch_Y)
            score_old = model.evaluate(minibatch_X, minibatch_Y, verbose=0)[0]
            for n in range (10):
                a = np.array(model.get_weights())
                b = np.array(model.get_weights())
                if flag_add==0:
                    j=np.random.randint(0, len(b))
                    adding_random=np.random.normal(0,0.01,b[j].shape)
                b[j]=b[j]+adding_random
                model.set_weights(b)
                score_new=model.evaluate(minibatch_X, minibatch_Y, verbose=0)[0]
                if score_new<score_old*(var1+i*var2/epochs):
                    if (count_non_update>5 and score_new>0.9999*score_old) or (score_new<=0.9999*score_old):
                        score_old=score_new
                        count_effective=count_effective+1
                        flag_add=1
                        loss_min=score_new
                        count_non_update=0
                else:
                    model.set_weights(a)
                    flag_add = 0
            count_batches = count_batches +1

        record[i , 0], record[i , 1] = model.evaluate(x_train,y_train,verbose=0)
        record[i, 2], record[i , 3] = model.evaluate(x_test,   y_test,verbose=0)
        if loss_min>record[i , 0]:
            loss_min=record[i , 0]
            count_non_update=0
        else:
            count_non_update=count_non_update+1;

        print('Train loss:', record[i,0])
        print('Train accuracy:', record[i,1])
        print('Test loss:',record[i,2])
        print('Test accuracy',record[i,3])
        print('number of effective local search:', count_effective)
        if i>700 and i%100==0:
            df = pd.DataFrame(record, columns= ['train_loss', 'train_acc','test_loss', 'test_acc'])
            df.to_csv('minist_ga_non_break_local'+str(i)+str(num_samples)+str(var1)+str(var2)+'.csv', index = None, header=True)

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
        '--num-samples',
        help='epochs',
        type=int,
        required=True
    )
    parser.add_argument(
        '--var1',
        help='var-base-req',
        type=float,
        required=True
    )
    parser.add_argument(
        '--var2',
        help='var_shrin',
        type=float,
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)
