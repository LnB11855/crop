import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
plt.ion() # interactive mode on
axisx1=[]
axisx2=[]
axisy1=[]
axisy2=[]
line, = plt.plot(axisx1,axisy1)  # plot the data and specify the 2d line
line2, = plt.plot(axisx2,axisy2,color = 'red', linestyle = '--')
ax = plt.gca() # get most of the figure elements
plt.legend(handles = [line, line2,], labels = ['all', 'average '],loc=1)
 # pause to show the figure
with open('D:\challenge\Crop\XYgcd33.pickle', 'rb') as handle:
    X_train,Y_train= pickle.load(handle)
X_train=X_train[:,1:]
n_pop=100
n_elite=20
n_output = 1
model_mean=0
model_sd=0.05
num_iteration=1000
batch_size = 5000
n_sample=X_train.shape[0]
num_minibatches = int( math.floor(n_sample/ batch_size))
print('number of minibatches is',num_minibatches)
np.random.seed(1)
n_input = X_train.shape[1]
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_hidden_3 = 128  # 2nd layer number of neurons
n_hidden_4 = 64  # 2nd layer number of neurons
n_hidden_5 = 32  # 2nd layer number of neurons
n_hidden_6 = 16  # 2nd layer number of neurons
n_hidden_7 = 16  # 2nd layer number of neurons
n_hidden_8 = 16  # 2nd layer number of neurons
n_hidden_9 = 16  # 2nd layer number of neurons
n1 = n_input * n_hidden_1
n2 = n1 + n_hidden_1 * n_hidden_2
n3 = n2 + n_hidden_2 * n_hidden_3
n4 = n3 + n_hidden_3 * n_hidden_4
n5 = n4 + n_hidden_4 * n_hidden_5
n6 = n5 + n_hidden_5 * n_hidden_6
n7 = n6 + n_hidden_6 * n_hidden_7
n8 = n7 + n_hidden_7 * n_hidden_8
n9 = n8 + n_hidden_8 * n_hidden_9
n10 = n9 + n_output * n_hidden_9
n11 = n10 + n_hidden_1
n12 = n11 + n_hidden_2
n13 = n12 + n_hidden_3
n14 = n13 + n_hidden_4
n15 = n14 + n_hidden_5
n16 = n15 + n_hidden_6
n17 = n16 + n_hidden_7
n18 = n17 + n_hidden_8
n19 = n18 + n_hidden_9
fit=np.zeros((n_pop,1))
n_para=(n_input+1)*n_hidden_1+(n_hidden_1+1)*n_hidden_2+(n_hidden_2+1)*n_hidden_3+(n_hidden_3+1)*n_hidden_4\
       +(n_hidden_4+1)*n_hidden_5+(n_hidden_5+1)*n_hidden_6+(n_hidden_6+1)*n_hidden_7+(n_hidden_7+1)*n_hidden_8\
       +(n_hidden_8+1)*n_hidden_9+n_hidden_9*n_output
#initialize xavier
def ini(n_pop,n_para):
    # pop = np.random.randn(n_pop, n_para)
    pop = np.random.normal(0,0.1,(n_pop, n_para))
    count = 0
    pop[:,0:n_input * n_hidden_1]=np.random.uniform(-np.sqrt(6/(n_input+n_hidden_1)), np.sqrt(6/(n_input+n_hidden_1)),(n_pop,n_input * n_hidden_1))
    count += n_input * n_hidden_1
    pop[:, count:count + n_hidden_1 * n_hidden_2] = np.random.uniform(-np.sqrt(6 / (n_hidden_2 + n_hidden_1)),
                                                       np.sqrt(6 / (n_hidden_2 + n_hidden_1)), (n_pop,n_hidden_2 * n_hidden_1))
    count += n_hidden_1 * n_hidden_2
    pop[:, count:count + n_hidden_2 * n_hidden_3] = np.random.uniform(-np.sqrt(6 / (n_hidden_2 + n_hidden_3)),
                                                                      np.sqrt(6 / (n_hidden_2 + n_hidden_3)),
                                                                      (n_pop,n_hidden_2 * n_hidden_3))
    count += n_hidden_3 * n_hidden_2
    pop[:, count:count + n_hidden_4 * n_hidden_3] = np.random.uniform(-np.sqrt(6 / (n_hidden_4 + n_hidden_3)),
                                                                      np.sqrt(6 / (n_hidden_4 + n_hidden_3)),
                                                                      (n_pop,n_hidden_4 * n_hidden_3))
    count += n_hidden_3 * n_hidden_4
    pop[:, count:count + n_hidden_4 * n_hidden_5] = np.random.uniform(-np.sqrt(6 / (n_hidden_4 + n_hidden_5)),
                                                                      np.sqrt(6 / (n_hidden_4 + n_hidden_5)),
                                                                      (n_pop,n_hidden_4 * n_hidden_5))
    count += n_hidden_5 * n_hidden_4
    pop[:, count:count + n_hidden_6 * n_hidden_5] = np.random.uniform(-np.sqrt(6 / (n_hidden_6 + n_hidden_5)),
                                                                      np.sqrt(6 / (n_hidden_6 + n_hidden_5)),
                                                                      (n_pop,n_hidden_6 * n_hidden_5))
    count += n_hidden_5 * n_hidden_6
    pop[:, count:count + n_hidden_6 * n_hidden_7] = np.random.uniform(-np.sqrt(6 / (n_hidden_6 + n_hidden_7)),
                                                                      np.sqrt(6 / (n_hidden_6 + n_hidden_7)),
                                                                      (n_pop,n_hidden_6 * n_hidden_7))
    count += n_hidden_7 * n_hidden_6
    pop[:, count:count + n_hidden_8 * n_hidden_7] = np.random.uniform(-np.sqrt(6 / (n_hidden_8 + n_hidden_7)),
                                                                      np.sqrt(6 / (n_hidden_8 + n_hidden_7)),
                                                                      (n_pop,n_hidden_8 * n_hidden_7))
    count += n_hidden_7 * n_hidden_8
    pop[:, count:count + n_hidden_8 * n_hidden_9] = np.random.uniform(-np.sqrt(6 / (n_hidden_8 + n_hidden_9)),
                                                                      np.sqrt(6 / (n_hidden_8 + n_hidden_9)),
                                                                      (n_pop,n_hidden_8 * n_hidden_9))
    count += n_hidden_9 * n_hidden_8
    pop[:, count:count + n_output * n_hidden_9] = np.random.uniform(-np.sqrt(6 / (n_output + n_hidden_9)),
                                                                      np.sqrt(6 / (n_output + n_hidden_9)),
                                                                    (n_pop,n_output * n_hidden_9))
    count += n_hidden_9 * n_output
    return pop
def fitness(individual,X,Y):
    w1=individual[0:n1].reshape(n_input, n_hidden_1)
    w2 = individual[n1:n2].reshape(n_hidden_1, n_hidden_2)
    w3 = individual[n2:n3].reshape(n_hidden_2, n_hidden_3)
    w4 = individual[n3:n4].reshape(n_hidden_3, n_hidden_4)
    w5 = individual[n4:n5].reshape(n_hidden_4, n_hidden_5)
    w6 = individual[n5:n6].reshape(n_hidden_5, n_hidden_6)
    w7 = individual[n6:n7].reshape(n_hidden_6, n_hidden_7)
    w8 = individual[n7:n8].reshape(n_hidden_7, n_hidden_8)
    w9 = individual[n8:n9].reshape(n_hidden_8, n_hidden_9)
    wout = individual[n9:n10].reshape(n_hidden_9, n_output)
    b1= individual[n10:n11].reshape(n_hidden_1, )
    b2 = individual[n11:n12].reshape(n_hidden_2, )
    b3 = individual[n12:n13].reshape(n_hidden_3, )
    b4 = individual[n13:n14].reshape(n_hidden_4, )
    b5 = individual[n14:n15].reshape(n_hidden_5, )
    b6 = individual[n15:n16].reshape(n_hidden_6, )
    b7 = individual[n16:n17].reshape(n_hidden_7, )
    b8 = individual[n17:n18].reshape(n_hidden_8, )
    b9 = individual[n18:n19].reshape(n_hidden_9, )
    Z1 = np.dot(X, w1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, w2) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(A2, w3) + b3
    A3 = np.tanh(Z3)
    Z4 = np.dot(A3, w4) + b4
    A4 = np.tanh(Z4)
    Z5 = np.dot(A4, w5) + b5
    A5 = np.tanh(Z5)
    Z6 = np.dot(A5, w6) + b6
    A6 = np.tanh(Z6)
    Z7 = np.dot(A6, w7) + b7
    A7 = np.tanh(Z7)
    Z8 = np.dot(A7, w8) + b8
    A8 = np.tanh(Z8)
    Z9 = np.dot(A8, w9) + b9
    A9 = np.tanh(Z9)
    out_layer = np.dot(A9, wout)
    fit=np.sqrt(np.mean((out_layer-Y)**2))
    return fit
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation].reshape((X.shape[0],1))
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    # if m % mini_batch_size != 0:
    #     mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
    #     mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,:]
    #     mini_batch = (mini_batch_X, mini_batch_Y)
    #     mini_batches.append(mini_batch)

    return mini_batches
pop=ini(n_pop,n_para)
for i in range(n_pop):
    fit[i]=fitness(pop[i,:],X_train[0:100,],Y_train[0:100])
print(fit.min())
count_update=0
def sor(A,B):
    index_A=np.argsort(A,0)
    C=A[index_A,:].reshape((n_pop,1))
    D=B[index_A,:].reshape((n_pop,n_para))
    return C, D

minibatches = random_mini_batches(X_train, Y_train, batch_size, 0)

for iter in range(num_iteration):
    aver_fit = 0
    for minibatch in minibatches:
        newfit = copy.deepcopy(fit)
        newpop = copy.deepcopy(pop)
        (minibatch_X, minibatch_Y) = minibatch

        for i in range( 0, n_pop):
            if i<n_elite:
                fit[i] = fitness(pop[i, :], minibatch_X, minibatch_Y)
            else:
                value_rand = np.random.uniform(0, 1)
                index_r=int(value_rand*n_pop)
                pop[i, :] = newpop[index_r, :] + np.random.normal(model_mean, model_sd, n_para)
                fit[i] = fitness(pop[i, :], minibatch_X, minibatch_Y)

        fit, pop = sor(fit, pop)
        print(iter, fit[0,0])
        axisx1 = np.append(axisx1, count_update)
        axisy1 = np.append(axisy1, fit[0,0])
        line.set_xdata(axisx1)
        line.set_ydata(axisy1)
        ax.relim()  # renew the data limits
        ax.autoscale_view(True, True, True)  # rescale plot view
        plt.draw()  # plot new figure# #
        plt.pause(1)
        aver_fit += fit[0, 0] / (num_minibatches)
        count_update = count_update + 1

    axisx2 = np.append(axisx2, (iter+1)*num_minibatches-1)
    axisy2 = np.append(axisy2, aver_fit)
    line2.set_xdata(axisx2)
    line2.set_ydata(axisy2)
    ax.relim()  # renew the data limits
    ax.autoscale_view(True, True, True)  # rescale plot view
    plt.draw()  # plot new figure# #
    plt.pause(1)
    print('%ith iteration: average fitness: %f' % (iter, aver_fit))
plt.savefig('GANN' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')





