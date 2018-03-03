# CS155 machine learning homework 4
# 
# Code for problem 2 training on MNIST dataset

# Revision History
#    01/30/18    Tim Liu    started file
#    01/30/18    Tim Liu    wrote spec_neural and build_neural

import imp                    #let us update without exiting Python
import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
#dataset we're using
from keras.datasets import mnist
# Visualizing an image
import matplotlib.pyplot as plt
from neural import *

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = keras.utils.np_utils.to_categorical(y_train, 10)             #change to one hot encoding
X_train = np.reshape(X_train, (len(X_train), len(X_train[0][0]) **2))  #input is 1D array
X_train = X_train/255                                                  #normalize the data


y_test = keras.utils.np_utils.to_categorical(y_test, 10)             #change to one hot encoding
X_test = np.reshape(X_test, (len(X_test), len(X_test[0][0]) **2))     #input is 1D array
X_test = X_test/255                                                  #normalize the data
NUM_DIM = len(X_train[0])

def show_image():
    '''show an example of an image'''
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.show()
    return


def spec_neural(m_class):
    '''specifies different neural networks to build and calls build_neural
    inputs: m_class - letter specifying which class of models to build; 
    each class corresponds to a specific problem'''
    
    model_c = []    #models for problem c; 100 hidden units no bound on hidden
    model_d = []	#models for problem c; 100 hidden units no bound on hidden
    model_e = [] 	#models for problem c; 100 hidden units no bound on hidden
    


    #fill model classes
    model_c.append(neural_net(100, 2))
    model_c.append(neural_net(100, 4))
    model_c.append(neural_net(100, 5))
    model_c.append(neural_net(100, 10))
    model_c.append(neural_net(100, 20))

    model_c.append(neural_net(100, 5, dropout = 0.01))
    model_c.append(neural_net(100, 5, dropout = 0.05))
    model_c.append(neural_net(100, 5, dropout = 0.1))
    model_c.append(neural_net(100, 5, dropout = 0.2))
    
    #fill model class d
    
    model_d.append(neural_net(200, 2))
    model_d.append(neural_net(200, 2, dropout = 0.01))    
    model_d.append(neural_net(200, 2, epochs = 20))    
    model_d.append(neural_net(200, 4))   
    
    #fill model class e
    

    model_e.append(neural_net(1000, 4, epochs = 30, dropout = 0.1)) 
    model_e.append(neural_net(1000, 4, epochs = 30, dropout = 0.05)) 
    model_e.append(neural_net(1000, 4, epochs = 30, dropout = 0.05, b_size = 1024)) 
    model_e.append(neural_net(1000, 4, epochs = 30, b_size = 1024)) 
    

    

    model_dic = {'c': model_c, 'd': model_d, 'e': model_e}
    
    for m in model_dic[m_class]:
        test_acc = build_neural(m)
        m.test_error = test_acc
        
    for m in model_dic[m_class]:
        print(m)

    return

def build_neural(neural_net):
    '''create reate neural network based on the given specifications
    and returns the results'''
    model = Sequential()
    nn_instr = neural_net.get_specs()      #declare a new model
    
    for layer in nn_instr:
        if type(layer) == int:            #integer specifies dense layer
            model.add(Dense(layer, input_dim = NUM_DIM))
        elif type(layer) == float:        #float specifies dropout
            model.add(Dropout(layer))
        elif type(layer) == str:          #string specifies activation
            model.add(Activation(layer))
        else:
            print("Unknown layer!")
    
    #extract neural network properties from object
    opt = neural_net.optimizer    #optimizer type
    n_epochs = neural_net.epochs  #number of epochs
    b_size = neural_net.b_size    #batch size of test
    
    #compile the model
    
    model.compile(loss = 'categorical_crossentropy', optimizer=opt,\
                  metrics = ['accuracy'])
    
    #fit the model
    fit = model.fit(X_train, y_train, batch_size=b_size, epochs=n_epochs)
    score = model.evaluate(X_test, y_test, verbose=0)
    
    return score[1]     #return test accuracy