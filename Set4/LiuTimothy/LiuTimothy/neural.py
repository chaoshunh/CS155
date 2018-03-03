# file defining class neural_net object
# neural_net class is interpreted by other functions to build a model

# Revision History
# 01/30/18    Tim Liu    wrote file
# 01/30/18    Tim Liu    retyped file to fix indentation problems

class neural_net():
    def __init__(self, units, layers, activation = 'relu', dropout = 0.0, \
                 b_size = 32, epochs = 10, optimizer = 'Adam'):
        self.units = units
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.b_size = b_size
        self.epochs = epochs
        self.test_error = 0
        self.optimizer = optimizer
        
    def get_specs(self):
        '''returns a list representing the specifications for a neural net
        Each element in list represents a lyaer and layer types are
        specified by element type:
        integers - dense layer with specified hidden units
        float - dropout with specified fraction weights set to zero
        string - activation'''
        
        net = []    #list representation of neural net
        units_per_layer = int(self.units/self.layers)  #units in a hidden layer
        
        for i in range(self.layers):
            net.append(units_per_layer)       #add dense layer
            net.append(self.activation)        #add activation
            net.append(self.dropout)          #add dropout
            
        net.append(10)         #add final layer(not hidden)
        net.append('softmax')  #establish a single prediction
        
        return net
        
    def __repr__(self):
        '''prints data specifying the neural net'''
        
        info = "\nTotal Units: "
        info += str(self.units)
        info += "\nHidden Layers: "
        info += str(self.layers)
        info += "\nActivation:"
        info += self.activation
        info += "\ndropout: "
        info += str(self.dropout)
        info += "\nbatch size: "
        info += str(self.b_size)
        info += "\nepochs:"
        info += str(self.epochs)
        info += "\nOptimizer: "
        info += self.optimizer
        info += "\nTest Error: "
        info += str(self.test_error)
        
        return info
        