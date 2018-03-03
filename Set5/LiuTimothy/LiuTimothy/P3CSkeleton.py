# Solution set for CS 155 Set 6, 2017
# Authors: Suraj Nair, Sid Murching

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from P3CHelpers import *
import sys

def get_word_repr(word_to_index, word):
    """
    Returns one-hot-encoded feature representation of the specified word given
    a dictionary mapping words to their one-hot-encoded index.

    Arguments:
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        word:          String containing word whose feature representation we wish to compute.

    Returns:
        feature_representation:     Feature representation of the passed-in word.
    """
    unique_words = word_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_words))
    feature_representation[word_to_index[word]] = 1
    return feature_representation    

def generate_traindata(word_list, word_to_index, window_size=4):
    """
    Generates training data for Skipgram model.

    Arguments:
        word_list:     Sequential list of words (strings).
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        window_size:   Size of Skipgram window.
                       (use the default value when running your code).

    Returns:
        (trainX, trainY):     A pair of matrices (trainX, trainY) containing training 
                              points (one-hot-encoded vectors representing individual words) and 
                              their corresponding labels (also one-hot-encoded vectors representing words).

                              For each index i, trainX[i] should correspond to a word in
                              <word_list>, and trainY[i] should correspond to one of the words within
                              a window of size <window_size> of trainX[i].
    """
    trainX = []
    trainY = []
    
    n_words = len(word_list)
    
    for i in range(n_words):   #iterate through the list of words
        min_j = max(0, i - window_size)                 #start of window
        max_j = min(n_words, i + window_size + 1)  #end of window
        
        for j in range(min_j, max_j):
            if word_list[i] != word_list[j]:         #must be different words
                #zero array for each point
                x = np.zeros(len(word_to_index))
                y = np.zeros(len(word_to_index))
                #set up one hot bit
                x[word_to_index[word_list[i]]] = 1
                y[word_to_index[word_list[j]]] = 1
                #add to the lists
                trainX.append(x)
                trainY.append(y)                
                          
    return (np.array(trainX), np.array(trainY))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python P3C.py <path_to_textfile> <num_latent_factors>")
        sys.exit(1)

    filename = sys.argv[1]
    num_latent_factors = int(sys.argv[2])
    # Load in a list of words from the specified file; remove non-alphanumeric characters
    # and make all chars lowercase.
    sample_text = load_word_list(filename)

    # Create dictionary mapping unique words to their one-hot-encoded index
    word_to_index = generate_onehot_dict(sample_text)
    # Create training data using default window size
    print("Building training set....")
    trainX, trainY = generate_traindata(sample_text, word_to_index)
    
    
    # vocab_size = number of unique words in our text file. Will be useful when adding layers
    # to your neural network
    print("Building neural net...")
    vocab_size = len(word_to_index)
    model = Sequential()
    model.add(Dense(10, input_dim = vocab_size))

    model.add(Dense(vocab_size, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer='Adam',\
                  metrics = ['accuracy'])  
    
    print("Fitting model...")
    
    fit = model.fit(trainX, trainY, epochs = 10)
    
    weights = model.layers[0].get_weights()[0]
    out_weights = model.layers[1].get_weights()[0]
    
    for i in range(len(model.layers)):
        print(np.array(model.layers[i].get_weights()).shape)
    
    
    #print(weights)
    #print(out_weights)
    
    print("Hidden Layer: ", weights.shape)
    print("Output Layer: ", out_weights.shape)

    # Find and print most similar pairs
    similar_pairs = most_similar_pairs(weights, word_to_index)
    for pair in similar_pairs[:30]:
        print(pair, ' \\\\')
