########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import math
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(M+1):
            #fill in a row (additional step in sequence)
            for j in range(self.L):
                #evaluate for each state
                if i == 0:
                    pass
                elif i == 1:
                    #first row is transition probability times emission
                    probs[i][j] = self.A_start[j] * self.O[j][x[i-1]]
                    seqs[i][j] = str(j)
                    
                else:
                    #array of transition * emission probabilities * last prob
                    tran = [self.A[past][j] * self.O[j][x[i-1]] *\
                            probs[i-1][past] for past in range(self.L)]

                    probs[i][j] = max(tran)     #maximum likelihood
                    #back pointer to previous state
                    back_ptr = tran.index(probs[i][j])
                    #append to sequence
                    seqs[i][j] = seqs[i-1][back_ptr] + str(j)
                    
        max_prob = max(probs[M])
        prob_index = probs[M].index(max_prob)
        max_seq = seqs[M][prob_index]
                
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M+1)]
        #print("M: ", M)
        
        #is z the observation?
        for i in range(M):
            for z in range(self.L):              

                if i == 0:
                    alphas[1][z] = self.O[z][x[0]]*self.A_start[z]
                                      
                else:
                    products = [alphas[i][j] * self.A[j][z]\
                                for j in range(self.L)]
                    alphas[i+1][z] = self.O[z][x[i]] * sum(products)
                    
            if (normalize):
                norm_sum = sum(alphas[i+1])
                if norm_sum != 0:
                    alphas[i+1] = [x/norm_sum for x in alphas[i+1]]
        
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M+1)]
        
        for i in reversed(range(M+1)):
            for z in range(self.L):
                if i == M:
                    #initialize last row
                    betas[i][z] = 1
                    
                else:
                    products = [betas[i+1][j] * self.A[z][j] * \
                                self.O[j][x[i]] for j in range(self.L)]
                    betas[i][z] = sum(products)
            if normalize:
                norm_sum = sum(betas[i])
                if norm_sum != 0:
                    betas[i] = [x/norm_sum for x in betas[i]]            

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        
        #create empty matrix
        self.A = [[0 for x in range(self.L)] for y in range(self.L)]
        
        y_b = [0 for x in range(self.L)]     #count the number of each state
        for seq in Y:                 #iterate through each sequence
            for i in range(len(seq)-1):
                b = seq[i]         #current state
                a = seq[i+1]       #state transitioned to
                self.A[b][a] += 1  #increment transition from b to a
                y_b[b] += 1        #count number of b
                
                
        for b in range(self.L):
            for a in range(self.L):
                self.A[b][a] /= y_b[b]    #divide by occurences of initial state
                
        # Calculate each element of O using the M-step formulas.
        
        self.O = [[0 for x in range(self.D)] for y in range(self.L)]
        
        y_z = [0 for x in range(self.L)]   #count number of each state
        
        for i in range(len(Y)):
            seq = Y[i]
            for j in range(len(seq)):
                #i, j correspond to a state/output in X and Y matrix
                x = X[i][j]
                y = Y[i][j]
                self.O[y][x] += 1   #increment state y emitting x
                y_z[y] += 1         #increment occurrence of state
                
        for state in range(self.L):
            for emit in range(self.D):
                self.O[state][emit] /= y_z[state]   #divide by number of states
                
        return


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        print("Total Iterations: ", N_iters)
        #loop through all iterations
        for n in range(N_iters):
            #define indices
            N = len(X)                #number of training examples N
            
            #print("A: ", self.A)
            #print("O: ", self.O)   
            #print(self.D)

            print("Iteration: ", n)
            
            # *****E step find marginals ***** #
            #length of longest training sequence
            max_M = max([len(x) for x in X])
            min_M = min([len(x) for x in X])
            #indexing training example, j, a
            P_ya = np.zeros((N, max_M+1, self.L))          #marginal tensor P(yj = a, x)
            #indexing training xample, j, a, b
            P_yab = np.zeros((N, max_M+1, self.L, self.L)) #marginal tensor P(yj = a, yj+1 = b, x)
            
            #iterate through all training examples
            for i in range(N):
                x = X[i]    #specific training example
                alphas = self.forward(x, normalize = True)
                betas = self.backward(x, normalize = True)

                #iterate through all jth element in training example
                M = len(X[i])
                for j in range(M):
                    denom_1 = 0    #denominator of first marginal
                    denom_2 = 0    #denominator of second marginal
                    if j == 0:     #indexing of alpha/beta starts at 1
                        continue
                    
                    #iterate through a (initial state)
                    for a in range(self.L):
                        #set to numerator of marginal
                        num_1 = alphas[j][a] * betas[j][a]
                        P_ya[i][j][a] = num_1
                        #build denominator of P_ya
                        denom_1 += num_1
                        
                        #iterate through b values (state transitioned to)
                        for b in range(self.L):
                            #find the numerator
                            num_2 = alphas[j][a] * betas[j+1][b] *\
                                self.O[b][x[j]] * self.A[a][b]
                            #num_2 = self.O[b][x[j]] * self.A[a][b]                            
                            P_yab[i, j, a, b] = num_2
                            #update denom_2
                            denom_2 += num_2
                    
                    #divide by the denominator to get final marginals                     
                    if denom_1 != 0:
                        P_ya[i][j] /= denom_1
                    if denom_2 != 0:
                        P_yab[i][j] /= denom_2
                                
            #******M step update parameters********#            
            #update transition matrix
            for a in range(self.L):
                denom = 0   #initialize denominator for new a
                
                #iterate through state transitioning to               
                for b in range(self.L):
                    P_ab = 0    #transition probability from a to b
                    
                    #for each training example
                    for i in range(N):
                        
                        M = len(X[i])  #length of training example                      
                        #for each element in single training example
                        for j in range(M):
                            if j == 0:
                                continue
                            P_ab += P_yab[i, j, a, b] #sum up numerator
                            if b == 0:
                                denom += P_ya[i, j, a] #only sum denom once
                    if denom == 0:
                        self.A[a][b] = 0
                    else:
                        self.A[a][b] = P_ab/denom
            
            #update emission matrix (separate loops for readability)
            #loop over all states
            for a in range(self.L):
                den = 0   #initialize value for denominator for new a
                                
                #loop over all emissions
                for w in range(self.D):
                    num = 0   #initialize value for numerator
                    
                    #loop over all input sequences N
                    for i in range(N):
                        
                        M = len(X[i])  #length of training example                                              
                        #loop over all letters in single training example
                        for j in range(M):
                            #skip; indexing starts at j = 1
                            if j == 0:
                                continue
                            #string x is zero indexed not 1 indexed
                            if X[i][j-1] == w:
                                num += P_ya[i][j][a]   #find numerator
                            if w == 0:
                                den += P_ya[i][j][a]   #only sum denom once
                    #write to emission matrix

                    if den == 0:
                        self.O[a][w] = 0
                    else:
                        self.O[a][w] = num/den



    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        for m in range(M):
            if m == 0:
                #randomly intitialize first state from uniform distribution
                states.append(random.choices(range(self.L))[0])
            else:
                states.append(random.choices(range(self.L), \
                                             weights = self.A[states[-1]])[0])
            emission.append(random.choices(range(self.D), \
                                           weights = self.O[states[-1]])[0])
                

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
