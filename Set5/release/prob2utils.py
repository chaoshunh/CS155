# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

# Revision History
# 02/22/18    Tim Liu

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    grad = reg * Ui - Vj * (Yij - np.dot(Ui, Vj))
    return eta * grad

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    grad = reg * Vj - Ui * (Yij - np.dot(Vj, Ui))
    return eta * grad

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    total_err = 0
    for p in range(len(Y)):
        i = Y[p][0]    #extract coordinates
        j = Y[p][1]
        
        predict = np.dot(U[i], V[j])           #come up with predicted value
        total_err += (Y[p][2] - predict) *  (Y[p][2] - predict) #prediction error
    
    #add in regularization error
    total_err += reg * np.linalg.norm(U)     #frobenius norm of matrix U
    total_err += reg * np.linalg.norm(V)     #frobenius norm of matrix V
    
    return total_err/2


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    
    U = np.random.uniform(low = -0.5, high = 0.5, size = (M, K))
    V = np.random.uniform(low = -0.5, high = 0.5, size = (N, K))
    
    start_err = get_err(U, V, Y, reg)
    
    
    order = np.arange(len(Y))     #create array for order to perform SGD
    
    for e in range(max_epochs):
        np.random.shuffle(order)
        for point in order:              #now iterate through points
            Yij = Y[point]               #look up point to perform SGD on
            i = Yij[0]
            j = Yij[1]
            
            Ui = U[i]
            Vj = V[j]
                      
            U[i] -= grad_U(U[i], Yij[2], Vj, reg, eta) #update U
            V[j] -= grad_V(V[j], Yij[2], Ui, reg, eta) #update V
                
        if e == 0:
            #calculate first reduction in error
            new_err = get_err(U, V, Y, reg)    #calculate error
            first_drop = start_err - new_err   #calculate first reduction in err
        else:
            old_err = new_err                  #swap and track past state
            new_err = get_err(U, V, Y, reg)
            if (old_err - new_err)/first_drop < eps:
                break
        if e % 10 == 0:
            print("Current Epoch: ", e, "K: ", K)
        
            
    return U, V, new_err
