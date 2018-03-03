# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

# Revision History
# 02/24/18    Tim Liu    modified so user/movies are zero indexed
# 02/24/18    Tim Liu    normalizing training and test set by size
# 02/24/18    Tim Liu    modified data path

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
		
def main():
    Y_train = np.loadtxt('./data/train.txt').astype(int)
    Y_test = np.loadtxt('./data/test.txt')	.astype(int)
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    Ks = [10,20,30,50,100]
	
    regs = [10**-4, 10**-3, 10**-2, 10**-1, 1]
    eta = 0.03 # learning rate
    E_ins = []
    E_outs = []
    
    #convert to zero indexing
    for i in range(len(Y_train)):
        Y_train[i][0] -= 1
        Y_train[i][1] -= 1
        
    for i in range(len(Y_test)):
        Y_test[i][0] -= 1
        Y_test[i][1] -= 1
	
    # Use to compute Ein and Eout
    for reg in regs:
        E_ins_for_lambda = []
        E_outs_for_lambda = []
        
        for k in Ks:
            print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, k, eta, reg))
            U,V, e_in = train_model(M, N, k, eta, reg, Y_train)
            E_ins_for_lambda.append(e_in)
            eout = get_err(U, V, Y_test)
            E_outs_for_lambda.append(eout)
            
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)
        
    #normalize by size of the training and test sets    
    E_ins = np.array(E_ins)/len(Y_train)
    E_outs = np.array(E_outs)/len(Y_test)
	

    # Plot values of E_in across k for each value of lambda
    for i in range(len(regs)):
        plt.plot(Ks, E_ins[i], label='$E_{in}, \lambda=$'+str(regs[i]))
    plt.title('$E_{in}$ vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('2e_ein.png')	
    plt.clf()

    # Plot values of E_out across k for each value of lambda
    for i in range(len(regs)):
        plt.plot(Ks, E_outs[i], label='$E_{out}, \lambda=$'+str(regs[i]))
    plt.title('$E_{out}$ vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()	
    plt.savefig('2e_eout.png')		

if __name__ == "__main__":
    main()
