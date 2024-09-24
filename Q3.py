import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

# this is from Problem 2
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    w=np.linalg.inv(X.T @ X)@X.T@y
    
	
    # IMPLEMENT THIS METHOD                                                   
    return w

# Problem 3 begins
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    w=np.linalg.inv(lambd*np.identity(X.shape[1])+X.T@X)@X.T@y                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    y_predict=Xtest@w
    mse=np.mean((ytest-y_predict)**2)
    
    # IMPLEMENT THIS METHOD

    return mse

# Main script
if __name__ == "__main__":
    
    # Problem 2
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

    # add intercept
    X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

    w = learnOLERegression(X,y)
    mle = testOLERegression(w,Xtest,ytest)
    mle_train=testOLERegression(w,X,y) # for training, without intercept

    w_i = learnOLERegression(X_i,y)
    mle_i = testOLERegression(w_i,Xtest_i,ytest)
    mle_train_i=testOLERegression(w_i,X_i,y) # for training, with intercept

    print('MSE for training without intercept '+str(mle_train))
    print('MSE for training with intercept '+str(mle_train_i))

    print('MSE for testing without intercept '+str(mle))
    print('MSE for testing with intercept '+str(mle_i))

    # Problem 3
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses3_train = np.zeros((k,1))
    mses3 = np.zeros((k,1))

    # ridge_norms for each lambda
    ridge_norms=[]
    for lambd in lambdas:
        w_l = learnRidgeRegression(X_i,y,lambd)
        mses3_train[i] = testOLERegression(w_l,X_i,y)
        mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        norm=np.linalg.norm(w_l) # calculate norm for each lambda
        ridge_norms.append(norm)
        i = i + 1

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')

    plt.show()

    # calculate the relative magnitude of weights learnt using OLE and ridge regression
    norm_OLE=np.linalg.norm(w)
    norm_OLE_bias=np.linalg.norm(w_i)
    print("L2 norm for weights learnt using OLE without bias : "+str(norm_OLE))
    print("L2 norm for weights learnt using OLE with bias : "+str(norm_OLE_bias))

    # draw plot for ridge_norms vs lambda
    fig1=plt.figure(figsize=[12,6])
    plt.plot(lambdas,ridge_norms)
    plt.xlabel("lambda")
    plt.ylabel("L2 Norm")
    plt.title('Norms for Weight Learnt Using Ridge')
    plt.show()

    # find the optimal lambda
    index_minMLE=np.argmin(mses3)
    optimal_lambda=lambdas[index_minMLE]
    print("the optimal lambda is: "+str(optimal_lambda))


    