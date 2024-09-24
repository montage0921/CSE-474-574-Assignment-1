import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    w=np.linalg.inv(X.T @ X)@X.T@y
    
	
    # IMPLEMENT THIS METHOD                                                   
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