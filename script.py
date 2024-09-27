import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys



# PROBLEM 2 #
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    w=np.linalg.inv(X.T @ X)@X.T@y
    
	
    # IMPLEMENT THIS METHOD                                                   
    return w

# PROBLEM 3 #
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

# PROBLEM 4 #
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    error_grad = error_grad.flatten()                                
    return error, error_grad


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
    mle_train=testOLERegression(w,X,y) # OWN_CODE for training, without intercept 

    w_i = learnOLERegression(X_i,y)
    mle_i = testOLERegression(w_i,Xtest_i,ytest)
    mle_train_i=testOLERegression(w_i,X_i,y) # OWN_CODE for training, with intercept

    print('MSE for training without intercept '+str(mle_train)) # OWN_CODE
    print('MSE for training with intercept '+str(mle_train_i)) #OWN CODE

    print('MSE without intercept '+str(mle))
    print('MSE with intercept '+str(mle_i))

    # Problem 3
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses3_train = np.zeros((k,1))
    mses3 = np.zeros((k,1))
      
    ridge_norms=[] # OWN_CODE ridge_norms for each lambda
    for lambd in lambdas:
        w_l = learnRidgeRegression(X_i,y,lambd)
        mses3_train[i] = testOLERegression(w_l,X_i,y)
        mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        norm=np.linalg.norm(w_l) # OWN_CODE calculate norm for each lambda
        ridge_norms.append(norm) # OWN_CODE

        i = i + 1

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')

    plt.show()

    # OWN_CODE BEGINS #
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
     # OWN_CODE ENDS#


    # Problem 4
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses4_train = np.zeros((k,1))
    mses4 = np.zeros((k,1))
    opts = {'maxiter' : 20}    # Preferred value.                                                
    w_init = np.ones((X_i.shape[1],1)).flatten()
    for lambd in lambdas:
        args = (X_i, y, lambd)
        w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
        w_l = np.transpose(np.array(w_l.x))
        w_l = np.reshape(w_l,[len(w_l),1])
        mses4_train[i] = testOLERegression(w_l,X_i,y)
        mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses4_train)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])

    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses4)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])
    plt.show()



