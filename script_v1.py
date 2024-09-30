import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    n = X.shape[0]
    d = X.shape[1]
    k = np.unique(y)
    u = np.empty((d, k.shape[0]))
    covmat = np.empty((d, d))
    index = 0
    yarray = np.empty((n))
    for i in range(0, n):
        yarray[i] = y[i][0]
    for i in k:
        u[:, index] = np.mean(X[yarray == i], axis=0)
        index += 1
    covmat = np.cov(X, rowvar=False)
    means = u

    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    means, c = ldaLearn(X, y)
    n = X.shape[0]
    d = X.shape[1]
    k = np.unique(y)
    covmats = np.empty((k.shape[0], d, d))
    index = 0
    yarray = np.empty((n))
    for i in range(0, n):
        yarray[i] = y[i][0]
    for i in k:
        covmats[index] = np.cov(X[yarray == i], rowvar=False)
        index += 1
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    n = Xtest.shape[0]
    d = Xtest.shape[1]
    k = means.shape[1]
    yhat = np.empty((n,1))
    accL = np.empty((n,1))
    covinv = np.linalg.inv(covmat)
    for i in range(0,n):
        max = 0
        index = 0
        for j in range(0,k):
            meank = np.empty((d))
            for a in range(0,d):
                meank[a] = means[a][j]
            #temp = Xtest[i] * np.linalg.inv(covmat)*means[:,j] - means[:,j].T*np.linalg.inv(covmat)*means[:,j]/2
            temp1 = np.zeros((d))
            temp2 = 0;
            temp3 = np.zeros((d))
            temp4 = 0;
            for aa in range(0,d):
                for a2 in range(0, d):
                    temp1[aa]+=Xtest[i][a2]*covinv[aa][a2]
            for aa in range(0, d):
                temp2 += temp1[aa]*meank[aa]  #x.T*Σ−1*μk
            for aa in range(0,d):
                for a2 in range(0, d):
                    temp3[aa]+=meank[a2]*covinv[aa][a2]
            for aa in range(0, d):
                temp4 += temp3[aa] * meank[aa] # μk.T*Σ−1*μk

            temp = temp2-temp4/2
            if max<temp:
                max = temp
                index = j
        yhat[i] = index+1
        if yhat[i] == ytest[i]:
            accL[i]=1
        else:
            accL[i]=0
        acc = np.mean(accL)
        ypred = yhat
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    n = Xtest.shape[0]
    d = Xtest.shape[1]
    k = means.shape[1]
    yhat = np.empty((n, 1))
    accL = np.empty((n, 1))
    for i in range(0, n):
        max = -99999999999
        index = 0
        for j in range(0, k):
            meank = np.empty((d))
            covk = covmats[j]
            covkinv = np.linalg.pinv(covk)
            for a in range(0, d):
                meank[a] = means[a][j]
            # temp = -ln(np.linalg.det（covk）)/2 - (Xtest[i] - meank).T*covkinv*(Xtest[i] - meank)/2
            temp1 = np.log(np.linalg.det(covkinv))
            temp2 = np.zeros((d))
            temp3 = np.zeros((d))
            temp4 = 0
            for a in range(0, d):
                temp2[a] = Xtest[i][a] - meank[a]
            for a in range(0, d):
                for aa in range(0, d):
                    temp3[a] += temp2[aa] * covkinv[a][aa]
            for a in range(0, d):
                temp4 += temp3[a] * temp2[a]
            temp = -temp1 / 2 - temp4 / 2
            if max < temp:
                max = temp
                index = j
        yhat[i] = index + 1
        if yhat[i] == ytest[i]:
            accL[i] = 1
        else:
            accL[i] = 0
        acc = np.mean(accL)
        ypred = yhat

    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1
    #w = (X.t*X)−1 *X.t*y
    w = np.linalg.inv(X.T@X)@X.T@y
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    #w = (X.t*X+λI)−1*X.t*y
    w = np.linalg.inv(X.T@X+lambd*np.eye(X.shape[1]))@ X.T@y
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    #mse = (y-y^)^2/n
    mse = np.sum((ytest-Xtest@w)**2)/ytest.shape[0]
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    d = w.shape[0]
    diff = X@w-y
    normSquare = 0
    for i in range(0,diff.shape[0]):
        normSquare+=diff[i]**2
    normSquarew=0
    for i in range(0,d):
        normSquarew += w[i]**2
    error = normSquare/2+normSquarew*lambd/2
    #error_grad = X.T@X@w - X.T@y + lambd*w
    temp1 = X.T@X
    temp2 = np.zeros((d,1))
    for i in range(0,d):
        for j in range(0,d):
            temp2[i]+=temp1[i][j]*w[j]
    temp3 = X.T@y
    temp4 =np.zeros((d,1))
    for i in range(0, d):
        temp4[i] = lambd*w[i]
    error_grad = temp2 - temp3 +temp4
    # IMPLEMENT THIS METHOD
    error_grad = error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    n = x.shape[0]  # Number of samples
    Xp = np.zeros((n, p + 1))
    for i in range(0, p + 1):
        Xp[:, i] = x ** i
    # IMPLEMENT THIS METHOD
    return Xp

# Main script
if __name__ == "__main__":
    # Problem 1
    # load the sample data                                                                 
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

    # LDA
    means,covmat = ldaLearn(X,y)
    ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
    print('LDA Accuracy = '+str(ldaacc))
    # QDA
    means,covmats = qdaLearn(X,y)
    qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
    print('QDA Accuracy = '+str(qdaacc))

    # plotting boundaries
    x1 = np.linspace(-5,20,100)
    x2 = np.linspace(-5,20,100)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.ravel()
    xx[:,1] = xx2.ravel()

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)

    zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('LDA')

    plt.subplot(1, 2, 2)

    zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('QDA')

    plt.show()
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

    w_i = learnOLERegression(X_i,y)
    mle_i = testOLERegression(w_i,Xtest_i,ytest)

    print('MSE without intercept '+str(mle))
    print('MSE with intercept '+str(mle_i))

    # Problem 3
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses3_train = np.zeros((k,1))
    mses3 = np.zeros((k,1))
    for lambd in lambdas:
        w_l = learnRidgeRegression(X_i,y,lambd)
        mses3_train[i] = testOLERegression(w_l,X_i,y)
        mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')

    plt.show()
    '''
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
'''

    # Problem 5
    pmax = 7
    lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
    mses5_train = np.zeros((pmax,2))
    mses5 = np.zeros((pmax,2))
    for p in range(pmax):
        Xd = mapNonLinear(X[:,2],p)
        Xdtest = mapNonLinear(Xtest[:,2],p)
        w_d1 = learnRidgeRegression(Xd,y,0)
        mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
        mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
        w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
        mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
        mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(range(pmax),mses5_train)
    plt.title('MSE for Train Data')
    plt.legend(('No Regularization','Regularization'))
    plt.subplot(1, 2, 2)
    plt.plot(range(pmax),mses5)
    plt.title('MSE for Test Data')
    plt.legend(('No Regularization','Regularization'))
    plt.show()
