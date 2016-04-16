import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import pickle
import time

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);  # epsilon is 
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid_z = 1.0/(1.0 + np.exp(-z)) 
    
    return sigmoid_z  #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    
    #Pick a reasonable size for validation data
    #Your code here
    #-----Train Data------
    temp_train_stack = np.zeros((0,784))  # matrix dimension 60000X784
    temp_train_labels_stack = np.zeros((0,1),dtype=np.int64) # 60000X1 matrix dimension
    for i in range(10): 
        mat_get_curr = mat.get('train'+str(i));
        temp_train_stack = np.concatenate((temp_train_stack,mat_get_curr))
        num_get_curr = mat_get_curr.shape[0]
        temp = np.zeros((num_get_curr,1))
        temp.fill(i)
        temp_train_labels_stack = np.concatenate((temp_train_labels_stack,temp))
        
    #temp_train_stack 60000X784
    #temp_train_labels_stack 60000X784
   
    # Feature Extraction
    res_common_flag = np.all(temp_train_stack == temp_train_stack[0,:],axis=0)  # find all features which have equal values
    
    delete_cols = np.where(res_common_flag == True)    # column numbers whose values are same are stored into delete_cols
    
    temp_train_stack = np.delete(temp_train_stack,delete_cols[0],axis=1)   # delete columns which are redundant
    
    train_data = np.zeros((50000,temp_train_stack.shape[1])) #50000Xno. of features
    train_label = np.zeros((50000,1),dtype=np.int64) # 50000X1
   
    s = random.sample(range(temp_train_stack.shape[0]),60000)
    for i in range(50000):
        train_data[i,:] = temp_train_stack[s[i],:]
        train_label[i,0] = temp_train_labels_stack[s[i],0]
        
    train_data = train_data.astype(np.float64) # Float Conversion
    train_data = train_data / 255.0 # Normalization
    
   
    
    #-----Validation Data-----
    validation_data = np.zeros((10000,temp_train_stack.shape[1])) # 10000Xno of features
    validation_label = np.zeros((10000,1),dtype=np.int64) # 10000X1      
    for j in range(50000,60000):
        validation_data[j-50000,:] = temp_train_stack[s[j],:]
        validation_label[j-50000,0] = temp_train_labels_stack[s[j],0]  
          
    validation_data = validation_data.astype(np.float64)   # Float Conversion
    validation_data = validation_data / 255.0 # Normalization
       
    
    #------Test Data------    
    test_data = np.zeros((0,784)) 
    test_label = np.zeros((0,1),dtype=np.int64)
    for i in range(10):
        mat_get_curr = mat.get('test'+str(i));
        test_data = np.concatenate((test_data,mat_get_curr))
        num_get_curr = mat_get_curr.shape[0]
        temp = np.zeros((num_get_curr,1))
        temp.fill(i)
        test_label = np.concatenate((test_label,temp))  
    #test_data 10000X784
    #test_label 10000X1
    
    test_data = test_data.astype(np.float64) #Float Conversion
    test_data = test_data / 255.0 # Normalization
    
    test_data = np.delete(test_data,delete_cols[0],axis=1)
                                            
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))      # 4 X 718
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))      # 10 X 5
    obj_val = 0.0  
    #Your code here
    #Adding the bias input node
    
    bias_node = np.ones((training_data.shape[0],1), dtype=np.float64)
    training_data = np.append(training_data,bias_node,axis=1)
    grad_w1 = np.zeros(w1.shape,dtype=np.float64)
    grad_w2 = np.zeros(w2.shape,dtype=np.float64)
    neg_log_likelihood_error = 0.0
    

    #Feedforward Propagation
    #Input to Hidden Layer
            
    z_hidden_output =  np.dot(training_data, w1.transpose())                   
            
    z_hidden_output = sigmoid(z_hidden_output)
        
    hidden_bias = np.ones((training_data.shape[0],1), dtype=np.float64)
    z_hidden_output = np.append(z_hidden_output,hidden_bias,axis=1)  # hidden layer bias node
        
    #Hidden to Output layer
    o_output = np.dot(z_hidden_output, w2.transpose());
                    
    o_output = sigmoid(o_output)    # dimension is (50000 X 10)
        
    #Calculating the error at the output layer
        
    y = np.copy(o_output)
    y.fill(0.0)
    for n in range(training_data.shape[0]):    
        y[n, training_label[n][0]] = 1.0    #could be 50000 X 10
        
    # y is (50000X10) , o_output is (50000X10), 
    neg_log_likelihood_error = np.sum((y * np.log(o_output)) + ((1.0-y)*np.log(1.0-o_output)))
            
    # Partial derivative of the objective  function wrt weight from input to hidden layer
        
    grad_w1_temp = np.dot((o_output-y),w2[:,0:n_hidden])   #o_output - y is (50000X10) , w2 is (10 X 5) , final dimension will be 50000 X 5
        
    grad_w1 += np.dot(((1-z_hidden_output[:,0:n_hidden]) * z_hidden_output[:,0:n_hidden] * grad_w1_temp).transpose() , training_data)
   
    # Partial derivative of the objective  function wrt weight from hidden to output layer
    grad_w2 = np.dot(( o_output - y ).transpose(), z_hidden_output)
                
    #Calculating final error function 
    obj_val = (-1.0/training_data.shape[0]) * neg_log_likelihood_error
    
    #Regularization term
    
    reg_term =   np.sum(np.square(w1)) + np.sum(np.square(w2))

    #Final error function with regularization
    obj_val = obj_val + ((lambdaval/(2*training_data.shape[0]))*reg_term)   

                          
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    grad_w1 = np.add(grad_w1 , (lambdaval*w1))/training_data.shape[0]
    grad_w2 = np.add(grad_w2 , (lambdaval*w2))/training_data.shape[0]
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.zeros((data.shape[0],1))
    bias_node = np.ones((data.shape[0],1), dtype=np.float64)
    data = np.append(data,bias_node,axis=1)
    for n in range(data.shape[0]): 
        #Input to Hidden Layer           
        z_hidden_output =  np.dot(w1 , data[n][:].transpose())                   
                
        z_hidden_output = sigmoid(z_hidden_output)
            
        z_hidden_output = np.append(z_hidden_output,[1],axis=0)  # hidden layer bias node
            
        #Hidden to Output layer
        o_output = np.dot(w2 , z_hidden_output);
                        
        o_output = sigmoid(o_output)
                
        o_output = sigmoid(o_output)
        labels[n][0] = np.argmax(o_output)
                      
    return labels
    




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4;


args = (n_input, n_hidden, n_class, train_data[0:50000], train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 100}    # Preferred value.

start_time = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
print("---Training Time : %s seconds ---" % (time.time() - start_time))

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
pickle.dump((n_input,n_hidden,w1,w2,lambdaval) , open( "params_hidden20.pickle", "wb" ))

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
