import numpy as np
import matplotlib.pyplot as plt
from activation_functions import * 

def nn_model(X, Y, layer_dims, activations, learning_rate, num_iterations, print_cost=False):
    """
    Implements a L-layer neural network.

    Arguments:
    X -- data, numpy array of shape (number of input units, number of examples)
    Y -- true "label" matrix, of shape (number of output units, number of examples)
    layers_dims -- list containing the input size and each layer size
    activations -- a list specifying activation function used in each layer
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # keep track of costs
    L = len(activations)  # number of layers

    # Parameters initialization
    parameters = _initialize_parameters(layer_dims)

    # Gradient descent loop
    for i in range(num_iterations):

        # Forward propagation
        cache = _forward_propagation(X, parameters, activations)
        AL = cache['A' + str(L)]

        # Compute cost
        cost = _compute_cost(AL, Y)

        # Backward propagation
        grads = _backward_propagation(parameters, cache, activations, Y)

        # Update parameters
        parameters = _update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters, activations):
    """
    This function is used to predict the results of a L-layer neural network.

    Arguments:
    X -- data, numpy array of shape (number of input units, number of examples)
    Y -- true "label" matrix, numpy array of shape (number of output units, number of examples)
    parameters -- a dictionary containing parameters W and b for each layer
    activations -- a list specifying activation function used in each layer

    Returns:
    predictions -- predictions for the given dataset X
    """

    m = X.shape[1]
    L = len(parameters) // 2 # number of layers in the neural network
    cache = _forward_propagation(X, parameters, activations)

    predictions = cache['A' + str(L)] > 0.5

    print('Accuracy: ' + str(np.sum((p == Y)/m)))
        
    return predictions


def _initialize_parameters(layer_dims):

    # number of layers in the network
    L = len(layer_dims) 
    parameters = {}
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def _forward_propagation(X, parameters, activations):

    cache = {}
    A = X
    cache['A0'] = X
    L = len(parameters) // 2

    for l in range(1, L+1):
        
        A_prev = A

        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        Z = np.dot(W, A_prev) + b

        if activations[l-1] == 'sigmoid':
            A = sigmoid(Z)
        elif activations[l-1] == 'relu':
            A = relu(Z)
        elif activations[l-1] == 'l_relu':
            A = l_relu(Z)
        elif activations[l-1] == 'tanh':
            A = tanh(Z)
        
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A

    return cache


def _compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))

    return np.squeeze(cost)

        
def _backward_propagation(parameters, cache, activations, Y):

    grads = {}
    L = len(parameters) // 2  # the number of layers
    m = Y.shape[1]  # number of training examples
    AL = cache['A' + str(L)]  # activations of the output layer
    
    # Derivative of the cost function with respect to AL
    grads['dA' + str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    for l in reversed(range(L)):
    
        # lth layer gradients
        W = parameters['W' + str(l+1)]
        b = parameters['b' + str(l+1)]

        # Derivative of the cost function with respect to Z 
        # (Z is an input to the activation function of a given layer)
        if activations[l] == 'sigmoid':
            dZ = sigmoid_backward(grads['dA' + str(l + 1)], cache['Z' + str(l + 1)])
        elif activations[l] == 'relu':
            dZ = relu_backward(grads['dA' + str(l + 1)], cache['Z' + str(l + 1)])
        elif activations[l] == 'l_relu':
            dZ = l_relu_backward(grads['dA' + str(l + 1)], cache['Z' + str(l + 1)])
        elif activations[l] == 'tanh':
            dZ = tanh_backward(grads['dA' + str(l + 1)], cache['Z' + str(l + 1)])

        grads["dA" + str(l)] = np.dot(W.T, dZ)
        grads["dW" + str(l + 1)] = (1/m) * np.dot(dZ, cache['A' + str(l)].T)
        grads["db" + str(l + 1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

    return grads


def _update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters