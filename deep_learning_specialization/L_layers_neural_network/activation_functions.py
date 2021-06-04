import numpy as np


def sigmoid(Z):
    """ Implements the sigmoid activation function """
    
    return 1/(1+np.exp(-Z)) 


def relu(Z):
    """ Implements the RELU activation function """
   
    return np.maximum(0,Z)


def l_relu(Z):
    """ Implements the Leaky RELU activation function """

    return np.maximum(0.01*Z,Z)


def tanh(Z):
    """ Implements the tanh activation function """

    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def sigmoid_backward(dA, Z):
    """
    Implements the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- output of the linear layer, of any shape

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

    
def relu_backward(dA, Z):
    """
    Implements the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- output of the linear layer, of any shape

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, we set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ


def l_relu_backward(dA, Z):
    """
    Implements the backward propagation for a single Leaky RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- output of the linear layer, of any shape

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, we set dz to 0.01 as well. 
    dZ[Z <= 0] = 0.01
    
    return dZ


def tanh_backward(dA, Z):
    """
    Implements the backward propagation for a single TANH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- output of the linear layer, of any shape

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    t = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    dZ = dA * (1 - t**2)

    return dZ