import numpy as np

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from Assignment1.Assignment1 import LoadBatch, NormalizeData, softmax
from torch_gradient_computations import ComputeGradsWithTorch

# Set the random seed for reproducibility
rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state



def networkInitialize(K, d, m):
    # randomly initialize a 2-layer network
    small_net = {}
    small_net['W'] = {}
    small_net['b'] = {}
    std1 = 1 / np.sqrt(d)
    std2 = 1 / np.sqrt(m)
    small_net['W'][0] = std1*rng.standard_normal(size = (m, d))
    small_net['b'][0] = np.zeros((m, 1))
    small_net['W'][1] = std2*rng.standard_normal(size = (K, m))
    small_net['b'][1] = np.zeros((K, 1))
    return small_net


def Apply2LayerNetwork(X, network):
    # Layer 1
    W1 = network['W'][0]
    b1 = network['b'][0]
    s1 = W1 @ X + b1
    # ReLU
    X = np.maximum(0, s1)
    # Layer 2
    W2 = network['W'][1]
    b2 = network['b'][1]
    s2 = W2 @ X + b2
    # Softmax
    P = softmax(s2)
    return P

def Backward2LayerPass(X, Y, P, network, lam):
    grads = {}
    grads['W'] = {}
    grads['b'] = {}
    nb = X.shape[1]
    G = -(Y - P)
    W1 = network['W'][0]
    W2 = network['W'][1]
    b1 = network['b'][0]
    # Gradient of the loss with respect to the output layer
    H = np.maximum(W1 @ X + b1, 0)
    dL_dW2 = (G @ H.transpose())/ nb
    dL_db2 = np.mean(G, axis=1, keepdims=True)
    # set the gradient 
    grads['W'][1] = dL_dW2 + 2*lam*W2
    grads['b'][1] = dL_db2
    # Backpropagation
    G = W2.transpose() @ G
    G = G * (H > 0)
    # Gradient of the loss with respect to the hidden layer
    dL_dW1 = (G @ X.transpose())/ nb
    dL_db1 = np.mean(G, axis=1, keepdims=True)
    # set the gradient
    grads['W'][0] = dL_dW1 + 2*lam*W1
    grads['b'][0] = dL_db1
    return grads

def CheckGradsWithTorch(trainX, trainY, trainy):
    d_small = 5
    n_small = 3
    m = 6
    lam = 0
    small_net = networkInitialize(10, d_small, m)
    X_small = trainX[0:d_small, 0:n_small]
    Y_small = trainY[:, 0:n_small]
    fp_data = Apply2LayerNetwork(X_small, small_net)
    my_grads = Backward2LayerPass(X_small, Y_small, fp_data, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, trainy[0:n_small], small_net)
    return [my_grads, torch_grads]


trainX, trainY, trainy = LoadBatch("data_batch_1")
valX, valY, valy = LoadBatch("data_batch_2")
testX, testY, testy = LoadBatch("test_batch")
trainX, valX, testX = NormalizeData(trainX, valX, testX)

[d,n] = trainX.shape


def relative_error2(grads1, grads2, eps=1e-6):
    rel_errors = {}

    for layer_key in grads1:  # e.g., 0 and 1
        ga = grads1[layer_key]  # analytical gradient (numpy array)
        gn = grads2[layer_key]  # numerical gradient (numpy array)

        if isinstance(ga, np.ndarray) and isinstance(gn, np.ndarray):
            num = np.abs(ga - gn)
            denom = np.maximum(eps, np.abs(ga) + np.abs(gn))
            rel_error = num / denom
            rel_errors[layer_key] = rel_error
        else:
            raise TypeError(f"Expected numpy arrays for gradients, got {type(ga)} and {type(gn)} at key {layer_key}")
    
    return rel_errors

my_grads, torch_grads = CheckGradsWithTorch(trainX, trainY, trainy)

# For weights
rel_errs_W = relative_error2(torch_grads['W'], my_grads['W'])
print("Relative errors in weights:")
for layer in rel_errs_W:
    print(f"  Layer {layer}: max rel error = {np.max(rel_errs_W[layer]):.2e}")

# For biases
rel_errs_b = relative_error2(torch_grads['b'], my_grads['b'])
print("Relative errors in biases:")
for layer in rel_errs_b:
    print(f"  Layer {layer}: max rel error = {np.max(rel_errs_b[layer]):.2e}")