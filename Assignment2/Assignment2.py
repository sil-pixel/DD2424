import numpy as np
import copy 
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from Assignment1.Assignment1 import ComputeAccuracy, ComputeCost, ComputeLoss, LoadBatch, NormalizeData, softmax
from torch_gradient_computations import ComputeL2GradsWithTorch

# Set the random seed for reproducibility
rng2 = np.random.default_rng()
BitGen = type(rng2.bit_generator)
seed = 2025
rng2.bit_generator.state = BitGen(seed).state



def networkInitialize(K, d, m):
    # randomly initialize a 2-layer network
    small_net = {}
    small_net['W'] = {}
    small_net['b'] = {}
    std1 = 1 / np.sqrt(d)
    std2 = 1 / np.sqrt(m)
    small_net['W'][0] = std1*rng2.standard_normal(size = (m, d))
    small_net['b'][0] = np.zeros((m, 1))
    small_net['W'][1] = std2*rng2.standard_normal(size = (K, m))
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
    d_small = 100
    n_small = 5
    m = 6
    lam = 0.01
    small_net = networkInitialize(10, d_small, m)
    X_small = trainX[0:d_small, 0:n_small]
    Y_small = trainY[:, 0:n_small]
    fp_data = Apply2LayerNetwork(X_small, small_net)
    my_grads = Backward2LayerPass(X_small, Y_small, fp_data, small_net, lam)
    torch_grads = ComputeL2GradsWithTorch(X_small, trainy[0:n_small], lam, small_net)
    return [my_grads, torch_grads]


trainX, trainY, trainy = LoadBatch("data_batch_1")
valX, valY, valy = LoadBatch("data_batch_2")
testX, testY, testy = LoadBatch("test_batch")
trainX, valX, testX = NormalizeData(trainX, valX, testX)

[d,n] = trainX.shape


def relative_error2(my_grads_layer, torch_grads_layer, eps=1e-6):
    rel_errors = {}

    for i in range(len(my_grads_layer)):
        ga = my_grads_layer[i]
        gn = torch_grads_layer[i]  # indexed access in torch_grads

        if isinstance(ga, np.ndarray) and isinstance(gn, np.ndarray):
            num = np.abs(ga - gn)
            denom = np.maximum(eps, np.abs(ga) + np.abs(gn))
            rel_error = num / denom
            rel_errors[i] = rel_error
        else:
            raise TypeError(f"Expected numpy arrays for gradients, got {type(ga)} and {type(gn)} at index {i}")
    return rel_errors

#my_grads, torch_grads = CheckGradsWithTorch(trainX, trainY, trainy)


# Check the relative errors
def print_relative_errors(my_grads, torch_grads):
    rel_errs_W = relative_error2(my_grads['W'], torch_grads['W'])
    rel_errs_b = relative_error2(my_grads['b'], torch_grads['b'])

    print("Relative errors in W:")
    for key in rel_errs_W:
        print(f"Layer {key}: max relative error = {np.max(rel_errs_W[key]):.2e}")

    print("\nRelative errors in b:")
    for key in rel_errs_b:
        print(f"Layer {key}: max relative error = {np.max(rel_errs_b[key]):.2e}")

#print_relative_errors(my_grads, torch_grads)

def Compute2LayerCost(loss, network, lam):  
    # Compute the cost for a 2-layer network
    W1 = network['W'][0]
    W2 = network['W'][1]
    cost = ComputeCost(loss, W2, lam) + ComputeCost(loss, W1, lam)
    return cost

def MiniBatch2LayerGD(X, Y, y, GDparams, init_net, lam, rng, valX=None, valy = None, testX=None, testy = None):
    loss_history = []
    cost_history = []
    val_loss_history = []
    val_cost_history = []
    trained_net = copy.deepcopy(init_net)
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']
    n = X.shape[1]
    for epoch in range(n_epochs):
        # Shuffle indices for current epoch
        perm = rng.permutation(n)
        X = X[:, perm]
        Y = Y[:, perm]
        y = y[perm]
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]
            # Forward Pass
            Pbatch = Apply2LayerNetwork(Xbatch, trained_net)
            # Backward Pass
            grads = Backward2LayerPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # Gradient descent update
            trained_net['W'][0] -= eta*grads['W'][0]
            trained_net['b'][0] -= eta*grads['b'][0]
            trained_net['W'][1] -= eta*grads['W'][1]
            trained_net['b'][1] -= eta*grads['b'][1]

        # Compute training loss and accuracy on full dataset (once per epoch)
        P_full = Apply2LayerNetwork(X, trained_net)
        loss = ComputeLoss(P_full, y)
        loss_history.append(loss)
        cost = Compute2LayerCost(loss, trained_net, lam)
        cost_history.append(cost)
        acc = ComputeAccuracy(P_full, y)
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")
        # Compute validation loss and accuracy (if validation data is provided)
        if valX is not None and valy is not None:
            P_val = Apply2LayerNetwork(valX, trained_net)
            val_loss = ComputeLoss(P_val, valy)
            val_loss_history.append(val_loss)
            val_cost = Compute2LayerCost(val_loss, trained_net, lam)
            val_cost_history.append(val_cost)
            val_acc = ComputeAccuracy(P_val, valy)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")
    acc = ComputeAccuracy(Apply2LayerNetwork(X, trained_net), y)
    print(f"Accuracy for training : {acc * 100:.2f}%")
    val_acc = ComputeAccuracy(Apply2LayerNetwork(valX, trained_net), valy)
    print(f"Accuracy for validation : {val_acc * 100:.2f}%")
    test_acc = ComputeAccuracy(Apply2LayerNetwork(testX, trained_net), testy)
    print(f"Accuracy for testing : {test_acc * 100:.2f}%")
    return [trained_net, loss_history, cost_history, val_loss_history, val_cost_history]


# Train the network
def Train2LayerNet(trainX, trainY, trainy, valX, valy, testX, testy, GDparams, init_net, lam, rng):
    # Train the network
    return MiniBatch2LayerGD(trainX, trainY, trainy, GDparams, init_net,lam, rng, valX=valX, valy=valy, testX=testX, testy=testy)


# GDparams = {
#     'n_batch': 100,
#     'eta': 0.001,
#     'n_epochs': 200
# }
# lam = 0.001
# init_net = networkInitialize(10, d, 100)
# [trained_net, train_loss_history, train_cost_history, val_loss_history, val_cost_history] = (
#     Train2LayerNet(trainX, trainY, trainy, valX, valy, testX, testy, GDparams, init_net, lam, rng))


def CyclicEta(n_s, eta_max, eta_min, t):
    l = int(t // (2 * n_s))  # determine current cycle
    t_mod = t - 2 * l * n_s  # position within current cycle

    if t_mod <= n_s:
        eta_t = eta_min + (t_mod / n_s) * (eta_max - eta_min)
    else:
        eta_t = eta_max - ((t_mod - n_s) / n_s) * (eta_max - eta_min)

    return eta_t      

# train the network with cyclic learning rate
def MiniBatch2LayerGDCyclic(X, Y, y, GDparams, init_net, lam, rng, valX=None, valy = None, testX=None, testy = None):
    loss_history = []
    cost_history = []
    acc_history = []
    steps_history = []
    val_loss_history = []
    val_cost_history = []
    val_acc_history = []
    eta_history1 = []
    eta_history2 = []
    step = 0
    trained_net = copy.deepcopy(init_net)
    n_batch = GDparams['n_batch']
    eta_max = GDparams['eta_max']
    eta_min = GDparams['eta_min']
    n_epochs = GDparams['n_epochs']
    n_s = GDparams['n_s']
    n = X.shape[1]
    for epoch in range(n_epochs):
        # Shuffle indices for current epoch
        perm = rng.permutation(n)
        X = X[:, perm]
        Y = Y[:, perm]
        y = y[perm]
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]
            ybatch = y[inds]
            # Forward Pass
            Pbatch = Apply2LayerNetwork(Xbatch, trained_net)
            # Backward Pass
            grads = Backward2LayerPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # Gradient descent update
            # Compute the current learning rate
            eta_t0 = CyclicEta(n_s, eta_max, eta_min, step)  # for layer 0
            eta_t1 = CyclicEta(n_s, eta_max, eta_min, step)  # for layer 1
            #print(f"step {step}: learning rate: {eta_t0}, {eta_t1}")
            trained_net['W'][0] -= eta_t0 * grads['W'][0]
            trained_net['b'][0] -= eta_t0 * grads['b'][0]
            trained_net['W'][1] -= eta_t1 * grads['W'][1]
            trained_net['b'][1] -= eta_t1 * grads['b'][1]
            # Store the current learning rate
            eta_history1.append(eta_t0)
            eta_history2.append(eta_t1)

            # Store the current step and learning rate
            if step % 100 == 0 or step == 999:
                steps_history.append(step)
                loss = ComputeLoss(Pbatch, ybatch)
                loss_history.append(loss)
                cost = Compute2LayerCost(loss, trained_net, lam)
                cost_history.append(cost)
                acc = ComputeAccuracy(Pbatch, ybatch)
                acc_history.append(acc)

                val_loss = ComputeLoss(Apply2LayerNetwork(valX, trained_net), valy)
                val_cost = Compute2LayerCost(val_loss, trained_net, lam)
                val_loss_history.append(val_loss)
                val_cost_history.append(val_cost)
                val_acc = ComputeAccuracy(Apply2LayerNetwork(valX, trained_net), valy)
                val_acc_history.append(val_acc)
            

            step += 1

    # Compute training loss and accuracy on full dataset 
    acc = ComputeAccuracy(Apply2LayerNetwork(X, trained_net), y)
    print(f"Accuracy for training : {acc * 100:.2f}% with steps {step}")
    val_acc = ComputeAccuracy(Apply2LayerNetwork(valX, trained_net), valy)
    print(f"Accuracy for validation : {val_acc * 100:.2f}% with steps {step}")
    test_acc = ComputeAccuracy(Apply2LayerNetwork(testX, trained_net), testy)
    print(f"Accuracy for testing : {test_acc * 100:.2f}% with steps {step}")
    return [trained_net, loss_history, cost_history, acc_history, val_loss_history, val_cost_history, val_acc_history, steps_history, eta_history1, eta_history2]

# Train the network with cyclic learning rate
def Train2LayerNetCyclic(trainX, trainY, trainy, valX, valy, testX, testy,
             GDparams, init_net, lam, rng):
    # Train the network
    return MiniBatch2LayerGDCyclic(trainX, trainY, trainy, GDparams, init_net, lam, rng, 
                                   valX=valX, valy=valy, testX=testX, testy=testy)


GDparams = {
    'n_batch': 100,
    'eta_max': 1e-1,
    'eta_min': 1e-5,
    'n_epochs': 10,
    'n_s': 500   
}
lam = 0.01
init_net = networkInitialize(10, d, 100)    
[trained_net, train_loss_history, train_cost_history, train_acc_history, 
 val_loss_history, val_cost_history, val_acc_history, train_steps_history, eta_history1, eta_history2] = (
    Train2LayerNetCyclic(trainX, trainY, trainy, valX, valy, testX, testy,
             GDparams, init_net, lam, rng2))



def plot_training_curves(log_steps, log_loss, log_cost, log_acc,
                         val_loss=None, val_cost=None, val_acc=None):
    plt.figure(figsize=(18, 4))

    # Cost
    plt.subplot(1, 3, 1)
    plt.plot(log_steps, log_cost, label='training', color='green')
    if val_cost:
        plt.plot(log_steps, val_cost, label='validation', color='red')
    plt.title("Cost plot")
    plt.xlabel("update step")
    plt.ylabel("cost")
    plt.xlim(0, 1000)
    plt.ylim(0, max(log_cost) * 1.1)
    plt.legend()

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(log_steps, log_loss, label='training', color='green')
    if val_loss:
        plt.plot(log_steps, val_loss, label='validation', color='red')
    plt.title("Loss plot")
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.xlim(0, 1000)
    plt.ylim(0, max(log_loss) * 1.1)
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(log_steps, log_acc, label='training', color='green')
    if val_acc:
        plt.plot(log_steps, val_acc, label='validation', color='red')
    plt.title("Accuracy plot")
    plt.xlabel("update step")
    plt.ylabel("accuracy")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("cyclic_training_curves.jpg", dpi=300)
    plt.show()

plot_training_curves(train_steps_history, train_loss_history, train_cost_history, train_acc_history,
                     val_loss=val_loss_history, val_cost=val_cost_history, val_acc=val_acc_history)

def plot_eta(eta1, eta2):
    plt.figure(figsize=(12, 4))
    log_steps = np.arange(0, len(eta1))
    plt.plot(log_steps, eta1, label='layer 0', color='blue')
    plt.plot(log_steps, eta2, label='layer 1', color='orange')
    plt.title("Learning rate plot")
    plt.xlabel("update step")
    plt.ylabel("learning rate")
    plt.xlim(0, 1000)
    plt.ylim(0, max(max(eta1), max(eta2)) * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cyclic_eta.jpg", dpi=300)
    plt.show()

#plot_eta(eta_history1, eta_history2)