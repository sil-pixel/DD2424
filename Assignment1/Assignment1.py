import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

#from torch_gradient_computations import ComputeL2GradsWithTorch

# Constants
CIFAR_DIR = 'Datasets/cifar-10-batches-py/'
K = 10 # number of classes

#####################################################################################################

# Q1
def LoadBatch(filename):
    # Load a batch of training data
    with open(CIFAR_DIR + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # Extract the image data and cast to float from the dict dictionary
    X = dict[b'data'].astype(np.float64) / 255.0
    X = X.transpose()

    # Extract the image labels
    y = np.array(dict[b'labels'])

    # One-hot encode the labels
    n = len(y)
    Y = np.zeros((K, n), dtype=np.float64)
    # one-hot encoding
    for i in range(n):
        Y[y[i], i] = 1.0
    return [X, Y, y]

#####################################################################################################
# Load training, validation and testing data
trainX, trainY, trainy = LoadBatch("data_batch_1")
valX, valY, valy = LoadBatch("data_batch_2")
testX, testY, testy = LoadBatch("test_batch")
[d,n] = trainX.shape

#####################################################################################################

# Q2
def NormalizeData(trainX, valX, testX):
    d = trainX.shape[0]
    mean_X = np.mean(trainX, axis=1).reshape(d, 1)
    std_X = np.std(trainX, axis=1).reshape(d, 1)

    trainX_norm = (trainX - mean_X) / std_X
    valX_norm = (valX - mean_X) / std_X
    testX_norm = (testX - mean_X) / std_X

    return trainX_norm, valX_norm, testX_norm

trainX, valX, testX = NormalizeData(trainX, valX, testX)
#####################################################################################################

# Q3
rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state
init_net = {}
init_net['W'] = .01*rng.standard_normal(size = (K, d))
init_net['b'] = np.zeros((K, 1))

#####################################################################################################

# Q4

def softmax(s):
    # to avoid inf/inf, for stability remove a constant value from exponentiation
    exp_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    P = exp_s/ np.sum(exp_s, axis=0, keepdims=True)
    return P

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    P = softmax(s)
    return P

#P = ApplyNetwork(trainX[:, 0:100], init_net)

#####################################################################################################

# Q5

# Cross-entropy loss
def ComputeLoss(P, y):
    n = P.shape[1]
    log_P = -np.log(P[y, np.arange(n)])  # using the one-hot coding formula for y
    L = np.mean(log_P)  # average over all samples
    return L

#L = ComputeLoss(P, trainy[0:100])

#####################################################################################################

# Q6

def ComputeAccuracy(P, y):
    n = P.shape[1]
    predictions = np.argmax(P, axis=0)
    correct = np.sum(predictions == y)
    accuracy = correct/n
    return accuracy

#acc = ComputeAccuracy(P, trainy[0:100])

#####################################################################################################

# Q7

def BackwardPass(X, Y, P, network, lam):
    nb = X.shape[1]
    G = -(Y - P)
    dL_dW = (G @ X.transpose())/ nb
    dL_db = np.mean(G, axis=1, keepdims=True)
    grads = {}
    W = network['W']
    grads['W'] = dL_dW + 2*lam*W
    grads['b'] = dL_db
    return grads

#grads = BackwardPass(trainX[:, 0:100], trainY[:, 0:100], P, init_net, 0)

#####################################################################################################

# check grads with torch

# def CheckGradsWithTorch(trainX, trainY, trainy):
#     d_small = 10
#     n_small = 3
#     lam = 0
#     small_net = {}
#     small_net['W'] = .01*rng.standard_normal(size = (10, d_small))
#     small_net['b'] = np.zeros((10, 1))
#     X_small = trainX[0:d_small, 0:n_small]
#     Y_small = trainY[:, 0:n_small]
#     P = ApplyNetwork(X_small, small_net)
#     my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
#     torch_grads = ComputeL2GradsWithTorch(X_small, trainy[0:n_small], lam, small_net)
#     return [my_grads, torch_grads]


def relative_error(grad_analytical, grad_numerical, eps=1e-6):
    rel_errors = {}

    for key in grad_analytical:
        ga = grad_analytical[key]
        gn = grad_numerical[key]

        # Compute element-wise relative error
        num = np.abs(ga - gn)
        denom = np.maximum(eps, np.abs(ga) + np.abs(gn))
        rel_error = num / denom

        # Store maximum relative error for that variable
        rel_errors[key] = np.max(rel_error)

    return rel_errors

# [my_grads, torch_grads] = CheckGradsWithTorch(trainX, trainY, trainy)
# rel_errs = relative_error(torch_grads, my_grads)

# for key, val in rel_errs.items():
#    print(f"Max relative error in {key}: {val:.2e}")

#####################################################################################################

# Compute cost function
def ComputeCost(loss, W, lam):
    # L2 regularization
    reg_term = lam * np.sum(W**2)
    cost = loss + reg_term
    return cost

#####################################################################################################

# Q8

def MiniBatchGD(X, Y, y, GDparams, init_net, lam, rng, valX=None, valy=None, testX=None, testy=None):
    loss_history = []
    cost_history = []
    val_loss_history = []
    val_cost_history = []
    # Initialize the network
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
            Pbatch = ApplyNetwork(Xbatch, trained_net)
            # Backward Pass
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # Gradient descent update
            trained_net['W'] -= eta*grads['W']
            trained_net['b'] -= eta*grads['b']

        # Compute training loss and accuracy on full dataset (once per epoch)
        P_full = ApplyNetwork(X, trained_net)
        loss = ComputeLoss(P_full, y)
        loss_history.append(loss)
        cost = ComputeCost(loss, trained_net['W'], lam)
        cost_history.append(cost)
        acc = ComputeAccuracy(P_full, y)
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")
        # Compute validation loss and accuracy
        if valX is not None and valy is not None:
            P_val = ApplyNetwork(valX, trained_net)
            val_loss = ComputeLoss(P_val, valy)
            val_cost = ComputeCost(val_loss, trained_net['W'], lam)
            val_loss_history.append(val_loss)
            val_cost_history.append(val_cost)
            val_acc = ComputeAccuracy(P_val, valy)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")
    acc = ComputeAccuracy(ApplyNetwork(X, trained_net), y)
    print(f"Accuracy for training : {acc * 100:.2f}%")
    val_acc = ComputeAccuracy(ApplyNetwork(valX, trained_net), valy)
    print(f"Accuracy for validation : {val_acc * 100:.2f}%")
    # Compute test loss and accuracy
    if testX is not None and testy is not None:
        test_acc = ComputeAccuracy(ApplyNetwork(testX, trained_net), testy)
        print(f"Accuracy for test : {test_acc * 100:.2f}%")
    return [trained_net, loss_history, cost_history, val_loss_history, val_cost_history]


# Train the network
def TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams, lam, rng):
    # Train the network
    return MiniBatchGD(trainX, trainY, trainy, GDparams, init_net, lam, rng, valX=valX, valy=valy, testX=testX, testy=testy)


GDparams = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40
}
# [trained_net, train_loss_history, train_cost_history, val_loss_history, val_cost_history] = (
#     TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams, 0.001, rng))


# Plot the loss curves

def PlotTrainingCurves(train_loss_history, val_loss_history, train_cost_history, val_cost_history,
                       loss_filename, cost_filename):

    epochs = range(1, len(train_loss_history) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_history, label="Training Loss", color='green')
    plt.plot(epochs, val_loss_history, label="Validation Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_filename)
    plt.show()

    # Plot Cost
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_cost_history, label="Training Cost", color='green')
    plt.plot(epochs, val_cost_history, label="Validation Cost", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training and Validation Cost Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(cost_filename)
    plt.show()


# Plot the training and validation loss and cost
# PlotTrainingCurves(train_loss_history, val_loss_history, train_cost_history,
#                   val_cost_history,"loss_plot_2_3.jpg", "cost_plot_2_3.jpg")


#####################################################################################################

# Visualise W matrix

def VisualizeFilters(W, filename):
    """
    Visualize the weight matrix W as 10 class template images for CIFAR-10.

    Parameters:
        W (np.ndarray): Weight matrix of shape (K, 3072) where K=10.
        filename (str): Output file name to save the filter visualization.
    """
    Ws = W.transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))  # shape: (32, 32, 3, 10)

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))  # normalize for display
        axes[i].imshow(w_im_norm)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


#VisualizeFilters(trained_net['W'], "filters_all_in_one.png")

def Assignment1():
    # case 1
    GDparams1 = {
        'n_batch': 100,
        'eta': 0.1,
        'n_epochs': 40
    }
    [trained_net1, train_loss_history1, train_cost_history1, val_loss_history1, val_cost_history1] = (
        TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams1, 0, rng))
    PlotTrainingCurves(train_loss_history1, val_loss_history1, train_cost_history1,
                      val_cost_history1, "loss_plot1.jpg", "cost_plot1.jpg")
    VisualizeFilters(trained_net1['W'], "filters_all_in_one1.png")
    # case 2
    GDparams2 = {
        'n_batch': 100,
        'eta': 0.001,
        'n_epochs': 40
    }
    [trained_net2, train_loss_history2, train_cost_history2, val_loss_history2, val_cost_history2] = (
        TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams2, 0, rng))
    PlotTrainingCurves(train_loss_history2, val_loss_history2, train_cost_history2,
                      val_cost_history2, "loss_plot2.jpg", "cost_plot2.jpg")
    VisualizeFilters(trained_net2['W'], "filters_all_in_one2.png")
    # case 3
    [trained_net3, train_loss_history3, train_cost_history3, val_loss_history3, val_cost_history3] = (
        TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams2, 0.1, rng))
    PlotTrainingCurves(train_loss_history3, val_loss_history3, train_cost_history3,
                      val_cost_history3, "loss_plot3.jpg", "cost_plot3.jpg")
    VisualizeFilters(trained_net3['W'], "filters_all_in_one3.png")
    # case 4
    [trained_net4, train_loss_history4, train_cost_history4, val_loss_history4, val_cost_history4] = (
        TrainNet(trainX, trainY, trainy, valX, valy, testX, testy, init_net, GDparams2, 1, rng))
    PlotTrainingCurves(train_loss_history4, val_loss_history4, train_cost_history4,
                      val_cost_history4, "loss_plot4.jpg", "cost_plot4.jpg")
    VisualizeFilters(trained_net4['W'], "filters_all_in_one4.png") 


#Assignment1()



def PlotHistogram(P, testy, title, filename):
    # Get predicted class for each example
    preds = np.argmax(P, axis=0)
    correct_mask = (preds == testy)
    incorrect_mask = ~correct_mask

    # Extract the predicted probability for the true class
    true_class_probs = P[testy, np.arange(len(testy))]

    correct_probs = true_class_probs[correct_mask]
    incorrect_probs = true_class_probs[incorrect_mask]

    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(correct_probs, bins=20, alpha=0.7, label="Correctly Classified", color='green')
    plt.hist(incorrect_probs, bins=20, alpha=0.7, label="Incorrectly Classified", color='red')
    plt.xlabel("Predicted Probability of True Class")
    plt.ylabel("Number of Examples")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

#PlotHistogram(ApplyNetwork(testX, trained_net), testy, "Histogram (Softmax + CE)", "histogram_ce.png")