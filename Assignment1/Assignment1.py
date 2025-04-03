import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch_gradient_computations import ComputeL2GradsWithTorch

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

#####################################################################################################

# Q2
# Compute mean and std for the training data
[d,n] = trainX.shape
mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

# Normalise training, validation and testing data wrt mean_X, std_X
trainX = (trainX - mean_X)/std_X

valX = (valX - mean_X)/std_X

testX = (testX - mean_X)/std_X

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

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    # to avoid inf/inf, for stability remove a constant value from exponentiation
    exp_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    P = exp_s/ np.sum(exp_s, axis=0, keepdims=True)
    return P

P = ApplyNetwork(trainX[:, 0:100], init_net)

#####################################################################################################

# Q5

# Cross-entropy loss
def ComputeLoss(P, y):
    n = P.shape[1]
    log_P = -np.log(P[y, np.arange(n)])  # using the one-hot coding formula for y
    L = np.mean(log_P)  # average over all samples
    return L

L = ComputeLoss(P, trainy[0:100])

#####################################################################################################

# Q6

def ComputeAccuracy(P, y):
    n = P.shape[1]
    predictions = np.argmax(P, axis=0)
    correct = np.sum(predictions == y)
    accuracy = correct/n
    return accuracy

acc = ComputeAccuracy(P, trainy[0:100])

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

d_small = 10
n_small = 3
lam = 0
small_net = {}
small_net['W'] = .01*rng.standard_normal(size = (10, d_small))
small_net['b'] = np.zeros((10, 1))
X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]
P = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
torch_grads = ComputeL2GradsWithTorch(X_small, trainy[0:n_small], lam, small_net)


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

rel_errs = relative_error(torch_grads, my_grads)

for key, val in rel_errs.items():
    print(f"Max relative error in {key}: {val:.2e}")

#####################################################################################################

# Q8

def MiniBatchGD(X, Y, y, GDparams, init_net, lam, rng):
    loss_history = []
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
        acc = ComputeAccuracy(P_full, y)
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")
    return [trained_net, loss_history]

GDparams = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40
}
lam = 1
seed = 42
rng = np.random.default_rng(seed)
[trained_net, train_loss_history] = MiniBatchGD(trainX, trainY, trainy, GDparams, init_net, lam, rng)
[trained_net_val, val_loss_history] = MiniBatchGD(valX, valY, valy, GDparams, init_net, lam, rng)

# Plot the loss curves

epochs = range(1, len(train_loss_history) + 1)
plt.plot(epochs, train_loss_history, label="Training Loss", color='green')
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.jpg")
plt.show()

#####################################################################################################

# Visualise W matrix


Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
W_im = np.transpose(Ws, (1, 0, 2, 3))
# Plot all 10 filters side by side
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    w_im = W_im[:, :, :, i]
    w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
    axes[i].imshow(w_im_norm)
    axes[i].axis('off')  # hide axes

# Adjust spacing
plt.tight_layout()

# Save the figure with all 10 filters
plt.savefig("filters_all_in_one.png", dpi=300)
plt.show()