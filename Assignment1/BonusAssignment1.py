import copy
import pandas as pd
import numpy as np


from Assignment1 import ApplyNetwork, BackwardPass, ComputeLoss, ComputeCost, ComputeAccuracy, MiniBatchGD, BitGen, \
    TrainNet, PlotTrainingCurves, VisualizeFilters, NormalizeData, LoadBatch, PlotHistogram

CIFAR_DIR = 'Datasets/cifar-10-batches-py/'
K = 10  # number of classes

#####################################################################################################

# Q1.a
# Load all available training data

# Load all 5 batches
X_list, Y_list, y_list = [], [], []
for i in range(1, 6):
    X_batch, Y_batch, y_batch = LoadBatch(f"data_batch_{i}")
    X_list.append(X_batch)
    Y_list.append(Y_batch)
    y_list.append(y_batch)

# Concatenate all training data (50,000 examples)
full_trainX = np.concatenate(X_list, axis=1)
full_trainY = np.concatenate(Y_list, axis=1)
full_trainy = np.concatenate(y_list, axis=0)

# Shuffle and split off 1000 examples for validation
n_total = full_trainX.shape[1]
perm = np.random.permutation(n_total)
val_size = 1000
val_idx = perm[:val_size]
train_idx = perm[val_size:]

# Final training set (~49,000)
trainX = full_trainX[:, train_idx]
trainY = full_trainY[:, train_idx]
trainy = full_trainy[train_idx]

# Validation set (1000)
valX = full_trainX[:, val_idx]
valY = full_trainY[:, val_idx]
valy = full_trainy[val_idx]

# Load test data
testX, testY, testy = LoadBatch("test_batch")

trainX, valX, testX = NormalizeData(trainX, valX, testX)


GDparams = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40
}
seed = 42
rng = np.random.default_rng()
rng.bit_generator.state = BitGen(seed).state
init_net = {}
[d,n] = trainX.shape
init_net['W'] = .01*rng.standard_normal(size = (K, d))
init_net['b'] = np.zeros((K, 1))

[trained_net, train_loss_history, train_cost_history, val_loss_history, val_cost_history] = (
        TrainNet(trainX, trainY, trainy, valX, valy, init_net, 0, rng))

PlotTrainingCurves(train_loss_history, val_loss_history, train_cost_history,
                  val_cost_history,"loss_plot_2_1.jpg", "cost_plot_2_1.jpg")

VisualizeFilters(trained_net['W'], "filters_all_in_one_2_1.png")
#####################################################################################################

# Q1.b

#Compute flipping indices once
aa = np.int32(np.arange(32)).reshape((32, 1))
bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
vv = np.tile(32 * aa, (1, 32))
ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
inds_flip = np.vstack((inds_flip, 2048 + ind_flip))
inds_flip = inds_flip.flatten()

def MiniBatchGDWithAugmentation(X, Y, y, GDparams, init_net, lam, rng, setting):
    loss_history = []
    cost_history = []
    trained_net = copy.deepcopy(init_net)
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']
    n = X.shape[1]

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X = X[:, perm]
        Y = Y[:, perm]
        y = y[perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)

            Xbatch = X[:, inds].copy()  # copy to not flip original
            Ybatch = Y[:, inds]

            # Data Augmentation: Horizontal Flip with 0.5 probability
            for i in range(Xbatch.shape[1]):
                if rng.random() < 0.5:
                    Xbatch[:, i] = Xbatch[inds_flip, i]

            # Forward pass
            Pbatch = ApplyNetwork(Xbatch, trained_net)
            # Backward pass
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # Gradient update
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        # Compute full epoch loss & cost
        P_full = ApplyNetwork(X, trained_net)
        loss = ComputeLoss(P_full, y)
        cost = ComputeCost(loss, trained_net['W'], lam)
        loss_history.append(loss)
        cost_history.append(cost)

    acc = ComputeAccuracy(ApplyNetwork(X, trained_net), y)
    print(f"Accuracy for {setting} : {acc * 100:.2f}%")
    return [trained_net, loss_history, cost_history]

lam = 0.001

# Initialize network parameters
network_params = {
    'W': np.random.randn(K, 3072) * 0.001,
    'b': np.zeros((K, 1))
}

# Train the network
def TrainNetWithAugmentation(trainX, trainY, trainy, valX, valY, valy, testX, testY, testy,
             GDparams, init_net, lam, rng):
    # Train the network with augmentation
    [trained_net, train_loss_history, train_cost_history] = MiniBatchGDWithAugmentation(trainX, trainY, trainy,
                                                                        GDparams, init_net,
                                                                        lam, rng, "training")
    # Validate the network without augmentation
    [val_net, val_loss_history, val_cost_history] = MiniBatchGD(valX, valY, valy,
                                                                GDparams, trained_net,
                                                                lam, rng, "validation")
    # Test the network without augmentation
    [tested_net, test_loss_history, test_cost_history] = MiniBatchGD(testX, testY, testy,
                                                                    GDparams, trained_net,
                                                                    lam, rng, "testing")
    return [trained_net, train_loss_history, train_cost_history,
           val_net, val_loss_history, val_cost_history,
           tested_net, test_loss_history, test_cost_history]


[aug_trained_net, train_loss_history, train_cost_history, val_net, val_loss_history, val_cost_history,
    tested_net, test_loss_history, test_cost_history] = (
        TrainNet(trainX, trainY, trainy,valX, valY, valy, testX, testY, testy,
             GDparams, init_net, lam, rng))

PlotTrainingCurves(train_loss_history, val_loss_history, train_cost_history,
                  val_cost_history,"loss_plot_2_2.jpg", "cost_plot_2_2.jpg")

VisualizeFilters(aug_trained_net['W'], "filters_all_in_one_2_2.png")

#####################################################################################################

# Q1.c

def GridSearch(trainX, trainY, trainy, init_net, valy, rng):
    # Grid of values to search
    lambdas = [0, 0.001, 0.1, 1]
    etas = [0.001, 0.005, 0.01]
    batch_sizes = [50, 100, 200]

    # best_acc = 0
    # best_params = {}
    results = []

    for lam in lambdas:
        for eta in etas:
            for n_batch in batch_sizes:
                print(f"\nTraining with lam={lam}, eta={eta}, n_batch={n_batch}")

                GDparams = {
                    'n_batch': n_batch,
                    'eta': eta,
                    'n_epochs': 40  # fewer epochs for speed in search
                }

                # Train
                [trained_net, _, _ ] = (MiniBatchGD
                                        (trainX, trainY, trainy, GDparams, init_net, lam, rng, "training"))

                # Validate
                P_val = ApplyNetwork(valX, trained_net)
                val_acc = ComputeAccuracy(P_val, valy)

                results.append((lam, eta, n_batch, val_acc*100))
    return results


results  = GridSearch(trainX, trainY, trainy, init_net, valy, rng)

df = pd.DataFrame(results, columns=['Lambda', 'Eta', 'BatchSize', 'ValAccuracy'])
df = df.sort_values(by='ValAccuracy', ascending=False)
print(df.head(10))  # Top 10 configurations

#####################################################################################################

# Q 1.d

def MiniBatchGDStepDecay(X, Y, y, GDparams, init_net, lam, rng, decay_every=10, decay_rate=0.1):
    loss_history = []
    cost_history = []
    val_acc_history = []

    trained_net = copy.deepcopy(init_net)
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']
    n = X.shape[1]

    for epoch in range(n_epochs):
        # Decay learning rate every decay_every epochs
        if epoch > 0 and epoch % decay_every == 0:
            eta *= decay_rate
            print(f"Epoch {epoch}: Reducing learning rate to {eta}")

        # Shuffle for each epoch
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
            Pbatch = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        # Track performance
        P_full = ApplyNetwork(X, trained_net)
        loss = ComputeLoss(P_full, y)
        cost = ComputeCost(loss, trained_net['W'], lam)
        acc = ComputeAccuracy(P_full, y)

        loss_history.append(loss)
        cost_history.append(cost)
        val_acc_history.append(acc)

        print(f"Epoch {epoch + 1}/{n_epochs} - LR: {eta:.5f} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    return trained_net, loss_history, cost_history, val_acc_history

BestGDparams = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40
}

seed = 42
rng = np.random.default_rng(seed)

trained_net, loss_hist, cost_hist, acc_hist = MiniBatchGDStepDecay(
    trainX, trainY, trainy, BestGDparams, init_net, lam=0.001, rng=rng,
    decay_every=10, decay_rate=0.1
)

#####################################################################################################

# Q2

# define sigmoid function
def sigmoid(s):
    return 1 / (1 + np.exp(-s + 1e-15))

# define BCE loss function
def ComputeBCELoss(P, Y):
    loss_matrix = (1 - Y) * np.log(1 - P) + Y * np.log(P)
    return -np.mean(np.sum(loss_matrix, axis=0, keepdims=True))

# define the training function for the sigmoid network
def ApplySigmoidNetwork(X, network):
    W = network['W']
    b = network['b']
    s = W @ X + b
    P = sigmoid(s)
    return P


def MiniGDWithBCE(X, Y, y, BCEparams, init_net, lam, rng, setting):
    loss_history = []
    cost_history = []
    trained_net = copy.deepcopy(init_net)
    n_batch = BCEparams['n_batch']
    eta = BCEparams['eta']
    n_epochs = BCEparams['n_epochs']
    n = X.shape[1]

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_perm = X[:, perm]
        Y_perm = Y[:, perm]
        y_perm = y[perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            Xbatch = X_perm[:, inds]
            Ybatch = Y_perm[:, inds]
            # Forward Pass
            Pbatch = ApplySigmoidNetwork(Xbatch, trained_net)
            # Backward Pass
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            # Update
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        # Compute training loss and accuracy on full dataset (once per epoch)
        P_full = ApplySigmoidNetwork(X_perm, trained_net)
        loss = ComputeBCELoss(P_full, Y_perm)
        loss_history.append(loss)
        cost = ComputeCost(loss, trained_net['W'], lam)
        cost_history.append(cost)

    accuracy = ComputeAccuracy(ApplySigmoidNetwork(X, trained_net), y)
    print(f"Accuracy for {setting} : {accuracy * 100:.2f}%")

    return [trained_net, loss_history, cost_history]

BCEparams = {
    'n_batch': 100,
    'eta': 0.01,
    'n_epochs': 40
}

[trained_net, train_loss_history, train_cost_history] = (
    MiniGDWithBCE(trainX, trainY, trainy, BCEparams, init_net, 0.001, rng, "training"))

[val_net, val_loss_history, val_cost_history] = (
    MiniGDWithBCE(valX, valY, valy, BCEparams, init_net, 0.001, rng, "validation"))

[tested_net, test_loss_history, test_cost_history] = (
    MiniGDWithBCE(testX, testY, testy, BCEparams, init_net, 0.001, rng, "testing"))

PlotTrainingCurves(train_loss_history, val_loss_history, train_cost_history,
                   val_cost_history,"loss_plot_2_4.jpg", "cost_plot_2_4.jpg")

PlotHistogram(ApplySigmoidNetwork(testX, tested_net), testy, "Histogram (Sigmoid + BCE)", "histogram_bce.png")