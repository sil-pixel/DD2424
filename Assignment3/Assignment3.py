import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import traceback
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Assignment1.Assignment1 import softmax, ComputeLoss, ComputeAccuracy
from torch_gradient_computations import verify_data_gradients_with_torch, verify_gradients_with_torch

np.random.seed(2025)

debug_file = 'debug_conv_info.npz'
load_data = np.load(f"Assignment3/" + debug_file)
X = load_data['X']
Fs = load_data['Fs']

n = X.shape[1] # 5
f = Fs.shape[0] # 4
nf = Fs.shape[3] # 2

n_patches = (32//f)**2 # 64 patches of size 4x4x3
MX = np.zeros((n_patches, (f*f*3), n))

def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a), np.abs(b) + 1e-8))

def exercise1():
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3)) # (32, 32, 3, 5)
    X_conv = np.zeros((32//f, 32//f, nf, n))
    for i in range(n):
        # sub patch of X_ims of size f x f x 3
        for j in range(32//f):
            for k in range(32//f):
                sub_patch = X_ims[j*f:(j+1)*f, k*f:(k+1)*f, :, i]
                for l in range(nf): 
                    X_conv[j, k, l, i] = np.sum(sub_patch * Fs[:, :, :, l])

            
    conv_outputs = load_data['conv_outputs'] # (8, 8, 2, 5)

    # compare X_conv and conv_outputs
    print(f"Max difference between X_conv and conv_outputs is : {relative_error(X_conv, conv_outputs)}")   

    # populate MX matrix 
    for i in range(n):
        patch_idx = 0
        for j in range(32//f):
            for k in range(32//f):
                X_patch = X_ims[j*f:(j+1)*f, k*f:(k+1)*f, :, i]
                MX[patch_idx, :, i] = X_patch.reshape((1, f*f*3), order='C')   # (64, 48, 5)
                patch_idx += 1


    Fs_flat = Fs.reshape((f*f*3, nf), order='C')   # (48, 2)
    conv_outputs_mat = np.einsum('ijn, jl->iln', MX, Fs_flat, optimize=True)  # (64, 2, 5)
    conv_outputs_flat = conv_outputs.reshape((n_patches, nf, n), order='C')  # (64, 2, 5)

    # compare conv_outputs_mat and conv_outputs_flat
    print(f"Max difference between conv_outputs_mat and conv_outputs_flat is : {relative_error(conv_outputs_mat, conv_outputs_flat)}")   
    return conv_outputs_mat, Fs_flat


def ForwardPass(conv_outputs_mat, W1, W2, b1, b2):
    conv_flat = np.fmax(conv_outputs_mat.reshape((n_patches* nf, n), order='C'), 0)  # (128, 5)
    # first layer
    x = W1 @ conv_flat + b1  # (10, 5)
    h = np.maximum(0, x) # ReLU activation
    # second layer
    s = W2 @ h + b2 # (10, 5)
    # softmax activation
    p = softmax(s)
    return conv_flat, h, p
    
def BackwardPass(Y, P, h, W1, W2, conv_flat, MX):
    G_batch = -(Y - P)  # (10, 5)
    dL_dW2 = (G_batch @ h.T)/n 
    dL_db2 = np.mean(G_batch, axis=1, keepdims=True)  # (10, 1)
    # back propogation 
    G_batch = W2.T @ G_batch  # (10, 5) @ (10, 5) -> (10, 5)
    G_batch = G_batch * (h > 0)  # ReLU derivative
    # gradient wrt hidden layer
    dL_dW1 = (G_batch @ conv_flat.T)/n  # (10, 5) @ (128, 5) -> (10, 128)
    dL_db1 = np.mean(G_batch, axis=1, keepdims=True)  # (10, 1)
    # gradient wrt conv_flat
    G_batch = W1.T @ G_batch  # (10, 128) @ (10, 5) -> (128, 5)
    # reshape for backpropagation
    GG = G_batch.reshape((n_patches, nf, n), order='C')  # (64, 2, 5)
    MXt = np.transpose(MX, (1, 0, 2))  # (48, 64, 5)
    grads_Fs_flat = np.einsum('ijn,jln->il', MXt, GG, optimize=True)  # (48, 2)
    grads = {
        'Fs_flat': grads_Fs_flat,
        'W1': dL_dW1,
        'b1': dL_db1,
        'W2': dL_dW2,
        'b2': dL_db2
    }
    return grads

def LabelSmoothing(Y, eps, smoothing):
    if not smoothing:
        return Y
    K = Y.shape[0]  # number of classes
    Y_smooth = np.copy(Y)
    for i in range(Y.shape[1]): 
        true_class_idx = np.argmax(Y[:, i])
        Y_smooth[:, i] = eps / (K - 1)  # All classes get small probability
        Y_smooth[true_class_idx, i] = 1 - eps  # True class gets high probability
    return Y_smooth

def exercise2(conv_outputs_mat):
    W1 = load_data['W1']      # (10, 128)
    W2 = load_data['W2']      # (10, 10)
    b1 = load_data['b1']      # (10, 1)
    b2 = load_data['b2']      # (10, 1)
    conv_flat, h, p = ForwardPass(conv_outputs_mat, W1, W2, b1, b2)
    data_conv_flat = load_data['conv_flat']  
    # compare conv_flat and data_conv_flat
    print(f"Max difference for conv_flat is : {relative_error(conv_flat, data_conv_flat)}")  
    # compare x1 and load_data['x1']
    x1_data = load_data['X1']  # (10, 5)
    print(f"Max difference for h is : {relative_error(h, x1_data)}")  
    # compare p and load_data['p']  
    p_data = load_data['P']  # (10, 5)
    print(f"Max difference for p is : {relative_error(p, p_data)}")  
    Y = load_data['Y']   # (10, 5)
    Y = LabelSmoothing(Y, 0.1, False)  # Smooth the labels
    grads = BackwardPass(Y, p, h, W1, W2, conv_flat, MX)
    for key in ['Fs_flat', 'W1', 'W2', 'b1', 'b2']:
        print(f"Max difference for {key} is: {relative_error(grads[key], load_data[f'grad_{key}'])}") 
    
    verify_gradients_with_torch(grads, load_data)
    verify_data_gradients_with_torch(load_data)

#  ----------------------------------------------------------------------------------------------- #

class ConvNet:
    def __init__(self, f=4, nf=10, nh=50):
        self.f = f
        self.nf = nf
        self.nh = nh
        self.n_p = (32 // f) ** 2
        self.input_dim = 3072
        self.output_dim = 10

        self.__initialise_parameters()
    
    def __initialise_parameters(self):
        filter_size = self.f * self.f * 3
        self.Fs = np.random.normal(0, 1/np.sqrt(filter_size), (self.f, self.f, 3, self.nf))
        self.Fs_flat = self.Fs.reshape((filter_size, self.nf), order='C')
        self.W1 = np.random.normal(0, 1/np.sqrt(self.n_p * self.nf), (self.nh, self.n_p * self.nf))
        self.W2 = np.random.normal(0, 1/np.sqrt(self.nh), (self.output_dim, self.nh))
        self.b1 = np.zeros((self.nh, 1))
        self.b2 = np.zeros((self.output_dim, 1))
        self.patch_indices = [[c*32*32 + (i+ii)*32 + (j+jj)
                        for ii in range(self.f)
                        for jj in range(self.f)
                        for c in range(3)]
                        for i in range(0,32,self.f)
                        for j in range(0,32,self.f)]
    
    def forward(self, X):
        n = X.shape[1]
        MX = np.zeros((self.n_p, self.f * self.f * 3, n), dtype=X.dtype)
        for idx, patch_indices in enumerate(self.patch_indices):
            MX[idx, :, :] = X[patch_indices, :] #.reshape((self.f * self.f * 3, n), order='C')
        # MX = np.stack([X[idxs, :] for idxs in self.patch_indices], axis=0)
        # print(MX.shape)
        # print(self.Fs_flat.shape)  
        conv_outputs_mat = np.einsum('ijn, jl->iln', MX, self.Fs_flat, optimize='optimal')
        conv_relu = np.maximum(conv_outputs_mat, 0)  # ReLU activation
        conv_flat = conv_relu.reshape((self.n_p * self.nf, n), order='C')
        # first layer
        x = self.W1 @ conv_flat + self.b1
        h = np.maximum(0, x)
        # second layer
        s = self.W2 @ h + self.b2
        # softmax activation
        p = softmax(s)
        cache = dict(X=X, MX=MX, conv_outputs=conv_outputs_mat, conv_flat=conv_flat, x=x, s=s, h=h)
        return p, cache
    
    def backward(self, Y, P, l2, cache):
        n = Y.shape[1]
        G_batch = -(Y - P)
        dL_dW2 = (G_batch @ cache['h'].T) / n + 2 * l2 * self.W2
        dL_db2 = np.mean(G_batch, axis=1, keepdims=True)
        # back propagation
        G_batch = self.W2.T @ G_batch
        G_batch = G_batch * (cache['h'] > 0)
        # gradient wrt hidden layer
        dL_dW1 = (G_batch @ cache['conv_flat'].T) / n + 2 * l2 * self.W1
        dL_db1 = np.mean(G_batch, axis=1, keepdims=True)
        # gradient wrt conv_flat
        G_batch = self.W1.T @ G_batch
        GG = G_batch.reshape((self.n_p, self.nf, n), order='C')
        GG = GG * (cache['conv_outputs'] > 0)  # ReLU derivative
        MXt = np.transpose(cache['MX'], (1, 0, 2))
        grads_Fs_flat = np.einsum('ijn,jln->il', MXt, GG, optimize='optimal')/ n + 2 * l2 * self.Fs_flat
        grad_Fs = grads_Fs_flat.reshape((self.f, self.f, 3, self.nf), order='C')
        grads = {
            'Fs_flat': grads_Fs_flat,
            'Fs': grad_Fs,
            'W1': dL_dW1,
            'b1': dL_db1,
            'W2': dL_dW2,
            'b2': dL_db2
        }
        return grads
    
    def compute_loss_accuracy(self, X, y):
        P, _ = self.forward(X)
        loss = ComputeLoss(P, y)
        accuracy = ComputeAccuracy(P, y)
        return loss, accuracy
    
    def update_step(self, grads, eta):
        self.Fs -= eta * grads['Fs']
        self.W1 -= eta * grads['W1']
        self.b1 -= eta * grads['b1']
        self.W2 -= eta * grads['W2']
        self.b2 -= eta * grads['b2']
        # update Fs_flat
        self.Fs_flat = self.Fs.reshape((self.f * self.f * 3, self.nf), order='C')
    
def trainNetwork(model, X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, lam, smoothing=False):
    n = X_train.shape[1]
    n_batch = GDParams['n_batch']
    eta_max = GDParams['eta_max']
    eta_min = GDParams['eta_min']
    n_cycles = GDParams['n_cycles']
    n_s = GDParams['n_s']
    log_freq = n_s/2
    steps = [n_s*2**i for i in range(n_cycles)]
    total_steps =  sum(2*s for s in steps)
    Y_train = LabelSmoothing(Y_train, 0.1, smoothing)
    history = {i:[] for i in ['loss_train', 'acc_train', 'loss_val', 'acc_val', 'update_steps', 'learning_rates', 'training_time']}
    update_steps, curr_cycle, cycle_steps = 0, 0, 0
    loss_val, acc_val = model.compute_loss_accuracy(X_val, y_val)
    loss_train, acc_train = model.compute_loss_accuracy(X_train, y_train)
    print(f"Initial validation loss: {loss_val:.4f}, Initial validation accuracy: {acc_val:.4f}")
    print(f"Initial training loss: {loss_train:.4f}, Initial training accuracy: {acc_train:.4f}")
    for k, v in zip(['loss_train', 'acc_train', 'loss_val', 'acc_val', 'update_steps', 'learning_rates'],
                    [loss_train, acc_train, loss_val, acc_val, update_steps, eta_min]):
        history[k].append(v)

    started_time = time.time()
    batch_idx = [(i*n_batch, min((i+1)*n_batch, n)) for i in range(n//n_batch)]
    while update_steps < total_steps:
        perm = np.random.permutation(n)
        X_perm, Y_perm, y_perm = X_train[:, perm], Y_train[:, perm], y_train[perm]
        for start, end in batch_idx:
            if update_steps >= total_steps:
                break
            X_batch = X_perm[:, start:end]
            Y_batch = Y_perm[:, start:end]
            y_batch = y_perm[start:end]
            P, cache = model.forward(X_batch)
            grads = model.backward(Y_batch, P, lam, cache)
            cycle_step = steps[curr_cycle]
            eta = (eta_min + (eta_max-eta_min)*cycle_steps/cycle_step
                   if cycle_steps < cycle_step else
                   eta_max - (eta_max-eta_min)*(cycle_steps-cycle_step)/cycle_step)
            model.update_step(grads, eta)
            cycle_steps += 1
            update_steps += 1
            if update_steps % log_freq == 0:
                ltr, atr = model.compute_loss_accuracy(X_batch, y_batch)
                lval, aval = model.compute_loss_accuracy(X_val, y_val)
                for k, v in zip(['loss_train', 'acc_train', 'loss_val', 'acc_val', 'update_steps', 'learning_rates'],
                        [ltr, atr, lval, aval, update_steps, eta]):
                    history[k].append(v)
                print(f"Update step {update_steps}/{total_steps}, "
                      f"Cycle {curr_cycle+1}/{n_cycles}, "
                      f"Cycle step {cycle_steps}/{steps[curr_cycle]}, "
                      f"Learning rate: {eta:.6f}, "
                      f"Train loss: {ltr:.4f}, Train accuracy: {atr:.4f}, "
                      f"Validation loss: {lval:.4f}, Validation accuracy: {aval:.4f}")
            if cycle_steps >= 2*steps[curr_cycle]:
                curr_cycle += 1
                cycle_steps = 0
                if curr_cycle >= n_cycles:
                    break
    ended_time = time.time()
    training_time = ended_time - started_time
    history['training_time'] = training_time
    print(f"Training completed in {training_time:.2f} seconds.")
    return history

def load_cifar(path, n_train=49000, dtype=np.float32):
    Xtr, Ytr, ytr = [], [], []
    for i in range(1,6):
        X, Y, y = load_batch(i, path, dtype)
        Xtr.append(X); Ytr.append(Y); ytr.append(y)
    Xtr, Ytr, ytr = np.concatenate(Xtr, axis=1), np.concatenate(Ytr, axis=1), np.concatenate(ytr, axis=0)
    Xtr_raw, Ytrain, ytrain = Xtr[:, :n_train], Ytr[:, :n_train], ytr[:n_train]
    Xv_raw, Yv, yv = Xtr[:, n_train:], Ytr[:, n_train:], ytr[n_train:]
    Xt_raw, Yt, yt = load_batch("test_batch", path, dtype)
    Xtr, Xv, Xt = preprocess(Xtr_raw, Xv_raw, Xt_raw)
    # print(f"Loaded {Xtr.shape[1]} train, {Xv.shape[1]} val, {Xt.shape[1]} test samples")
    return Xtr, Ytrain, ytrain, Xv, Yv, yv, Xt, Yt, yt

def load_batch(batch, path="Datasets/cifar-10-batches-py", dtype=np.float32):
    fname = os.path.join(path, batch if isinstance(batch,str) else f"data_batch_{batch}")
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].astype(dtype).T/255.
    y = np.array(d[b'labels'])
    K, n = 10, X.shape[1]
    Y = np.zeros((K, n), dtype=dtype)
    Y[y, np.arange(n)] = 1
    return X, Y, y

def preprocess(Xtr, Xv, Xt):
    mu, sigma = Xtr.mean(1, keepdims=True), Xtr.std(1, keepdims=True)
    return (Xtr-mu)/sigma, (Xv-mu)/sigma, (Xt-mu)/sigma

def save_results(args, GD_Params, lambda_reg, hist, test_acc, arch, logdir="Assignment3/results"):
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f'architecture_{arch}_train_log.txt')
    with open(logfile, 'a') as f:
        f.write("\n"+"="*50+"\nTRAINING SUMMARY\n"+"="*50+"\n")
        f.write(f"Arch {arch}\nTime: {hist['training_time']:.2f}s\n")
        f.write(f"Final val acc: {hist['acc_val'][-1]:.4f}\n")
        f.write(f"Final test acc: {test_acc:.4f}\n")
        f.write(f"Train samples: {args.n_train}\nThreads: {args.num_threads}\n")
        f.write(f"LR: {GD_Params['eta_min']}–{GD_Params['eta_max']}\nCycles: {GD_Params['n_cycles']}\nBatch: {GD_Params['n_batch']}\nLambda: {lambda_reg}\n")

def plotting(history, logdir="Assignment3/results", arch="convnet"):
    os.makedirs(logdir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['update_steps'], history['loss_train'], label='Train Loss')
    plt.plot(history['update_steps'], history['loss_val'], label='Val Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Update Steps')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['update_steps'], history['acc_train'], label='Train Accuracy')
    plt.plot(history['update_steps'], history['acc_val'], label='Val Accuracy')
    plt.xlabel('Update Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Update Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'architecture_{arch}_training_plot.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history['update_steps'], history['learning_rates'], label='Learning Rate')
    plt.xlabel('Update Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Update Steps')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'architecture_{arch}_learning_rate_plot.png'))
    plt.show()

def plot_bar_graph(arch_test_acc, logdir="Assignment3/results"):
    os.makedirs(logdir, exist_ok=True)
    architectures = arch_test_acc['architectures']
    test_acc = arch_test_acc['test_acc']
    training_time = arch_test_acc['training_time']

    plt.figure(figsize=(10, 6))
    plt.bar(architectures, test_acc, color='skyblue', alpha=0.7, label='Test Accuracy', width=0.4)
    plt.xlabel('Architectures')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy per Architecture')
    plt.xticks(architectures)
    
    for i, acc in enumerate(test_acc):
        plt.text(architectures[i], acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'architecture_test_accuracy.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(architectures, training_time, color='lightgreen', alpha=0.7, label='Training Time', width=0.4)
    plt.xlabel('Architectures')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time per Architecture')
    plt.xticks(architectures)
    for i, time in enumerate(training_time):
        plt.text(architectures[i], time + 0.1, f'{time:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'architecture_training_time.png'))
    plt.show()

#  ----------------------------------------------------------------------------------------------- #

def exercise3(Xtr, Ytr, ytr, Xv, Yv, yv, Xt, Yt, yt, architectures, arch_cfg, GD_Params, l2, logdir="Assignment3/results", num_threads=4):
    arch_test_acc = {i:[] for i in ['architectures', 'test_acc', 'training_time']}

    for arch in architectures:
        cfg = arch_cfg[arch]
        log_file = os.path.join(logdir, f'architecture_{arch}_train_log.txt')
        orig_stdout = sys.stdout
        sys.stdout = open(log_file, 'w')

        try:
            print(f"\nExercise 3, arch {arch}, {n_train} train, {num_threads} threads")
            print(f"Training Arch {arch}: f={cfg['f']} nf={cfg['nf']} nh={cfg['nh']}")
            model = ConvNet(f=cfg['f'], nf=cfg['nf'], nh=cfg['nh'])
            hist = trainNetwork(model, Xtr, Ytr, ytr, Xv, Yv, yv, GD_Params, l2, smoothing=False)
            test_loss, test_acc = model.compute_loss_accuracy(Xt, yt)
            arch_test_acc['architectures'].append(f"Arch{arch}")
            arch_test_acc['test_acc'].append(test_acc)
            arch_test_acc['training_time'].append(hist['training_time'])
            print(f"Final test acc: {test_acc:.4f}")
            save_results(
                type('Args', (object,), {
                    "n_train": n_train,
                    "num_threads": num_threads
                })(),  # create a dummy args object
                GD_Params, l2, hist, test_acc, arch
            )
            plotting(hist, logdir, arch)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            sys.stdout.close()
            sys.stdout = orig_stdout
            print(f"Training log saved to {log_file}")
            print("Training completed for architecture", arch)
            print("Plots/results saved in", logdir)

    plot_bar_graph(arch_test_acc)
    print("Exercise 3 done.")

#  ----------------------------------------------------------------------------------------------- #


def exercise4(Xtr, Ytr, ytr, Xv, Yv, yv, Xt, Yt, yt, arch_cfg, GD_Params, l2, logdir="Assignment3/results", num_threads=4):
    for smoothing in [False, True]:
        arch = "ConvNet_smoothing" if smoothing else "ConvNet_no_smoothing"
        log_file = os.path.join(logdir, f'{arch}_train_log.txt')
        orig_stdout = sys.stdout
        sys.stdout = open(log_file, 'w')

        try:
            print(f"\nExercise 4, {arch}, {n_train} train, {num_threads} threads")
            print(f"Training {arch}: f={arch_cfg['f']} nf={arch_cfg['nf']} nh={arch_cfg['nh']}")
            model = ConvNet(f=arch_cfg['f'], nf=arch_cfg['nf'], nh=arch_cfg['nh'])
            hist = trainNetwork(model, Xtr, Ytr, ytr, Xv, Yv, yv, GD_Params, l2, smoothing=smoothing)
            test_loss, test_acc = model.compute_loss_accuracy(Xt, yt)
            print(f"Final test acc: {test_acc:.4f}")
            save_results(
                type('Args', (object,), {
                    "n_train": n_train,
                    "num_threads": num_threads
                })(),  # create a dummy args object
                GD_Params, l2, hist, test_acc, arch
            )
            plotting(hist, logdir, arch)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            sys.stdout.close()
            sys.stdout = orig_stdout
            print(f"Training log saved to {log_file}")
            print("Training completed for architecture", arch)
            print("Plots/results saved in", logdir)
    print("Exercise 4 done.")

#  ----------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    conv_outputs_mat, Fs_flat = exercise1()
    exercise2(conv_outputs_mat)
    architectures = [1, 2, 3, 4]
    n_train = 49000
    data_path = 'Datasets/cifar-10-batches-py'
    num_threads = 4
    logdir = "Assignment3/results"
    os.makedirs(logdir, exist_ok=True)

    # Model configs for each architecture
    arch_cfg = {
        1: dict(f=2, nf=3, nh=50),
        2: dict(f=4, nf=10, nh=50),
        3: dict(f=8, nf=40, nh=50),
        4: dict(f=16, nf=160, nh=50)
    }
    GD_Params = dict(n_batch=100, eta_min=1e-5, eta_max=1e-1, n_s=800, n_cycles=3)
    l2 = 0.003

    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    np.set_printoptions(precision=3)

    # Load data once, use for all architectures
    print(f"Loading data with {n_train} training samples...")
    Xtr, Ytr, ytr, Xv, Yv, yv, Xt, Yt, yt = load_cifar(data_path, n_train)
    print(f"Loaded {Xtr.shape[1]} train, {Xv.shape[1]} val, {Xt.shape[1]} test samples")
    exercise3(Xtr, Ytr, ytr, Xv, Yv, yv, Xt, Yt, yt, 
              architectures, arch_cfg, GD_Params, l2, logdir, num_threads)
    cfg = dict(f=4, nf=40, nh=300)
    l2 = 0.0025
    GD_Params = dict(n_batch=100, eta_min=1e-5, eta_max=1e-1, n_s=800, n_cycles=4)
    exercise4(Xtr, Ytr, ytr, Xv, Yv, yv, Xt, Yt, yt, cfg, GD_Params, l2, logdir, num_threads)
    print("All exercises completed successfully.")
    
