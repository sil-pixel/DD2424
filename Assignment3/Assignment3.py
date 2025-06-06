from json import load
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sympy import true
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Assignment1.Assignment1 import softmax
from torch_gradient_computations import verify_data_gradients_with_torch, verify_gradients_with_torch

# set seed = 42
np.random.seed(42)

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


if __name__ == "__main__":
    conv_outputs_mat, Fs_flat = exercise1()
    exercise2(conv_outputs_mat)












