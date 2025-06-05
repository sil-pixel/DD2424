import numpy as np
import matplotlib.pyplot as plt


debug_file = 'debug_conv_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']

n = X.shape[1] # 5
f = Fs.shape[0] # 4
nf = Fs.shape[3] # 2

n_patches = (32//f)**2
MX = np.zeros((n_patches, (f*f*3), n))

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
    print(np.allclose(X_conv, conv_outputs))   # True

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
    print(np.allclose(conv_outputs_mat, conv_outputs_flat))   # True


if __name__ == "__main__":
    exercise1()










