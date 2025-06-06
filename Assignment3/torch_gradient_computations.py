import torch
import torch.nn.functional as F
import numpy as np

def ComputeConvGradsWithTorch(X, Y, Fs, W1, W2, b1, b2, use_label_smoothing=False, eps=0.1):
    """
    Compute gradients for 3-layer network: Conv -> FC -> FC with PyTorch
    
    Args:
        X: Input images (3072, n) - flattened CIFAR images
        Y: One-hot labels (10, n)
        Fs: Conv filters (4, 4, 3, 2)
        W1, W2: FC layer weights
        b1, b2: FC layer biases
        use_label_smoothing: Whether to apply label smoothing
        eps: Label smoothing parameter
    """
    
    # Convert to torch tensors
    n = X.shape[1]
    f = Fs.shape[0]  # 4
    nf = Fs.shape[3]  # 2
    
    # Reshape input to image format
    X_torch = torch.from_numpy(X.T.reshape(n, 3, 32, 32))  # (n, 3, 32, 32)
    
    # Convert parameters to torch tensors with gradients
    Fs_torch = torch.from_numpy(Fs).permute(3, 2, 0, 1)  # (nf, 3, f, f) = (2, 3, 4, 4)
    Fs_torch.requires_grad_(True)
    
    W1_torch = torch.tensor(W1, requires_grad=True)  # (10, 128)
    W2_torch = torch.tensor(W2, requires_grad=True)  # (10, 128)
    b1_torch = torch.tensor(b1, requires_grad=True)
    b2_torch = torch.tensor(b2, requires_grad=True)

    
    # b1_torch = torch.from_numpy(b1.squeeze()).float()
    # b1_torch.requires_grad_(True)
    
    # b2_torch = torch.from_numpy(b2.squeeze()).float()
    # b2_torch.requires_grad_(True)
    
    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)
    apply_conv = torch.nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=f, stride=f, bias=False)
    apply_conv.weight = torch.nn.Parameter(Fs_torch)

    # Forward pass
    # Convolution with stride=4 (same as filter width)
    conv_out = apply_conv(X_torch)  # (n, nf, 8, 8)
    # conv_out = F.conv2d(X_torch, Fs_torch, stride=f)  # (n, nf, 8, 8)
    # ReLU activation
    conv_relu = apply_relu(conv_out) # (n, nf, 8, 8)
    # Flatten for FC layers
    conv_flat = conv_relu.view(n, -1).T  # (128, n) to match your format
    # First FC layer
    h = apply_relu(W1_torch @ conv_flat + b1_torch.unsqueeze(1))  # (10, n)
    # Second FC layer
    scores = W2_torch @ h + b2_torch.unsqueeze(1)  # (10, n)
    # Softmax
    P = apply_softmax(scores)  # (10, n)
    
    # Prepare labels
    if use_label_smoothing:
        # Apply label smoothing
        Y_smooth = torch.from_numpy(Y).float()
        K = Y.shape[0]
        for i in range(n):
            true_class = torch.argmax(Y_smooth[:, i])
            Y_smooth[:, i] = eps / (K - 1)
            Y_smooth[true_class, i] = 1 - eps
        targets = Y_smooth
    else:
        targets = torch.from_numpy(Y).float()
    
    # Cross-entropy loss (manual implementation to match your format)
    loss = -torch.mean(targets * torch.log(P + 1e-15)) 
    
    # Backward pass
    loss.backward()
    
    # Extract gradients
    grads = {}
    
    # Convert filter gradients back to original format
    Fs_grad = apply_conv.weight.grad.permute(2, 3, 1, 0).detach().cpu().numpy()  # (f, f, 3, nf)
    grads['Fs_flat'] = Fs_grad.reshape(f*f*3, nf, order='C')  # (48, 2)
    
    grads['W1'] = W1_torch.grad.numpy()
    grads['W2'] = W2_torch.grad.numpy()
    grads['b1'] = b1_torch.grad.numpy()    #.reshape(-1, 1)
    grads['b2'] = b2_torch.grad.numpy()   #.reshape(-1, 1)
    
    return grads

def verify_data_gradients_with_torch(load_data):
    """
    Verify data gradients against PyTorch implementation
    """
    X = load_data['X']
    Y = load_data['Y']
    Fs = load_data['Fs']
    W1 = load_data['W1']
    W2 = load_data['W2']
    b1 = load_data['b1']
    b2 = load_data['b2']
    
    # Compute gradients with PyTorch
    torch_grads = ComputeConvGradsWithTorch(X, Y, Fs, W1, W2, b1, b2, 
                                          use_label_smoothing=False, eps=0.1)
    
    # Load reference gradients
    ref_grads = {
        'Fs_flat': load_data['grad_Fs_flat'],
        'W1': load_data['grad_W1'],
        'W2': load_data['grad_W2'],
        'b1': load_data['grad_b1'],
        'b2': load_data['grad_b2']
    }

    print("PyTorch vs Reference Gradients:")
    for key in ['Fs_flat', 'W1', 'W2', 'b1', 'b2']:
        error = relative_error(torch_grads[key], ref_grads[key])
        print(f"Max difference for {key}: {error}")
    
    return torch_grads

def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a), np.abs(b) + 1e-8))
    

def verify_gradients_with_torch(grads, load_data):
    X = load_data['X']
    Y = load_data['Y']
    Fs = load_data['Fs']
    W1 = load_data['W1']
    W2 = load_data['W2']
    b1 = load_data['b1']
    b2 = load_data['b2']
    
    # Compute gradients with PyTorch
    torch_grads = ComputeConvGradsWithTorch(X, Y, Fs, W1, W2, b1, b2, 
                                          use_label_smoothing=False, eps=0.1)
    
    print("PyTorch vs your Gradients:")
    for key in ['Fs_flat', 'W1', 'W2', 'b1', 'b2']:
        error = relative_error(torch_grads[key], grads[key])
        print(f"Max difference with torch for {key}: {error}")
    