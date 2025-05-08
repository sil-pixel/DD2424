import numpy as np
from torch_gradient_computations_column_wise import ComputeGradsWithTorch

m = 100
eta = 0.001
seq_length = 25

# Set the random seed for reproducibility
rng = np.random.default_rng(2025)
BitGen = type(rng.bit_generator)
seed = 2025
rng.bit_generator.state = BitGen(seed).state


def softmax(s):
    # to avoid inf/inf, for stability remove a constant value from exponentiation
    exp_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    P = exp_s/ np.sum(exp_s, axis=0, keepdims=True)
    return P

def readFile(book_fname):
    fid = open(book_fname, 'r')
    book_data = fid.read()
    fid.close()
    # Count the number of characters in the book
    #num_chars = len(book_data)
    #print('Number of characters in the book:', num_chars)
    unique_chars = list(set(book_data))
    #print('Number of unique characters:', len(unique_chars))
    #print('Unique characters:', unique_chars)
    return book_data, unique_chars

def listToDict(unique_chars):
    # Create a dictionary to map each character to an index
    char_to_ind = {char: index for index, char in enumerate(unique_chars)}
    #print('Character to index mapping:', char_to_ind)

    # Create a dictionary to map each index to a character
    ind_to_char = {index: char for index, char in enumerate(unique_chars)}
    #print('Index to character mapping:', ind_to_char)
    return char_to_ind, ind_to_char

def networkInitialize(m, K):
    RNN = {}
    RNN['b'] = np.zeros((m, 1))
    RNN['c'] = np.zeros((K, 1))
    RNN['U'] = (1/ np.sqrt(2*K)) * rng.standard_normal((m, K))
    RNN['W'] = (1/ np.sqrt(2*m)) * rng.standard_normal((m, m))
    RNN['V'] = (1/ np.sqrt(m)) * rng.standard_normal((K, m))
    return RNN

def applyNetwork(RNN, h, x):
    # Compute the next hidden state and output
    a = np.dot(RNN['U'], x) + np.dot(RNN['W'], h) + RNN['b']
    h = np.tanh(a)
    o = np.dot(RNN['V'], h) + RNN['c']
    p = softmax(o)
    return p, h

def sampleNextCharacter(p):
    # Sample the next character from the output distribution
    cp = np.cumsum(p, axis=0)
    a_rand = rng.uniform(size=1)
    ii = np.argmax(cp > a_rand)
    return ii


def synthesizeText(RNN, h0, x0, n):
    h = h0
    x = np.zeros((K, 1))
    x = x0
    Y = np.zeros((K, n))

    for t in range(n):
        p, h = applyNetwork(RNN, h, x)
        ii = sampleNextCharacter(p)
        # Update the input for the next time step based on the sampled character
        x = np.zeros((K, 1))
        x[ii] = 1
        Y[ii, t] = 1
    # return one-hot encoding of the sampled character
    return Y

def generateText(RNN, h0, x0, ind_to_char):
    # Generate a sequence of characters
    Y = synthesizeText(RNN, h0, x0, seq_length)
    # Convert the one-hot encoding back to characters
    generated_text = ''.join([ind_to_char[np.argmax(Y[:, i])] for i in range(seq_length)])
    # Print the generated text
    print(generated_text)
    return generated_text
    
def forwardPass(RNN, X_one_hot, Y_one_hot, h0):
    loss = 0
    n = X_one_hot.shape[1]
    h = h0
    output = np.zeros((K, n))
    P = np.zeros((K, n))
    for t in range(n):
        p, h = applyNetwork(RNN, h, X_one_hot[:, t])
        loss -= np.dot(Y_one_hot[:, t].T, np.log(p))
        output[:, t] = h
        P[:, t] = p
    return loss, output, P

def backwardPass(RNN, X_one_hot, Y_one_hot, h0):
    n = X_one_hot.shape[1]
    grads = {}
    grads['U'] = np.zeros_like(RNN['U'])
    grads['W'] = np.zeros_like(RNN['W'])
    grads['V'] = np.zeros_like(RNN['V'])
    grads['b'] = np.zeros_like(RNN['b'])
    grads['c'] = np.zeros_like(RNN['c'])
    loss, H, P = forwardPass(RNN, X_one_hot, Y_one_hot, h0)
    for t in range(n-1, 1, -1):
        dL_do = - (Y_one_hot[:, t] - P[:, t]).T
        grads['c'] += dL_do.T
        grads['V'] += np.dot(dL_do.T, H[:, t].T)
        dL_dh = np.dot(dL_do, RNN['V'])
        if t != n-1:
            dL_dh += dL_at @ grads['W'] 
        grads['b'] += (1 - H[:, t]**2).reshape(-1,1) * dL_dh.T
        dL_at = dL_dh * (1 - H[:, t]**2)
        if t != 1:
            h_prev = H[:, t-1]
        else:
            h_prev = h0
        grads['W'] += dL_at.T @ h_prev.T
        grads['U'] += dL_at.T @ X_one_hot[:, t].T
    
    H = H[:, 1:]
    return grads, loss, H



def checkGradsWithTorch(RNN, X_one_hot, Y_one_hot, h0):
    # check relative error between my grads and torch grads
    grads = {}
    grads, loss, H = backwardPass(RNN, X_one_hot, Y_one_hot, h0)
    torch_grads = ComputeGradsWithTorch(X_one_hot, Y_one_hot, h0, RNN)
    for kk in grads.keys():
        ga = grads[kk]
        gn = torch_grads[kk]
        num = np.abs(ga - gn)
        denom = np.maximum(eps, np.abs(ga) + np.abs(gn))
        rel_error = num / denom
        rel_errors[kk] = rel_error
    print(rel_errors)
    return rel_errors



if __name__ == "__main__":
    book_data, unique_chars = readFile('goblet_book.txt')
    char_to_ind, ind_to_char = listToDict(unique_chars)
    K = len(unique_chars)
    RNN = networkInitialize(m, K)
    h0 = np.zeros((m, 1))
    x0_index = char_to_ind[unique_chars[0]]
    x0 = np.zeros((K, 1))
    x0[x0_index] = 1
    generated_text = generateText(RNN, h0, x0, ind_to_char)
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]
    X_one_hot = np.zeros((K, seq_length))
    Y_one_hot = np.zeros((K, seq_length))
    for i in range(seq_length):
        X_one_hot[:, i] = np.eye(K)[char_to_ind[X_chars[i]]]
        Y_one_hot[:, i] = np.eye(K)[char_to_ind[Y_chars[i]]]
    checkGradsWithTorch(RNN, X_one_hot, Y_one_hot, h0)






