import numpy as np
import matplotlib.pyplot as plt

from torch_gradient_computations_column_wise import ComputeGradsWithTorch

# Set random seed for reproducibility
rng = np.random.default_rng(42)
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

# ------------------ Utility Functions ------------------

def softmax(s):
    """Compute softmax values for each set of scores in s."""
    # Numerical stability: subtract max value before exponentiating
    exp_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)

def oneHotChar(idx, K):
    """Convert character index to one-hot vector."""
    vec = np.zeros((K, 1))
    vec[idx] = 1
    return vec

def oneHotEncode(chars, K, char_to_ind):
    """Convert a sequence of characters to one-hot encoded vectors."""
    onehots = np.zeros((K, len(chars)))
    for i, ch in enumerate(chars):
        onehots[char_to_ind[ch], i] = 1
    return onehots

def sampleNextCharacter(p):
    """Sample the next character index based on probability distribution p."""
    cp = np.cumsum(p, axis=0)
    a = rng.uniform(size=1)
    return np.argmax(cp > a)

# ------------------ RNN Functions ------------------

def forwardPass(RNN, X, Y, h0):
    """
    Forward pass through the RNN.
    
    Args:
        RNN: Dictionary containing network parameters
        X: Input data (K x seq_length)
        Y: Target data (K x seq_length)
        h0: Initial hidden state
        
    Returns:
        loss: Cross-entropy loss
        caches: Cached values for backward pass
    """
    m, K, n = RNN['W'].shape[0], X.shape[0], X.shape[1]
    
    H = np.zeros((m, n + 1))
    A = np.zeros((m, n))
    P = np.zeros((K, n))
    loss = 0
    H[:, [0]] = h0

    for t in range(n):
        A[:, [t]] = RNN['W'] @ H[:, [t]] + RNN['U'] @ X[:, [t]] + RNN['b']
        H[:, [t + 1]] = np.tanh(A[:, [t]])
        O = RNN['V'] @ H[:, [t + 1]] + RNN['c']
        P[:, [t]] = softmax(O)
        loss -= np.log(np.dot(Y[:, [t]].T, P[:, [t]]) + 1e-12).item()

    return loss / n, {'H': H, 'A': A, 'P': P, 'X': X}

def backwardPass(RNN, X, Y, h0):
    """
    Backward pass through the RNN using backpropagation through time.
    
    Args:
        RNN: Dictionary containing network parameters
        X: Input data (K x seq_length)
        Y: Target data (K x seq_length)
        h0: Initial hidden state
        
    Returns:
        grads: Gradients for all parameters
        h_last: Last hidden state
        loss: Cross-entropy loss
    """
    loss, cache = forwardPass(RNN, X, Y, h0)
    H, A, P, n = cache['H'], cache['A'], cache['P'], X.shape[1]

    dW, dU, dV = np.zeros_like(RNN['W']), np.zeros_like(RNN['U']), np.zeros_like(RNN['V'])
    db, dc = np.zeros_like(RNN['b']), np.zeros_like(RNN['c'])
    dh_next = np.zeros((RNN['W'].shape[0], 1))

    for t in reversed(range(n)):
        dp = P[:, [t]] - Y[:, [t]]
        dV += dp @ H[:, [t + 1]].T
        dc += dp
        dh = RNN['V'].T @ dp + dh_next
        da = (1 - H[:, [t + 1]] ** 2) * dh
        dW += da @ H[:, [t]].T
        dU += da @ X[:, [t]].T
        db += da
        dh_next = RNN['W'].T @ da

    grads = { 'W': dW / n, 'U': dU / n, 'V': dV / n, 'b': db / n, 'c': dc / n }
    return grads, H[:, [-1]], loss
 
def synthesizeSequence(RNN, h0, x0, n, K, ind_to_char, temperature=1.0):
    """
    Synthesize a sequence of characters using the RNN.
    
    Args:
        RNN: Dictionary containing network parameters
        h0: Initial hidden state
        x0: Initial input character (one-hot encoded)
        n: Length of sequence to generate
        K: Number of unique characters
        ind_to_char: Mapping from indices to characters
        temperature: Temperature parameter for sampling
        
    Returns:
        generated_text: Generated text as a string
    """
    h = h0.copy()
    x = x0.copy()
    indices = []
    
    for t in range(n):
        # Forward pass
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        h = np.tanh(a)
        o = RNN['V'] @ h + RNN['c']
        
        # Apply temperature to logits
        if temperature != 1.0:
            o = o / temperature
            
        p = softmax(o)
        
        # Sample next character
        idx = sampleNextCharacter(p)
        indices.append(idx)
        
        # Update input for next step
        x = np.zeros((K, 1))
        x[idx] = 1
    
    # Convert indices to characters
    generated_text = ''.join(ind_to_char[idx] for idx in indices)
    return generated_text

# ------------------ Gradient Checking ------------------

def checkGradsWithTorch(RNN, X, Y, h0):
    """Compare analytical gradients with PyTorch gradients."""
    print("\n[Gradient Check]")
    analytic_grads, _, _ = backwardPass(RNN, X, Y, h0)
    torch_grads = ComputeGradsWithTorch(X, Y, h0, RNN)

    eps = 1e-8
    for key in RNN.keys():
        print(f"Key: {key}")
        ga = analytic_grads[key]
        gn = torch_grads[key]
        # Relative error
        num = np.abs(ga - gn)
        denom = np.maximum(eps, np.abs(ga) + np.abs(gn))
        rel_error = num / denom
        print(f"Mean relative error in {key}: {np.mean(rel_error):.2e}\n")

# ------------------ Adam Optimizer ------------------

def adamUpdate(RNN, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update RNN parameters using Adam optimizer.
    
    Args:
        RNN: Dictionary containing network parameters
        grads: Dictionary containing gradients
        m: First moment estimate
        v: Second moment estimate
        t: Iteration count (starting from 1)
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small value for numerical stability
        
    Returns:
        RNN: Updated parameters
        m: Updated first moment
        v: Updated second moment
    """
    # Update parameters with Adam
    for key in RNN:
        # Update biased first moment estimate
        m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
        
        # Update biased second raw moment estimate
        v[key] = beta2 * v[key] + (1 - beta2) * (grads[key]**2)
        
        # Correct bias in first moment
        m_hat = m[key] / (1 - beta1**t)
        
        # Correct bias in second moment
        v_hat = v[key] / (1 - beta2**t)
        
        # Update parameters
        RNN[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return RNN, m, v

# ------------------ Training Function ------------------

def trainRNN(RNN, book_data, char_to_ind, ind_to_char, h0, num_epochs, seq_length, learning_rate, print_every=10000):
    """
    Train the RNN on book data.
    
    Args:
        RNN: Dictionary containing network parameters
        book_data: Text data for training
        char_to_ind: Mapping from characters to indices
        ind_to_char: Mapping from indices to characters
        h0: Initial hidden state template (for resetting)
        num_epochs: Number of epochs
        seq_length: Length of training sequences
        learning_rate: Learning rate for Adam
        print_every: How often to print progress
        
    Returns:
        RNN: Trained network parameters
        loss_history: History of smoothed loss values
    """
    K = len(char_to_ind)
    e = 0  # Position in book
    smooth_loss = None
    loss_history = []
    hurrah_count = 0
    # Initialize Adam optimizer parameters
    m = {k: np.zeros_like(v) for k, v in RNN.items()}
    v = {k: np.zeros_like(v) for k, v in RNN.items()}
    t = 0  # Adam iteration counter
    
    h_prev = h0.copy()
    
    # Sample text at beginning for comparison
    print("Initial text sample (before training):")
    initial_x = oneHotChar(char_to_ind[book_data[0]], K)
    initial_text = synthesizeSequence(RNN, h0, initial_x, 200, K, ind_to_char)
    print(initial_text)
    print("\nStarting training...\n")
    
    synthesized_texts = []  # Store synthesized texts for reporting
    RNN_best = None
    loss_best = float('inf')
    # Training loop
    batch_size = 50000
    num_updates = batch_size * num_epochs
    print(f"Number of updates: {num_updates}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        for update in range(1, batch_size + 1):
            # Check if we're at the end of the book
            if e + seq_length + 1 >= len(book_data):
                e = 0  # Reset position
                h_prev = h0.copy()  # Reset hidden state
            
            # Get the next batch of sequences
            X_chars = book_data[e:e+seq_length]
            Y_chars = book_data[e+1:e+seq_length+1]
            
            # Convert to one-hot encoding
            X = oneHotEncode(X_chars, K, char_to_ind)
            Y = oneHotEncode(Y_chars, K, char_to_ind)
            
            # Compute gradients and loss
            grads, h_prev, loss = backwardPass(RNN, X, Y, h_prev)
            
            # Apply Adam update
            t += 1  # Increment Adam iteration counter
            RNN, m, v = adamUpdate(RNN, grads, m, v, t, learning_rate)
            
            # Update smooth loss
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            
            loss_history.append(smooth_loss)
            
            # Print progress and synthesize text
            if update == 1 or update % print_every == 0 or update in [1000, 4000, 30000, 150000]:
                update_iter = update + epoch * batch_size
                print(f"Iteration: {update_iter}, Smooth Loss: {smooth_loss:.6f}")
                
                # Generate sample text
                x0 = X[:, 0:1]  # Use first character of current sequence as seed
                generated_text = synthesizeSequence(RNN, h_prev, x0, 200, K, ind_to_char)
                print(f"Sample text:\n{generated_text}\n")

                # call hurrah when we see Harry, Potter, etc characters and places
                hurrah_text = ['Harry', 'Potter', 'Hermoine', 'Ron', 'Dumbledore', 'Voldemort', 'Hogwarts', 'Malfoy', 'Snape', 'Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff', 'Hagrid', 'Dementors', 'Quidditch']
                if any(text in generated_text for text in hurrah_text):
                    print("Hurrah!")
                    hurrah_count += 1
                
                # Save sample for later reporting
                if update == 1 or update % 10000 == 0:
                    synthesized_texts.append((update_iter, generated_text))
            
            # Move to next position in the book
            e += seq_length

            if smooth_loss < loss_best:
                loss_best = smooth_loss
                RNN_best = RNN.copy()

    
    print(f"Hurrah count: {hurrah_count}")
    # Generate a longer text sample from the trained model
    print(f"Best loss: {loss_best:.6f}")
    final_sample = synthesizeSequence(RNN_best, h0, initial_x, 1000, K, ind_to_char)
    
    return loss_history, synthesized_texts, final_sample

# ------------------ Main Function ------------------

def main():
    # Load book data
    with open("goblet_book.txt", "r", encoding="utf-8") as f:
        book_data = f.read()
    
    # Create character mappings
    unique_chars = list(set(book_data))
    K = len(unique_chars)
    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    
    print(f"Text length: {len(book_data)} characters")
    print(f"Unique characters: {K}")
    
    # Model hyperparameters
    m = 100  # Hidden layer size
    seq_length = 25
    learning_rate = 0.001
    
    # Initialize RNN parameters
    h0 = np.zeros((m, 1))
    RNN = {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m, K)),
        'W': (1/np.sqrt(2*m)) * rng.standard_normal(size=(m, m)),
        'V': (1/np.sqrt(m)) * rng.standard_normal(size=(K, m))
    }
    
    # Gradient checking with smaller network
    print("Performing gradient check with smaller network...")
    m_check = 10
    RNN_check = {
        'b': np.zeros((m_check, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal(size=(m_check, K)),
        'W': (1/np.sqrt(2*m_check)) * rng.standard_normal(size=(m_check, m_check)),
        'V': (1/np.sqrt(m_check)) * rng.standard_normal(size=(K, m_check))
    }
    h0_check = np.zeros((m_check, 1))
    
    # Use first seq_length characters for gradient checking
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]
    X_check = oneHotEncode(X_chars, K, char_to_ind)
    Y_check = oneHotEncode(Y_chars, K, char_to_ind)
    
    # Check gradients
    checkGradsWithTorch(RNN_check, X_check, Y_check, h0_check)
    
    # Train the RNN
    print("\nStarting RNN training...")
    
    loss_history, synthesized_texts, final_sample = trainRNN(RNN, book_data, char_to_ind, ind_to_char, h0, 3, seq_length, learning_rate)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history)
    plt.title("Smooth Loss During Training")
    plt.xlabel("Update Steps")
    plt.ylabel("Smooth Loss")
    plt.grid(True)
    plt.savefig("smooth_loss_evolution.png")
    plt.show()
    
    # Save synthesis evolution to file
    with open("synthesis_evolution.txt", "w", encoding="utf-8") as f:
        for update, text in synthesized_texts:
            f.write(f"Iteration {update}:\n{text}\n\n")
    
    # Save final long sample
    with open("final_synthesis_1000chars.txt", "w", encoding="utf-8") as f:
        f.write(final_sample)
    
    print("\nTraining complete!")
    print("Loss history plotted and saved to 'smooth_loss_evolution.png'")
    print("Synthesis evolution saved to 'synthesis_evolution.txt'")
    print("Final 1000-character sample saved to 'final_synthesis_1000chars.txt'")

if __name__ == "__main__":
    main()