import numpy as np

# Set dimensions according to d > n > r condition
d = 200  # Feature space dimensionality
n = 100   # Number of observations
r = 5   # Rank (/row ?)

# Generate a low-rank matrix representation of the training data (U and V)
U = np.random.randn(n, r) * np.sqrt(1/r)
'''
Generate a matrix V of dimensions d x r with orthonormal columns. 
One way to compute that is by generating a random matrix, say C of 
dimensions n x d (say Gaussian). Take its SVD decomposition as  
numpy.linalg.svd. this will create the decomposition C = USV', 
where V is a d x d matrix. V will have all the columns orthonormal. 
Keep r of them and this will be your V.
'''


C = np.random.randn(n, d)
_, _, V = np.linalg.svd(C)

V = np.transpose(V)
V = np.delete(V, np.s_[r:], axis = 1)


# Generate the true parameter z*
z_star = np.random.randn(r)

# Compute A and b
A = U @ np.transpose(V)
b = U @ z_star

# Define the IST algorithm for linear regression
def iterative_soft_thresholding(A, b, lambda_reg, iterations, learning_rate):
    """
    Perform the IST algorithm to solve Ax = b with L2 regularization and a learning rate.
    
    Args:
    - A: The design matrix
    - b: The target vector
    - lambda_reg: The regularization parameter
    - iterations: Number of iterations to run IST
    - learning_rate: The learning rate for the gradient descent step
    
    Returns:
    - x: The solution vector after IST
    """
    x = np.zeros(A.shape[1])
    for _ in range(iterations):
        # Compute the gradient
        grad = A.T @ (A @ x - b)
        
        # Soft thresholding
        x_thresholded = np.sign(x) * np.maximum(np.abs(x) - lambda_reg, 0)
        
        # Gradient descent step with learning rate
        x = x_thresholded - learning_rate * grad
        
        # Clamp the values of x to prevent overflow
        x = np.clip(x, -1e10, 1e10)
    return x
# Set regularization parameter and iterations
lambda_reg = 0.1
iterations = 100


# Set learning rate
learning_rate = 1e-6

# Run the IST algorithm with the updated learning rate
x_ist = iterative_soft_thresholding(A, b, lambda_reg, iterations, learning_rate)

# Compute the Mean Squared Error and the error || Vx - z* ||^2 again
mse = np.mean((A @ x_ist - b) ** 2)
error = np.linalg.norm(V @ x_ist - z_star) ** 2

print(mse, error)

# -----------
# Sanity Check

def distributed_ist(A, b, lambda_reg, local_iterations, learning_rate, p):
    """
    Distributed IST algorithm where the matrix A is divided among p workers.

    Args:
    - A: The design matrix
    - b: The target vector
    - lambda_reg: The regularization parameter
    - local_iterations: Number of local iterations for each worker
    - learning_rate: The learning rate for the gradient descent step
    - p: Number of workers

    Returns:
    - aggregated_x: The aggregated solution vector from all workers
    """
    n, d = A.shape
    # Split A and b into p parts
    A_splits = np.array_split(A, p)
    b_splits = np.array_split(b, p)
    
    # Each worker solves its local problem
    x_locals = [iterative_soft_thresholding(A_split, b_split, lambda_reg, local_iterations, learning_rate) 
                for A_split, b_split in zip(A_splits, b_splits)]

    # Aggregate the results by averaging
    aggregated_x = np.mean(x_locals, axis=0)

    return aggregated_x

# Parameters for distributed IST
p = n // 100  # Number of workers (each with 100 entries)
local_iterations = 1000  # Large number of local iterations
global_iterations = 1    # Number of global iterations (sanity check)

# Run distributed IST algorithm
aggregated_x = distributed_ist(A, b, lambda_reg, local_iterations, learning_rate, p)

# Compute the Mean Squared Error and the error || Vx - z* ||^2 for the aggregated result
mse_aggregated = np.mean((A @ aggregated_x - b) ** 2)
error_aggregated = np.linalg.norm(V @ aggregated_x - z_star) ** 2

print(mse_aggregated, error_aggregated)


