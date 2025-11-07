import numpy as np
from scipy.linalg import svd

def principal_angles(A, B):
    """
    Compute the principal angles between two subspaces spanned by columns of A and B.

    Args:
        A (numpy.ndarray): An m x n1 matrix where columns form an orthonormal basis for subspace A.
        B (numpy.ndarray): An m x n2 matrix where columns form an orthonormal basis for subspace B.

    Returns:
        numpy.ndarray: An array containing the principal angles in radians.
    """
    # 1) basis for A
    r_A = np.linalg.matrix_rank(A)
    print('rank of A:', r_A)
    print('A.shape before SVD:', A.shape)
    U, S, Vt = svd(A, full_matrices=True)
    print('singular values of A:', S)
    A = U[:, :r_A]
    print('A.shape after SVD:', A.shape) 
    print('check orthogonality:',np.linalg.norm(A.T @ A - np.eye(r_A)))  # should be ~0

    # 2) basis for B
    r_B = np.linalg.matrix_rank(B)
    print('rank of B:', r_B)
    print('B.shape before SVD:', B.shape)
    U, S, Vt = svd(B, full_matrices=True)
    print('singular values of B:', S)
    print('Vt:', Vt)
    B = Vt.T[:,  r_B:]
    print('B.shape after SVD:', B.shape)


    # print(np.linalg.norm(A.T @ A - np.eye(r_C)))  # should be ~0
    # 2) dimension check
    # print("A.shape =", A.shape)
    # print('rank of A:', np.linalg.matrix_rank(A))
    

    # Compute the matrix C = A.T @ B
    C = A.T @ B
    
    # Perform SVD on C
    U, s, Vh = svd(C)
    print('singular values:', s)
    
    # The singular values (s) are the cosines of the principal angles
    principal_angles = np.arccos(np.clip(s, -1, 1))  # Clip to handle numerical issues
    # principal_angles = np.arccos(s)  # Clip to handle numerical issues
    res = np.linalg.norm((np.eye(72) - B @ B.T) @ A, ord='fro')
    print("Frobenius norm of residual (should be ~0 if A ⊆ B):", res)
    
    return principal_angles

def friedrichs_angle(A, B):
    """
    Compute the Friedrichs angle between two subspaces.

    Args:
        A (numpy.ndarray): An m x n1 matrix where columns form an orthonormal basis for subspace A.
        B (numpy.ndarray): An m x n2 matrix where columns form an orthonormal basis for subspace B.

    Returns:
        float: The Friedrichs angle in radians.
    """
    # Get the principal angles
    angles = principal_angles(A, B)
    
    # The Friedrichs angle is the largest principal angle
    friedrichs_angle = np.min(angles)
    
    return friedrichs_angle

# Example usage
if __name__ == "__main__":
    # Define two random orthonormal matrices A and B (using QR decomposition for example)
    m = 5  # Dimension of the Hilbert space
    n1 = 3  # Dimension of subspace A
    n2 = 2  # Dimension of subspace B

    # Generate random matrices
    A_rand = np.random.randn(m, n1)
    B_rand = np.random.randn(m, n2)

    # Orthonormalize the columns of A_rand and B_rand using QR decomposition
    A, _ = np.linalg.qr(A_rand)
    B, _ = np.linalg.qr(B_rand)

    # Compute the Friedrichs angle
    theta = friedrichs_angle(A, B)
    
    print(f"The Friedrichs angle between the two subspaces is: {np.degrees(theta):.2f} degrees")
