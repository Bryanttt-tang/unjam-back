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
    # basis for A
    # U, S, Vt = svd(A, full_matrices=False)
    # A = U[:, :r_C] 

    # Compute the matrix C = A.T @ B
    C = np.dot(A.T, B)
    
    # Perform SVD on C
    U, s, Vh = svd(C)
    
    # The singular values (s) are the cosines of the principal angles
    principal_angles = np.arccos(np.clip(s, -1, 1))  # Clip to handle numerical issues
    
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
    friedrichs_angle = np.max(angles)
    
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
