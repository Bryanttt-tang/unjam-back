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
    # Compute the matrix C = A.T @ B
    C = np.dot(A.T, B)
    
    # Perform SVD on C
    U, s, Vh = svd(C)
    
    # The singular values (s) are the cosines of the principal angles
    principal_angles = np.arccos(np.clip(s, -1, 1))  # Clip to handle numerical issues
    
    return principal_angles

def friedrichs_angle(A, B, tol=1e-10):
    """
    Compute the Friedrichs angle between two subspaces, ignoring intersection directions.

    Args:
        A (numpy.ndarray): An m x n1 matrix where columns form an orthonormal basis for subspace A.
        B (numpy.ndarray): An m x n2 matrix where columns form an orthonormal basis for subspace B.
        tol (float): Tolerance for considering an angle as zero (intersection).

    Returns:
        float: The Friedrichs angle in radians.
    """
    angles = principal_angles(A, B)
    # Ignore angles close to zero (intersection directions)
    nonzero_angles = angles[angles > tol]
    if nonzero_angles.size == 0:
        # Subspaces are identical or only intersect
        return 0.0
    friedrichs_angle = np.max(nonzero_angles)
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
