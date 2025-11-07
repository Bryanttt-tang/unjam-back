import numpy as np
from scipy.linalg import svd

def orthonormal_basis_from_matrix(M, tol=None):
    """Return an orthonormal basis (columns) for col(M) and numeric rank r."""
    U, S, Vt = svd(M, full_matrices=False)
    if tol is None:
        tol = max(M.shape) * np.finfo(S.dtype).eps * (S[0] if S.size>0 else 1.0)
    r = int(np.sum(S > tol))
    return U[:, :r], r, S

def nullspace_basis_via_svd(M, tol=None):
    """Return orthonormal basis for ker(M) (shape n x (n-r))."""
    U, S, Vt = svd(M, full_matrices=False)
    if tol is None:
        tol = max(M.shape) * np.finfo(S.dtype).eps * (S[0] if S.size>0 else 1.0)
    r = int(np.sum(S > tol))
    V = Vt.T
    V_null = V[:, r:]   # RIGHT: columns r..end span kernel
    return V_null, (M.shape[1] - r), S

def principal_angles(A_in, B_in, verbose=True):
    """
    Compute principal angles (radians) between subspaces spanned by columns of A_in and B_in.
    A_in, B_in may be arbitrary matrices (not necessarily orthonormal bases).
    Returns: angles (ascending), cosines (descending), and orthonormal bases A,B used.
    """
    # 1) orthonormal bases for the two subspaces (SVD-based)
    A, rA, S_A = orthonormal_basis_from_matrix(A_in)
    B, rB, S_B = orthonormal_basis_from_matrix(B_in)
    if verbose:
        print("A basis shape, rank:", A.shape, rA)
        print("B basis shape, rank:", B.shape, rB)
        print("singular values of A (top few):", S_A[:min(10, S_A.size)])
        print("singular values of B (top few):", S_B[:min(10, S_B.size)])

    if rA == 0 or rB == 0:
        # one subspace is trivial
        return np.array([]), np.array([]), A, B

    # 2) compute cross matrix and its SVD
    M = A.T @ B                 # rA x rB
    s = svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)    # numerical safety
    angles = np.arccos(s)       # radians, ascending because s sorted desc

    if verbose:
        print("principal cosines (first 10):", s[:min(10, s.size)])
        print("principal angles deg (first 10):", np.degrees(angles[:min(10, angles.size)]))

    return angles, s, A, B

def friedrichs_angle(A_in, B_in, verbose=True):
    """
    Friedrichs angle between span(A_in) and span(B_in).
    Returns theta_F (radians) and cos(theta_F).
    """
    angles, sigmas, A, B = principal_angles(A_in, B_in, verbose=verbose)
    if sigmas.size == 0:
        # degenerate case: one subspace zero -> treat as orthogonal
        return np.pi/2.0, 0.0, A, B
    sigma1 = float(sigmas[0])            # largest cosine
    thetaF = float(np.arccos(np.clip(sigma1, -1.0, 1.0)))
    if verbose:
        print("Friedrichs cos (sigma1) =", sigma1)
        print("Friedrichs angle (deg) =", np.degrees(thetaF))
    return thetaF, sigma1, A, B

# -------------------------
# Example usage with your matrices:
# H : 72 x 167 (your Hankel)
# big_G : 24 x 72 (kron(I_12, G))
# -------------------------
# Replace H and big_G with your actual arrays.
# Example:
# thetaF_rad, cos_thetaF, A_basis, B_basis = friedrichs_angle(H, big_G)
