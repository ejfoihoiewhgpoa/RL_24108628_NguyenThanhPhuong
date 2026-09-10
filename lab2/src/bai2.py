import numpy as np

P = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

def validate_transition_matrix(P, tol=1e-10):
    P = np.asarray(P, dtype=float)
    
    if P.shape[0] != P.shape[1]:
        return False
    if np.any(P < 0) or np.any(P > 1):
        return False
    if not np.allclose(P.sum(axis=1), 1.0, atol=tol):
        return False
    
    return True

print(validate_transition_matrix(P))