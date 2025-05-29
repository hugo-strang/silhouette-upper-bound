import numpy as np

def _row_f(row: np.ndarray, kappa: int, n: int) -> float:
    
    x = np.sum(row[:kappa - 1])
        
    y = np.sum(row[kappa - 1:])

    q = (x / (kappa - 1)) / (y / (n - kappa))

    for delta in range(kappa + 1, n - kappa + 1):

        d_to_move = row[delta - 2]

        x += d_to_move
        y -= d_to_move
        
        q_candidate = (x / (delta - 1)) / (y / (n - delta))

        if q_candidate < q:
            q = q_candidate
    
    return 1 - q 
