import numpy as np
from silhouette_upper_bound import upper_bound

def test_basic():

    d1 = np.array([[0, 1, 1, 1, 1], 
                   [1, 0, 1, 1, 1], 
                   [1, 1, 0, 1, 1], 
                   [1, 1, 1, 0, 1], 
                   [1, 1, 1, 1, 0]])

    assert upper_bound(d1) == 0.0 

    d2 = np.array([[0, 1, 5, 5, 5],  
                   [1, 0, 5, 5, 5],  
                   [5, 5, 0, 1, 1],  
                   [5, 5, 1, 0, 1],  
                   [5, 5, 1, 1, 0]]) 

    assert upper_bound(d2) == 1 - 1 / 5
