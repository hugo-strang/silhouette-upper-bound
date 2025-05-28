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

    d3 = np.array([[0, 1, 4, 5.5, 7],  
                   [1, 0, 5, 4.5, 6.88],  
                   [4, 5, 0, 0.5, 0.78],  
                   [5.5, 4.5, 0.5, 0, 0.5],  
                   [7, 6.88, 0.78, 0.5, 0]]) 
    
    d3f = np.array([3 / (4+5.5+7), 
                    3 / (5+4.5+6.88), 
                    (0.5 + 0.78) / (4 + 5), 
                    (0.5 + 0.5) / (5.5 + 4.5), 
                    (0.78 + 0.5) / (7 + 6.88)])
    
    assert np.abs(upper_bound(d3) - np.mean(1 - d3f)) < 1e-15
