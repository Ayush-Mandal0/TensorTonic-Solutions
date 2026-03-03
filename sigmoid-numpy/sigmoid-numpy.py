import numpy as np


def sigmoid(x) -> np.ndarray:
   
    x = np.asarray(x, dtype=np.float64) 

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = np.empty_like(x, dtype=np.float64)

    
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

   
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)

    return result