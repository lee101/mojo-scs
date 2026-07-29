import numpy as np
from scipy import sparse

import mojo_scs as scs

A = sparse.csc_matrix([[1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
data = {
    "A": A,
    "b": np.array([2.0, 0.0, 0.0]),
    "c": np.array([-3.0, -2.0]),
}

solution = scs.solve(data, {"l": 3}, verbose=False)
print(solution["info"]["status"])
print(solution["x"])
