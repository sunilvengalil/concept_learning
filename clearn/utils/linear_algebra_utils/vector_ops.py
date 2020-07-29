import numpy as np
from scipy.linalg import null_space
from copy import deepcopy
# ref http://mlwiki.org/index.php/Gram-Schmidt_Process
# orth_matrix = null_space(kernel_flattened)


def normalize(v):
    return v / np.sqrt(v.dot(v))


def orthogonalize(v, inplace=True):
    if inplace:
        x = v
    else:
        x = deepcopy(v)

    n = len(x)
    x[:, 0] = normalize(x[:, 0])

    for i in range(1, n):
        x_i = x[:, i]
        for j in range(0, i):
            x_j = x[:, j]
            t = x_i.dot(x_j)
            x_i = x_i - t * x_j
        x[:, i] = normalize(x_i)
    return x

# @param v a vector of dimension d
# @return a
def to_do_give_a_name_to_this(v):
    #Find the null space of dimension d-1
    perpendicular_plane = null_space(v)


