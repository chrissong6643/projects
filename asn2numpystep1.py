import numpy as np
import numpy.linalg.linalg

a, b = np.random.rand(3, 3), np.random.rand(3, 3)
ab = np.dot(a, b)
element_wise = np.multiply(a, b)
c = np.dot(np.dot(a.T, b), np.linalg.inv(a))
av = a.copy()
bv = b.copy()
av.resize((9, 1))
bv.resize((9, 1))
cv = np.c_[av, bv]
l2 = np.linalg.norm(cv)
I = np.random.uniform(0, 255, size=(256, 256))
M = np.random.uniform(0, 1, size=(256, 256))
for i in range(len(I)):
    for j in range(len(I)):
        if I[i][j] == M[i][j]:
            I[i][j] = 0
