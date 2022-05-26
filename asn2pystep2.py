import random

import numpy as np


def func(a: np.matrix, b: np.matrix) -> np.ndarray:
    if a.size != b.size:
        print("Matrices need to be of same size!")
    else:
        return np.dot(np.dot(a.T, b), np.linalg.inv(a))


def func2() -> int:
    r = random.randint(0, 3)
    if r >= 2:
        return 1
    else:
        return -1


if __name__ == '__main__':
    a = np.matrix('1 ,2; 3,4')
    b = np.matrix('5,6;7,8')
print(func(a, b))
print(func2())
rand1 = [random.randint(-1, 11) for i in range(10)]
for i in range(len(rand1)):
    if rand1[i] >= 5 and rand1[1] <= 7:
        rand1[i] = 0
rand2 = [random.randint(-1, 11) for i in range(10)]
for i in range(len(rand2)):
    if rand2[i] % 2 == 0:
        rand2[i] == "Even"
    else:
        rand2[i] == "Odd"
I = np.random.uniform(0, 255, size=(256, 256))
Is = [[[i for i in range(256)] for i in range(256)] for i in range(2)]
Is.append(I.tolist())