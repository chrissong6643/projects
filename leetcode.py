import math
from typing import List


def pushDominoes(dominoes: str) -> str:
    if "L" not in dominoes and "R" not in dominoes:
        return dominoes
    if len(dominoes) == 1:
        return dominoes
    dominoes = list(dominoes)
    dir = []
    for i in range(len(dominoes)):
        if dominoes[i] == "L" or dominoes[i] == "R":
            dir.append([dominoes[i], i])
    i = 0
    while i in range(len(dir) - 1):
        if i < len(dir) - 1 and dir[i + 1][0] == "L" and dir[i][0] == "R":
            n = dir[i + 1][1] - dir[i][1] + 1
            if n % 2 == 0:
                s = list(("R" * (n // 2)) + ("L" * (n // 2)))
                dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
            else:
                s = list(("R" * (n // 2)) + "." + ("L" * (n // 2)))
                dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        elif dir[i][0] == "L" and dir[i + 1][0] == "L":
            s = list("L" * (dir[i + 1][1] - dir[i][1] + 1))
            dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        elif dir[i][0] == "R" and dir[i + 1][0] == "R":
            s = list("R" * (dir[i + 1][1] - dir[i][1] + 1))
            dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        i += 1
    if dir[0][0] == "L":
        dominoes = list("L" * (dir[0][1] + 1)) + dominoes[dir[0][1] + 1:]
    if dir[-1][0] == "R":
        dominoes = dominoes[:dir[-1][1]] + list("R" * (len(dominoes) - dir[-1][1]))
    return "".join(dominoes)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
    def dfs(res: List[int], node: TreeNode, target: int, k: int, depth: int) -> int:
        if node == None:
            return 0
        if depth == k:
            res.append(node.val)
            return 0
        left, right = 0, 0
        if node.val == target or depth > 0:
            left = dfs(res, node.left, target, k, depth + 1)
            right = dfs(res, node.right, target, k, depth + 1)
        else:
            left = dfs(res, node.left, target, k, depth)
            right = dfs(res, node.right, target, k, depth)

        if node.val == target:
            return 1

        if left == k or right == k:
            res.append(node.val)
            return 0

        if left > 0:
            dfs(res, node.right, target, k, left + 1);
            return left + 1

        if right > 0:
            dfs(res, node.left, target, k, right + 1)
            return right + 1

        return 0

    res = []
    if k == 0:
        res.append(target.val)
    else:
        dfs(res, root, target.val, k, 0)
    return res


def checkPowersOfThree(n: int) -> bool:
    tp = []
    i = 0
    while 3 ** i <= n:
        tp.insert(0, 3 ** i)
        i += 1
    for i in tp:
        if i <= n:
            n -= i
        if n == 0:
            return True
    if n == 0:
        return True
    return False


def maximumProduct(nums: List[int]) -> int:
    nums.sort()
    if nums[0] < 0 and nums[1] < 0 and nums[0] * nums[1] > nums[len(nums) - 3] * nums[len(nums) - 2] and nums[-1] > 0:
        return nums[0] * nums[1] * nums[-1]
    return math.prod(nums[len(nums) - 3:])


def videoStitching(clips: List[List[int]], time: int) -> int:
    clips.sort(key=lambda x: x[0])
    i = 0
    max = clips[0]
    while i < len(clips) and clips[i][0] == 0:
        if clips[i][1] > max[1]:
            max = clips[i]
        i += 1
    itvl = [max]
    while i < len(clips):
        if itvl[-1][1] >= time and itvl[0][0] == 0:
            return len(itvl)
        if clips[i][0] > itvl[-1][1]:
            if max[0] <= itvl[-1][1]:
                itvl.append(max)
        if clips[i][1] > max[1]:
            max = clips[i]
        i += 1
    if max[0] <= itvl[-1][1] and itvl[-1][1] < time:
        itvl.append(max)
    if itvl[-1][1] >= time and itvl[0][0] == 0:
        return len(itvl)
    return -1


videoStitching([[0, 2]], 2)
