import bisect
import itertools
from typing import List
from itertools import permutations
from decimal import *


def largestValsFromLabels(values: List[int], labels: List[int], numWanted: int, useLimit: int) -> int:
    m = {}
    u = {}
    r = []
    for i in range(len(values)):
        if values[i] not in m:
            m[values[i]] = [labels[i]]
        else:
            m[values[i]].append(labels[i])
    k = sorted(m.keys(), reverse=True)
    i = 0
    while i < len(k) and len(r) < numWanted:
        z = m.get(k[i])
        if len(z) > 1:
            j = 0
            while len(r) < numWanted and j < len(z):
                if z[j] not in u:
                    u[z[j]] = 1
                    r.append(k[i])
                elif u[z[j]] < useLimit:
                    r.append(k[i])
                    u[z[j]] += 1
                j += 1
        else:
            if z[0] not in u:
                u[z[0]] = 1
                r.append(k[i])
            elif u[z[0]] < useLimit:
                r.append(k[i])
                u[z[0]] += 1
        i += 1
    return sum(r)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def recoverTree2(root: [TreeNode]) -> None:
    t = [None] * 2

    def mtd(node: TreeNode, t) -> (TreeNode, TreeNode):
        min_node = node
        max_node = node
        if node.left:
            left_min, left_max = mtd(node.left, t)
            # if done:
            #     return (None, None, done)
            if left_max and left_max.val > node.val:
                t[0] = left_max
                t[1] = node
            min_node = node if left_min.val > node.val else left_min
        if node.right:
            right_min, right_max = mtd(node.right, t)

            if right_min and right_min.val < node.val:
                t[0] = right_min
                t[1] = node

            max_node = node if right_max.val < node.val else right_max

        return (min_node, max_node)

    mtd(root, t)
    tv = t[0].val
    t[0].val = t[1].val
    t[1].val = tv


def recoverTree(root: [TreeNode]) -> None:
    cur, prev, drops, stack = root, TreeNode(float('-inf')), [], []
    while cur or stack:
        while cur or stack:
            while cur:
                while cur:
                    stack.append(cur)
                    cur = cur.left
            node = stack.pop()
            if node.val < prev.val: drops.append((prev, node))
            prev, cur = node, node.right
    drops[0][0].val, drops[-1][1].val = drops[-1][1].val, drops[0][0].val

    t1 = TreeNode(1, None, None)
    t2 = TreeNode(2, None, None)
    t4 = TreeNode(4, t2, None)
    t3 = TreeNode(3, t1, t4)
    recoverTree(t3)


def are_they_equal(array_a: List[int], array_b: List[int]):
    for i in range(len(array_a)):
        j = i
        while j < len(array_a) and array_a[i] != array_b[j]:
            j += 1
        if j < len(array_a) and array_a[i] == array_b[j] and j != i:
            array_a = array_a[:i] + array_a[i:j + 1][::-1] + array_a[j + 1:]

    return array_a == array_b


def minDepth(root: TreeNode) -> int:
    if not root:
        return 0
    lvl = 1
    if not root.left and not root.right:
        return lvl
    levelnodes = [[root.left, root.right]]
    while [None, None] not in levelnodes:
        ta = []
        for i in levelnodes:
            if i[0]:
                ta.append([i[0].left, i[0].right])
            if i[1]:
                ta.append([i[1].left, i[1].right])
        levelnodes = ta.copy()
        lvl += 1
    return lvl


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def increasingBST(root: TreeNode) -> TreeNode:
    nodelst = []

    def Inorder(root):
        if root:
            Inorder(root.left)
            nodelst.append(TreeNode(root.val))
            Inorder(root.right)

    Inorder(root)
    nt = nodelst[0]
    ctr = nt
    for i in nodelst[1:]:
        ctr.right = i
        ctr = ctr.right
    return nt


def increasingBST2(root: TreeNode) -> TreeNode:
    result = None

    def Inorder(node):
        if node is None:
            return None

        leftresult = Inorder(node.left)
        if leftresult:
            leftresult.right = node
            node.left = None
        else:
            nonlocal result
            if not result:
                result = node
        right_result = Inorder(node.right)
        if not right_result:
            return node
        else:
            return right_result

    Inorder(root)
    return result


def increasingBST3(root, tail=None):
    if not root:
        return tail
    res = increasingBST3(root.left, root)
    root.left = None
    root.right = increasingBST3(root.right, tail)
    return res


t1 = TreeNode(1)
t7 = TreeNode(7)
t9 = TreeNode(9)
t4 = TreeNode(4)
t2 = TreeNode(2, t1)
t8 = TreeNode(8, t7, t9)
t3 = TreeNode(3, t2, t4)
t6 = TreeNode(6, None, t8)
t5 = TreeNode(5, t3, t6)


def searchInsert(nums: List[int], target: int) -> int:
    if target <= nums[0]:
        return 0
    l = 0
    h = len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if target < nums[m]:
            h = m - 1
        elif target > nums[m]:
            l = m + 1
        elif target == nums[m]:
            return m
    if target > nums[-1]:
        return m + 1
    z = Decimal(l + h) / 2
    return int(z.to_integral_value(rounding=ROUND_HALF_UP))


def sumRootToLeaf(root: TreeNode) -> int:
    nums = []
    cn = []
    def Preorder(node):
        if node:

            cn.append(node.val)
            if not node.left and not node.right:
                ts = [str(j) for j in cn]
                nums.append("".join(ts))
                cn.pop(-1)
            Preorder(node.left)
            if node == root and len(cn) > 1:
                cn.pop(-1)
            Preorder(node.right)

    Preorder(root)
    s = 0
    for i in nums:
        s += int(i,2)
    return s










