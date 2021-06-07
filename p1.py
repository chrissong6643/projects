import bisect
from typing import List
import itertools


def restoreIpAddresses(self, s: str):
    ret = []
    self.dfs(ret, s, [])
    return ret


def dfs(self, ret, s, path_list):
    if len(path_list) == 4:
        if s == '':
            ret.append('.'.join(path_list))
        return
    if not s:
        return
    if s[0] == '0':
        temp_path_list = path_list[::]
        temp_path_list.append('0')
        self.dfs(ret, s[1:], temp_path_list)
        return
    for i in range(len(s)):
        if i >= 3:
            break
        if int(s[:i + 1]) <= 255:
            temp_path_list = path_list[::]
            temp_path_list.append(s[:i + 1])
            self.dfs(ret, s[i + 1:], temp_path_list)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def Preorder(n, l):
    if (n == None):
        l.append("None")
        return;
    l.append(n.val)
    Preorder(n.left, l);
    Preorder(n.right, l);


def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    pv = []
    qv = []
    Preorder(p, pv)
    Preorder(q, qv)
    if pv == qv:
        return True
    return False


p2 = TreeNode(2)

q2 = TreeNode(2)

p1 = TreeNode(1, p2)

q1 = TreeNode(1, None, q2)


# def search(nums: List[int], target: int) -> bool:
#     if target in nums:
#         return True
#     return False

def numberOfSubstrings(s: str) -> int:
    r = 0
    a = []
    b = []
    c = []
    for i in range(len(s)):
        if s[i] == "a":
            a.append(i)
        if s[i] == "b":
            b.append(i)
        if s[i] == "c":
            c.append(i)
    for i in range(len(s)):
        num = []
        if len(a) < 1 or len(b) < 1 or len(c) < 1 or i > a[-1] or i > b[-1] or i > c[-1]:
            break
        num.append(a[bisect.bisect_left(a, i)])
        num.append(b[bisect.bisect_left(b, i)])
        num.append(c[bisect.bisect_left(c, i)])
        r += len(s) - (max(num) + 1) + 1
    return r


def maxDistance(nums1: List[int], nums2: List[int]) -> int:
    n1p = 0
    dsts = []
    if nums1[-1] > nums2[0]:
        return 0
    for i in range(len(nums2)):
        while nums1[n1p] > nums2[i] and n1p <= i:
            if n1p != len(nums1) - 1:
                n1p += 1
            else:
                break
        if nums1[n1p] <= nums2[i]:
            dsts.append(i - n1p)
    if len(dsts) < 1:
        return 0
    return (max(max(dsts), 0))


def canJump(nums: List[int]) -> bool:
    maxind = 0
    i = 0
    while i <= maxind and maxind < len(nums) - 1:
        if i + nums[i] > maxind:
            maxind = i + nums[i]
        i += 1
    if maxind >= len(nums) - 1:
        return True
    return False


def maxSubarraySumCircular(nums: List[int]) -> int:
    r = []
    min_ending_here = min_so_far = float('inf')
    max_so_far = -float('inf') - 1
    max_ending_here = 0
    for i in range(len(nums)):
        if (min_ending_here > 0):
            min_ending_here = nums[i]
        else:
            min_ending_here += nums[i]
        min_so_far = min(min_so_far, min_ending_here)
        max_ending_here = max_ending_here + nums[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here
        if max_ending_here < 0:
            max_ending_here = 0
    if max_so_far < 0:
        return max_so_far
    r.append(sum(nums) - min(min_so_far, min_ending_here))
    r.append(max_so_far)
    return max(r)


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    l = list(itertools.chain.from_iterable(matrix))
    if target in l:
        return True
    return False


def corpFlightBookings(bookings: List[List[int]], n: int) -> List[int]:
    l = [0] * n
    for i in bookings:
        l[i[0] - 1] += i[2]
        if i[1] <= n - 1:
            l[i[1]] -= i[2]
    i = 1
    while i in range(1, len(l)):
        l[i] += l[i - 1]
        i += 1
    return l


def canReorderDoubled(arr: List[int]) -> bool:
    if len(arr) == 0:
        return True
    if len(arr) < 2:
        return False
    arr = sorted(arr, key=abs)
    p2 = 1
    while len(arr) > 0:
        while 2 * arr[0] != arr[p2]:
            p2 += 1
            if p2 == len(arr):
                return False
        arr.pop(0)
        arr.pop(p2 - 1)
        p2 = max(1, p2 - 2)
    return True


def numDifferentIntegers(word: str) -> int:
    r = ""
    for i in word:
        if i.isdigit():
            r += i
        else:
            r += " "
    r = r.split()
    r = [i.lstrip("0") for i in r]
    r = set(r)
    return len(r)


def findMinHeightTrees(n: int, edges: List[List[int]]) -> List[int]:
    if n == 1:
        return [0]
    t = [set() for i in range(n)]
    for i in edges:
        t[i[0]].add(i[1])
        t[i[1]].add(i[0])
    nl = len(t)
    leaves = []
    for i in range(len(t)):
        if len(t[i]) == 1:
            leaves.append(i)
    while nl > 2:
        newleaf = []
        nl -= len(leaves)
        for i in leaves:
            p = t[i].pop()
            t[p].remove(i)
            if len(t[p]) == 1:
                newleaf.append(p)
        leaves = newleaf
    return leaves


def minPairSum(nums: List[int]) -> int:
    s = []
    nums.sort()
    for i in range(len(nums) // 2):
        s.append(nums[i] + nums[len(nums) - 1 - i])
    return max(s)


def fourSum(nums, target):
    def findNsum(l, r, target, N, result, results):
        if r - l + 1 < N or N < 2 or target < nums[l] * N or target > nums[r] * N:  # early termination
            return
        if N == 2:  # two pointers solve sorted 2-sum problem
            while l < r:
                s = nums[l] + nums[r]
                if s == target:
                    results.append(result + [nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
        else:  # recursively reduce N
            for i in range(l, r + 1):
                if i == l or (i > l and nums[i - 1] != nums[i]):
                    findNsum(i + 1, r, target - nums[i], N - 1, result + [nums[i]], results)

    nums.sort()
    results = []
    findNsum(0, len(nums) - 1, target, 4, [], results)
    return results


def findSubstringInWraproundString(p: str) -> int:
    s = "zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    d = {}
    prev = p[0]
    d[prev] = 1
    ctr = 1
    for i in range(1, len(p)):
        if p[i] == s[s.index(prev) + 1]:
            ctr += 1
        else:
            ctr = 1
        if p[i] in d:
            d[p[i]] = max(d[p[i]], ctr)
        else:
            d[p[i]] = ctr
        prev = p[i]
    s = 0
    for i in d.values():
        s += i
    return s


def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    if len(intervals) == 0:
        return [newInterval]
    if intervals[len(intervals) - 1][1] < newInterval[0]:
        intervals.append(newInterval)
        return intervals
    if intervals[0][0] > newInterval[1]:
        intervals.insert(0, newInterval)
        return intervals
    r = []
    for i in range(len(intervals)):
        if intervals[i][1] >= newInterval[0]:
            j = i
            while j < len(intervals) and intervals[j][0] <= newInterval[1]:
                j += 1
            if j == len(intervals):
                j -= 1
            if intervals[j][0] > newInterval[1]:
                j -= 1
            r = intervals[:i]
            r.append([min(intervals[i][0], newInterval[0]), max(intervals[j][1], newInterval[1])])
            r.extend(intervals[j + 1:])
            break
    return r


insert([[1, 5]], [0, 1])
