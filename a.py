import math
import sys
from typing import List


# def solution(a: List[int]):
#     c = 0
#     for i in a:
#         s = str(i)
#         if len(s) % 2 == 0:
#             c += 1
#     return c


# def solution2(elements: List[int]):
#     if elements == sorted(elements):
#         return 0
#     shifts = 1
#     element_copy = elements.copy()
#     element_copy.append(element_copy.pop(0))
#     while elements != element_copy and element_copy != sorted(elements):
#         element_copy.append(element_copy.pop(0))
#         shifts += 1
#     if element_copy == sorted(elements):
#         return shifts
#     return -1


# def solution3(bubbles: List[List[int]]):
#     coordinatelist = []
#     for i in range(len(bubbles)):
#         for j in range(len(bubbles[i])):
#             left, right, up, down = False, False, False, False
#             if j - 1 >= 0 and bubbles[i][j - 1] == bubbles[i][j]:
#                 left = True
#             if j + 1 < len(bubbles[i]) and bubbles[i][j + 1] == bubbles[i][j]:
#                 right = True
#             if i - 1 >= 0 and bubbles[i - 1][j] == bubbles[i][j]:
#                 up = True
#             if i + 1 < len(bubbles) and bubbles[i + 1][j] == bubbles[i][j]:
#                 down = True
#             ar = [left, right, up, down]
#             if ar.count(True) >= 2:
#                 for k in range(len(ar)):
#                     if ar[0]:
#                         coordinatelist.append([i, j - 1])
#                     if ar[1]:
#                         coordinatelist.append([i, j + 1])
#                     if ar[2]:
#                         coordinatelist.append([i - 1, j])
#                     if ar[3]:
#                         coordinatelist.append([i + 1, j])
#                     coordinatelist.append([i, j])
#     for i in coordinatelist:
#         bubbles[i[0]][i[1]] = 0
#     i = len(bubbles) - 1
#     while i >= 0:
#         t = 1
#         if i + 1 <
#             while i + t < len(bubbles) and bubbles[i][j] != 0 and bubbles[i + 1][j] == 0:
#                 t += 1
#             bubbles[i + t][j] = bubbles[i][j]
#             bubbles[i][j] = 0
#         j += 1
#
#     return bubbles
#
#
# # solution3([
# #     [3, 1, 2, 1],
# #     [1, 1, 1, 4],
# #     [3, 1, 2, 2],
# #     [3, 3, 3, 4]
# # ])
# def solution4(strings: List[str]):
#     pairs = {}
#     for i in strings:
#         removed = strings.copy()
#         removed.remove(i)
#         for j in removed:
#             if len(i) < len(j) and i not in pairs.keys():
#                 if j.endswith(i):
#                     pairs[i] = j
#             elif j not in pairs.keys():
#                 if i.endswith(j):
#                     pairs[j] = i
#     return len(pairs.keys())


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def deleteMiddle(head: ListNode) -> ListNode:
    l = []
    nl = None
    while n:
        l.append(n.val)
        n = n.next
    tn = None
    for i in range(len(l)):
        if len(l) % 2 == 0:
            num = math.ceil(len(l) / 2)
        else:
            num = math.floor(len(l) / 2)
        if i != num:
            if nl == None:
                nl = ListNode(l[i])
                tn = nl
            else:
                tn.next = ListNode(l[i])
                tn = tn.next
    return nl


def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right


# def find(n: TreeNode, val: int, path: List[str]) -> bool:
#     if n.val == val:
#         return True
#     if n.left and find(n.left, val, path):
#         path += "L"
#     elif n.right and find(n.right, val, path):
#         path += "R"
#     return path
#
# s, d = [], []
# find(root, startValue, s)
# find(root, destValue, d)
# while len(s) and len(d) and s[-1] == d[-1]:
#     s.pop()
#     d.pop()
# return "".join("U" * len(s)) + "".join(reversed(d))


def checkRecord(s: str) -> bool:
    ac = 0
    lc = 0
    for i in range(len(s)):
        if s[i] == "A":
            ac += 1
        if s[i] == "L":
            if i > 0 and s[i - 1] == "L":
                lc += 1
            else:
                lc = 1
        if ac == 2:
            return False
        if lc == 3:
            return False
    return True


def calPoints(ops: List[str]) -> int:
    rcd = []
    for i in range(len(ops)):
        if ops[i].isdigit() or ops[i].startswith("-"):
            if ops[i].startswith("-"):
                rcd.append(-1 * int(ops[i].lstrip("-")))
            else:
                rcd.append(int(ops[i]))
        if ops[i] == "C":
            rcd.pop()
        if ops[i] == "D":
            rcd.append(2 * rcd[-1])
        if ops[i] == "+":
            rcd.append(rcd[-2] + rcd[-1])
    return sum(rcd)


def maxWidthRamp(nums: List[int]) -> int:
    s = []
    for i in range(len(nums)):
        if len(s) == 0 or nums[i] < nums[s[-1]]:
            s.append(i)
    res = 0
    for i in range(len(nums) - 1, -1, -1):
        while len(s) > 0 and nums[s[-1]] <= nums[i]:
            res = max(res, i - s.pop())

    return res;


def findClosestNumber(nums: List[int]) -> int:
    n = None
    d = sys.maxsize
    for i in nums:
        if abs(i) <= d:
            if abs(i) == d and i < n:
                pass
            else:
                n = i
                d = abs(i)
    return n


def maxCount(m: int, n: int, ops: List[List[int]]) -> int:
    if len(ops) > 0:
        r = sys.maxsize
        c = sys.maxsize
        for i in ops:
            r = min(r, i[0])
            c = min(c, i[1])
        return r * c
    return m * n


def threeSum(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1
    return res


from shapely.geometry import box


def area(coords: List[int]):
    ta = 0
    for i in coords:
        ta += ((i[2] - i[0]) * (i[3] - i[1]))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            ta -= int(box(coords[i][0], coords[i][1], coords[i][2], coords[i][3]).intersection(box(coords[j][0], coords[j][1], coords[j][2], coords[j][3])).area)
    print(ta)


temp = [None for i in range(30)]
temp[0] = "h";
temp[1] = "e";
temp[2] = "l";
temp[3] = "l";
temp[4] = "o";
id = 4
add = 3
for i in range(add):
    for j in range(id, -1, -1):
        temp[j + 1] = temp[j]
        if j == 0:
            temp[j + 1] = temp[j]
            temp[j] = ' '
            length += 1
    id += 1
    temp += " "
    length += 1
None
