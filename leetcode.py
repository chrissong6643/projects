import bisect
import math
import traceback
from typing import List
import collections
from math import comb
from itertools import permutations


def canArrange(arr: List[int], k: int) -> bool:
    # mds = [i % k for i in arr]
    # mds.sort()
    # while len(mds)>0 and mds[0] == 0:
    #     mds.pop(0)
    # i = len(mds) - 1
    # while len(mds) > 0:
    #     while (i > 0 and mds[0] + mds[i]) % k != 0:
    #         i -= 1
    #     if i != 0:
    #         mds.pop(i)
    #         mds.pop(0)
    #         i = len(mds) - 1
    #     else:
    #         return False
    # return True
    mds = {}
    for i in range(k):
        mds[i] = []
    for i in arr:
        mds[i % k].append(i)
    mds = collections.OrderedDict(sorted(mds.items()))
    j = 0
    for i in mds.keys():
        if j > math.ceil(len(mds) / 2) - 1:
            break
        if i == 0:
            if len(mds.get(i)) % 2 != 0:
                return False
        elif len(mds.get(i)) != len(mds.get(k - i)):
            return False
        j += 1
    return True


def countValidWords(sentence: str) -> int:
    sentence = sentence.split()
    w = 0
    for i in sentence:
        if i[0] != "-" and i[-1] != "-":
            hc = 0
            if i[-1] == "!" or i[-1] == "." or i[-1] == ",":
                i = i[:-1]
            for j in range(len(i)):
                if i[j].isdigit():
                    w -= 1
                    break
                if i[j] == "-":
                    hc += 1
                    if hc > 1 or j == len(i) - 1:
                        w -= 1
                        break
                if i[j] == "!" or i[j] == "." or i[j] == ",":
                    w -= 1
                    break
            w += 1
    return w


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def visible_nodes(root: TreeNode):
    def height(root):
        if root == None:
            return 0
        else:
            lheight = height(root.left)
            rheight = height(root.right)
            if lheight > rheight:
                return lheight + 1
            else:
                return rheight + 1

    return height(root)


def delNodes(root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
    r = []

    def Preorder(node: TreeNode, parent_deleted: bool):
        if not node:
            return None
        if parent_deleted and node.val not in to_delete:
            r.append(node)
        node.left = Preorder(node.left, node.val in to_delete)
        node.right = Preorder(node.right, node.val in to_delete)
        return None if (node.val in to_delete) else node

    Preorder(root, True)
    return r


def count_of_nodes(root: TreeNode, queries, s):
    r = 0
    nds = [[root]]
    nn = 0
    nr = None
    nrf = False
    while all(i != [None, None] for i in nds):
        if nn + len(nds) * 2 >= queries[0]:
            r += 1
            for i in nds:
                if nn + 1 == queries[0]:
                    nr = i[0]
                    break
                elif nn + 2 == queries[0]:
                    nr = i[1]
                    break
            break
        else:
            nn += len(nds) * 2

    else:
        nn += 1
    ta = []
    for i in nds:
        if i:
            ta.append([i[0].left, i[0].right])
            ta.append([i[1].left, i[1].right])


def maxSubArray(nums: List[int]) -> int:
    dp = [[] for i in range(len(nums))]
    dp[0] = nums[0]
    m = dp[0]
    for i in range(1, len(nums)):
        dp[i] = nums[i] + (dp[i - 1] if dp[i - 1] > 0 else 0)
        m = max(m, dp[i])
    return m


# def minSubarray(nums: List[int], p: int) -> int:
#     s = sum(nums)
#     if s % p == 0:
#         return 0
#     i = (s % p)
#     pa = []
#
#     while i <= s:
#         ca = []
#         for j in nums:
#             if sum(ca) == i:
#                 pa.append(ca)
#             while len(ca) > 0 and sum(ca) + j > i:
#                 ca.pop(0)
#             ca.append(j)
#         if sum(ca) == i:
#             pa.append(ca)
#         i += p
#     for k in range(len(nums) - 3 + 1):
#         print(k, s - sum(nums[k:k + 3]), sum(nums[k:k + 3]), (s - sum(nums[k:k + 3])) % p)
#     ml = len(min(pa, key=len))
#     return ml if len(pa) > 0 else -1

def minSubarray(nums: List[int], p: int) -> int:
    if sum(nums) < p:
        return -1
    rem = sum(nums) % p
    if rem == 0:
        return 0
    rmndrs = {}
    rmndrs[0] = -1
    currem = 0
    ans = len(nums)
    for i in range(len(nums)):
        currem = (currem + nums[i]) % p
        rmndrs[currem] = i
        if currem - rem in rmndrs:
            ans = min(ans, i - rmndrs.get(currem - rem))
    uuu = ans if ans < len(nums) else -1
    return uuu


def invertTree2(root: TreeNode) -> TreeNode:
    lvl = [root]
    while not all(i is None for i in lvl):
        ta = []
        for i in lvl:
            if i:
                tempnode = i.right
                i.right = i.left
                i.left = tempnode
                ta.append(i.left)
                ta.append(i.right)
        lvl = ta.copy()
    return root


def invertTree(root: TreeNode) -> TreeNode:
    def inverse_helper(node):
        if node is None:
            return node
        inverse_helper(node.left)
        inverse_helper(node.right)
        l = node.left
        node.left = node.right
        node.right = l
        return node

    inverse_helper(root)
    return root


t1 = TreeNode(1)
t3 = TreeNode(3, t1)
t2 = TreeNode(2, t3)


# t1 = TreeNode(1)
# t3 = TreeNode(3)
# t6 = TreeNode(6)
# t9 = TreeNode(9)
# t2 = TreeNode(2, t1, t3)
# t7 = TreeNode(7, t6, t9)
# t4 = TreeNode(4, t2, t7)


def maxUniqueSplit(s: str) -> int:
    # split = [i for i in s]
    # i = 0
    # while len(set(split)) != len(split):
    #     while i < len(split) and split[i] in split[:i]:
    #         split[i] = "".join(split[i - 1:i + 1])
    #         split.pop(i - 1)
    #     i += 1
    # return len(split)
    #
    #
    def helper(s, index, subarray):
        if index >= len(s):
            return len(set(subarray))

        if subarray:
            l_string = subarray[-1]
            subarray[-1] = subarray[-1] + s[index]
            r1 = helper(s, index + 1, subarray)

            subarray[-1] = l_string
            subarray.append(s[index])
            r2 = helper(s, index + 1, subarray)
            subarray.pop()

            return max(r1, r2)
        else:
            subarray.append(s[index])
            r2 = helper(s, index + 1, subarray)

            return r2

    result = helper(s, 0, [])
    return result

    def f1(s: str, ind: int, subarray: List[str]):
        if ind >= len(s):
            return len(subarray)
        if len(subarray) > 0:
            ls = subarray[-1]
            subarray[-1] += s[ind]
            r1 = f1(s, ind + 1, subarray)
            subarray[-1] = ls
            subarray.append(s[ind])
            r2 = f1(s, ind + 1, subarray)
            subarray.pop()
            return max(r1, r2)
        else:
            subarray.append(s[ind])
            r2 = f1(s, ind + 1, subarray)
            return r2

    return f1(s, 0, [])


def lastStoneWeight(stones: List[int]) -> int:
    while len(stones) > 1:
        stones.sort()
        n = min(stones[-2], stones[-1])
        stones[-1] -= n
        stones[-2] -= n
        while 0 in stones:
            del stones[stones.index(0)]
    return stones[0] if len(stones) > 0 else 0


def getTotalTime(arr: List[int]):
    a = []
    while len(arr) > 1:
        arr.sort()
        n = arr[-2] + arr[-1]
        a.append(n)
        arr = arr[:len(arr) - 2]
        arr.append(n)
    return sum(a)


def repeatedNTimes(nums: List[int]) -> int:
    d = {}
    for i in nums:
        if i in d:
            return i
        else:
            d[i] = 1
    return 0


def goodDaysToRobBank(security: List[int], time: int) -> List[int]:
    r = []
    if time == 0:
        return [i for i in range(len(security))]
    if len(security) <= time:
        return []
    bef = security[:time]
    for i in range(time, len(security) - time):
        aft = security[i + 1:i + time + 1]
        befs = sorted(bef, reverse=True)
        afts = sorted(aft)
        if befs == bef and afts == aft and befs[-1] >= security[i] and security[i] <= afts[0]:
            r.append(i)
        if len(bef) > 0:
            bef.pop(0)
            bef.append(security[i])
    return r


def getDirections(root: TreeNode, startValue: int, destValue: int) -> str:
    start_path = []
    dest_path = []

    def find_path(node, v, path):
        if node is None:
            return False

        if node.val == v:
            path.append(('', node.val))
            return True

        left = find_path(node.left, v, path)
        if left:
            path.append(('L', node.val))
            return left

        right = find_path(node.right, v, path)
        if right:
            path.append(('R', node.val))
            return right

        return False

    find_path(root, startValue, start_path)
    find_path(root, destValue, dest_path)

    dest_set = set()
    for (_, node_v) in dest_path:
        dest_set.add(node_v)
    # common_parent = None
    for i, (_, node_v) in enumerate(start_path):
        if node_v in dest_set:
            common_parent = node_v
            index = i
            break

    res = ""
    for i in range(index):
        res += "U"

    for i, (_, node_v) in enumerate(dest_path):
        if node_v == common_parent:
            index = i
            break
    for i in range(index, -1, -1):
        res += dest_path[i][0]

    return res


def getDirections2(root: TreeNode, startValue: int, destValue: int) -> str:
    def find(node: TreeNode, val: int, path: List[int]):
        if node:
            return False
        left = find(node.left, val, path)
        if left:
            path.append(["L", node.val])
            return left
        right = find(node.right, val, path)
        if right:
            path.append(["R", node.val])
            return right

    p1 = []
    p2 = []
    find(root, startValue, p1)
    find(root, destValue, p2)
    setp2 = set(p2)
    common = next((i for i in p1 if a in setp2), None)


def flipAndInvertImage(image: List[List[int]]) -> List[List[int]]:
    nm = []
    for i in image:
        ta = []
        i.reverse()
        for j in i:
            if j == 0:
                ta.append(1)
            else:
                ta.append(0)
        nm.append(ta)
    return nm


def shortestToChar(s: str, c: str) -> List[int]:
    r = []
    inds = []
    for i in range(len(s)):
        if s[i] == c:
            inds.append(i)
    for i in range(len(s)):
        j = bisect.bisect_left(inds, i)
        if j < len(inds):
            r.append(min(abs(i - inds[j]), abs(i - inds[j - 1])))
        else:
            r.append(abs(i - inds[-1]))
    return r


def lemonadeChange(bills: List[int]) -> bool:
    chge = {}
    for i in bills:
        if i == 20:
            if 10 in chge and chge[10] > 0 and 5 in chge and chge[5] > 0:
                chge[10] -= 1
                chge[5] -= 1
            elif 5 in chge and chge[5] > 2:
                chge[5] -= 3
            else:
                return False
        if i == 10:
            if 5 in chge and chge[5] > 0:
                chge[5] -= 1
            else:
                return False
        if i in chge:
            chge[i] += 1
        else:
            chge[i] = 1
    return True


def backspaceCompare(s: str, t: str) -> bool:
    while "#" in s:
        if s[0] == "#":
            s = s[1:]
        else:
            s = s[:s.index("#") - 1] + s[s.index("#") + 1:]
    while "#" in t:
        if t[0] == "#":
            t = t[1:]
        else:
            t = t[:t.index("#") - 1] + t[t.index("#") + 1:]
    return s == t


def leafSimilar(root1: TreeNode, root2: TreeNode) -> bool:
    t1l = []
    t2l = []

    def Preorder(root: TreeNode, l: List[int]):
        if root:
            if not root.left and not root.right:
                l.append(root.val)
            Preorder(root.left, l)
            Preorder(root.right, l)

    Preorder(root1, t1l)
    Preorder(root2, t2l)
    return t1l == t2l


def uncommonFromSentences(s1: str, s2: str) -> List[str]:
    s1 = s1.split()
    s2 = s2.split()
    d = {}
    r = []
    for i in s1:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1

    for j in s2:
        if j not in d:
            d[j] = 1
        else:
            d[j] += 1
    for i in d.keys():
        if d[i] == 1:
            r.append(i)
    return r


def findSecondMinimumValue(root: TreeNode) -> int:
    l = []

    def preorder(node: TreeNode, l: List[int]):
        if node:
            l.append(node.val)
            preorder(node.left, l)
            preorder(node.right, l)

    preorder(root, l)
    l = list(set(l))
    l.sort()
    return l[1] if len(l) > 1 else -1


def wateringPlants(plants: List[int], capacity: int) -> int:
    s = 0
    c = capacity
    for i in range(len(plants)):
        if c - plants[i] < 0:
            s += 2 * i + 1
            c = capacity - plants[i]
        else:
            s += 1
            c -= plants[i]
    return s


def checkIfExist(arr: List[int]) -> bool:
    for i in arr:
        if i * 2 in arr:
            if i * 2 == i:
                if arr.count(i) > 1:
                    return True
            else:
                return True
    return False


checkIfExist([-2, 0, 10, -19, 4, 6, -8])


def maxSubsequence(nums: List[int], k: int) -> List[int]:
    r = []
    nc = sorted(nums)
    ga = nc[-k:]
    for i in nums:
        if i in ga:
            r.append(i)
            ga.remove(i)
    return r


def sumOfLeftLeaves(root: TreeNode) -> int:
    s = [0]

    def preorder(node: TreeNode):
        if node:
            if node.left and not node.left.left and not node.left.right:
                s[0] += node.left.val
            preorder(node.left)
            preorder(node.right)

    preorder(root)
    return s[0]


def mostCommonWord(paragraph: str, banned: List[str]) -> str:
    d = {}
    tw = ""
    for i in paragraph:
        if str.isalpha(i):
            tw += i
        elif len(tw) > 0:
            if tw.lower() in d:
                d[tw.lower()] += 1
            else:
                d[tw.lower()] = 1
            tw = ""
    if tw.lower() in d:
        d[tw.lower()] += 1
    else:
        d[tw.lower()] = 1
    r = []
    for i in d.keys():
        r.append([i, d.get(i)])
    r.sort(key=lambda x: x[1], reverse=True)
    while len(r) > 1 and r[0][0] in banned:
        r.pop(0)
    return r[0][0]


def numEquivDominoPairs(dominoes: List[List[int]]) -> int:
    d = {}
    r = 0
    for i in dominoes:
        i.sort()
        t = tuple(i)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1
    for i in d.keys():
        if d.get(i) > 1:
            r += comb(d.get(i), 2)
    return r


def findMaxLength(nums: List[int]) -> int:
    cl = {0: [-1]}
    md = 0
    c = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            c -= 1
        else:
            c += 1
        if c in cl:
            cl[c].append(i)
        else:
            cl[c] = [i]
    for i in cl.keys():
        if cl.get(i)[-1] - cl.get(i)[0] > md:
            if cl.get(i)[-1] - cl.get(i)[0] > md:
                md = cl.get(i)[-1] - cl.get(i)[0]
    return md


def checkValid(matrix: List[List[int]]) -> bool:
    clms = [[] for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i].count(matrix[i][j]) > 1:
                return False
            if clms[j].count(matrix[i][j]) > 0:
                return False
            else:
                clms[j].append(matrix[i][j])
    return True


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def nextLargerNodes(head: ListNode) -> List[int]:
    vals = []
    q = []

    while head:
        vals.append(head.val)
        head = head.next
    cm = 0
    r = [0 for i in range(len(vals))]
    for i in range(len(vals)):
        while len(q) > 0 and q[-1][0] < vals[i]:
            r[q[-1][1]] = vals[i]
            q.pop(-1)
        q.append([vals[i], i])
    return r


def modifyString(s: str) -> str:
    if s == "?":
        return "a"
    r = ""
    alph = "abcdefghijklmnopqrstuvqxyz"
    for i in range(len(s)):
        if s[i] == "?":

            if i > 0 and i < len(s) - 1:
                j = 0
                while alph[j] == r[-1] or alph[j] == s[i + 1]:
                    j += 1
                r += alph[j]
            elif i == 0:
                j = 0
                while alph[j] == s[i + 1]:
                    j += 1
                r += alph[j]
            else:
                j = 0
                while alph[j] == r[-1]:
                    j += 1
                r += alph[j]
        else:
            r += s[i]
    return r


def evaluate(s: str, knowledge: List[List[str]]) -> str:
    d = {}
    rs = ""
    for i in knowledge:
        d[i[0]] = i[1]
    inp = False
    for i in s:
        if i == "(":
            inp = True
            inpw = ""
        elif i == ")":
            inp = False
            if inpw in d:
                rs += d[inpw]
            else:
                rs += "?"
        elif inp:
            inpw += i
        else:
            rs += i
    return rs


def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    r = []
    n2 = {}
    for i in nums2:
        if i in n2:
            n2[i] += 1
        else:
            n2[i] = 1
    for i in nums1:
        if i in n2 and n2[i] > 0:
            r.append(i)
            n2[i] -= 1
    return r


def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    q = []
    inds = {}
    r2 = []
    r = [-1 for i in range(len(nums2))]
    for i in range(len(nums2)):
        while len(q) > 0 and q[-1][0] < nums2[i]:
            r[q[-1][1]] = nums2[i]
            q.pop(-1)
        inds[nums2[i]] = i
        q.append([nums2[i], i])
    for i in nums1:
        r2.append(r[inds[i]])
    return r2


def findLengthOfLCIS(nums: List[int]) -> int:
    m = 0
    si = 0
    for i in range(1, len(nums)):
        if nums[i - 1] >= nums[i]:
            m = max(m, i - si)
            si = i
    m = max(m, len(nums) - si)
    return m


def secondHighest(s: str) -> int:
    n = set()
    for i in s:
        if i.isdigit():
            n.add(int(i))
    n = sorted(list(n))
    return n[-2] if len(n) > 1 else -1


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m = [([0] * (len(text2) + 1)) for i in range(len(text1) + 1)]
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
            else:
                m[i][j] = max(m[i - 1][j], m[i][j - 1])
    return m[-1][-1]


def firstUniqChar(s: str) -> int:
    d = {}
    r = -1
    for i in range(len(s)):
        if s[i] in d:
            d[s[i]][1] += 1
        else:
            d[s[i]] = [i, 1]
    for i in d.keys():
        if d[i][1] < 2:
            return d[i][0]
    return r


def canBeIncreasing(nums: List[int]) -> bool:
    r = False
    i = 1
    if len(nums) == 2:
        return True
    while i in range(len(nums)):
        if i > 0 and nums[i] <= nums[i - 1]:
            if r:
                return False
            r = True
            a1 = nums.copy()
            a2 = nums.copy()
            a1.pop(i - 1)
            a2.pop(i)
            r1 = True
            r2 = True
            for j in range(1, len(a1)):
                if a1[j] <= a1[j - 1]:
                    r1 = False
                    break
            for j in range(1, len(a2)):
                if a2[j] <= a2[j - 1]:
                    r2 = False
                    break
            if not r1 and not r2:
                return False
            if r1:
                nums.pop(i - 1)
            else:
                nums.pop(i)
            i = max(-1, i - 2)
        i += 1
    return True


def addToArrayForm(num: List[int], k: int) -> List[int]:
    return [i for i in str(int("".join([str(i) for i in num])) + k)]


def mergeNodes(head: ListNode) -> ListNode:
    head = head.next
    n = ListNode(0)
    z = n
    ts = 0
    while head != None:
        if head.val == 0:
            z.next = ListNode(ts)
            z = z.next
            ts = 0
        else:
            ts += head.val
        head = head.next
    return n.next


def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    def match(r1: TreeNode, r2: TreeNode) -> bool:
        if r1 and r2:
            return r1.val == r2.val and match(r1.left, r2.left) and match(r1.right, r2.right)
        return r1 == r2

    if not root:
        return False
    return match(root, subRoot) or isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)


def findJudge(n: int, trust: List[List[int]]) -> int:
    m = {}
    judge = 0
    if len(trust) < 1:
        if n == 1:
            return 1
        return -1
    for i in trust:
        if i[0] not in m:
            m[i[0]] = [i[1]]
        else:
            m[i[0]].append(i[1])
    if len(m.keys()) < n - 1:
        return -1
    for i in m.values():
        found = False
        for j in i:
            if j not in m:
                judge = j
                found = True
                break
        if not found:
            return -1

    return judge


def searchbst(n: TreeNode, target) -> TreeNode:
    if not n:
        return None
    if n.val == target:
        return n
    if n.val > target:
        return searchbst(n.left, target)
    else:
        return searchbst(n.right, target)


def mergeTwoLists(list1: ListNode, list2: ListNode) -> ListNode:
    if not list1 and not list2:
        return None
    if not list1:
        return list2
    if not list2:
        return list1
    if list1.val < list2.val:
        newhead = ListNode(list1.val)
        list1 = list1.next
    else:
        newhead = ListNode(list2.val)
        list2 = list2.next
    p = newhead
    while list1 and list2:
        p.next = ListNode()
        p = p.next
        if list1.val < list2.val:
            p.val = list1.val
            list1 = list1.next
        else:
            p.val = list2.val
            list2 = list2.next
    while list1:
        p.next = ListNode(list1.val)
        p = p.next
        list1 = list1.next
    while list2:
        p.next = ListNode(list2.val)
        p = p.next
        list2 = list2.next
    return newhead


def sumRootToLeaf(root: TreeNode) -> int:
    temppath = []
    allpath = []

    def paths(root, path, pathLen, allp):
        if root:
            if (len(path) > pathLen):
                path[pathLen] = str(root.val)
            else:
                path.append(str(root.val))

            pathLen += 1

            if not root.left and not root.right:
                allp.append("".join(path)[:pathLen])
            else:
                paths(root.left, path, pathLen, allp)
                paths(root.right, path, pathLen, allp)

    paths(root, temppath, 0, allpath)
    return sum([int(i, 2) for i in allpath])


t1 = TreeNode(0)
t9 = TreeNode(1)
t10 = TreeNode(1, None, t9)
t2 = TreeNode(0, None, t10)
t3 = TreeNode(0)
t5 = TreeNode(1, t1)
t6 = TreeNode(0, t2, t3)
t7 = TreeNode(0, t5, t6)
sumRootToLeaf(t7)
