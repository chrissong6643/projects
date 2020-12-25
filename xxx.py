from typing import List
from fractions import Fraction


def fractionAddition(expression: str) -> str:
    s = 0
    nums = []
    denoms = []
    inds = [i for i in range(len(expression)) if expression.startswith("/", i)]
    for i in inds:
        if i == 1:
            nums.append(int(expression[i - 1]))
        elif expression[i - 2] == "-":
            nums.append(-1 * int(expression[i - 1]))
        elif expression[i - 2].isdigit():
            if expression[i - 3] == "-":
                nums.append(-1 * int(expression[i - 2:i]))
            else:
                nums.append(int(expression[i - 2:i]))
        else:
            nums.append(int(expression[i - 1]))
        if i == len(expression) - 2 or not expression[i + 2].isdigit():
            denoms.append(int(expression[i + 1]))
        else:
            denoms.append(int(expression[i + 1:i + 3]))
    for i in range(len(nums)):
        s += nums[i] / denoms[i]
    f = Fraction(s).limit_denominator()
    return str(f.numerator) + "/" + str(f.denominator)


def maxLength(arr: List[str]) -> int:
    l = 0
    nl = []
    for i in arr:
        if len(i) == len(set(i)):
            nl.append(set(i))
    ma = [["" for x in range(len(nl))] for y in range(len(nl) + 1)]
    res = 0
    for i in range(len(nl)):
        ma[0][i] = nl[i]
        res = max(res, len(nl[i]))
    for i in range(1, len(nl)):
        for j in range(len(nl)):
            if nl[i] == nl[j]:
                ma[i][j] = nl[i]
                res = max(res, len(nl[i]))
            else:
                s = ""
                for k in range(i - 1, -1, -1):
                    if len(nl[i] & ma[k][j]) == 0:
                        if len(ma[k][j] | nl[i]) > len(s):
                            s = ma[k][j] | nl[i]
                ma[i][j] = set(s)
                res = max(len(s), res)
    return res


def accountsMerge(accounts: List[List[str]]) -> List[List[str]]:
    l = accounts
    out = []
    while len(l) > 0:
        first, *rest = l
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r[1:]))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(first)
        l = rest
    i = 0
    while i in range(len(out)):
        t = sorted(list(out[i]))
        out[i] = t
        i += 1
    return out


def longestPalindromeSubseq(s: str) -> int:
    # n = [1]
    # for i in range(len(s)):
    #     if s[i + 1:].count(s[i]) > 0:
    #         ta = s[i:s.rindex(s[i]) + 1]
    #         if len(ta) % 2 == 0:
    #             if len(ta) == 2:
    #                 n.append(2)
    #             else:
    #                 n1 = 0
    #                 j = 0
    #                 if ta[int(len(ta) / 2 - 1)] != ta[int(len(ta) / 2)]:
    #                     n1 += 1
    #                 while j in range(int(len(ta) / 2 + 1)):
    #                     if len(ta) > 0 and ta[j + 1:].count(ta[j]) > 0:
    #                         n1 += 2
    #                         ta = ta[:ta.rindex(ta[j])]
    #                         ta = ta[:j] + ta[j + 1:]
    #                         j -= 1
    #                     j += 1
    #                 n.append(n1)
    #         else:
    #             n1 = 0
    #             j = 0
    #             while j in range(int(len(ta) // 2 + 1)):
    #                 if len(ta) > 0 and ta[j + 1:].count(ta[j]) > 0:
    #                     n1 += 2
    #                     ta = ta[:ta.rindex(ta[j])]
    #                     ta = ta[:j] + ta[j + 1:]
    #                     j -= 1
    #                 j += 1
    #             if len(ta)>0:
    #                 n1+=1
    #             n.append(n1)
    # return max(n)
    m = [[0 for i in range(len(s))] for j in range(len(s))]
    for i in range(len(s)):
        m[i][i] = 1
        for j in range(i - 1, -1, -1):
            if s[i] == s[j]:
                m[i][j] = m[i - 1][j + 1] + 2
            elif i > 0 and j < len(s) - 1:
                m[i][j] = max(m[i - 1][j], m[i][j + 1])
    return m[len(s) - 1][0]


longestPalindromeSubseq("a")
