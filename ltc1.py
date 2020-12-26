from typing import List


def getSumAbsoluteDifferences(nums: List[int]) -> List[int]:
    res = []
    fs = 0
    bs = sum(nums)
    for i in range(len(nums)):
        if i > 0:
            fs += nums[i - 1]
        bs -= nums[i]
        res.append((nums[i] * i - fs) + (bs - nums[i] * (len(nums) - i - 1)))
    return res


getSumAbsoluteDifferences([1,4,6,8,10])
