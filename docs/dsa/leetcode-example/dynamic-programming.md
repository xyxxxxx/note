# 动态规划

## [45.跳跃游戏II](https://leetcode.cn/problems/jump-game-ii/)

* 动态规划

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        l = len(nums)
        if l == 1:
            return 0

        step = 0
        prev_idx = 0
        farthest_idx = 0
        while True:
            step += 1
            if farthest_idx > prev_idx:
                for i in range(prev_idx + 1, farthest_idx + 1):
                    idx = i + nums[i]
                    if idx > farthest_idx:
                        prev_idx = i
                        farthest_idx = idx
            else:
                farthest_idx += nums[farthest_idx]
            if farthest_idx >= l - 1:
                return step
# 44ms 15.8MB
```

## [72.编辑距离](https://leetcode.cn/problems/edit-distance/)

* 动态规划

!!! tip "提示"
    考虑子问题：将 `word1` 的前缀转换成 `word2` 的前缀，推知递归关系为 `minDistance(word1[:i], word2[:j])` 为

    * `minDistance(word1[:i-1], word2[:j-1]) + 1`
    * `minDistance(word1[:i-1], word2[:j]) + 1`
    * `minDistance(word1[:i], word2[:j-1]) + 1`

    三者的最小值。下面是一个示例：

    |     | -   | e   | x   | e   | c   | u   | t   | i   | o   | n   |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | -   | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
    | i   | 1   | 1   | 2   | 3   | 4   | 5   | 6   | 6   | 7   | 8   |
    | n   | 2   | 2   | ... |     |     |     |     |     |     |     |
    | t   | 3   | 3   |     |     |     |     |     |     |     |     |
    | e   | 4   | 3   |     |     |     |     |     |     |     |     |
    | n   | 5   | 4   |     |     |     |     |     |     |     |     |
    | t   | 6   | 5   |     |     |     |     |     |     |     |     |
    | i   | 7   | 6   |     |     |     |     |     |     |     |     |
    | o   | 8   | 7   |     |     |     |     |     |     |     |     |
    | n   | 9   | 8   |     |     |     |     |     |     |     |     |

    如果不允许<u>替换</u>一个字符，则去掉上面的 `minDistance(word1[:i-1], word2[:j-1]) + 1` 这一项。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        if not word1:
            return len(word2)
        if not word2:
            return len(word1)

        row = len(word1) + 1
        column = len(word2) + 1
        table = [[0] * column for _ in range(row)]
        table[0] = list(range(column))
        for row_i, row_list in enumerate(table):
            row_list[0] = row_i

        for row_i in range(1, row):
            for column_i in range(1, column):
                if word1[row_i - 1] == word2[column_i - 1]:
                    table[row_i][column_i] = table[row_i - 1][column_i - 1]
                else:
                    table[row_i][column_i] = min(
                        table[row_i - 1][column_i - 1] + 1,
                        table[row_i][column_i - 1] + 1,
                        table[row_i - 1][column_i] + 1)

        return table[row_i][column_i]
# 156ms 18.7MB
```

## [174.地下城游戏](https://leetcode.cn/problems/dungeon-game/)

* 动态规划

!!! tip "提示"
    与 [72.编辑距离](#72编辑距离httpsleetcodecnproblemsedit-distance) 类似。下面是一个示例：

    地下城

    | -2 (K) | -3   | 3      |
    | ------ | ---- | ------ |
    | -5     | -10  | 1      |
    | 10     | 30   | -5 (P) |

    需要最小生命值

    | 7    | 5    | 2    |
    | ---- | ---- | ---- |
    | 6    | 11   | 5    |
    | 1    | 1    | 6    |

```python
import copy

class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        row = len(dungeon)
        column = len(dungeon[0])
        table = [[99999] * (column + 1) for _ in range(row + 1)]
        table[-2][-1] = 1

        for row_i in range(row - 1, -1, -1):
            for column_i in range(column - 1, -1, -1):
                table[row_i][column_i] = max(min(table[row_i][column_i + 1], table[row_i + 1][column_i]) - dungeon[row_i][column_i], 1)

        return table[0][0]
# 36ms 15.6MB
```

## [887.鸡蛋坠落](https://leetcode.cn/problems/super-egg-drop/)

* 动态规划

!!! tip "提示"
    ……

    | 6    | 6    | 3    | 3    |
    | ---- | ---- | ---- | ---- |
    | 5    | 5    | 3    | 3    |
    | 4    | 4    | 3    | 3    |
    | 3    | 3    | 2    | 2    |
    | 2    | 2    | 2    | 2    |
    | 1    | 1    | 1    | 1    |
    | n/k  | 1    | 2    | 3    |

    | f\k  | 1    | 2    | 3    |
    | ---- | ---- | ---- | ---- |
    | 1    | 1    | 1    | 1    |
    | 2    | 2    | 3    | 3    |
    | 3    | 3    | 6    | 7    |
    | 4    | 4    | 10   | 14   |
    | 5    | 5    | 15   | 25   |
    | 6    | 6    | 21   | 41   |

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        if n == 1:
            return 1
        if k == 1:
            return n

        f = 1
        max_floor_wrt_k = [1] * k
        while True:
            f += 1
            new_max_floor_wrt_k = [
                max_floor_wrt_k[i - 1] + v + 1 if i > 0 else v + 1
                for i, v in enumerate(max_floor_wrt_k)
            ]
            if new_max_floor_wrt_k[-1] >= n:
                return f
            max_floor_wrt_k = new_max_floor_wrt_k
# 52ms 14.9MB
```
