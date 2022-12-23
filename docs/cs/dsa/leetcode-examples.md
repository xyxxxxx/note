# Leetcode 例题

## 迭代和递归

### [3.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

* 迭代（双指针法）

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        s_len = len(s)
        p1 = 0
        p2 = 0
        sub = {s[0]}
        max_len = 1
        while p2 < s_len - 1:
            p2 += 1
            if s[p2] in sub:
                sub_len = len(sub)
                if sub_len > max_len:
                    max_len = sub_len
                while True:
                    if s[p1] == s[p2]:
                        p1 += 1
                        break
                    else:
                        sub.remove(s[p1])
                        p1 += 1
            else:
                sub.add(s[p2])

        sub_len = len(sub)
        if sub_len > max_len:
            max_len = sub_len

        return max_len
# 48ms 15.1MB
```

### [21.合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

数据结构：链表

* 迭代

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        list0 = ListNode()
        p = list0
        while True:
            if list1.val > list2.val:
                p.next = list2
                if list2.next:
                    list2 = list2.next
                else:
                    p.next.next = list1
                    return list0.next
            else:
                p.next = list1
                if list1.next:
                    list1 = list1.next
                else:
                    p.next.next = list2
                    return list0.next
            p = p.next
# 40ms 15.1MB
```

* 递归

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        if list1.val > list2.val:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
        else:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
# 44ms 15.1MB
```

### [23.合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

数据结构：链表

设链表数为 $k$，总节点数为 $n$。

* 迭代（每次迭代移除各链表首节点最小者）

时间复杂度：$<O(nk)$

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return
        if None in lists:
            return self.mergeKLists([l for l in lists if l])
        list0 = ListNode()
        p = list0
        while lists:
            minimum = 10000
            for i, l in enumerate(lists):
                if l.val < minimum:
                    minimum = l.val
                    indexs = [i]
                elif l.val == minimum:
                    indexs.append(i)
            indexs.reverse()
            for i in indexs:
                p.next = lists[i]
                if lists[i].next:
                    lists[i] = lists[i].next
                else:
                    lists.pop(i)
                p = p.next
        return list0.next
# 164ms 18MB
```

* 迭代（维护一个由各链表首节点组成的堆）

时间复杂度：$O(k\log k)+nO(\log k)=O((n+k)\log k)$

```python
import heapq

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:    
        if not lists:
            return
        if None in lists:
            return self.mergeKLists([l for l in lists if l])
        list0 = ListNode()
        p = list0
        heap = [(l.val, id(l), l) for l in lists]  # id(l) as tiebreaker
        heapq.heapify(heap)
        while heap:
            _, _, l = heapq.heappop(heap)
            p.next = l
            p = p.next
            if l.next:
                heapq.heappush(heap, (l.next.val, id(l.next), l.next))
        return list0.next
# 88ms 18.2MB
```

* 迭代（定义合并两个有序链表的函数并归约所有链表）

时间复杂度：$<O(nk/2)$

```python
from functools import reduce

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:    
        if not lists:
            return
        if None in lists:
            return self.mergeKLists([l for l in lists if l])
        return reduce(self.mergeTwoLists, lists)
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        list0 = ListNode()
        p = list0
        while True:
            if list1.val > list2.val:
                p.next = list2
                if list2.next:
                    list2 = list2.next
                else:
                    p.next.next = list1
                    return list0.next
            else:
                p.next = list1
                if list1.next:
                    list1 = list1.next
                else:
                    p.next.next = list2
                    return list0.next
            p = p.next
# 2852ms 17.9MB
```

* 迭代（定义合并两个有序链表的函数，每次迭代合并相邻的链表）

时间复杂度：$O(n\log k)$

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:    
        if not lists:
            return
        if None in lists:
            return self.mergeKLists([l for l in lists if l])
        while True:
            next_lists = []
            while len(lists) > 1:
                next_lists.append(self.mergeTwoLists(lists.pop(), lists.pop()))
            next_lists.extend(lists)
            if len(next_lists) == 1:
                return next_lists[0]
            lists = next_lists
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        list0 = ListNode()
        p = list0
        while True:
            if list1.val > list2.val:
                p.next = list2
                if list2.next:
                    list2 = list2.next
                else:
                    p.next.next = list1
                    return list0.next
            else:
                p.next = list1
                if list1.next:
                    list1 = list1.next
                else:
                    p.next.next = list2
                    return list0.next
            p = p.next
# 68ms 17.9MB
```

* 递归（移除各链表首节点最小者并递归）

时间复杂度：$<O(nk)$

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return
        if None in lists:
            return self.mergeKLists([l for l in lists if l])
        minimum = 10000
        for i, l in enumerate(lists):
            if l.val < minimum:
                minimum = l.val
                indexs = [i]
            elif l.val == minimum:
                indexs.append(i)
        list0 = ListNode()
        p = list0
        for i in indexs:
            p.next = lists[i]
            lists[i] = lists[i].next
            p = p.next
        p.next = self.mergeKLists(lists)
        return list0.next
# 148ms 22.8MB
```

### [25.K个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

* 迭代（翻转分组组装）

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1:
            return head
        if not (head and head.next):
            return head

        list_len = 2
        current_node = head
        next_node = head.next
        current_node.next = None
        while True:
            if next_node.next:
                list_len += 1
                next_next_node = next_node.next
                next_node.next = current_node
                current_node = next_node
                next_node = next_next_node
            else:
                next_node.next = current_node
                head = next_node
                break

        if list_len <= k:
            return head

        remainder = list_len % k
        result_head = ListNode()
        for _ in range(remainder):
            tail = result_head.next
            result_head.next = head
            head = head.next
            result_head.next.next = tail

        while head:
            tail = result_head.next
            result_head.next = head
            for _ in range(k - 1):
                head = head.next
            next_head = head.next
            head.next = tail
            head = next_head

        return result_head.next
# 48ms 16.1MB
```

* 迭代（顺序分组翻转）

只使用 $O(1)$ 的额外内存空间。

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1:
            return head
        if not (head and head.next):
            return head

        def reverse_group(group_head: ListNode):
            current_node = group_head
            next_node = group_head.next
            group_head.next = None
            while True:
                if next_node.next:
                    next_next_node = next_node.next
                    next_node.next = current_node
                    current_node = next_node
                    next_node = next_next_node
                else:
                    next_node.next = current_node
                    return next_node

        group_head = head
        current_node = head
        next_group_head = None
        prev_group_tail = None
        while True:
            to_break = False
            for _ in range(k - 1):
                current_node = current_node.next
                if not current_node:
                    to_break = True
                    break
            if to_break:
                if prev_group_tail:
                    prev_group_tail.next = group_head
                break

            next_group_head = current_node.next
            current_node.next = None
            new_group_head = reverse_group(group_head)
            if prev_group_tail:
                prev_group_tail.next = new_group_head
            else:
                head = new_group_head
            if next_group_head:
                prev_group_tail = group_head
                group_head = next_group_head
                current_node = next_group_head
            else:
                break

        return head
# 64ms 15.8MB
```

### [42.接雨水](https://leetcode.cn/problems/trapping-rain-water/)

!!! tip "提示"
    任意索引位置接雨水的量为其两侧的最大值中的较小值与该位置的值的差值。

设列表长度为 $n$。

* 迭代

时间复杂度：$O(2n)$

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        right_max = 0
        right_maxs = []
        for i in height[::-1]:
            right_maxs.append(right_max)
            right_max = max(right_max, i)
        right_maxs.reverse()
        rain = 0
        left_max = 0
        for h, right_max in zip(height, right_maxs):
            m = min(left_max, right_max)
            if h < m:
                rain += m - h
            left_max = max(left_max, h)
        return rain
# 64ms 16.4MB
```

## 分治算法

## 动态规划

### 45.跳跃游戏

### 72.编辑距离

### [887.鸡蛋坠落](https://leetcode.cn/problems/super-egg-drop/)

```python

```

### 刷木板

## 贪心算法

## 回溯算法

### [37.解数独](https://leetcode.cn/problems/sudoku-solver/)

* 回溯算法

模仿人类解数独的行为，为每个未确定值的位置维护了一个候选值的集合，每次迭代从最短集合开始处理。这种做法增加了内存消耗，但加快了解题速度，即以空间换取时间。

```python
import copy

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        elements = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

        def column_list(board: List[List[Any]], column: int) -> List[Any]:
            return [board[row][column] for row in range(9)]

        def block_list(board: List[List[Any]], row: int, column: int) -> List[Any]:
            block_row = row // 3
            block_column = column // 3
            blist = []
            for row in range(block_row * 3, block_row * 3 + 3):
                blist.extend(board[row][block_column * 3:block_column * 3 + 3])
            return blist

        for row, row_list in enumerate(board):
            for column in range(9):
                if row_list[column] == '.':
                    candidates = elements.copy()
                    for i in row_list:
                        candidates.discard(i)
                    for i in column_list(board, column):
                        candidates.discard(i)
                    for i in block_list(board, row, column):
                        candidates.discard(i)
                    row_list[column] = candidates

        def update_candidate(board: List[List[Any]], row: int, column: int, value: str) -> bool:
            # row remove candidate
            for i in board[row]:
                if isinstance(i, set):
                    i.discard(value)
                    if len(i) == 0:
                        return False

            # column remove candidate
            for row_i in range(9):
                if isinstance(board[row_i][column], set):
                    board[row_i][column].discard(value)
                    if len(board[row_i][column]) == 0:
                        return False
                    
            # block remove candidate
            block_row = row // 3
            block_column = column // 3
            for row_i in range(block_row * 3, block_row * 3 + 3):
                for column_i in range(block_column * 3, block_column * 3 + 3):
                    if isinstance(board[row_i][column_i], set):
                        board[row_i][column_i].discard(value)
                        if len(board[row_i][column_i]) == 0:
                            return False

            return True

        def try_solution(board: List[List[Any]]) -> None:
            updated = False
            for set_len in range(1, 9):
                for row, row_list in enumerate(board):
                    for column in range(9):
                        if isinstance(row_list[column], set) and len(row_list[column]) == set_len:
                            updated = True
                            if set_len > 1:
                                board_backup = copy.deepcopy(board)
                                candidates = row_list[column]
                                for candidate in candidates:
                                    for row_i, row_list in enumerate(board):
                                        row_list.clear()
                                        for i in board_backup[row_i]:
                                            if isinstance(i, set):
                                                row_list.append(i.copy())
                                            else:
                                                row_list.append(i)
                                    board[row][column] = candidate
                                    ok = update_candidate(board, row, column, candidate)
                                    if not ok:
                                        continue
                                    ok = try_solution(board)
                                    if ok:
                                        return True  # return
                                return False         # backtrace
                            else:
                                value = row_list[column].pop()
                                row_list[column] = value
                                ok = update_candidate(board, row, column, value)
                                if not ok:
                                    return False     # backtrace
                if updated:
                    break
                else:
                    done = True
                    for row_list in board:
                        for element in row_list:
                            if isinstance(element, set):
                                done = False
                                break
                        if not done:
                            break
                    if done:
                        return True                  # return

            return try_solution(board)

        try_solution(board)
# 44ms 15.4MB
```

### [51.N皇后](https://leetcode.cn/problems/n-queens/)

* 回溯算法

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        solutions = []
        solution = [None] * n

        def try_solution(row: int) -> None:
            for candidate in range(n):
                if candidate in solution[:row]:
                    continue
                
                to_continue = False
                for prev_row in range(row):
                    if abs(candidate - solution[prev_row]) == row - prev_row:
                        to_continue = True
                        break
                if to_continue:
                    continue
                    
                if row < n - 1:
                    solution[row] = candidate
                    try_solution(row + 1)
                else:
                    solution[row] = candidate
                    solutions.append(format_solution())

        def format_solution() -> List[str]:
            return [ i * '.' + 'Q' + (n - 1 - i) * '.' for i in solution]

        try_solution(0)

        return solutions
# 68ms 15.1MB
```

### [679.24点游戏](https://leetcode.cn/problems/24-game/)



## 组合数学

### 排两队
