# 迭代和递归

## [3.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

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

## [21.合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

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

## [23.合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

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

## [25.K个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

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

## [42.接雨水](https://leetcode.cn/problems/trapping-rain-water/)

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

## [50.Pow(x, n)](https://leetcode.cn/problems/powx-n/)

!!! tip "提示"
    设 $n$ 的二进制展开为 $b_1b_2\cdots b_k$，则有

    $$x^n=((1×x^{b_1})^2×x^{b_2})^2\cdots ×x^{b_k}$$

    由此可归纳出如下递推式：

    ```
    myPow(x, n) = myPow(x, n >> 1)**2 * x, if n & 1 == 1
                  myPow(x, n >> 1)**2    , if n & 1 == 0
    ```

* 递归

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.
        if n < 0:
            return self.myPow(1 / x, -n)

        return self.myPow(x, n >> 1)**2 * x if n & 1 else self.myPow(
            x, n >> 1)**2
# 32ms 14.8MB
```
