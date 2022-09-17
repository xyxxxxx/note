# Leetcode 例题

## 迭代和递归

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
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:


```

* 递归（移除各链表首节点最小者并递归）

时间复杂度：

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

## 分治算法

## 动态规划

### [887.鸡蛋坠落](https://leetcode.cn/problems/super-egg-drop/)

```python

```

### 刷木板

## 贪心算法

## 回溯算法

### [37.解数独]()

## 组合数学

### 排两队
