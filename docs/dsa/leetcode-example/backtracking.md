# 回溯算法

## [37.解数独](https://leetcode.cn/problems/sudoku-solver/)

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
                                return False         # backtrack
                            else:
                                value = row_list[column].pop()
                                row_list[column] = value
                                ok = update_candidate(board, row, column, value)
                                if not ok:
                                    return False     # backtrack
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

## [51.N皇后](https://leetcode.cn/problems/n-queens/)

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

## [679.24点游戏](https://leetcode.cn/problems/24-game/)

```python
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        if len(cards) == 1:
            a = cards[0]
            if isinstance(a, int):
                return a == 24
            else:
                return abs(a - 24) < 0.001

        if len(cards) == 2:
            a, b = cards
            if isinstance(a, int) and isinstance(b, int):
                if a * b == 0:
                    return a + b == 24
                else:
                    return a + b == 24 or abs(a - b) == 24 or a * b == 24 or (
                        a // b == 24 and a % b == 0) or (b // a == 24
                                                         and b % a == 0)
            else:
                if a * b == 0:
                    return abs(a * b - 24) < 0.001
                else:
                    return abs(a + b - 24) < 0.001 or abs(abs(
                        a - b) - 24) < 0.001 or abs(a * b - 24) < 0.001 or abs(
                            a / b - 24) < 0.001 or abs(b / a - 24) < 0.001

        def cards_after_operation(cards: List[int], first: int, second: int,
                                  op: str) -> List[int]:
            cards_ = cards.copy()
            a = cards_.pop(first)
            b = cards_.pop(second)
            if op == '+':
                cards_.append(a + b)
                return cards_
            if op == '-':
                diff = abs(a - b)
                if diff:
                    cards_.append(diff)
                return cards_
            if op == '*':
                cards_.append(a * b)
                return cards_
            if op == '/':
                cards_.append(a / b)
                return cards_
            if op == '/<':
                cards_.append(b / a)
                return cards_

        def try_operation(cards: List[int], first: int, second: int) -> bool:
            ok = self.judgePoint24(
                cards_after_operation(cards, first, second, '+'))
            if ok:
                return True
            ok = self.judgePoint24(
                cards_after_operation(cards, first, second, '-'))
            if ok:
                return True
            ok = self.judgePoint24(
                cards_after_operation(cards, first, second, '*'))
            if ok:
                return True
            ok = self.judgePoint24(
                cards_after_operation(cards, first, second, '/'))
            if ok:
                return True
            ok = self.judgePoint24(
                cards_after_operation(cards, first, second, '/<'))
            if ok:
                return True
            return False

        if len(cards) == 3:
            first = -1
            second = -1
            while True:
                ok = try_operation(cards, first, second)
                if ok:
                    return True
                if second > -2:
                    second -= 1
                elif first > -2:
                    first -= 1
                    second = first
                else:
                    break
            return False

        first = -1
        second = -1
        while True:
            ok = try_operation(cards, first, second)
            if ok:
                return True
            if second > -3:
                second -= 1
            elif first > -3:
                first -= 1
                second = first
            else:
                break
        return False
# 44ms 14.9MB
```
