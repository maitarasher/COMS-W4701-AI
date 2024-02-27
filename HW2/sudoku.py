from typing import Tuple, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from random import sample

"""
Sudoku board initializer
Credit: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
HOMEWORK BOILERPLATE CODE START: PLEASE DO NOT MODIFY ANYTHING IN THIS SECTION
"""
def generate(n: int, num_clues: int) -> dict:
    # Generate a sudoku problem of order n with "num_clues" cells assigned
    # Return dictionary containing clue cell indices and corresponding values
    # (You do not need to worry about components inside returned dictionary)
    N = range(n)

    rows = [g * n + r for g in sample(N, n) for r in sample(N, n)]
    cols = [g * n + c for g in sample(N, n) for c in sample(N, n)]
    nums = sample(range(1, n**2 + 1), n**2)

    S = np.array(
        [[nums[(n * (r % n) + r // n + c) % (n**2)] for c in cols] for r in rows]
    )
    indices = sample(range(n**4), num_clues)
    values = S.flatten()[indices]

    mask = np.full((n**2, n**4), True)
    mask[:, indices] = False
    i, j = np.unravel_index(indices, (n**2, n**2))

    for c in range(num_clues):
        v = values[c] - 1
        maskv = np.full((n**2, n**2), True)
        maskv[i[c]] = False
        maskv[:, j[c]] = False
        maskv[
            (i[c] // n) * n : (i[c] // n) * n + n, (j[c] // n) * n : (j[c] // n) * n + n
        ] = False
        mask[v] = mask[v] * maskv.flatten()

    return {"n": n, "indices": indices, "values": values, "valid_indices": mask}


def display(problem: dict):
    # Display the initial board with clues filled in (all other cells are 0)
    n = problem["n"]
    empty_board = np.zeros(n**4, dtype=int)
    empty_board[problem["indices"]] = problem["values"]
    print("Sudoku puzzle:\n", np.reshape(empty_board, (n**2, n**2)), "\n")


def initialize(problem: dict) -> npt.NDArray:
    # Returns a random initial sudoku board given problem
    n = problem["n"]
    S = np.zeros(n**4, dtype=int)
    S[problem["indices"]] = problem["values"]

    all_values = list(np.repeat(range(1, n**2 + 1), n**2))
    for v in problem["values"]:
        all_values.remove(v)
    all_values = np.array(all_values)
    np.random.shuffle(all_values)

    indices = [i for i in range(S.size) if i not in problem["indices"]]
    S[indices] = all_values
    S = S.reshape((n**2, n**2))

    return S


def successors(S: npt.NDArray, problem: dict) -> List[npt.NDArray]:
    # Returns list of all successor states of S by swapping two non-clue entries
    mask = problem["valid_indices"]
    indices = [i for i in range(S.size) if i not in problem["indices"]]
    succ = []

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            s = np.copy(S).flatten()
            if s[indices[i]] == s[indices[j]]:
                continue
            if not (
                mask[s[indices[i]] - 1, indices[j]]
                and mask[s[indices[j]] - 1, indices[i]]
            ):
                continue
            s[indices[i]], s[indices[j]] = s[indices[j]], s[indices[i]]
            succ.append(s.reshape(S.shape))

    return succ


"""
HOMEWORK BOILERPLATE CODE END
"""


"""
WRITE THIS FUNCTION
"""
def num_errors(S: npt.NDArray) -> int:
    # Given a current sudoku board state (2d NumPy array), compute and return total number of errors
    # Count total number of missing numbers from each row, column, and non-overlapping square blocks
    row_numbers: Set[int] = set()
    col_numbers: Set[int] = set()
    sub_numbers: Set[int] = set()
    rows, cols = S.shape
    erros = 0 
    """Go over rows and cols (square matrix)""" 
    for r in range (rows):
        row_numbers.clear()
        col_numbers.clear()
        for c in range (cols):
            if S[r][c] > 0:
                row_numbers.add(S[r][c])
            if S[c][r] > 0:
                col_numbers.add(S[c][r])
        erros += cols - len(row_numbers)
        erros += rows - len(col_numbers)
    """Go over subgrid"""
    for r in range(rows):
        for c in range (cols):
            if r % np.sqrt(rows) == 0 and c % np.sqrt(cols) == 0:
                """new subgrid"""
                sub_numbers.clear()
                for i in range(r, r+int(np.sqrt(rows))):
                    for j in range(c, c+int(np.sqrt(cols))):
                        if S[i][j] > 0:
                            sub_numbers.add(S[i][j])
                erros += rows - len(sub_numbers)
    return erros


"""
WRITE THIS FUNCTION
"""
def hill_climb(
    problem: dict,
    max_sideways: int = 0,
    max_restarts: int = 0
) -> Tuple[npt.NDArray, List[int]]:
    # Given: Sudoku problem and optional max sideways moves and max restarts parameters
    # Return: Board state solution (2d NumPy array), list of errors in each iteration of hill climbing search
    my_board = initialize(problem)
    erros_list: List[int] = []
    board_erros = num_errors(my_board)
    """for 3.4 sideways"""
    neighbors: List[npt.NDArray] = []
    sidemoves: int = 0 
    """for 3.5 random restarts"""
    restarts: int = 0 
    
    while board_erros != 0:
        erros_list.append(board_erros)
        """check corresponding num of erros of the states in the successors list"""
        succ = successors(my_board, problem)
        for s in succ:
            s_erros = num_errors(s)
            if s_erros < board_erros:
                my_board = s
                board_erros = s_erros
            """for 3.4 sideways"""
            if s_erros == board_erros:
                neighbors.append(s) 
        """for 3.4- if a better state wasn't found, 
        and best neighbors all have the same number of conflicts as the current state,
        we can move to a random one (without 3.4, I would break if condition holds)"""
        if erros_list[-1] == board_erros:
            if neighbors and sidemoves < max_sideways:
                my_board = sample(neighbors,1)[0]
                sidemoves += 1
            else: 
                """for 3.5- a better state wasn't found and/or sidemoves == max_sideways, 
                and didnt reach max restarts"""
                if restarts < max_restarts:
                    my_board = initialize(problem)
                    board_erros = num_errors(my_board)
                    restarts += 1
                    sidemoves = 0
                else:
                    break
    if board_erros == 0:
        erros_list.append(board_erros)

    return my_board,  erros_list


if __name__ == "__main__":
    n = 3
    clues = 40
    problem = generate(n, clues)
    display(problem)
    sol, errors = hill_climb(problem, 15, 20)
    print("Solution:\n", sol)
    plt.plot(errors)
    plt.show()
    """print("\nErros list:",errors)"""
    print("Num of erros:", num_errors(sol))

#My Tests
"""
test = np.array([[0,0,0,0],[4,1,3,2],[0,0,0,0],[0,0,2,0]])
print(num_errors(test))"""

#Tests for 3.3 and 3.4 3.5
"""
success = 0 
total_errors = 0
for i in range(100):
    print(i)
    n = 2
    clues = 5
    problem = generate(n, clues)
    display(problem)
    sol, errors = hill_climb(problem, 10,10)
    print("Solution:\n", sol)
    if num_errors(sol) == 0:
        success += 1
        print("sol found")
    else:
        print(errors[-1])
        total_errors += errors[-1]
print("Success rate- out of 100 tests", success, "times a solution was found (",success,"%)")
print("Average error- ", total_errors/100)
    """
