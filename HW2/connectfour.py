from typing import Pattern, Union, Tuple, List, Any

import numpy as np
import numpy.typing as npt
import re

"""
HOMEWORK BOILERPLATE CODE START: PLEASE DO NOT MODIFY ANYTHING IN THIS SECTION
"""
Numeric = Union[float, int, np.number, None]

"""
Utility function: Return a list of all consecutive board positions in state that satisfy regex
"""
def k_in_row(state: npt.NDArray, regex: str) -> List[str]:
    # Return a list of all consecutive board positions in state that satisfy regex
    flipped = np.fliplr(state)
    sequences = []

    for i in range(state.shape[0]):
        sequences.extend(re.findall(regex, "".join(state[i])))
        sequences.extend(re.findall(regex, "".join(np.diag(state, k=-i))))
        sequences.extend(re.findall(regex, "".join(np.diag(flipped, k=-i))))

    for j in range(state.shape[1]):
        sequences.extend(re.findall(regex, "".join(state[:, j])))
        if j != 0:
            sequences.extend(re.findall(regex, "".join(np.diag(state, k=j))))
            sequences.extend(re.findall(regex, "".join(np.diag(flipped, k=j))))

    return sequences


"""
Functions to be used by alpha-beta search
"""
def terminal(state: npt.NDArray, k: int) -> Numeric:
    # If the given state is terminal, return computed utility (positive for X win, negative for O win, 0 for draw)
    # Otherwise, return None
    if k_in_row(state, "X{" + str(k) + "}"):
        return 1
    if k_in_row(state, "O{" + str(k) + "}"):
        return -1
    if np.count_nonzero(state == ".") == 0:
        return 0
    return None


def eval(state: npt.NDArray, k: int) -> Numeric:
    # Evaluate a non-terminal state based on both players' potential for winning the game
    score = terminal(state, k)
    if score is not None:
        return score

    score = 0
    possible_Xseq = k_in_row(state, "[X\.]{" + str(k) + ",}")
    possible_Oseq = k_in_row(state, "[O\.]{" + str(k) + ",}")
    score += sum([len(x) * x.count("X") for x in possible_Xseq])
    score -= sum([len(o) * o.count("O") for o in possible_Oseq])
    if score != 0:
        maxstr = max(possible_Xseq + possible_Oseq, key=len)
        score /= k * len(maxstr) * (len(possible_Xseq) + len(possible_Oseq))

    return score


"""
HOMEWORK BOILERPLATE CODE END
"""


"""
4.1: WRITE THIS FUNCTION
"""
def successors(state: npt.NDArray, player: str) -> List[npt.NDArray]:
    # Given board state (2d NumPy array) and player to move, return list of all possible successor states
    succ = []
    rows, cols = state.shape
    for c in range(cols):
        for r in range(rows - 1, -1, -1):
            if state[r][c] == ".":
                s = np.copy(state)
                s[r][c] = player
                succ.append(s)
                break
    return succ


"""
Alpha-beta depth-limited search
Params: Board state (2d NumPy array), player ('X' or 'O'), connect-k value, optional maximum search depth
Return: Value and best successor state
PLEASE DO NOT MODIFY ANYTHING IN THIS FUNCTION
"""
def alpha_beta_search(
    state: npt.NDArray,
    player: str,
    k: int,
    max_depth: Numeric
) -> Tuple[Numeric, npt.NDArray]:
    if player == "X":
        value, next = max_value(state, -float("inf"), float("inf"), k, 0, max_depth)
    else:
        value, next = min_value(state, -float("inf"), float("inf"), k, 0, max_depth)
    return value, next


"""
4.2: WRITE THIS FUNCTION
""" 
def max_value(
    state: npt.NDArray,
    alpha: float,
    beta: float,
    k: int,
    depth: int,
    max_depth: Numeric,
) -> Tuple[Numeric, npt.NDArray]:
    if terminal(state,k) != None:
        return terminal(state,k), None
    if depth == max_depth:
        return eval(state,k), None
    v = -float("inf")
    move: npt.NDArray = None
    """sort successors from largest to smallest"""
    succ = successors(state, "X")
    sort_succ: List[(npt.NDArray, Numeric)] = []
    for s in  succ:
        sort_succ.append((s,eval(s,k)))
    sort_succ.sort(key = lambda x: x[1], reverse = True)
    for s, x in sort_succ:
        v2, move2 = min_value(s,alpha,beta,k,depth+1,max_depth)
        if v2 > v:
            v, move = v2, s 
            alpha = max(alpha, v)
        if v >= beta:
            return v, move
    return v, move


"""
4.2: WRITE THIS FUNCTION
"""
def min_value(
    state: npt.NDArray,
    alpha: float,
    beta: float,
    k: int,
    depth: int,
    max_depth: Numeric,
) -> Tuple[Numeric, npt.NDArray]:
    if terminal(state,k) != None:
        return terminal(state,k), None
    if depth == max_depth:
        return eval(state,k), None
    v = float("inf")
    move: npt.NDArray = None
    """sort successors from smallest to largest"""
    succ = successors(state, "O")
    sort_succ: List[(npt.NDArray, Numeric)] = []
    for s in succ:
        sort_succ.append((s,eval(s,k)))
    sort_succ.sort(key = lambda x: x[1])
    #print(sort_succ)
    for s, x in sort_succ:
        v2, move2 = max_value(s,alpha,beta,k,depth+1,max_depth)
        if v2 < v:
            v, move = v2, s
            beta = min(beta, v)
        if v <= alpha:
            return v, move 
    return v, move

"""
Set parameters in main function, which will call game_loop to simulate a game
PLEASE DO NOT MODIFY ANYTHING IN THIS FUNCTION
"""
def game_loop(
    m: int,
    n: int,
    k: int,
    x_max_depth: Numeric = float("inf"),
    o_max_depth: Numeric = float("inf")
):
    # Play a Connect-k game given grid size (mxn)
    # Optional search depth parameters for player X and player O
    state = np.full((m, n), ".")
    print("Connect", k, "on a", m, "by", n, "board")
    player = "X"

    while state is not None:
        print(np.matrix(state), "\n")
        if player == "X":
            value, state = alpha_beta_search(state, player, k, x_max_depth)
            player = "O"
        else:
            value, state = alpha_beta_search(state, player, k, o_max_depth)
            player = "X"

    if value > 0:
        print("X wins!")
    elif value < 0:
        print("O wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    m, n, k = 6, 7, 4
    game_loop(m, n, k, 6, 2)
    
"""
test = np.array([["X",".",".","."],["X",".",".","."],["X",".",".","."],["X","X","O","O"]])
for s in successors(test, "X"):
    print(s)"""

