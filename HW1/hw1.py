import math
import string
from dataclasses import dataclass
from heapq import heapify, heappush, heappop
from collections import deque
from typing import Callable, Tuple, Dict, Set, List, Optional, cast

# Mark this as `False` if you run into problems installing enchant
USE_ENCHANT = False

"""
HOMEWORK BOILERPLATE CODE START: PLEASE DO NOT MODIFY ANYTHING IN THIS SECTION
"""
INTINF = cast(int, math.inf)


def get_wordchecker(use_enchant: bool) -> Callable[[str], bool]:
    if use_enchant:
        import enchant

        ench_dict = enchant.Dict("en_US")
        wordchecker = lambda word: ench_dict.check(word)
    else:
        import nltk
        from nltk.corpus import words

        nltk.download("words")
        nltk_dict = set(words.words())
        wordchecker = lambda word: word in nltk_dict
    return wordchecker


wordchecker = get_wordchecker(USE_ENCHANT)

# What is `@dataclass`?? Read about this here:
# https://realpython.com/python-data-classes/
@dataclass
class Node:
    state: str
    parent: Optional["Node"]
    cost: int


def successors(state: str) -> List[Tuple[int, str]]:
    child_states: List[Tuple[int, str]] = []
    for i in range(len(state)):
        new = [state[:i] + x + state[i + 1 :] for x in string.ascii_lowercase]
        words = [x for x in new if wordchecker(x) and x != state]
        child_states = child_states + [(i, word) for word in words]

    return child_states


def sequence(node: Node) -> List[str]:
    words = [node.state]
    while node.parent:
        node = node.parent
        words.insert(0, node.state)

    return words


"""
HOMEWORK BOILERPLATE CODE END
"""


"""
5.1: Depth-limited depth-first search
> You may use the `expand()` function in other sections. 
"""


def depth_limited_dfs(
    start: str, goal: str, depth: int
) -> Tuple[Optional[Node], int, List[int]]:
    node = Node(state=start, parent=None, cost=0)
    frontier: List[Node] = []
    frontier_size: List[int] = []
    reached: Set[str] = set()
    nodes_expanded = 0
    goal_node = None

    # YOUR CODE HERE
    frontier.append(node)
    reached.add(node.state)
    while len(frontier) != 0:
        frontier_size.append(len(frontier))
        popped_node = frontier.pop(-1)
        if popped_node.cost < depth:
            nodes_expanded += 1
            for child in expand(popped_node):
                if child.state == goal:
                    goal_node = child
                    return goal_node, nodes_expanded, frontier_size
                if child.state not in reached:
                        reached.add(child.state)
                        frontier.append(child)
                        
    return goal_node, nodes_expanded, frontier_size


def expand(node: Node) -> List[Node]:
    succs: List[Node] = []
    
    # YOUR CODE HERE
    words = successors(node.state)
    for word in words:
        succs.append(Node(word[1], node, node.cost+1))

    return succs


"""
5.1 END
"""


"""
5.2: Iterative deepening
"""


def iterative_deepening(
    start: str, goal: str, max_depth: int
) -> Tuple[Optional[Node], int, List[int]]:
    node = Node(state=start, parent=None, cost=0)
    frontier_size: List[int] = []
    nodes_expanded = 0
    goal_node = None

    # YOUR CODE HERE
    for i in range(1,max_depth+1):
        dfs = depth_limited_dfs(start,goal,i)
        frontier_size.extend(dfs[2])
        nodes_expanded += dfs[1]
        if dfs[0] != None:
            goal_node = dfs[0]
            return goal_node, nodes_expanded, frontier_size
   
    return goal_node, nodes_expanded, frontier_size


"""
5.2 END
"""


"""
5.3: A* search
"""

def astar_search(start: str, goal: str) -> Tuple[Optional[Node], int, List[int]]:
    node = Node(state=start, parent=None, cost=0)
    frontier: List[Tuple[int, str, Node]] = []
    frontier_size: List[int] = [0]
    reached: Dict[str, Node] = {}
    nodes_expanded = 0
    goal_node = None

    # YOUR CODE HERE
    heappush(frontier, (0, node.state, node))
    reached[node.state] = node
    while len(frontier) != 0:
        frontier_size.append(len(frontier))
        popped_item = heappop(frontier)
        if popped_item[1] == goal:
            goal_node = popped_item[2]
            return goal_node, nodes_expanded, frontier_size
        nodes_expanded += 1
        for child in expand(popped_item[2]):
            f_child_value = child.cost + compare_words(child.state,goal)
            if child.state in reached:
                current_node = reached.get(child.state)
                f_current_node = current_node.cost + compare_words(current_node.state,goal)
                if f_child_value < f_current_node:
                    reached[child.state] = child
                    """"Update item in frontier as well"""
                    frontier.remove((f_current_node,current_node.state,current_node))
                    heappush(frontier,(f_child_value,child.state, child))
            else: 
                reached[child.state]=child
                heappush(frontier,(f_child_value,child.state, child))
    

    return goal_node, nodes_expanded, frontier_size

def compare_words(str_1: str, str_2: str):
    count = 0
    for i in range(len(str_1)):
        if str_1[i] != str_2[i]:
            count += 1
    return count
        

"""
5.3 END
"""

if __name__ == "__main__":
    sol = depth_limited_dfs("fat", "cop",6)
    if sol[0]:
        print("DFS:", sequence(sol[0]), sol[1])


import numpy as np
import matplotlib.pyplot as plt

# My Tests
"""
sol_1 = depth_limited_dfs("fat", "cop",8)
print(sequence(sol_1[0]))
if sol_1[0]:
    print("Solution length:", len(sequence(sol_1[0]))-1)
    #print("Frontier size:", sol_1[2])
    print("Node depth:", sol_1[0].cost)
print("Nodes expanded:", sol_1[1])
sol_2 = depth_limited_dfs("cold", "warm",8)
if sol_2[0]:
    print(sequence(sol_2[0]))
    print("Solution length:", len(sequence(sol_2[0]))-1)
    #print("Frontier size:", sol_2[2])
    print("Node depth:", sol_2[0].cost)
print("Nodes expanded:", sol_2[1])
sol_3 = depth_limited_dfs("small", "large",8)
if sol_3[0]:
    print(sequence(sol_3[0]))
    print("Solution length:", len(sequence(sol_3[0]))-1)
    #print("Frontier size:", sol_3[2])
    print("Len frontier size:", len(sol_3[2]))
    print("Node depth:", sol_3[0].cost)
print("Nodes expanded:", sol_3[1])

#x = np.arange(0, len(sol_1)+1, 1)
#y1 = np.array(sol_1[2])
#y2 = np.array(sol_2[2])
y3 = np.array(sol_3[2])
#plt.plot(y1)
#plt.plot(y2)
plt.plot(y3)
#plt.show()
"""