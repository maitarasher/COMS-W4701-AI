"""
BOILERPLATE CODE: DO NOTE MODIFY
"""
from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

np.random.seed(4701)

def test_random_seed():
    rands = [np.random.randint(100) for _ in range(10)]
    rands_t = [20, 51, 4, 27, 4, 14, 35, 62, 19, 8]
    
    for r1, r2 in zip(rands, rands_t):
        assert r1 == r2
    
    rands = np.array(rands) 
    choices_t = [
        [62, 14, 4, 4, 4],
        [4, 62, 8, 4, 27],
        [4, 14, 51, 4, 62],
        [62, 35, 4, 35, 62],
        [20, 51, 35, 20, 8],
    ]
    for i in range(5):
        choices = np.random.choice(rands, 5)
        for c1, c2 in zip(choices, choices_t[i]):
            assert c1 == c2

    print("Random test done!")

test_random_seed()

"""
BOILERPLATE CODE END
"""

"""
5.1 Value iteration
"""

def value_iteration(
    V0: npt.NDArray, 
    lr: float, 
    gamma:float, 
    epsilon: float=1e-12
) -> npt.NDArray:
    max_norm = float('inf')
    draw_sum: float
    while max_norm > epsilon:
        Vi: npt.NDArray = np.zeros(V0.size)
        for s in range(V0.size):
            draw_sum = 0
            """possibile states from given state"""
            for x in range(1,11):
                if  x == 10:
                    p = 4/13
                else:
                    p = 1/13
                if s+x >= V0.size:
                    draw_sum += p*(lr + gamma*0)
                else:
                    draw_sum += p*(lr + gamma*V0[s+x])
            max_action = max(draw_sum,s)
            Vi[s] = max_action
        max_norm = np.max(np.subtract(Vi, V0))     
        V0 = Vi
    return V0

"""
5.1 END
"""


"""
5.2 Policy extraction
"""

def value_to_policy(
        V: npt.NDArray, lr: float, gamma: float
) -> npt.NDArray:
    draw_sum: float
    policy = np.zeros(V.size)
    for s in range (V.size):
        draw_sum = 0
        for x in range(1,11):
            if  x == 10:
                p = 4/13
            else:
                p = 1/13
            if s+x >= V.size:
                draw_sum += p*(lr + gamma*0)
            else:
                draw_sum += p*(lr + gamma*V[s+x])
        if draw_sum > s:
            policy[s] = 1
    return policy


"""
5.2 END
"""


"""
5.4 Temporal difference learning
"""

def Qlearn(
    Q0: npt.NDArray, 
    lr: float, 
    gamma: float, 
    alpha: float, 
    epsilon: float, 
    N: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    records = np.zeros((N,3))
    state = 0
    for i in range(N):
        """epsilon gridy policy"""
        if np.random.rand() > epsilon:
            action = np.argmax(Q0[state])
        else:
            action = np.random.randint(2)
        """check actions"""
        if  action == 1:
            records[i] = [state,action,lr]
            succ_state = state + draw()
        else:
            records[i] = [state,action,state]
            succ_state = 0
        """check successor state and update Q values"""
        if succ_state > 21 or action == 0:
            Q0[state,action] = Q0[state,action] + alpha*(records[i,2] - Q0[state,action])
            state = 0
        else:
            Q0[state,action] = Q0[state,action] + alpha*(records[i,2] + gamma*np.max(Q0[succ_state])-Q0[state,action])
            state = succ_state
    return Q0, records

"""
5.4 END
"""


"""
HOMEWORK BOILERPLATE CODE START: PLEASE DO NOT MODIFY ANYTHING IN THIS SECTION
"""

def draw() -> int:
    probs = 1/13*np.ones(10)
    probs[-1] *= 4
    return np.random.choice(np.arange(1,11), p=probs)


def RL_analysis():
    lr, gamma, alpha, epsilon, N = 0, 1, 0.1, 0.1, 10000
    visits = np.zeros((22,6))
    rewards = np.zeros((N,6))
    values = np.zeros((22,6))

    for i in range(6):
        _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, epsilon, 10000*i)
        vals, counts = np.unique(record[:,0], return_counts=True)
        visits[vals.astype(int),i] = counts
        _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, 0.2*i, N)
        rewards[:,i] = record[:,2]
        vals, _ = Qlearn(np.zeros((22,2)), lr, gamma, min(0.2*i+0.1,1), epsilon, N)
        values[:,i] = np.max(vals, axis=1)

    plt.figure()
    plt.plot(visits)
    plt.legend(['N=0', 'N=10k', 'N=20k', 'N=30k' ,'N=40k', 'N=50k'])
    plt.title('Number of visits to each state')

    plt.figure()
    plt.plot(np.cumsum(rewards, axis=0))
    plt.legend(['e=0.0', 'e=0.2', 'e=0.4' ,'e=0.6', 'e=0.8', 'e=1.0'])
    plt.title('Cumulative rewards received')

    plt.figure()
    plt.plot(values)
    plt.legend(['a=0.1' ,'a=0.3', 'a=0.5', 'a=0.7', 'a=0.9', 'a=1.0'])
    plt.title('Estimated state values');

"""
HOMEWORK BOILERPLATE CODE END
"""

if __name__ == "__main__":
    lr, gamma = 0, 1
    Vstar = value_iteration(np.zeros(22), lr, gamma)
    pistar = value_to_policy(Vstar, lr, gamma)
    #My Tests
    "print(Vstar)"
    plt.plot(Vstar)
    plt.show()
    "print(pistar)"
    plt.plot(pistar)
    plt.show()
    RL_analysis()
    """
    lr, gamma, alpha, epsilon, N = 0, 1, 0.3, 0.1, 10000
    Q0, records = Qlearn(np.zeros((22,2)), lr, gamma, alpha, epsilon, N)
    print(Q0)"""
