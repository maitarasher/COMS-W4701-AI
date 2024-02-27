from typing import Pattern, Union, Tuple, List, Dict, Any

import numpy as np
import numpy.typing as npt

"""
Some type annotations
"""
Numeric = Union[float, int, np.number, None]


"""
Global list of parts of speech
"""
POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
       'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

"""
Utility functions for reading files and sentences
"""
def read_sentence(f):
    sentence = []
    while True:
        line = f.readline()
        if not line or line == '\n':
            return sentence
        line = line.strip()
        word, tag = line.split("\t", 1)
        sentence.append((word, tag))

def read_corpus(file):
    f = open(file, 'r', encoding='utf-8')
    sentences = []
    while True:
        sentence = read_sentence(f)
        if sentence == []:
            return sentences
        sentences.append(sentence)


"""
3.1: Supervised learning
Param: data is a list of sentences, each of which is a list of (word, POS) tuples
Return: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities} 
"""
def learn_model(data:List[List[Tuple[str]]]
                ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str,npt.NDArray]]:
    X0 = np.zeros(17)
    Tprob = np.zeros((17,17))
    Oprob: Dict[str,npt.NDArray] = {}
    for sentence in data:
        for i in range(len(sentence)):
            word = sentence[i][0]
            pos =  sentence[i][1]
            index = POS.index(pos)
            """count POS in first word"""
            if i == 0:
                X0[index] += 1
            """count the number of POS to POS transitions"""
            if i < (len(sentence) - 1):
                next_index = POS.index(sentence[i+1][1])
                Tprob[next_index][index] += 1
            """count the number of POS to word observations"""
            if word not in Oprob:
                pos_array = np.zeros(17)
                pos_array[index] += 1
                Oprob[word] = pos_array
            else:
                Oprob[word][index] += 1
    """normalize X0"""
    for i in range(17):
        X0[i] /= len(data)
    """normalize Tprob"""
    for c in range(17):
        s = Tprob[:,c].sum()
        for r in range(17):
            Tprob[r][c] /= s
    """normalize Oprob"""
    for key in Oprob:
        value = Oprob[key]
        s = np.sum(value)
        for i in range (17):
            value[i] /= s
        Oprob[key] = value
    #print(Oprob["have"])
    return X0, Tprob, Oprob


"""
3.2: Viterbi forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: m, 1D array; pointers, 2D array
"""
def viterbi_forward(X0:npt.NDArray,
                    Tprob:npt.NDArray,
                    Oprob:Dict[str,npt.NDArray],
                    obs:List[str]
                    ) -> Tuple[npt.NDArray, npt.NDArray]:
    n = X0.size
    t = len(obs)
    pointers = np.zeros((t,n))
    pointers = pointers.astype(int)
    m = X0.copy()
    for i in range(t):
        """calculate m prime and pointer first"""
        mp = np.zeros(n)
        for r in range(n):
            max_p = 0
            for c in range(n):
                s = Tprob[r,c]*m[c]
                if s > max_p:
                    max_p = s
                    mp[r] = s
                    pointers[i,r] = c
        """create observation matrix"""
        if obs[i] in Oprob:
            v = Oprob[obs[i]]   
        else: 
            v = np.ones(n)
        o = np.diag(v)
        m = np.matmul(o,mp)     
    return m, pointers

"""
3.2: Viterbi backward algorithm
Param: m, 1D array; pointers, 2D array
Return: List of most likely POS (strings)
"""
def viterbi_backward(m:npt.NDArray,
                     pointers:npt.NDArray
                     ) -> List[str]:
    t = pointers.shape[0]
    sequence = np.zeros(t)
    sequence = sequence.astype(int)
    x = np.argmax(m)
    sequence[-1] = x
    for i in reversed(range(t-1)):
        sequence[i] = pointers[i+1,x]
    pos_list = []
    for i in range(t):
        pos_list.append(POS[sequence[i]])
    return pos_list


"""
3.3: Evaluate Viterbi by predicting on data set and returning accuracy rate
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of (word,POS) pairs
Return: Prediction accuracy rate

 Your Viterbi implementation can now be used for prediction. evaluate viterbi takes in the HMM
parameters, along with a data list in the same format as in learn model. Complete the function so
that it runs Viterbi on each sentence separately on the data set. Then compare all returned POS
predictions with the true POS in data. Compute and return the accuracy rate as the proportion
of correct predictions (this should be a number between 0 and 1). """
def evaluate_viterbi(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     data:List[List[Tuple[str]]]
                     ) -> float:
    correct_predictions = 0
    for sentence in data:
        words = []
        result = []
        for word, pos in sentence:
            words.append(word)
            result.append(pos)
        m, pointers = viterbi_forward(X0,Tprob,Oprob,words)
        viterbi_prediction = viterbi_backward(m,pointers)
        sentence_predictions = 0
        for i in range(len(sentence)):
            if viterbi_prediction[i] == result[i]:
                sentence_predictions += 1
        sentence_predictions /= len(sentence)
        correct_predictions += sentence_predictions
    correct_predictions /= len(data)
    return correct_predictions


"""
3.4: Forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: P(XT, e_1:T)
"""
def forward(X0:npt.NDArray,
            Tprob:npt.NDArray,
            Oprob:Dict[str,npt.NDArray],
            obs:List[str]
            ) -> npt.NDArray:
    n = X0.size
    t = len(obs)
    alpha = X0.copy()
    for i in range(t):
        alpha_tag = np.matmul(Tprob,alpha)
        if obs[i] in Oprob:
            v = Oprob[obs[i]]   
        else: 
            v = np.ones(n)
        o = np.diag(v)
        alpha = np.matmul(o,alpha_tag)
    return alpha

"""
3.4: Backward algorithm
Param: Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(e_k+1:T | Xk)
"""
def backward(Tprob:npt.NDArray,
             Oprob:Dict[str,npt.NDArray],
             obs:List[str],
             k:int
             ) -> npt.NDArray:
    n = Tprob.shape[0] 
    t = len(obs)
    beta = np.ones(n)
    """traverse from the end of the list (t) to k (including k)"""
    for i in range(t-1,k,-1):
        if obs[i] in Oprob:
            v = Oprob[obs[i]]   
        else: 
            v = np.ones(n)
        o = np.diag(v)
        beta_tag = np.matmul(o,beta)
        beta = np.matmul(np.transpose(Tprob),beta_tag)
    return beta

"""
3.4: Forward-backward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(Xk | e_1:T)
"""
def forward_backward(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     obs:List[str],
                     k:int
                     ) -> npt.NDArray:
    alpha = forward(X0,Tprob,Oprob,obs[0:k+1])
    beta = backward(Tprob,Oprob,obs,k)
    v = np.multiply(alpha,beta)
    """normalize v"""
    s = np.sum(v)
    for i in range(len(v)):
        v[i] /= s
    return v


if __name__ == "__main__":
    # Run below for 3.3
    train = read_corpus('train.upos.tsv')
    test = read_corpus('test.upos.tsv')
    X0, T, O = learn_model(train)
    print("Train accuracy:", evaluate_viterbi(X0, T, O, train))
    print("Test accuracy:", evaluate_viterbi(X0, T, O, test))

    # Write all code for 3.5 below
    p = ["The", "group", "could","never","have", "guessed", "what", "was", "in", "the", "mansion", "."]
    f = forward(X0,T,O,p[0:5])
    fb = forward_backward(X0, T, O, p, 4)
    print("forward:", f, "\n", POS[np.argmax(f)])
    print("forward backward:", fb, "\n", POS[np.argmax(fb)])
    
    
    """
    #My Tests
    x0 = np.array([0.5,0.5])
    t = np.array([[0.7,0.3],
                 [0.3,0.7]])
    o = {1: np.array([0.1,0.8]), 2:np.array([0.9,0.2]), 3:np.array([0.1,0.8])}
    obs = np.array([1,2,3])
    #Viterbi
    m, pointers = viterbi_forward(x0,t,o,obs)
    print( "m:", m,"\n", pointers)
    viterbi_prediction = viterbi_backward(m,pointers)
    print(viterbi_prediction)
    #Forward
    a = forward(x0,t,o,obs)
    print(a)
    #Backward 
    b = backward(t, o, obs, 2)
    print(b)
    #Forward Backward
    ab = forward_backward(x0, t, o, obs, 2)
    print(ab)
    #p = ["Hello", "world", "!", "How", "are", "you","feeling","today","?"]
    #p = ["This", "is", "my", "first", "test", ":"]
    #p = ["This", "is", "a", "beautiful", "building"]
    #p = ["Jhon", "Walorski", "was", "making", "a", "great", "deal"]
    #p = ["Despite", "going", "through", "widespread", "devastation", "during", "world", "war", "2", ",", "west", "Germany", "and", "Japan", "experienced", "miraculus", "ecconomic", "growth"]
    """
    
    
    
   
    
    