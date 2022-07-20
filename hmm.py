# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    # initialize array alpha
    Alpha = np.zeros((N, T))
    for t in range(T):
        for n in range(N):
            if t == 0:  # given initial value
                Alpha[n][t] = pi[n] * B[n][O[t]]
            else:
                Alpha[n][t] = np.dot([alpha[t - 1] for alpha in Alpha], [a[n] for a in A]) * B[n][O[t]]
    # sum up those states
    prob = np.sum([alpha[T - 1] for alpha in Alpha])

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    Add:
        Beta:((N,T))) result array
    P.S.
        The time T of the beta matrix decreases from left to right, and the rightmost side is the initial time.
        There is no other reason, just because it is convenient to write code in this way.:)
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    Beta = np.zeros((N, T))
    for n in range(N): # initialize beta[T]
        Beta[n][0] = 1
    for t in range(1,T): # time(col) from t-1 to 1
        for n in range(N): # row
            Beta[n][t] = np.sum([A[n][j] * B[j][O[T - t]] * Beta[j][t - 1] for j in range(N)])

    # sum up initial states
    prob=np.sum([pi[n] * B[n][O[0]] * Beta[n][T-1] for n in range(N)])

    # End Assignment
    return prob


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    Add:
        result:(t2,t3,..,tT),temp array
        Delta:((N,T)),the probability of the current sequence
        Psi:((N,T)),the max probability sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    Delta = np.zeros((N, T), dtype=np.float64)
    Psi = np.zeros((N, T), dtype=np.int32)

    for n in range(N):  # initialize
        Delta[0][n] = pi[n] * B[n][0]
        for t in range(1, T):
            result = [Delta[t - 1][j] * A[j][t] for j in range(N)]
            Psi[t][n] = int(np.argmax(result))
            Delta[t][n] = result[Psi[t][n]] * B[n][O[t]]

    best_path = np.zeros((T), dtype=np.int32)

    best_path[T - 1] = np.argmax(Delta[T - 1])
    for t in range(T - 2, -1, -1):
        best_path[t] = Psi[t + 1][best_path[t + 1]]

    best_prob = Delta[best_path[1]][best_path[T-1]]
    # Convert array subscripts to matrix ordinal numbers
    best_path = [(val + 1) for val in best_path]

    # End Assignment
    return best_prob, best_path

if __name__ == "__main__":
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)
