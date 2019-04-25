from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

'''
def wp_similarity(x: list, y: list):
    pass
'''
#The Jaccard similarity
def jaccard_similarity(x: list, y: list):
    c11 = 0
    c10 = 0
    c01 = 0
    for i in range(len(x)):
        if x[i] == 1 and y[i] == 1:
            c11 = c11 + 1
        elif x[i] == 1 and y[i] == 0:
            c10 = c10 + 1
        elif x[i] == 0 and y[i] == 1:
            c01 = c01 + 1
    return c11/(c10 + c01 + c11)

def jaccard_distance(x: list, y: list):
    return 1 - jaccard_similarity(x, y)
####################################################################################
#The pearson correlation
def pear_corr(arr: np.array) -> np.array:
    return np.array(
        cosine_similarity(
            [
                list(map(lambda x:x-avg if x!=0 else 0, line))
                for line in arr
                for avg in [np.sum(line)/np.count_nonzero(line)]
            ]
        )
    )
####################################################################################

def save_dist_matrix(filename: str, matrix: np.array):
    np.savetxt(filename, matrix, delimiter=',')

def load_dist_matrix(filename: str):
    return np.genfromtxt(filename, delimiter=',')


save_dist_matrix(
    'test_mat.csv',
    np.array([
        [1, 3, 2],
        [10,30,20]
    ])
)

print(load_dist_matrix('test_mat.csv'))