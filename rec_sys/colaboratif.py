from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import dbscan
from rec_sys.read_csv import load_data
#from semantic import um
import numpy as np
import fractions

#np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

def colaboratif_distance(arr: np.array) -> np.array:
    return 0.5*np.array(
        cosine_distances(
            [
                list(map(lambda x:x-avg if x!=0 else 0, line))
                for line in arr
                for avg in [np.sum(line)/np.count_nonzero(line)]
            ]
        )
    )

class Colaboratif:
    def __init__(self, users: np.array):
        self.user_sim_matrix = self.pear_corr(users)
    
    def pear_corr(self, arr: np.array) -> np.array:
        return 0.5*np.array(
            cosine_distances(
                [
                    list(map(lambda x:x-avg if x!=0 else 0, line))
                    for line in arr
                    for avg in [np.sum(line)/np.count_nonzero(line)]
                ]
            )
        )

#co = Colaboratif(um)

'''
test_array = np.array(
    [[4.0,0.0,0.0,5.0,1.0,0.0,0.0],
     [5.0,5.0,4.0,0.0,0.0,0.0,0.0],
     [0.0,0.0,0.0,2.0,4.0,5.0,0.0],
     [0.0,3.0,0.0,0.0,0.0,0.0,3.0]])

c = Colaboratif(test_array)
#pc = c.pear_corr(test_array)
#cs = cosine_similarity(pc)
print(c.user_sim_matrix)
#print(cs)
'''
'''
mat = Colaboratif(load_data(
    "/home/imad/Desktop/PFE/db/ml-100k/ua.base",
    "/home/imad/Desktop/PFE/db/ml-100k/u.user",
    "/home/imad/Desktop/PFE/db/ml-100k/u.item"
)[0]).user_sim_matrix

core, labels = dbscan(mat, eps=0.415, min_samples=10, metric="precomputed")

print(core)
print(labels)
'''