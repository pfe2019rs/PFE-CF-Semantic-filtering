#from metrics import jaccard_similarity
import numpy as np
from rec_sys.read_csv import load_data

'''
def compute_item_matrix(items: np.array):
        item_sim_matrix = []
        
        for i in range(len(items)):
            item_sim_matrix.append(
                [
                    jaccard_similarity(items[i], items[j])
                    for j in range(len(items))
                ]
            )

        item_sim_matrix = np.array(item_sim_matrix)
        
        return item_sim_matrix


def interest_sim(u1: list, u2: list, item_sim_matrix: np.array):
        s1 = 0; s2 = 0
        for i in range(len(u1)):
            s2 += u1[i]
            s1 += u1[i]*max(
                [
                    item_sim_matrix[i][j]
                    for j in range(len(item_sim_matrix[i]))
                    if u2[j] > 0
                ]
            )
        
        for i in range(len(u2)):
            s2 += u2[i]
            s1 += u2[i]*max(
                [
                    item_sim_matrix[i][j]
                    for j in range(len(item_sim_matrix[i]))
                    if u1[j] > 0
                ]
            )

        return s1/s2

def compute_user_matrix(usage_matrix: list, item_sim_matrix: np.array):
        
        user_sim_matrix = []
        for user in usage_matrix:
            user_sim_matrix.append([
                interest_sim(user, ouser, item_sim_matrix)
                for ouser in usage_matrix
            ])
        
        return np.array(user_sim_matrix)


def semantic_distance(usage_matrix, item_matrix) -> np.array:
    return compute_user_matrix(usage_matrix, compute_item_matrix(item_matrix))
'''

class Semantic:

    def __init__(self, users: np.array = None, items: np.array = None):
        #self.compute_item_matrix(items)
        #self.compute_user_matrix(users)
        pass

    def get_interest_dict(self, users: np.array) ->dict:
        (N, M) = users.shape
        return {
            u: {
                m: users[u, m]
                for m in range(M)
                if users[u, m] > 0
            }
            for u in range(N)
        }

    def jaccard_distance(self, x: list, y: list):
        return 1 - self.jaccard_similarity(x, y)

    def jaccard_similarity(self, x: list, y: list):
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

    def compute_item_matrix(self, items: np.array):
        n = len(items)
        self.item_dist_matrix = np.zeros((n, n), dtype=tuple)
        
        for i in range(n):
            for j in range(n):
                self.item_dist_matrix[i, j] = (j, self.jaccard_distance(items[i], items[j]))
        
        return self.item_dist_matrix

    def get_sorted_item_matrix(self, item_dist_matrix: np.array) ->np.array:
        self.sorted_item_matrix = np.array(
            [
                sorted(line, key=lambda x: x[1])
                for line in item_dist_matrix
            ]
        )
    
    def interest_dist(self, u1: dict, u2: dict):
        s1 = 0; s2 = 0
        for m1 in u1:
            min1 = 1
            for (index, value) in self.sorted_item_matrix[m1]:
                if u2.get(index, 0) > 0:
                    min1 = value
                    break
            s1 += u1[m1]*min1
            s2 += u1[m1]
        
        for m2 in u2:
            min2 = 1
            for (index, value) in self.sorted_item_matrix[m2]:
                if u1.get(index, 0) > 0:
                    min2 = value
                    break
            s1 += u2[m2]*min2
            s2 += u2[m2]

        return s1/s2
   

    def compute_user_matrix(self, usage_matrix: np.array):
        if self.item_dist_matrix is None:
            print("no item similarity matrix found")
            pass

        n = len(usage_matrix)

        user_interest_dict = self.get_interest_dict(usage_matrix)

        self.user_dist_matrix = np.zeros((n, n))
        for i in range(n):
            print("line: ", i)
            for j in range(i, n):
                print("column: ",j)
                self.user_dist_matrix[i, j] = self.interest_dist(user_interest_dict[i], user_interest_dict[j])
                self.user_dist_matrix[j, i] = self.user_dist_matrix[i, j]
        
        return self.user_dist_matrix
            

def semantic_distance(usage_matrix: np.array, item_matrix: np.array) ->np.array:
    sem = Semantic()
    sem.compute_item_matrix(item_matrix)
    sem.get_sorted_item_matrix(sem.item_dist_matrix)
    return sem.compute_user_matrix(usage_matrix)
    
'''
a = np.array([
    [1,0,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0]
])

um = np.array([
    [3,1,0,0,4],
    [4,2,1,0,0],
    [0,1,0,0,4],
    [0,0,3,2,0],
    [1,0,0,4,2]  
])

(u_m, i_m) =  load_data(
    "/home/imad/Desktop/PFE/db/ml-100k/ua.base",
    "/home/imad/Desktop/PFE/db/ml-100k/u.user",
    "/home/imad/Desktop/PFE/db/ml-100k/u.item"
)
sem = Semantic(u_m, i_m)
sem.compute_item_matrix(i_m)

sem.get_sorted_item_matrix(sem.item_dist_matrix)

print(sem.sorted_item_matrix)

sem.compute_user_matrix(u_m)

print(sem.user_dist_matrix)
'''