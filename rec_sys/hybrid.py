from rec_sys.semantic import Semantic
#from colaboratif import co
import numpy as np


class WHybrid:
    def __init__(self, sem: np.array, col: np.array, alpha = 0.5, beta = 0.5):
        self.semantic_dist_matrix = sem
        self.colaboratif_dist_matrix = col
        self.compute_user_similarity_matrix(alpha, beta)

    
    def compute_user_similarity_matrix(self, alpha, beta):
        self.user_sim_matrix = alpha*self.semantic_dist_matrix + beta*self.colaboratif_dist_matrix
        #print(self.user_sim_matrix)



#wh = WHybrid(sem.user_sim_matrix, co.user_sim_matrix)

class MixHybrid:
    def __init__(self, sem: np.array, col: np.array):
        self.semantic_dist_matrix = sem
        self.colaboratif_dist_matrix = col

        self.compute_user_similarity_matrix()

    def compute_user_similarity_matrix(self):
        self.user_dist_matrix = np.zeros((len(self.semantic_dist_matrix), len(self.semantic_dist_matrix)))
        for i in range(len(self.semantic_dist_matrix)):
            for j in range(len(self.semantic_dist_matrix)):
                if self.colaboratif_dist_matrix[i, j] == 0:
                    self.user_dist_matrix[i, j] = self.semantic_dist_matrix[i, j] 
                else:
                    self.user_dist_matrix[i, j] = self.colaboratif_dist_matrix[i, j]
        #print(self.user_sim_matrix)


#mixH = MixHybrid(sem.user_sim_matrix, co.user_sim_matrix)

class ThirdHybrid:
    def __init__(self):
        pass
    
    def compute_user_similarity_matrix(self, users: np.array, item_dist_matrix: np.array):
        sem = Semantic()
        sem.item_dist_matrix = item_dist_matrix
        self.user_dist_matrix = sem.compute_user_matrix(users)
        print(self.user_dist_matrix)


#ThirdHybrid().compute_user_similarity_matrix(um, )