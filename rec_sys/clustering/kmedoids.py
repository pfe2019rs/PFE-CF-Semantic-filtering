# coding: utf-8
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import random
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

def getRating(clusterI,idMovie,idUser,matrice):
    rating=0
    for cpt in clusterI:
        rating=rating+matrice[cpt][idMovie]
    #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
    rating=rating-matrice[idUser][idMovie]#i know it is equal to zero but in the futur we might replace them with something (._. who knows !!)
    return rating
#######################################################
#create a dictionary of movies that we need to guess
def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = fields[1]
            rating = fields[2]
            movieDict[userID].append({ movieID:rating })
    return movieDict
############################################################################################
def MAE_RMSE(matrice,Clusters,testFile):
    #print(matrice)
    realValue=[]
    predection=[]
    movieDict=createDictTestMovies(testFile)
    for label in Clusters: 
        for idUser in Clusters[label]:
                listOfMovies=movieDict[idUser]
                for element in listOfMovies:
                    for idMovie in element:
                        #print(idUser+1,idMovie)
                        realValue.append(element[idMovie])#element[val] is rating and int(idMovie)-1 is the id of the movie
                        rating=getRating(Clusters[label],int(idMovie)-1,idUser,matrice)
                        if len(Clusters[label])==1:#there is only one element in the cluster no predection can be done
                            rating =0 #maybe we can add them to the nearest cluster ... we 'll see later
                            predection.append(rating)
                        else : 
                            rating=rating/(len(Clusters[label])-1)
                            if rating >5.:#warning! :add this to knn 
                                rating=5.
                            predection.append(round(rating))

    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=mean_squared_error(realValue,predection)
    resutls=[]
    resutls.append(mae)
    resutls.append(rmse)
    #print("real values:",realValue)
    #print("predection:",predection)
    print("mean_absolute_error and mean_squared_error=",resutls)
    return resutls
#####################################################################################""
def kMedoids(D, k, tmax=1000):#tmax is the number of max iterations
    # determine dimensions of distance matrix D
    m, n = D.shape
    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
#######################################################################################
ratings = pd.read_csv('ua.base',sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
data=usagematrix.values
D = pairwise_distances(data, metric='euclidean')#we must use only distances no similarity, so we shall reverse for pearson cor ! c:
# split into k clusters
M, C = kMedoids(D, 5)
"""
print('medoids:')
for point_idx in M:
    print( data[point_idx] )
print('clustering results:')
for label in C:
    print(len(C[label]))
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx]))
"""
MAE_RMSE(data,C,"ua.test")
