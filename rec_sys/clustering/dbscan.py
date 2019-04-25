import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
import numpy as  np
from numpy import *
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



def getRating(clusterI,idMovie,idUser,matrice):
    rating=0
    for cpt in clusterI:
        rating=rating+matrice[cpt][idMovie]#remove 1 cuz in matrix movies are from 0 to ...
    #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
    rating=rating-matrice[idUser][idMovie]#i know it is equal to zero but in the futur we might replace them with something (._. who knows !!)
    return rating
#######################################################
#read a test file and create a dictionary of movies that we need to guess
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
#read a file and create matrice user_user with cosin similarity
def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) 
    return matrice
#############################################################################################
def MinPtsNeighbor(matrice,p,eps):
    neighbors = []
    for cpt in range(len(matrice[p])):
        if cpt!=p:
            if ((matrice[p][cpt]<eps)):
                neighbors.append(cpt)
    return(neighbors)
########################################################################################
def CreatCluster(matrice, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C    
    i = 0
    #  The FIFO behavior is accomplished by using a while loop rather than a for loop
    while i < len(NeighborPts):    
        Pn = NeighborPts[i]   
        # A NOISE point may later be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change, from NOISE to something else)
        if labels[Pn] == -1:
           labels[Pn] = C
        #else it is a no visited point
        elif labels[Pn] == 0:
            labels[Pn] = C          
            PnNeighborPts = MinPtsNeighbor(matrice, Pn, eps)         
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1        
################################################################
def dbscan(matrice,minpts,eps):
    labels = [0]*len(matrice)   
    C = 0
    for P in range(len(matrice)):
        #if noise or clustered then continue
        if not (labels[P] == 0):
            continue
        #else get the neighbors
        NeighborPts = MinPtsNeighbor(matrice, P, eps)
        #if < minpts then noise
        if len(NeighborPts) < minpts:
                labels[P] = -1
        #else grow a cluster
        else: 
            C += 1
            CreatCluster(matrice, labels, P, NeighborPts, C,eps, minpts)
    return labels
#################################################################

def MAE_RMSE(usageMatrix,listeOfClusters,testDict):
#when sending the numpy_matrix to the function "createMatrice" it does change the values so im reloading them in this line
    
    #print(reloadedUsageMatrix)
    #arrays i'm sending to the python function MAE and RMSE
    realValue=[]
    predection=[]
    #movieDict=createDictTestMovies(testFile)
    #get the different ids of sets created from dbscan
    idCluster=set(array(listeOfClusters))
    for element in idCluster: #for each id_cluster if it is not -1 (a noise)
        if element != -1:
            clusterI = np.where(listeOfClusters == element)[0]#get all the indexes of a given id_cluster
        #i'm making sure to get the real value and its predection in the proper order
            for idUser in clusterI:
                #listOfMovies=movieDict[idUser]
                for element in testDict[idUser]:
                    for idMovie in element:
                    #print("movie",int(idMovie)-1)
                        realValue.append(element[idMovie])#element[val] is rating and int(idMovie)-1 is the id of the movie
                        rating=getRating(clusterI,int(idMovie)-1,idUser,usageMatrix)
                        rating=rating/(len(clusterI)-1)
                        if rating >5.:#add this to knn
                            rating=5.
                        predection.append(rating)

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
##############################################################################################
"""
ratings = pd.read_csv("myubase.base",sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
numpy_matrix = usagematrix .values
matrice=creatMatrice(numpy_matrix)
print(dbscan(matrice,3,0.3))
MAE_RMSE(ratings,dbscan(matrice,3,0.3),'myutest.test')


ratings = pd.read_csv("ua.base",sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
numpy_matrix = usagematrix .values
matrice=creatMatrice(numpy_matrix)
MAE_RMSE(ratings,dbscan(matrice,20,0.2),'ua.test')
"""