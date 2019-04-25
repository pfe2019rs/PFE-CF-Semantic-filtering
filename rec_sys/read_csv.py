import csv
import pandas as pd
import numpy as np
import sys

'''
def create_usage_matrix(filepath: str) -> np.array:
    #reading   
    ratings = pd.read_csv(filepath,sep='\t',names=['user','movie','rating','time'])
    mean_ratings = ratings.pivot_table(index='user', columns='movie', values='rating')

    #replace NaN with 0
    user_movie_rating = mean_ratings.fillna(0) 
    #get list movie titles
    user_movie_rating.columns.tolist()
    #get the usage matrix in an array 
    return user_movie_rating.values

print(len(create_usage_matrix('/home/imad/Desktop/PFE/db/ml-100k/u1.base')[100]))

def create_item_matrix(filepath: str, sep=",", cols=None) -> np.array:
    movies = pd.read_csv(filepath, sep=sep, usecols=cols, header=None, encoding="ISO-8859-1")
    return np.array(
        [
            (movie[0], movie[5:])
            for movie in movies.values
        ]
    )


print(len(create_item_matrix("/home/imad/Desktop/PFE/db/ml-100k/u.item", sep="|")))
'''

def load_data(user_ratings_path: str, users_path:str , movies_path: str) -> np.array:
    ratings = np.array(
        list(csv.reader(open(user_ratings_path, "r", encoding="ISO-8859-1"),delimiter='\t'))
    ).astype('int')

    movies = np.array(
        [
            row[5:]
            for row in csv.reader(open(movies_path, "r", encoding="ISO-8859-1"),delimiter='|')
        ]
    ).astype('int')

    users = np.array(
        list(
            csv.reader(open(users_path, "r", encoding="ISO-8859-1"),delimiter='\t')
        )
    )

    usage_matrix = np.zeros((len(users), len(movies)))
    for rating in ratings:
        usage_matrix[rating[0]-1, rating[1]-1] = rating[2]
    
    #print(len(usage_matrix[0]), usage_matrix[0])
    #print(len(movies), movies[0])
    return (usage_matrix, movies)

#load_data('/home/imad/Desktop/PFE/db/ml-100k/u.data', '/home/imad/Desktop/PFE/db/ml-100k/u.user', '/home/imad/Desktop/PFE/db/ml-100k/u.item')