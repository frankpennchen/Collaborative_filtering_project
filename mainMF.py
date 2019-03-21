import math
import pandas as pd
import numpy as np
import numba
import time

from surprise import Reader, Dataset
from surprise import SVD, evaluate
from surprise import NMF



start = time.time()

tags=pd.read_csv('./dataset/tags.csv')
movies=pd.read_csv('./dataset/movies.csv')
ratings=pd.read_csv('./dataset/ratings.csv')

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0.5, 5.0))
training_data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

training_data.split(n_folds=5)

# svd
algo = SVD()
evaluate(algo, training_data, measures=['RMSE'])

# nmf
algo = NMF()
evaluate(algo, training_data, measures=['RMSE'])





