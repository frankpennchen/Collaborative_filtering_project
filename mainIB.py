import math
import pandas as pd
import numpy as np
import numba
import time

knn_para=5
#knn_para=float('inf')

#IB-CF#########################################################################
#weighting matrix
@numba.jit
def weighting_matrix(adjusted_ratings):
    
    weight_matrix_columns = ['movie_1', 'movie_2', 'weight']
    weight_matrix=pd.DataFrame(columns=weight_matrix_columns)
    
    i = 0
    distinct_movies = np.unique(adjusted_ratings['movieId'])
    for movie_1 in distinct_movies:
        
        if i % 100 == 0:
            print(i , "out of ", len(distinct_movies))
        user_data = adjusted_ratings[adjusted_ratings['movieId'] == movie_1]
        distinct_users = np.unique(user_data['userId'])
        if len(distinct_users)>knn_para:
            distinct_users=distinct_users[:knn_para]
        comp_columns = ['userId', 'movie_1', 'movie_2', 'rating_adjusted_1', 'rating_adjusted_2']
        comp_movie_1_2 = pd.DataFrame(columns=comp_columns)
        for user_id in distinct_users:
            movie_1_rating = user_data[user_data['userId'] == user_id]['rating_adjusted'].iloc[0]
            this_user_data=adjusted_ratings[(adjusted_ratings['userId'] == user_id)\
                                            & (adjusted_ratings['movieId'] != movie_1)]
            this_distinct_movies = np.unique(this_user_data['movieId'])
            if len(this_distinct_movies)>knn_para:
                this_distinct_movies=this_distinct_movies[:knn_para]
            
            for movie_2 in this_distinct_movies:
                movie_2_rating = this_user_data[this_user_data['movieId'] == movie_2]\
                ['rating_adjusted'].iloc[0]
                comp = pd.Series([user_id, movie_1, movie_2, movie_1_rating, movie_2_rating],\
                                   index=comp_columns)
                comp_movie_1_2 = comp_movie_1_2.append(comp, ignore_index=True)
                
        distinct_movie_2 = np.unique(comp_movie_1_2['movie_2'])
            
        for movie_2 in distinct_movie_2:
            paired_movie_1_2 = comp_movie_1_2[comp_movie_1_2['movie_2'] == movie_2]
            sim_value_numerator = (paired_movie_1_2['rating_adjusted_1'] * \
                                       paired_movie_1_2['rating_adjusted_2']).sum()
            sim_value_denominator = np.sqrt(np.square(paired_movie_1_2['rating_adjusted_1'])\
                                                .sum()) * np.sqrt(np.square(paired_movie_1_2['rating_adjusted_2']).sum())
                 
            if sim_value_denominator != 0:
                sim_value_denominator = sim_value_denominator
            else:
                sim_value_denominator = 1e-8
            sim_value = sim_value_numerator / sim_value_denominator
            weight_matrix = weight_matrix.append(pd.Series([movie_1, movie_2, sim_value],\
                                                     index=weight_matrix_columns), ignore_index=True)
        i = i + 1
        
    return weight_matrix

#
@numba.jit
def predict_rating(userId, movieId, weight_matrix, adjusted_ratings, rating_mean):
    if rating_mean[rating_mean['movieId'] == movieId].shape[0] > 0:
        mean_rating = rating_mean[rating_mean['movieId'] == movieId]['rating_mean'].iloc[0]
    else:
        mean_rating = 2.5
    
    user_ratings = adjusted_ratings[adjusted_ratings['userId'] == userId]
    distinct_movies = np.unique(user_ratings['movieId'])
    if len(distinct_movies)>knn_para:
        distinct_movies=distinct_movies[:knn_para]
    sum_weighted_ratings = 0
    sum_weights = 0
    for movie_i in distinct_movies:
        if rating_mean[rating_mean['movieId'] == movie_i].shape[0] > 0:
            rating_mean_i = rating_mean[rating_mean['movieId'] == movie_i]['rating_mean'].iloc[0]
        else:
            rating_mean_i = 2.5

        weight_movie_1_2 = weight_matrix[(weight_matrix['movie_1'] == movieId) & (weight_matrix['movie_2'] == movie_i)]
        if weight_movie_1_2.shape[0] > 0:
            user_rating_i = user_ratings[user_ratings['movieId']==movie_i]
            sum_weighted_ratings += (user_rating_i['rating'].iloc[0] - rating_mean_i) * weight_movie_1_2['weight'].iloc[0]
            sum_weights += np.abs(weight_movie_1_2['weight'].iloc[0])


    if sum_weights == 0:
        predicted_rating = mean_rating
    else:
        predicted_rating = mean_rating + sum_weighted_ratings/sum_weights

    return predicted_rating    

@numba.jit    
def predict_eval(ratings_test, weight_matrix, adjusted_ratings, rating_mean):
    i = 0
    ratings_test = ratings_test.assign(predicted_rating = pd.Series(np.zeros(ratings_test.shape[0])))
    
    for index, row_rating in ratings_test.iterrows():
        if i % 100 == 0:
            print(i , "out of ", len(ratings_test))        
        predicted_rating = predict_rating(row_rating['userId'], row_rating['movieId'],\
                                   weight_matrix, adjusted_ratings, rating_mean)
        ratings_test.loc[index, 'predicted_rating'] = predicted_rating
        
        i += 1
    
    truepos = ratings_test.query('(rating >= 2.5) & (predicted_rating >= 2.5)').shape[0]
    falsepos = ratings_test.query('(rating < 2.5) & (predicted_rating >= 2.5)').shape[0]
    fneg = ratings_test.query('(rating >= 2.5) & (predicted_rating < 2.5)').shape[0]    
    
    type1error=1.0-falsepos/(truepos+falsepos)
    type2error=1.0-fneg/(truepos+fneg)
    return (type1error,type2error)

    


#loading dataset
start = time.time()
tags=pd.read_csv('./dataset/tags.csv')
movies=pd.read_csv('./dataset/movies.csv')
ratings=pd.read_csv('./dataset/ratings.csv')

num_users = 600
num_items = 9000

#data split
ratings_training = ratings.sample(frac=0.7)
ratings_test = ratings.drop(ratings_training.index)

#normalization
rating_mean = ratings_training.groupby(['movieId'], as_index = False, sort = False)\
.mean().rename(columns = {'rating': 'rating_mean'})[['movieId','rating_mean']]
adjusted_ratings = pd.merge(ratings_training,rating_mean,on = 'movieId', how = 'left', sort = False)
adjusted_ratings['rating_adjusted']=adjusted_ratings['rating']-adjusted_ratings['rating_mean']
adjusted_ratings.loc[adjusted_ratings['rating_adjusted'] == 0, 'rating_adjusted'] = 1e-8

weight_matrix = weighting_matrix(adjusted_ratings)
eval_result = predict_eval(ratings_test, weight_matrix, adjusted_ratings, rating_mean)
print('Evaluation result - precision: %f, recall: %f' % eval_result)
end = time.time()
print((end - start)/60.0,'min')





