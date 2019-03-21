import math
import pandas as pd
import numpy as np
import numba
import time

knn_para=5
#knn_para=float('inf')

#UB-CF#########################################################################
#weighting matrix
@numba.jit
def weighting_matrix(adjusted_ratings):
    
    weight_matrix_columns = ['userid_1', 'userid_2', 'weight']
    weight_matrix=pd.DataFrame(columns=weight_matrix_columns)
    
    i = 0
    distinct_userid = np.unique(adjusted_ratings['userId'])
    for userid_1 in distinct_userid:
        
        if i % 100 == 0:
            print(i , "out of ", len(distinct_userid))   
        movie_data = adjusted_ratings[adjusted_ratings['userId'] == userid_1]
        distinct_movies = np.unique(movie_data['movieId'])
        if len(distinct_movies)>knn_para:
            distinct_movies=distinct_movies[:knn_para]        
        
        comp_columns = ['movieId', 'userid_1', 'userid_2', 'rating_adjusted_1', 'rating_adjusted_2']
        comp_movie_1_2 = pd.DataFrame(columns=comp_columns)            
            
        for movie_id in distinct_movies:
            userid_1_rating = movie_data[movie_data['movieId'] == movie_id]['rating_adjusted'].iloc[0]
            this_movie_data=adjusted_ratings[(adjusted_ratings['movieId'] == movie_id)\
                                            & (adjusted_ratings['userId'] != userid_1)]
            this_distinct_userid = np.unique(this_movie_data['userId'])
            if len(this_distinct_userid)>knn_para:
                this_distinct_userid=this_distinct_userid[:knn_para]            
            for userid_2 in this_distinct_userid:
                userid_2_rating = this_movie_data[this_movie_data['userId'] == userid_2]\
                ['rating_adjusted'].iloc[0]
                comp = pd.Series([movie_id, userid_1, userid_2, userid_1_rating, userid_2_rating],\
                                   index=comp_columns)
                comp_movie_1_2 = comp_movie_1_2.append(comp, ignore_index=True)            

        distinct_userid_2 = np.unique(comp_movie_1_2['userid_2'])
            
        for userid_2 in distinct_userid_2:
            paired_userid_1_2 = comp_movie_1_2[comp_movie_1_2['userid_2'] == userid_2]
            sim_value_numerator = (paired_userid_1_2['rating_adjusted_1'] * \
                                       paired_userid_1_2['rating_adjusted_2']).sum()
            sim_value_denominator = np.sqrt(np.square(paired_userid_1_2['rating_adjusted_1'])\
                                    .sum()) * np.sqrt(np.square(paired_userid_1_2['rating_adjusted_2']).sum())
                 
            if sim_value_denominator != 0:
                sim_value_denominator = sim_value_denominator
            else:
                sim_value_denominator = 1e-8
            sim_value = sim_value_numerator / sim_value_denominator
            weight_matrix = weight_matrix.append(pd.Series([userid_1, userid_2, sim_value],\
                            index=weight_matrix_columns), ignore_index=True)
            
        i = i + 1
        
    
    return weight_matrix

@numba.jit
def predict_rating(userId, movieId, weight_matrix, adjusted_ratings, rating_mean):
    if rating_mean[rating_mean['userId'] == userId].shape[0] > 0:
        mean_rating = rating_mean[rating_mean['userId'] == userId]['rating_mean'].iloc[0]
    else:
        mean_rating = 2.5
    
    movie_ratings = adjusted_ratings[adjusted_ratings['movieId'] == movieId]
    distinct_userid = np.unique(movie_ratings['userId'])
    if len(distinct_userid)>knn_para:
        distinct_userid=distinct_userid[:knn_para]
        
    sum_weighted_ratings = 0
    sum_weights = 0
    for userid_i in distinct_userid:
        if rating_mean[rating_mean['userId'] == userid_i].shape[0] > 0:
            rating_mean_i = rating_mean[rating_mean['userId'] == userid_i]['rating_mean'].iloc[0]
        else:
            rating_mean_i = 2.5
        weight_userid_1_2 = weight_matrix[(weight_matrix['userid_1'] == userId) & (weight_matrix['userid_2'] == userid_i)]
        if weight_userid_1_2.shape[0] > 0:
            user_rating_i = movie_ratings[movie_ratings['userId']==userid_i]
            sum_weighted_ratings += (user_rating_i['rating'].iloc[0] - rating_mean_i) * weight_userid_1_2['weight'].iloc[0]
            sum_weights += np.abs(weight_userid_1_2['weight'].iloc[0])

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

ratings_training = ratings.sample(frac=0.7)
ratings_test = ratings.drop(ratings_training.index)

#normalization
rating_mean = ratings_training.groupby(['userId'], as_index = False, sort = False)\
.mean().rename(columns = {'rating': 'rating_mean'})[['userId','rating_mean']]
adjusted_ratings = pd.merge(ratings_training, rating_mean, on = 'userId', how = 'left', sort = False)
adjusted_ratings['rating_adjusted']=adjusted_ratings['rating']-adjusted_ratings['rating_mean']
adjusted_ratings.loc[adjusted_ratings['rating_adjusted'] == 0, 'rating_adjusted'] = 1e-8

weight_matrix=weighting_matrix(adjusted_ratings)
eval_result = predict_eval(ratings_test, weight_matrix, adjusted_ratings, rating_mean)
print('Evaluation result - precision: %f, recall: %f' % eval_result)
end = time.time()
print((end - start)/60.0,'min')







