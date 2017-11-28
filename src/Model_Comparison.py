
"""
Objective: Build the recommendation system using three approaches: Rank Factorization, Rating Factorization and Item based.
Details of various models can be found here (We should add them to the report):
1. https://turi.com/products/create/docs/generated/graphlab.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html
2. https://turi.com/products/create/docs/generated/graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender.html
3. https://turi.com/products/create/docs/generated/graphlab.recommender.factorization_recommender.create.html

We train all these three models to obtain the best model. All of these models take user-item rating matrix

Steps:
1. Split the data in train and validation using Graphlab library
2. Define all these models (here in: recommendation_modules method)
3. Train these models for various latent factors: Factors required in factorization
4. Optimizing for RMSE
5. Plot various models to obtain which model and factor performed the best. 


Results:
On validation:

1. Ranking Factormziation: Num Factors: 100  Score: 0.34862012987
2. Rating Factorization: Num Factors: 90  Score: 2.51836444805
3. Item Based Factorization: Num Factors: 90  Score: 1.43739853896

"""

import graphlab as gl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def score(df_true, df_pred):

    df = pd.concat([df_pred,
                    df_true], axis=1)

    g = df.groupby('user_id')

    top_5 = g.pred_rating.apply(
        lambda x: x >= x.quantile(.95)
    )

    return df_true[top_5==1].mean()['true_rating']

def load_data():
    # Input data
    sf = gl.SFrame("../data/ratings.dat", format='tsv')

    # Data to test predictions on
    df_sample = pd.read_csv("../data/sample_submission.csv")
    sf_sample = gl.SFrame(df_sample)

    return sf, sf_sample, df_sample

def recommendation_modules(sf, num_factors, regularization = None):
    ranking_model = gl.recommender.ranking_factorization_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     num_factors = num_factors,
                                                     regularization = regularization,
                                                     verbose = False,
                                                     random_seed = 42)
    
    factorization_model = gl.recommender.factorization_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     num_factors = num_factors,
                                                     regularization = regularization,
                                                     verbose = False,
                                                     random_seed = 42)
    
    item_sim_model_jaccard = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     #solver='auto',
                                                     #num_factors = num_factors,
                                                     #regularization = regularization,
                                                     verbose = False,
                                                     #random_seed = 42, 
                                                     similarity_type='jaccard')
    
    
    
    return ranking_model, factorization_model, item_sim_model_jaccard


if __name__ == "__main__":
    sf, sf_sample, df_sample = load_data()

    training_data, validation_data = gl.recommender.util.random_split_by_user(sf, 'user_id', 'joke_id')

    df_true = pd.DataFrame()
    df_pred = pd.DataFrame()

    df_true['user_id'] = validation_data['user_id']
    df_true['joke_id'] = validation_data['joke_id']

    df_true['true_rating'] = validation_data['rating']

    # Plot scores vs num_factors
    #num_factors = range(2,4)
    num_factors = [2, 4, 8, 16, 24, 32, 50, 64, 80, 90, 100]
    scores = []
    model_names = ["ranking_model", "factorization_model", "item_sim_model_jaccard"]
    
    for n in num_factors:
        temp_scores = []
        for m in recommendation_modules(training_data, num_factors = n):
            #m = create_factorization_recommender(training_data, num_factors = n)
            df_pred['pred_rating'] = m.predict(validation_data)
            rc = score(df_true, df_pred)
            temp_scores.append(rc)
            #scores.append(rc)
            print 'Num Factors:', n, ' Score:', rc
        scores.append(temp_scores)
        print "\n\n\n"
        
        
    scores = map(list, zip(*scores))
    for model_scores in scores:
        plt.plot(num_factors, model_scores)
    
    plt.legend(model_names, loc='upper center', bbox_to_anchor=(0.5, -0.30),
          fancybox=True, shadow=True, ncol=5)
  
    
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Score')
    plt.title('Score vs Number of Latent Features')
    plt.show()