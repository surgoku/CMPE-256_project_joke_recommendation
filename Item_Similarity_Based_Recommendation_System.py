
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
import re
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree



def extract_key_words(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    
    return continuous_chunk


def score(df_true, df_pred):
    df = pd.concat([df_pred,
                    df_true], axis=1)
    g = df.groupby('user_id')
    top_5 = g.pred_rating.apply(
        lambda x: x >= x.quantile(.95)
    )
    return df_true[top_5==1].mean()['true_rating']


def clean_joke(joke):
    joke = re.sub(r'([^\.\s\w]|_)+', '', joke).replace(".", ". ")
    joke = " ".join(extract_key_words(joke))
    return joke

def load_joke_classes_and_text():
    data = pd.read_csv("../data/Jokes_labelling.txt", delimiter="\t")
    data['Jokes'] = data['Jokes'].map(lambda j: clean_joke(j))
    data.drop('joke_category', axis=1, inplace=True)
    cat_feats = pd.get_dummies(data['joke_category_reduced'], prefix='cat')
    data = pd.concat([data['joke_id'], data['Jokes'], cat_feats], axis=1)
    
    data_sf = gl.SFrame(data)
    
    return data_sf
    

def load_data():
    # Input data
    sf = gl.SFrame("../data/ratings.dat", format='tsv')

    # Data to test predictions on
    df_sample = pd.read_csv("../data/sample_submission.csv")
    sf_sample = gl.SFrame(df_sample)

    return sf, sf_sample, df_sample

def load_joke_classes_text_and_glove_vectors():
    id_vectors = pd.read_csv("../data/Jokes_id_with_vectors.txt", delimiter="\t")
    data = pd.read_csv("../data/Jokes_labelling.txt", delimiter="\t")
    cat_feats = pd.get_dummies(data['joke_category_reduced'], prefix='cat')
    

    all_data = pd.merge(data, id_vectors, on='joke_id', how='inner').set_index('joke_id').reset_index()
    
    X = pd.concat([all_data, cat_feats], axis=1)
    #print X.columns
    #print X.describe(include='all')
    X.drop(['Jokes','joke_category', 'joke_category_reduced','Unnamed: 301'], axis=1, inplace=True)
    X = X.fillna(0)
    X = gl.SFrame(X)
    
    cat_feats['joke_id'] = range(1,151)
    cat_feats = gl.SFrame(cat_feats)
    return X, cat_feats

def recommendation_modules(sf, num_factors, regularization = None):
    
    joke_vector_and_cat, joke_cat = load_joke_classes_text_and_glove_vectors()
    
    item_sim_model_cosine = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     #solver='auto',
                                                     #num_factors = num_factors,
                                                     #regularization = regularization,
                                                     verbose = False,
                                                     #random_seed = 42, 
                                                     similarity_type='cosine')
    
    item_sim_model_pearson = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     #solver='auto',
                                                     #num_factors = num_factors,
                                                     #regularization = regularization,
                                                     verbose = False,
                                                     #random_seed = 42, 
                                                     similarity_type='pearson')
    
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
    
    item_sim_model_with_categories = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     #solver='auto',
                                                     #num_factors = num_factors,
                                                     #regularization = regularization,
                                                     verbose = False,
                                                     #random_seed = 42,
                                                     similarity_type='jaccard',
                                                     item_data=  joke_cat                                      
                                                     )
    item_sim_model_with_vectors_and_categories = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     #solver='auto',
                                                     #num_factors = num_factors,
                                                     #regularization = regularization,
                                                     verbose = False,
                                                     #random_seed = 42,
                                                     similarity_type='jaccard',
                                                     item_data=  joke_vector_and_cat                                      
                                                     )
    
    
    
    return item_sim_model_cosine, item_sim_model_pearson, item_sim_model_jaccard, item_sim_model_with_categories, item_sim_model_with_vectors_and_categories



if __name__ == "__main__":
    sf, sf_sample, df_sample = load_data()

    training_data, validation_data = gl.recommender.util.random_split_by_user(sf, 'user_id', 'joke_id')

    df_true = pd.DataFrame()
    df_pred = pd.DataFrame()

    df_true['user_id'] = validation_data['user_id']
    df_true['joke_id'] = validation_data['joke_id']

    df_true['true_rating'] = validation_data['rating']

    # Plot scores vs num_factors
    num_factors = range(2,100)
    #num_factors = [2, 4, 8, 16, 24, 32, 50, 64, 80, 90, 100]
    num_factors = [2, 4]
    #num_factors = range(2,100)
    scores = []
    model_names = ["item_sim_cosine", "item_sim_pearson", "item_sim_jaccard", "item_sim_with_joke_categories", "item_sim_with_joke_categories_joke_embeddings"]
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
