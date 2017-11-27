import string
import re
import graphlab
import numpy as np
%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def clean_data():
    #with open('../data/jester_dataset_2/jester_items.dat') as f:
    with open('../data/Jokes_labelling.txt') as f:
        #text = f.read().lower().split('</p>')
        text = f.read().lower().splitlines()
        text = [" ".join(i.split('\t')[1].split('|||')) for i in text[1:]]
        text = [re.sub(r'([^\.\s\w]|_)+', '', i).replace(".", ". ") for i in text]
        text = [line.replace('\r', '') for line in text]
        text = [line.replace('\n', '') for line in text]
        text = [line.replace('<br />', '') for line in text]
        text = [line.replace('<p>', '') for line in text]
        text = [line.replace('&quot;', '') for line in text]
        text = [line.replace('&#039;', '') for line in text]
        #text = [re.sub(" \d+", " ", line) for line in text]
        text = [line.split(':', 1)[-1] for line in text]
        #text = text[:150]
    return text

text = clean_data()
text[0]

ratings_data = pd.read_csv('../data/jester_dataset_2/jester_ratings.dat', sep='\t')
msk = np.random.rand(len(ratings_data)) < 0.8
user_ratings_train = ratings_data[msk]
user_ratings_test = ratings_data[~msk]
train_data = graphlab.SFrame(user_ratings_train)
test_data = graphlab.SFrame(user_ratings_test)

user_ratings_train.head(5)

sns.distplot(user_ratings_train['rating'])

filter_good_rating = user_ratings_train[user_ratings_train.rating > 8]
jokes_good_rating = filter_good_rating.joke_id.unique()
[text[i-1] for i in jokes_good_rating][:5]

filter_bad_rating = user_ratings_train[user_ratings_train.rating < -8]
jokes_bad_rating = filter_bad_rating.joke_id.unique()
jokes_bad_rating.max()
[text[i-1] for i in jokes_bad_rating][:5]

user_ratings_train.joke_id.describe().T


#See if there is a correlation between the length of the joke and its rating

text = [line.translate(None, string.punctuation) for line in text]
length_joke_id = np.zeros(user_ratings_train.shape[0])
#length_joke_id = []
print ratings_data.shape[0]
set_joke_id = set()

ratings_list = []
joke_length_list = []

for i, joke_id in enumerate(list(user_ratings_train.joke_id)):
    set_joke_id.add(joke_id)
    try:
        length_joke_id[i] = int(len(text[joke_id - 1].split()))
        #ratings_list.append()
    except:
        print i, joke_id
#print list(length_joke_id)
#print list(set_joke_id)
#print list(set_id)
print max(length_joke_id)
print user_ratings_train.shape
print length_joke_id.shape
user_ratings_train['length_joke_id'] = length_joke_id

user_ratings_train.head()
plt.scatter(x=user_ratings_train.length_joke_id, y=user_ratings_train.rating)