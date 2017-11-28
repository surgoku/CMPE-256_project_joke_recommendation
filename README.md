# CMPE-256_project_joke_recommendation
Recommendation System for Jokes and Clustering of Jokes

# Requirements
1. There is a need to download the following pre-trained GLove vectors using the following url: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit . Download this file and move it under models folder. Unzip this file in the same folder to obtain a ".bin" format file.

2. Install sklearn, numpy, scipy, pandas, matplotlib, graphlab, gensim. Graphlab is free for students. I have used the student licence to run graphlab. More instructions here for Mac environment:
https://turi.com/download/install-graphlab-create-command-line.html
 

# Running Instructions
All the code is under src folder. All the required data is under data folder. 

The following codes are used for various purposes:
1. Exploration.py : Exploratory analysis of the data
2. Model_Comparison.py : For comparing Item level, ranking, and rating based RS.
3. Item similarity based Recommendation: Item_Similarity_Based_Recommendation_System.py
4. Predicting_Joke_Category.ipynb : Predicting the category of a joke.
5. clean_jokes.py : For cleaning the data and obtaining and storing the joke vectors.



