import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Normalise Similarity Scores
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

import regex as re
import spacy
#Find best match with user input
import difflib


# load spaCy and the English model
nlp = spacy.load("en_core_web_md")
vectorizer = TfidfVectorizer()
scaler = MinMaxScaler()


df = pd.read_csv("movies.csv") #Read database
#Drop unnecessary Cols
df_clean = df.drop(columns=['homepage','original_language','original_title','production_countries','production_companies','spoken_languages',
                      'status','crew'])
#Fill NA with empty string
df_clean = df_clean.fillna('')

#Combine Text features into one
df_clean['combined_text'] = df_clean['genres'] + ' ' + df_clean['overview'] + ' ' + df_clean['keywords']+ ' ' + df_clean['tagline']+ ' ' + df_clean['cast']+ ' ' + df_clean['director']

#Cleaning Text
def clean_text(text):
    #Remove any \u
    text_clean = re.sub(r'\\u',' ',text)
    #Remove punctuation
    text_clean = re.sub(r'[^\w\s]','',text_clean)

    #Add spaces where words are attached together
    text_clean = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text_clean) #Matches a letter followed by a digit and captures them as separate groups
    text_clean = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text_clean) #Matches a digit followed by a letter and captures them as separate groups
    text_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_clean) # Matches a lowercase letter followed by an uppercase letter and captures them as separate groups.

    #Lowercasing
    text_clean = text_clean.lower()

    return text_clean


#Lemmatise and Remove Stop words
def convert_text(text):
    # Parse text with Spacy
    doc = nlp(text)
    
    # Lemmatize and remove stop words
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Join lemmas back into a string
    clean_text = ' '.join(lemmas)
    
    return clean_text


#Pre-Processing steps of the combined text
df_clean['short'] = df_clean['combined_text'].apply(clean_text)
df_clean['short'] = df_clean['short'].apply(convert_text)

feature_vectors = vectorizer.fit_transform(df_clean['short'])

#Similarity score
similarity = cosine_similarity(feature_vectors) #Finds which values are similar to each other based on the feature vectors

# user_input = input("Enter Movie: ")


#Create List that contains all movies in dataset
movie_list = df_clean['title'].tolist() #To compare with what the user inputted

#Find closest match
find_match = difflib.get_close_matches(user_input, movie_list)

#Just get the first match
close_match = find_match[0]

#Find the index of the movie
movie_index = df_clean[df_clean.title == close_match]['index'].values[0]

similarity_score = list(similarity[movie_index])

#Append Similarity Scores to the Dataframe
df_clean['similarity_scores'] = similarity_score



df_clean['weighted_sim_score'] = 0  # Initialize the weighted score column with zeros

#Get the Inputted Movie genres

movie_genres = df_clean['genres'].iloc[movie_index]
movie_genres = movie_genres.split(' ')


for index, row in df_clean.iterrows():
    genres = row['genres'].split(' ')
    similarity_score = row['similarity_scores']
    weighted_score = 0

    #Check how many genres defined for a movie are the same as the inputted movie
    common_genres = set(movie_genres) & set(genres)
    count_common = len(common_genres)

    #Based on how many are common increase the Weighted Similarity score
    weighted_score = similarity_score * (1 + (count_common/len(movie_genres)))
    
    df_clean.at[index, 'weighted_sim_score'] = weighted_score


#Sort by Weighted Similarity Scores (Including how relevant the genres are to each other)
df_sorted = df_clean[df_clean['weighted_sim_score'].notna()].sort_values('weighted_sim_score', ascending=False)



#Popularity

scaled_similarity = df_clean['weighted_sim_score']
scaled_similarity = scaled_similarity.values.reshape(-1,1) #Make 2D
scaled_similarity = scaler.fit_transform(scaled_similarity)


df_clean_2 = df_clean.copy()  # Create a copy of the original dataframe
df_clean_2['scaled_similarity_scores'] = scaled_similarity

#Constants for Wilson Score Interval
confidence_level = 0.95
z = norm.ppf((1+confidence_level) / 2) #Calculates the Z score for a specifide confidence interval

#Average Vote and # of Votes
vote_average = df_clean_2['vote_average'] / 10 #Make it between 0 and 1
vote_count = df_clean_2['vote_count']

# Calculate Wilson score interval for movie rating
df_clean_2['lower_bound'], df_clean_2['upper_bound'] = proportion_confint(vote_average * vote_count, vote_count, method='wilson')

#Incorperate Average Rating

#Scale Average Rating
scaled_avgrating = df_clean_2['vote_average']
scaled_avgrating = scaled_avgrating.values.reshape(-1,1) #Make 2D
scaled_avgrating = scaler.fit_transform(scaled_avgrating)

df_clean_2['scaled_avgrating'] = scaled_avgrating

#Weighted Average of The Movie Ratings and the rating Uncertainity
# To account for movies that have a high rating but low voting count, multiply the avg rating by the lower bound of the wilson score interval
weight_avg_ratings = 0.7
weight_rating_uncertainty = 0.3

df_clean_2['weighted_rating_uncertainty'] = (weight_avg_ratings * df_clean_2['scaled_avgrating']) + (weight_rating_uncertainty * df_clean_2['lower_bound'])

#Scale Average Rating
weighted_rating_uncertainty = df_clean_2['weighted_rating_uncertainty']
weighted_rating_uncertainty = weighted_rating_uncertainty.values.reshape(-1,1) #Make 2D

#Weighted Average - Combine Normalised Similarity Ratings and Average Ratings

#Create Different Weights. Similarity would weigh higher for recommending a movie
similarity_weight = 0.8
avgrating_weight = 0.2

#Weighted Average
weighted_avg = (similarity_weight * scaled_similarity) + (avgrating_weight * weighted_rating_uncertainty)

#Add new col for the weighted avg
df_clean_2['weighted_avg'] = weighted_avg

#Sort by Weighted avg
df_sorted = df_clean_2[df_clean_2['weighted_avg'].notna()].sort_values('weighted_avg', ascending=False)

df_sorted = df_sorted.reindex(columns=['index', 'title','weighted_avg','weighted_sim_score', 'similarity_scores', 'id', 'overview', 'budget', 'genres', 'keywords',
       'popularity', 'release_date', 'revenue', 'runtime', 'tagline',
       'vote_average', 'vote_count', 'cast', 'director', 'combined_text'
       ])

top_10 = df_sorted.iloc[1:11]