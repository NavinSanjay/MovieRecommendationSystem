from flask import Flask, request, render_template
import pandas as pd
import regex as re
import spacy
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

app = Flask(__name__)

# Load spaCy and the English model
nlp = spacy.load("en_core_web_md")
vectorizer = TfidfVectorizer()
scaler = MinMaxScaler()

df = pd.read_csv("movies.csv")  # Read database

# Cleaning Text
def clean_text(text):
    # Remove any \u
    text_clean = re.sub(r'\\u', ' ', text)
    # Remove punctuation
    text_clean = re.sub(r'[^\w\s]', '', text_clean)

    # Add spaces where words are attached together
    text_clean = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text_clean)  # Matches a letter followed by a digit and captures them as separate groups
    text_clean = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text_clean)  # Matches a digit followed by a letter and captures them as separate groups
    text_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_clean)  # Matches a lowercase letter followed by an uppercase letter and captures them as separate groups.

    # Lowercasing
    text_clean = text_clean.lower()

    return text_clean

# Lemmatize and Remove Stop words
def convert_text(text):
    # Parse text with Spacy
    doc = nlp(text)

    # Lemmatize and remove stop words
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Join lemmas back into a string
    clean_text = ' '.join(lemmas)

    return clean_text


# Pre-Processing steps of the combined text
df_clean = df.drop(columns=['homepage', 'original_language', 'original_title', 'production_countries',
                            'production_companies', 'spoken_languages', 'status', 'crew'])
df_clean = df_clean.fillna('')
df_clean['combined_text'] = df_clean['genres'] + ' ' + df_clean['overview'] + ' ' + df_clean['keywords'] + ' ' + \
                             df_clean['tagline'] + ' ' + df_clean['cast'] + ' ' + df_clean['director']
df_clean['short'] = df_clean['combined_text'].apply(clean_text)
df_clean['short'] = df_clean['short'].apply(convert_text)

feature_vectors = vectorizer.fit_transform(df_clean['short'])
similarity = cosine_similarity(feature_vectors)
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/user_input', methods=['POST'])

def user_input():
    user_input = request.form['text']

    movie_list = df_clean['title'].tolist()
    find_match = difflib.get_close_matches(user_input, movie_list)
    close_match = find_match[0]
    movie_index = df_clean[df_clean.title == close_match]['index'].values[0]
    similarity_score = list(similarity[movie_index])

    df_clean['similarity_scores'] = similarity_score
    df_clean['weighted_sim_score'] = 0

    movie_genres = df_clean['genres'].iloc[movie_index]
    movie_genres = movie_genres.split(' ')

    for index, row in df_clean.iterrows():
        genres = row['genres'].split(' ')
        similarity_score = row['similarity_scores']
        weighted_score = 0

        common_genres = set(movie_genres) & set(genres)
        count_common = len(common_genres)

        weighted_score = similarity_score * (1 + (count_common / len(movie_genres)))

        df_clean.at[index, 'weighted_sim_score'] = weighted_score

    df_sorted = df_clean[df_clean['weighted_sim_score'].notna()].sort_values('weighted_sim_score', ascending=False)

    scaled_similarity = df_clean['weighted_sim_score']
    scaled_similarity = scaled_similarity.values.reshape(-1, 1)
    scaled_similarity = scaler.fit_transform(scaled_similarity)

    df_clean_2 = df_clean.copy()
    df_clean_2['scaled_similarity_scores'] = scaled_similarity

    confidence_level = 0.95
    z = norm.ppf((1 + confidence_level) / 2)

    vote_average = df_clean_2['vote_average'] / 10
    vote_count = df_clean_2['vote_count']

    df_clean_2['lower_bound'], df_clean_2['upper_bound'] = proportion_confint(vote_average * vote_count, vote_count,
                                                                              method='wilson')

    scaled_avgrating = df_clean_2['vote_average']
    scaled_avgrating = scaled_avgrating.values.reshape(-1, 1)
    scaled_avgrating = scaler.fit_transform(scaled_avgrating)

    df_clean_2['scaled_avgrating'] = scaled_avgrating

    weight_avg_ratings = 0.7
    weight_rating_uncertainty = 0.3

    df_clean_2['weighted_rating_uncertainty'] = (weight_avg_ratings * df_clean_2['scaled_avgrating']) + (
                weight_rating_uncertainty * df_clean_2['lower_bound'])

    weighted_rating_uncertainty = df_clean_2['weighted_rating_uncertainty']
    weighted_rating_uncertainty = weighted_rating_uncertainty.values.reshape(-1, 1)

    similarity_weight = 0.8
    avgrating_weight = 0.2

    weighted_avg = (similarity_weight * scaled_similarity) + (avgrating_weight * weighted_rating_uncertainty)

    df_clean_2['weighted_avg'] = weighted_avg

    df_sorted = df_clean_2[df_clean_2['weighted_avg'].notna()].sort_values('weighted_avg', ascending=False)

    df_sorted = df_sorted.reindex(columns=['index', 'title', 'weighted_avg', 'weighted_sim_score', 'similarity_scores',
                                           'id', 'overview', 'budget', 'genres', 'keywords', 'popularity',
                                           'release_date', 'revenue', 'runtime', 'tagline', 'vote_average',
                                           'vote_count', 'cast', 'director', 'combined_text'])

    top_10 = df_sorted['title'].iloc[1:11]

    return render_template('index.html', top_10=top_10.to_frame().to_html())


if __name__ == "__main__":
    app.run(debug=True)