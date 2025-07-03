# AI-Powered-Movie-Recommendation-System
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# Preprocess data
movies['metadata'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['metadata'])

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(movie_title):
    movie_title = movie_title.lower()
    if movie_title not in movies['title'].str.lower().values:
        return []
    idx = movies[movies['title'].str.lower() == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# Streamlit UI
st.title("🎬 AI-Powered Movie Recommendation System")
movie_name = st.text_input("Enter a movie you like:", "")

if movie_name:
    recommendations = recommend(movie_name)
    if len(recommendations) > 0:
        st.subheader("Top 5 Recommendations:")
        for i, row in recommendations.iterrows():
            st.markdown(f"{row['title']} — *{row['genres']}*")
    else:
        st.error("Movie not found. Please try a different title.")
