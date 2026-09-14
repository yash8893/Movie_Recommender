import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    movie_data = joblib.load(BASE_DIR / "movie_data.pkl")
    indices = joblib.load(BASE_DIR / "indices.pkl")
    tfidf_vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.pkl")
    return movie_data, indices, tfidf_vectorizer, tfidf_matrix

try:
    df, indices, tfidf, tfidf_matrix = load_artifacts()
except FileNotFoundError as e:
    st.error("❌ A required pickle file is missing.")
    st.code(str(e))
    st.info("Required files: movie_data.pkl, indices.pkl, tfidf_vectorizer.pkl, tfidf_matrix.pkl")
    st.stop()
except Exception as e:
    st.error("❌ The recommendation system could not be loaded.")
    st.exception(e)
    st.stop()

movie_titles = df["title"].dropna().astype(str).drop_duplicates().tolist()
movie_titles_lower = [(title, title.lower()) for title in movie_titles]


def search_movies(searchterm: str):
    query = searchterm.strip().lower()
    if not query:
        return []

    starts_with = []
    contains = []
    for title, title_lower in movie_titles_lower:
        if title_lower.startswith(query):
            starts_with.append(title)
        elif query in title_lower:
            contains.append(title)

    return (starts_with + contains)[:10]


def recommend_movies(title: str, n: int = 10):
    if title not in indices.index:
        return pd.DataFrame()

    idx = indices[title]
    similarity_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[::-1][1:n + 1]

    recommendations = df.iloc[similar_indices].copy()
    recommendations["similarity"] = similarity_scores[similar_indices]
    return recommendations

st.title("🎬 Movie Recommendation System")
st.markdown("Find a movie and get recommendations based on its **overview, genres, and tagline** using TF-IDF and cosine similarity.")
st.divider()

st.subheader("🔎 Search for a Movie")
selected_movie = st_searchbox(
    search_movies,
    placeholder="Start typing a movie title...",
    label="Movie Title",
    key="movie_search",
    debounce=150,
    clear_on_submit=False,
    edit_after_submit="option",
)

if selected_movie:
    if selected_movie not in indices.index:
        st.warning("Please select a movie from the suggestions.")
    else:
        st.success(f"Selected movie: **{selected_movie}**")
        recommendations = recommend_movies(selected_movie, n=10)

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("🍿 Recommended Movies")
            for rank, (_, movie) in enumerate(recommendations.iterrows(), start=1):
                st.markdown(f"### {rank}. {movie['title']}")
                col1, col2 = st.columns([5, 1])
                with col1:
                    if movie.get("genres", ""):
                        st.write(f"**Genres:** {movie['genres']}")
                    if movie.get("tagline", ""):
                        st.write(f"_{movie['tagline']}_")
                    overview = str(movie.get("overview", ""))
                    if overview:
                        st.write(overview if len(overview) <= 300 else overview[:300] + "...")
                with col2:
                    st.metric("Similarity", f"{movie['similarity']:.1%}")
                    st.write(f"⭐ {float(movie['vote_average']):.1f}")
                    st.write(f"🔥 Popularity: {float(movie['popularity']):.1f}")
                st.divider()

st.sidebar.title("📋 About the Model")
st.sidebar.write("**Recommendation Type:** Content-Based")
st.sidebar.write("**Text Representation:** TF-IDF")
st.sidebar.write("**Similarity:** Cosine Similarity")
st.sidebar.write("**Text Features:** Overview + Genres + Tagline")
st.sidebar.write(f"**Movies:** {len(df):,}")
st.sidebar.write("**Recommendations:** 10")
st.sidebar.divider()
st.sidebar.caption("The recommendation logic follows the original Jupyter Notebook.")
