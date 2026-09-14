# Movie Recommender

A content-based movie recommendation app built with Streamlit, TF-IDF, and cosine similarity.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_movie_recommender.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select the repository and branch.
4. Set the main file path to `app_movie_recommender.py`.
5. Click **Deploy**.

The four `.pkl` files are required by the app and must be committed to the repository:

- `movie_data.pkl`
- `indices.pkl`
- `tfidf_vectorizer.pkl`
- `tfidf_matrix.pkl`