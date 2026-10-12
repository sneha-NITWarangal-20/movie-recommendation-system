# Movie Recommendation System

Content-based recommender on the TMDB 5000 movie dataset (about 4,800 films).

A movie is recommended because it shares metadata with the one you pick, not because other people watched both. The metadata used is genres, keywords, the top three cast members, the director, and the plot overview.

## How it works

1. Parse the JSON metadata and join movies with credits.
2. Build one text field per movie. Names are kept as a single token so a person is not split into first and last name.
3. `CountVectorizer` turns that text into word counts (English stop words removed, top 5,000 terms).
4. Cosine similarity scores every pair of movies.
5. The Streamlit app returns the closest movies that have at least 50 ratings, so a short unused listing does not outrank a real match.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The CSVs live in `data/`. The first load builds the similarity matrix and then caches it for the session.

## Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit.
