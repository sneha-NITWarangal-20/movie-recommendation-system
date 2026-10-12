"""Streamlit app: pick a movie, get content-based recommendations."""
import streamlit as st

from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model() -> MovieRecommender:
    return MovieRecommender.from_data()


model = load_model()
titles = model.titles()

st.title("Movie Recommendation System")
st.write(
    "Content-based recommendations from movie metadata. "
    "Genres, keywords, cast, director, and overview are combined into one text, "
    "counted with CountVectorizer, and compared with cosine similarity."
)

chosen = st.selectbox("Choose a movie", titles, index=titles.index("The Dark Knight") if "The Dark Knight" in titles else 0)
count = st.slider("How many recommendations", min_value=3, max_value=10, value=5)

selected = model.movies.loc[model.movies["title"] == chosen].iloc[0]
st.subheader(chosen)
meta = " · ".join(
    part
    for part in [
        selected["genres_label"],
        f"Director: {selected['director']}" if selected["director"] else "",
        f"Rating: {selected['vote_average']}",
    ]
    if part
)
st.caption(meta)
st.write(selected["overview"] or "No overview in the dataset.")

st.subheader("Because you picked this")
recs = model.recommend(chosen, n=count)
for _, row in recs.iterrows():
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{row['title']}**")
            st.caption(
                " · ".join(
                    part
                    for part in [
                        row["genres_label"],
                        row["cast_label"],
                        f"Director: {row['director']}" if row["director"] else "",
                    ]
                    if part
                )
            )
            overview = row["overview"] or ""
            st.write(overview[:320] + ("..." if len(overview) > 320 else ""))
        with right:
            st.metric("Similarity", f"{row['similarity']:.2f}")
            st.metric("Rating", f"{row['vote_average']}")
