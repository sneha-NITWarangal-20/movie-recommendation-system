"""Content-based movie recommender.

Each movie is turned into a text document from its metadata (genres,
keywords, top cast, director, and overview). CountVectorizer counts those
words, and cosine similarity finds movies with a similar mix.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"
TOP_CAST = 3
TOP_KEYWORDS = 8


def _parse_list(raw: object) -> list:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _names(raw: object, limit: int | None = None) -> list[str]:
    names = []
    for item in _parse_list(raw):
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
        if limit is not None and len(names) >= limit:
            break
    return names


def _director(raw: object) -> str:
    for person in _parse_list(raw):
        if isinstance(person, dict) and person.get("job") == "Director":
            return str(person.get("name") or "")
    return ""


def _token(name: str) -> str:
    """Keep a person's name as one token so 'Sam Worthington' is not two words."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    movies = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv")
    credits = credits.rename(columns={"movie_id": "id"})
    frame = movies.merge(credits[["id", "cast", "crew"]], on="id", how="inner")
    frame = frame.drop_duplicates(subset="id").copy()
    # Same title can appear twice. Keep the cut with more votes.
    frame = frame.sort_values("vote_count", ascending=False)
    frame = frame.drop_duplicates(subset="title", keep="first")
    frame = frame.sort_values("title").reset_index(drop=True)
    return frame


def build_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    movies = frame.copy()
    movies["overview"] = movies["overview"].fillna("")
    movies["genre_list"] = movies["genres"].apply(_names)
    movies["keyword_list"] = movies["keywords"].apply(lambda raw: _names(raw, TOP_KEYWORDS))
    movies["cast_list"] = movies["cast"].apply(lambda raw: _names(raw, TOP_CAST))
    movies["director"] = movies["crew"].apply(_director)

    def soup(row) -> str:
        pieces = [_token(name) for name in row["genre_list"]]
        pieces += [_token(name) for name in row["keyword_list"]]
        pieces += [_token(name) for name in row["cast_list"]]
        if row["director"]:
            pieces.append(_token(row["director"]))
        pieces.append(row["overview"].lower())
        return " ".join(piece for piece in pieces if piece)

    movies["tags"] = movies.apply(soup, axis=1)
    movies["genres_label"] = movies["genre_list"].apply(lambda names: ", ".join(names))
    movies["cast_label"] = movies["cast_list"].apply(lambda names: ", ".join(names))
    keep = [
        "id",
        "title",
        "overview",
        "genres_label",
        "cast_label",
        "director",
        "vote_average",
        "vote_count",
        "release_date",
        "tags",
    ]
    return movies[keep].reset_index(drop=True)


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, similarity):
        self.movies = movies
        self.similarity = similarity
        self._index = pd.Series(movies.index, index=movies["title"].str.lower())

    @classmethod
    def from_data(cls, data_dir: Path = DATA_DIR) -> "MovieRecommender":
        raw = load_movies(data_dir)
        movies = build_metadata(raw)
        # Bag-of-words on the engineered tags. English stop words are dropped
        # so words like "the" do not make unrelated films look similar.
        matrix = CountVectorizer(stop_words="english", max_features=5000).fit_transform(movies["tags"])
        similarity = cosine_similarity(matrix)
        return cls(movies, similarity)

    def titles(self) -> list[str]:
        return self.movies["title"].tolist()

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        key = title.strip().lower()
        if key not in self._index.index:
            raise KeyError(f"Movie not in the catalogue: {title}")
        idx = int(self._index.loc[key])
        if isinstance(idx, pd.Series):
            idx = int(idx.iloc[0])
        scores = list(enumerate(self.similarity[idx]))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        # Skip films almost nobody rated. A tiny overlap in a short overview
        # otherwise ranks them above real matches.
        picked = []
        for i, _score in scores:
            if i == idx:
                continue
            if int(self.movies.at[i, "vote_count"]) < 50:
                continue
            picked.append(i)
            if len(picked) == n:
                break
        result = self.movies.iloc[picked][
            ["title", "genres_label", "cast_label", "director", "vote_average", "overview"]
        ].copy()
        result["similarity"] = [round(float(self.similarity[idx, i]), 3) for i in picked]
        return result.reset_index(drop=True)
