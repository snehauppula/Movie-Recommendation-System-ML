"""
Movie recommendation: load data, TF-IDF search, collaborative filtering.
Uses local movies.csv/ratings.csv or auto-downloads from GroupLens.
"""
import os
import re
import shutil
import unicodedata
from typing import Optional, Tuple
import zipfile
import urllib.request
import tempfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def _data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)


def _download_data(target_dir: str) -> Tuple[str, str]:
    """Download MovieLens ml-latest-small from GroupLens; save movies.csv and ratings.csv to target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    movies_path = os.path.join(target_dir, "movies.csv")
    ratings_path = os.path.join(target_dir, "ratings.csv")
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return movies_path, ratings_path
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        urllib.request.urlretrieve(GROUPLENS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base == "movies.csv":
                    zf.extract(name, tmpdir)
                    shutil.copy2(os.path.join(tmpdir, name), movies_path)
                elif base == "ratings.csv":
                    zf.extract(name, tmpdir)
                    shutil.copy2(os.path.join(tmpdir, name), ratings_path)
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError("movies.csv or ratings.csv not found in zip")
    return movies_path, ratings_path


def load_data(
    movies_path: Optional[str] = None,
    ratings_path: Optional[str] = None,
    auto_download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and ratings. Uses local CSVs if present; otherwise downloads from GroupLens."""
    base = os.path.dirname(__file__)
    mp = movies_path or _data_path("movies.csv")
    rp = ratings_path or _data_path("ratings.csv")

    if not (os.path.exists(mp) and os.path.exists(rp)) and auto_download:
        mp, rp = _download_data(base)

    if not os.path.exists(mp) or not os.path.exists(rp):
        raise FileNotFoundError(
            f"movies.csv and/or ratings.csv not found in {base}. "
            "Download from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip "
            "and extract into this folder."
        )

    return pd.read_csv(mp), pd.read_csv(rp)


def clean_title(title: str) -> str:
    """Normalize title for search: lowercase, strip accents, alphanumeric + spaces."""
    if pd.isna(title):
        return ""
    title = str(title)
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    title = re.sub(r"[^0-9a-z ]+", " ", title.lower().strip())
    return re.sub(r"\s+", " ", title)


def build_vectorizer(corpus: pd.Series, ngram_range=(1, 2)):
    """TF-IDF vectorizer on cleaned titles."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        strip_accents="unicode",
        min_df=2,
    )
    vectors = vectorizer.fit_transform(corpus.fillna("").astype(str))
    return vectorizer, vectors


def search_movie(
    query: str,
    movies_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    vectors,
    top_n: int = 10,
    min_score: float = 0.2,
) -> pd.DataFrame:
    """Return top_n movies matching query by TF-IDF cosine similarity."""
    name = clean_title(query)
    if not name.strip():
        return pd.DataFrame(columns=list(movies_df.columns) + ["_score"])
    vec = vectorizer.transform([name])
    sims = cosine_similarity(vec, vectors).flatten()
    if sims.size == 0 or sims.max() < min_score:
        return pd.DataFrame(columns=list(movies_df.columns) + ["_score"])
    idx = sims.argsort()[-top_n:][::-1]
    out = movies_df.iloc[idx].copy()
    out["_score"] = sims[idx]
    return out


def find_similar_movies(
    movie_id: int,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    min_fraction: float = 0.10,
) -> pd.DataFrame:
    """Recommend movies based on users who liked this movie (collaborative filtering)."""
    similar_users = ratings_df.loc[
        (ratings_df["movieId"] == movie_id) & (ratings_df["rating"] >= 4), "userId"
    ].unique()
    if len(similar_users) == 0:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])

    sim_recs = ratings_df.loc[
        ratings_df["userId"].isin(similar_users) & (ratings_df["rating"] >= 4), "movieId"
    ]
    if sim_recs.empty:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])
    sim_frac = sim_recs.value_counts() / len(similar_users)
    sim_frac = sim_frac[sim_frac > min_fraction]
    if sim_frac.empty:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])

    all_recs = ratings_df.loc[
        ratings_df["movieId"].isin(sim_frac.index) & (ratings_df["rating"] >= 4)
    ]
    num_all = all_recs["userId"].nunique()
    if num_all == 0:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])
    all_frac = all_recs["movieId"].value_counts() / num_all

    rec = pd.concat([sim_frac, all_frac], axis=1).fillna(0)
    rec.columns = ["similar", "all"]
    rec["score"] = rec.apply(lambda r: r["similar"] / r["all"] if r["all"] > 0 else 0, axis=1)
    rec = rec.replace([np.inf, -np.inf], 0).sort_values("score", ascending=False).head(20)
    rec = rec.reset_index().rename(columns={"index": "movieId"})
    rec = rec.merge(movies_df, on="movieId", how="left")
    return rec[["score", "similar", "all", "movieId", "title", "genres"]]
