import os
import re
import unicodedata
from typing import Optional, Tuple
import zipfile
import shutil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _data_path(filename: str) -> str:
    base = os.path.dirname(__file__)
    return os.path.join(base, filename)


def download_kaggle_dataset(dataset_name: str = "grouplens/movielens-latest-small", 
                            cache_dir: str = ".kaggle_cache") -> Tuple[str, str]:
    """Download dataset from Kaggle and return paths to movies.csv and ratings.csv
    
    Args:
        dataset_name: Kaggle dataset identifier (default: grouplens/movielens-latest-small)
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Tuple of (movies_csv_path, ratings_csv_path)
    """
    import opendatasets as od
    
    # Create cache directory if it doesn't exist
    cache_path = _data_path(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    # Check if already downloaded
    dataset_folder = os.path.join(cache_path, dataset_name.split('/')[-1])
    
    if not os.path.exists(dataset_folder):
        print(f"📥 Downloading dataset from Kaggle: {dataset_name}")
        try:
            # Download using opendatasets (handles Kaggle authentication)
            kaggle_url = f"https://www.kaggle.com/datasets/{dataset_name}"
            od.download(kaggle_url, data_dir=cache_path)
            print("✅ Dataset downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading from Kaggle: {e}")
            print("\n💡 Please ensure you have Kaggle API credentials set up:")
            print("   1. Go to https://www.kaggle.com/settings")
            print("   2. Click 'Create New API Token'")
            print("   3. Save kaggle.json to your home directory")
            raise
    else:
        print(f"✅ Using cached dataset from {dataset_folder}")
    
    # Find movies.csv and ratings.csv in the downloaded folder
    movies_path = None
    ratings_path = None
    
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file == "movies.csv":
                movies_path = os.path.join(root, file)
            elif file == "ratings.csv":
                ratings_path = os.path.join(root, file)
    
    if not movies_path or not ratings_path:
        raise FileNotFoundError(
            f"Could not find movies.csv or ratings.csv in {dataset_folder}. "
            f"Please check the dataset structure."
        )
    
    return movies_path, ratings_path


def load_data(movies_path: Optional[str] = None, 
              ratings_path: Optional[str] = None,
              use_kaggle: bool = True,
              dataset_name: str = "grouplens/movielens-latest-small"):
    """Load `movies.csv` and `ratings.csv` from local files or Kaggle.
    
    Args:
        movies_path: Path to local movies.csv (if None, will try to download from Kaggle)
        ratings_path: Path to local ratings.csv (if None, will try to download from Kaggle)
        use_kaggle: If True and local files not found, download from Kaggle
        dataset_name: Kaggle dataset identifier
        
    Returns:
        Tuple of (movies_df, ratings_df)
    """
    # Try local paths first
    if movies_path is None:
        movies_path = _data_path("movies.csv")
    if ratings_path is None:
        ratings_path = _data_path("ratings.csv")
    
    # Check if local files exist
    local_files_exist = os.path.exists(movies_path) and os.path.exists(ratings_path)
    
    if not local_files_exist and use_kaggle:
        print("🔍 Local CSV files not found. Attempting to download from Kaggle...")
        try:
            movies_path, ratings_path = download_kaggle_dataset(dataset_name)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load data from local files or Kaggle. Error: {e}\n"
                f"Please either:\n"
                f"  1. Place movies.csv and ratings.csv in {os.path.dirname(__file__)}, OR\n"
                f"  2. Set up Kaggle API credentials to download automatically"
            )
    elif not local_files_exist:
        raise FileNotFoundError(
            f"movies.csv and/or ratings.csv not found in {os.path.dirname(__file__)}"
        )
    
    print(f"📂 Loading movies from: {movies_path}")
    print(f"📂 Loading ratings from: {ratings_path}")
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    print(f"✅ Loaded {len(movies)} movies and {len(ratings)} ratings")
    
    return movies, ratings


def clean_title(title: str) -> str:
    """Safer title cleaning: handle NaN, lowercase, strip accents, remove non-alphanumerics, collapse spaces."""
    if pd.isna(title):
        return ""
    # normalize unicode (accents)
    title = str(title)
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")
    title = title.lower().strip()
    # replace non-alphanumeric with spaces
    title = re.sub(r"[^0-9a-z ]+", " ", title)
    # collapse whitespace
    title = re.sub(r"\s+", " ", title)
    return title


def build_vectorizer(corpus: pd.Series, ngram_range=(1, 2)):
    """Create tuned TF-IDF vectorizer and fit on `corpus` (cleaned titles)."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        strip_accents="unicode",
        min_df=2,
    )
    vectors = vectorizer.fit_transform(corpus.fillna("").astype(str))
    return vectorizer, vectors


def search_movie(movie_name: str, movies_df: pd.DataFrame, vectorizer: TfidfVectorizer, vectors, top_n: int = 10, min_score: float = 0.2):
    """Return top_n rows from movies_df that best match movie_name using TF-IDF cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    name = clean_title(movie_name)
    if not name.strip():
        return pd.DataFrame(columns=movies_df.columns)
    movie_vector = vectorizer.transform([name])
    sims = cosine_similarity(movie_vector, vectors).flatten()
    if sims.size == 0:
        return pd.DataFrame(columns=movies_df.columns)
    if sims.max() < min_score:
        return pd.DataFrame(columns=movies_df.columns)
    idx = sims.argsort()[-top_n:][::-1]
    result = movies_df.iloc[idx].copy()
    result = result.assign(_score=sims[idx])
    return result


def find_similar_movies(movie_id: int, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, min_similar_fraction: float = 0.10):
    """Find movies recommended by users who liked the given movie.

    Returns a DataFrame with columns ['score', 'similar', 'all', 'movieId', 'title', 'genres'] sorted by score desc.
    """
    # users who liked the movie (>=4)
    similar_users = ratings_df.loc[(ratings_df["movieId"] == movie_id) & (ratings_df["rating"] >= 4), "userId"].unique()
    if len(similar_users) == 0:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])

    # recommendations among those similar users
    sim_recs = ratings_df.loc[ratings_df["userId"].isin(similar_users) & (ratings_df["rating"] >= 4), "movieId"]
    if sim_recs.empty:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])
    sim_frac = sim_recs.value_counts() / float(len(similar_users))
    sim_frac = sim_frac[sim_frac > min_similar_fraction]
    if sim_frac.empty:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])

    # fraction among all users who rated those movies (>=4)
    all_recs = ratings_df.loc[ratings_df["movieId"].isin(sim_frac.index) & (ratings_df["rating"] >= 4)]
    num_all_users = all_recs["userId"].nunique()
    if num_all_users == 0:
        return pd.DataFrame(columns=["score", "similar", "all", "movieId", "title", "genres"])
    all_frac = all_recs["movieId"].value_counts() / float(num_all_users)

    rec_percentages = pd.concat([sim_frac, all_frac], axis=1).fillna(0)
    rec_percentages.columns = ["similar", "all"]
    # avoid divide-by-zero; replace 0 with NaN so result becomes inf -> then clip/fill
    rec_percentages["score"] = rec_percentages.apply(lambda row: (row["similar"] / row["all"]) if row["all"] > 0 else np.nan, axis=1)
    rec_percentages = rec_percentages.replace([np.inf, -np.inf], np.nan).fillna(0)
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    df = rec_percentages.head(20).reset_index().rename(columns={"index": "movieId"})
    df = df.merge(movies_df, on="movieId", how="left")
    # keep useful columns
    out = df[["score", "similar", "all", "movieId", "title", "genres"]]
    return out
