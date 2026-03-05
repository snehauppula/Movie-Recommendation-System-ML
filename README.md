# CineMatch — Movie Recommender

Search movies by title and get recommendations (TF-IDF + collaborative filtering). Built with Streamlit and scikit-learn.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `movies.csv` and `ratings.csv` are missing, the app downloads MovieLens ml-latest-small from GroupLens on first run. Or download [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) and put the two CSVs in the project folder.

## Project layout

- **app.py** — Streamlit UI (search, browse, recommendations)
- **recommender.py** — Load data, TF-IDF search, collaborative filtering
- **requirements.txt** — Dependencies
