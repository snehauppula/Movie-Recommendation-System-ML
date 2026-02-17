# Kaggle Dataset Setup Guide

This guide explains how to set up Kaggle API credentials to automatically download movie datasets.

## Quick Start

The application will automatically download the dataset from Kaggle if local CSV files are not found. To enable this:

### 1. Create a Kaggle Account

Visit [kaggle.com](https://www.kaggle.com) and create a free account (if you don't have one).

### 2. Get Your API Token

1. Go to [Kaggle Settings](https://www.kaggle.com/settings)
2. Scroll down to the **API** section
3. Click **"Create New API Token"**
4. This will download a `kaggle.json` file

### 3. Place kaggle.json

**Windows:**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**Linux/Mac:**
```
~/.kaggle/kaggle.json
```

**Alternative (for this project only):**
You can also place `kaggle.json` in the project root directory.

### 4. Set Permissions (Linux/Mac only)

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Using the Application

### First Run (Auto-Download)

When you run the application for the first time without local CSV files:

```bash
python app.py
```

The application will:
1. Detect that local CSV files are missing
2. Download the dataset from Kaggle automatically
3. Cache it in `.kaggle_cache` folder
4. Load the data and start the app

### Subsequent Runs

The cached data will be used automatically - no re-download needed!

## Configuration

### Change the Dataset

By default, the app uses `grouplens/movielens-latest-small`. To use a different dataset, modify `app.py`:

```python
# In load_app_data() function
movies, ratings = load_data(
    use_kaggle=True,
    dataset_name="grouplens/movielens-20m-dataset"  # Change this
)
```

### Popular Movie Datasets

- `grouplens/movielens-latest-small` - 100k ratings (recommended for quick testing)
- `grouplens/movielens-20m-dataset` - 20M ratings (larger, more comprehensive)
- `rounakbanik/the-movies-dataset` - Alternative movie dataset

## Troubleshooting

### "Could not download from Kaggle"

**Problem:** Missing or incorrect Kaggle API credentials

**Solution:**
1. Verify `kaggle.json` is in the correct location
2. Check that the file contains valid credentials
3. Ensure you've accepted the dataset's terms on Kaggle website

### "Dataset not found"

**Problem:** The specified dataset doesn't exist or isn't accessible

**Solution:**
1. Visit the dataset page on Kaggle
2. Click "Download" button to accept terms
3. Verify the dataset name is correct

### Still Having Issues?

You can always use local CSV files instead:
1. Download the dataset manually from Kaggle
2. Extract `movies.csv` and `ratings.csv`
3. Place them in the project root directory (`d:\ds\recommendation\`)

## Manual Download (Alternative)

If you prefer not to use the Kaggle API:

1. Visit: https://www.kaggle.com/datasets/grouplens/movielens-latest-small
2. Click **Download** (you'll need to be logged in)
3. Extract the ZIP file
4. Copy `movies.csv` and `ratings.csv` to your project folder
5. The app will use these local files automatically

## How It Works

The data loading logic follows this priority:

```
1. Check for local CSV files in project directory
   ↓ (if not found)
2. Check cache directory (.kaggle_cache)
   ↓ (if not found)
3. Download from Kaggle using API
   ↓
4. Save to cache for future use
   ↓
5. Load data and start app
```

This ensures:
- ✅ No repeated downloads
- ✅ Works offline after first download
- ✅ Minimal disk space usage
- ✅ Easy setup

## Need Help?

If you encounter any issues, check:
- Kaggle API credentials are set up correctly
- You have internet connection
- You've accepted the dataset terms on Kaggle
- The dataset name is correct
