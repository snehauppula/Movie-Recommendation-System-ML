"""
Test script to verify Kaggle dataset integration
This script tests the load_data function with Kaggle downloading
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_data

def test_kaggle_download():
    """Test loading data from Kaggle"""
    print("=" * 60)
    print("Testing Kaggle Dataset Integration")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Try loading with Kaggle (will use cache if available)
        print("Test 1: Loading data (will use local files if present, otherwise Kaggle)")
        print("-" * 60)
        movies, ratings = load_data(use_kaggle=True)
        
        print()
        print(f"✅ Success! Loaded data:")
        print(f"   - Movies: {len(movies)} entries")
        print(f"   - Ratings: {len(ratings)} entries")
        print()
        print("Sample movies:")
        print(movies.head())
        print()
        print("Sample ratings:")
        print(ratings.head())
        print()
        
        # Test 2: Verify columns
        print("Test 2: Verifying data structure")
        print("-" * 60)
        expected_movie_cols = ['movieId', 'title', 'genres']
        expected_rating_cols = ['userId', 'movieId', 'rating']
        
        assert all(col in movies.columns for col in expected_movie_cols), \
            f"Missing columns in movies: {expected_movie_cols}"
        assert all(col in ratings.columns for col in expected_rating_cols), \
            f"Missing columns in ratings: {expected_rating_cols}"
        
        print("✅ All expected columns present")
        print()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. You can now delete the local movies.csv and ratings.csv files")
        print("2. Run 'python app.py' to start the Streamlit app")
        print("3. The app will use the cached Kaggle data automatically")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("If you see authentication errors, make sure:")
        print("1. You have a Kaggle account")
        print("2. You've downloaded kaggle.json from https://www.kaggle.com/settings")
        print("3. Place kaggle.json in your home directory:")
        print(f"   Windows: C:\\Users\\{os.getenv('USERNAME')}\\.kaggle\\kaggle.json")
        print()
        print("See KAGGLE_SETUP.md for detailed instructions")
        return False
    
    return True

if __name__ == "__main__":
    success = test_kaggle_download()
    sys.exit(0 if success else 1)
