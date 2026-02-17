"""
Helper script to set up Kaggle API credentials
Run this to create your kaggle.json file in the correct location
"""
import os
import json

def setup_kaggle_credentials():
    print("=" * 60)
    print("Kaggle API Credentials Setup")
    print("=" * 60)
    print()
    
    # Get user input
    print("Please provide your Kaggle credentials:")
    print()
    username = input("Kaggle Username: ").strip()
    
    # Use the key provided by the user
    key = "KGAT_e517dcc132307478ebdfb3752dd55bd4"
    print(f"API Key: {key[:20]}... (using the key you provided)")
    print()
    
    # Ask user if they want to enter a different key
    change_key = input("Is this the correct key? (y/n): ").strip().lower()
    if change_key == 'n':
        key = input("Enter your Kaggle API key: ").strip()
    
    # Create credentials dictionary
    credentials = {
        "username": username,
        "key": key
    }
    
    # Determine the path for kaggle.json
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    # Create .kaggle directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Write credentials to file
    with open(kaggle_json_path, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    print()
    print("=" * 60)
    print("✅ Kaggle credentials saved successfully!")
    print("=" * 60)
    print(f"Location: {kaggle_json_path}")
    print()
    print("Next steps:")
    print("1. Test the integration: python test_kaggle_integration.py")
    print("2. Run your app: python app.py")
    print()

if __name__ == "__main__":
    try:
        setup_kaggle_credentials()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
