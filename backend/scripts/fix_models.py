"""
Quick fix to properly train and save models
"""
import os
import sys
import pickle
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load ratings data manually
def load_ratings_data():
    """Load ratings from .dat file"""
    ratings = []
    data_path = Path("/app/data") if os.path.exists("/app/data") else Path("data")
    
    with open(data_path / 'ratings.dat', 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) == 4:
                ratings.append({
                    'user_id': int(parts[0]),
                    'movie_id': int(parts[1]),
                    'rating': float(parts[2])
                })
    
    return ratings

def create_pandas_dataframe(ratings):
    """Convert ratings to pandas DataFrame"""
    import pandas as pd
    return pd.DataFrame(ratings)

# Main execution
if __name__ == "__main__":
    print("Loading and converting data...")
    ratings = load_ratings_data()
    print(f"Loaded {len(ratings)} ratings")
    
    # Convert to pandas DataFrame
    import pandas as pd
    ratings_df = pd.DataFrame(ratings)
    
    # Import models
    from recommender.models_impl import CollaborativeFilteringModel, ALSModel
    import asyncio
    
    async def train_and_save():
        # Train Collaborative Filtering
        print("\nTraining Collaborative Filtering...")
        collab_model = CollaborativeFilteringModel()
        await collab_model.train(ratings_df)
        
        # Save the actual model object
        model_dir = Path("model_registry/collaborative/0.1.0")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(collab_model, f)
        
        # Save metadata
        metadata = {
            "name": "collaborative",
            "version": "0.1.0",
            "type": "collaborative",
            "trained_at": collab_model.metadata.get("trained_at", ""),
            "metrics": {
                "num_items": collab_model.metadata.get("n_movies", 0)
            },
            "parameters": {}
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest
        with open("model_registry/collaborative/latest.txt", 'w') as f:
            f.write("0.1.0")
        
        print("✓ Collaborative model saved")
        
        # Test the model
        print("\nTesting collaborative model...")
        recs = await collab_model.predict(100, k=5)
        print(f"Recommendations for user 100: {recs}")
        
        # Train ALS
        print("\nTraining ALS...")
        als_model = ALSModel()
        await als_model.train(ratings_df)
        
        # Save ALS model
        model_dir = Path("model_registry/als/0.1.0")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(als_model, f)
        
        # Save metadata
        metadata = {
            "name": "als",
            "version": "0.1.0", 
            "type": "als",
            "trained_at": als_model.metadata.get("trained_at", ""),
            "metrics": {
                "num_items": als_model.metadata.get("n_movies", 0)
            },
            "parameters": {}
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update latest
        with open("model_registry/als/latest.txt", 'w') as f:
            f.write("0.1.0")
            
        print("✓ ALS model saved")
        
        # Test the model
        print("\nTesting ALS model...")
        recs = await als_model.predict(100, k=5)
        print(f"Recommendations for user 100: {recs}")
    
    # Run the async function
    asyncio.run(train_and_save())
    print("\n✅ Models fixed!")