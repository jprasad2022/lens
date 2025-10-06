#!/usr/bin/env python3
"""
Train initial recommendation models from MovieLens data
This script REQUIRES data to be downloaded first!
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class ModelTrainer:
    def __init__(self, data_dir="./data", model_dir="./model_registry"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        # Data can be either in ml-1m subdirectory or directly in data directory
        if (self.data_dir / "ml-1m").exists():
            self.ml_path = self.data_dir / "ml-1m"
        else:
            self.ml_path = self.data_dir
        
    def check_data_exists(self):
        """Check if MovieLens data exists"""
        required_files = ['movies.dat', 'ratings.dat', 'users.dat']
        
        if not self.ml_path.exists():
            print(f"❌ Data directory not found: {self.ml_path}")
            print("\nPlease download data first:")
            print("  python scripts/download_movielens.py")
            return False
            
        missing_files = []
        for file in required_files:
            if not (self.ml_path / file).exists():
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing data files: {missing_files}")
            print("\nPlease download data first:")
            print("  python scripts/download_movielens.py")
            return False
            
        return True
    
    def load_data(self):
        """Load MovieLens data"""
        print("Loading MovieLens data...")
        
        # Load ratings
        ratings = pd.read_csv(
            self.ml_path / 'ratings.dat', 
            sep='::', 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Load movies
        movies = pd.read_csv(
            self.ml_path / 'movies.dat',
            sep='::',
            names=['item_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
        
        # Load users
        users = pd.read_csv(
            self.ml_path / 'users.dat',
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip'],
            engine='python'
        )
        
        print(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
        return ratings, movies, users
    
    def train_popularity_model(self, ratings, movies):
        """Train simple popularity-based model"""
        print("\n🏋️ Training Popularity Model...")
        
        # Calculate item popularity
        item_stats = ratings.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        item_stats.columns = ['item_id', 'rating_count', 'avg_rating']
        
        # Popularity score: combination of rating count and average rating
        # Using logarithmic scaling for counts to avoid extreme bias
        item_stats['popularity_score'] = (
            np.log1p(item_stats['rating_count']) * item_stats['avg_rating']
        )
        
        # Sort by popularity
        item_stats = item_stats.sort_values('popularity_score', ascending=False)
        
        # Merge with movie titles
        popular_items = item_stats.merge(movies[['item_id', 'title']], on='item_id')
        
        # Create model dictionary
        model = {
            'type': 'popularity',
            'version': '0.1.0',
            'popularity_scores': item_stats.set_index('item_id')['popularity_score'].to_dict(),
            'item_stats': item_stats.to_dict('records'),
            'top_items': popular_items.head(100)['item_id'].tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        print(f"✓ Popularity model trained with {len(model['popularity_scores'])} items")
        return model
    
    def train_collaborative_model(self, ratings):
        """Train simple collaborative filtering model"""
        print("\n🏋️ Training Collaborative Filtering Model...")
        
        # Create user-item matrix
        user_item = ratings.pivot_table(
            index='user_id',
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate item-item similarities (using Pearson correlation)
        # Note: This is a simplified version - production would use more sophisticated methods
        print("Calculating item similarities (this may take a moment)...")
        
        # Sample for faster training (use top 500 most rated items)
        item_counts = ratings['item_id'].value_counts()
        top_items = item_counts.head(500).index
        user_item_sample = user_item[top_items]
        
        # Calculate correlations using sklearn's cosine similarity (more efficient)
        from sklearn.preprocessing import StandardScaler
        
        # Normalize the data
        scaler = StandardScaler()
        user_item_normalized = scaler.fit_transform(user_item_sample.T)
        
        # Calculate cosine similarities
        item_sim_matrix = cosine_similarity(user_item_normalized)
        item_sim = pd.DataFrame(
            item_sim_matrix,
            index=user_item_sample.columns,
            columns=user_item_sample.columns
        )
        
        # Convert to dictionary format for storage
        similarities = {}
        for item in item_sim.index:
            # Get top 50 similar items for each item
            similar = item_sim[item].sort_values(ascending=False)[1:51]
            similarities[int(item)] = [
                (int(sim_item), float(score)) 
                for sim_item, score in similar.items() 
                if score > 0.1  # Only keep items with correlation > 0.1
            ]
        
        model = {
            'type': 'collaborative_filtering',
            'version': '0.1.0',
            'item_similarities': similarities,
            'num_items': len(similarities),
            'min_similarity': 0.1,
            'max_neighbors': 50,
            'training_date': datetime.now().isoformat()
        }
        
        print(f"✓ Collaborative model trained with {len(similarities)} items")
        return model
    
    def save_model(self, model, model_type):
        """Save model to model registry"""
        version = model['version']
        model_path = self.model_dir / model_type / version
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model pickle
        with open(model_path / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'version': version,
            'trained_at': model['training_date'],
            'framework': 'custom',
            'metrics': {
                'num_items': model.get('num_items', len(model.get('popularity_scores', [])))
            }
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create model card
        model_card = f"""# {model_type.title()} Model

**Version**: {version}  
**Trained**: {model['training_date']}  
**Type**: {model['type']}

## Description
{'Recommends items based on overall popularity and ratings.' if model_type == 'popularity' else 'Recommends items based on collaborative filtering using item-item similarities.'}

## Training Data
- Dataset: MovieLens 1M
- Ratings: 1,000,209
- Users: 6,040  
- Items: 3,706

## Usage
This model is automatically loaded by the LENS recommendation service.
"""
        
        with open(model_path / 'model_card.md', 'w') as f:
            f.write(model_card)
        
        # Create 'latest' pointer (Windows-compatible)
        latest_path = self.model_dir / model_type / 'latest'
        
        # On Windows, create a text file pointing to the latest version
        # On Unix, create a symlink
        if os.name == 'nt':  # Windows
            with open(f"{latest_path}.txt", 'w') as f:
                f.write(version)
        else:  # Unix/Linux
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(Path(version))
        
        print(f"✓ Saved {model_type} model to {model_path}")
    
    def train_all_models(self):
        """Train all initial models"""
        # Check data exists
        if not self.check_data_exists():
            return False
        
        # Load data
        ratings, movies, users = self.load_data()
        
        # Train models
        print("\n" + "="*50)
        print("Training Initial Models")
        print("="*50)
        
        # Train popularity model
        popularity_model = self.train_popularity_model(ratings, movies)
        self.save_model(popularity_model, 'popularity')
        
        # Train collaborative model
        collab_model = self.train_collaborative_model(ratings)
        self.save_model(collab_model, 'collaborative')
        
        print("\n" + "="*50)
        print("✅ Model training complete!")
        print("="*50)
        
        # Show model registry structure
        print("\nModel Registry Structure:")
        for model_type in ['popularity', 'collaborative']:
            model_path = self.model_dir / model_type / '0.1.0'
            if model_path.exists():
                print(f"\n{model_type}/")
                print(f"  0.1.0/")
                for file in model_path.iterdir():
                    size_kb = file.stat().st_size / 1024
                    print(f"    {file.name} ({size_kb:.1f} KB)")
                print(f"  latest -> 0.1.0")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train initial recommendation models")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--model-dir", default="./model_registry", help="Model registry path")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.data_dir, args.model_dir)
    success = trainer.train_all_models()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()