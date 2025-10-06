#!/usr/bin/env python3
"""
Train initial recommendation models without external ML dependencies
Uses only Python standard library + basic packages
"""

import os
import json
import pickle
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import math

class SimpleModelTrainer:
    def __init__(self, data_dir="./data", model_dir="./model_registry"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.ml_path = self.data_dir
        
    def check_data_exists(self):
        """Check if MovieLens data exists"""
        required_files = ['movies.dat', 'ratings.dat', 'users.dat']
        
        if not self.ml_path.exists():
            print(f"❌ Data directory not found: {self.ml_path}")
            return False
            
        missing_files = []
        for file in required_files:
            if not (self.ml_path / file).exists():
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing data files: {missing_files}")
            return False
            
        return True
    
    def load_ratings(self):
        """Load ratings data using csv module"""
        print("Loading ratings data...")
        ratings = []
        
        with open(self.ml_path / 'ratings.dat', 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) == 4:
                    ratings.append({
                        'user_id': int(parts[0]),
                        'item_id': int(parts[1]),
                        'rating': float(parts[2]),
                        'timestamp': int(parts[3])
                    })
        
        print(f"Loaded {len(ratings)} ratings")
        return ratings
    
    def load_movies(self):
        """Load movies data"""
        print("Loading movies data...")
        movies = {}
        
        with open(self.ml_path / 'movies.dat', 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) == 3:
                    movies[int(parts[0])] = {
                        'item_id': int(parts[0]),
                        'title': parts[1],
                        'genres': parts[2]
                    }
        
        print(f"Loaded {len(movies)} movies")
        return movies
    
    def train_popularity_model(self, ratings, movies):
        """Train simple popularity-based model"""
        print("\n🏋️ Training Popularity Model...")
        
        # Calculate item statistics
        item_stats = defaultdict(lambda: {'count': 0, 'sum': 0})
        
        for rating in ratings:
            item_id = rating['item_id']
            item_stats[item_id]['count'] += 1
            item_stats[item_id]['sum'] += rating['rating']
        
        # Calculate popularity scores
        popularity_scores = {}
        for item_id, stats in item_stats.items():
            avg_rating = stats['sum'] / stats['count']
            # Popularity score combines rating count and average rating
            # Using log to prevent extreme bias toward high-count items
            popularity_score = math.log1p(stats['count']) * avg_rating
            popularity_scores[item_id] = {
                'score': popularity_score,
                'avg_rating': avg_rating,
                'rating_count': stats['count']
            }
        
        # Sort by popularity
        sorted_items = sorted(popularity_scores.items(), 
                            key=lambda x: x[1]['score'], 
                            reverse=True)
        
        # Get top 100 items
        top_items = [item_id for item_id, _ in sorted_items[:100]]
        
        # Create model dictionary
        model = {
            'type': 'popularity',
            'version': '0.1.0',
            'popularity_scores': {k: v['score'] for k, v in popularity_scores.items()},
            'item_stats': popularity_scores,
            'top_items': top_items,
            'num_items': len(popularity_scores),
            'training_date': datetime.now().isoformat()
        }
        
        print(f"✓ Popularity model trained with {len(popularity_scores)} items")
        return model
    
    def train_collaborative_model(self, ratings):
        """Train simple item-based collaborative filtering"""
        print("\n🏋️ Training Collaborative Filtering Model...")
        
        # Create user-item mapping
        user_items = defaultdict(dict)
        item_users = defaultdict(dict)
        
        for rating in ratings:
            user_id = rating['user_id']
            item_id = rating['item_id']
            score = rating['rating']
            
            user_items[user_id][item_id] = score
            item_users[item_id][user_id] = score
        
        # Get item rating counts
        item_counts = Counter(rating['item_id'] for rating in ratings)
        
        # Calculate similarities for top 500 most rated items (for performance)
        top_items = [item for item, _ in item_counts.most_common(500)]
        
        print("Calculating item similarities...")
        similarities = {}
        
        for i, item1 in enumerate(top_items):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(top_items)} items processed")
            
            item1_users = item_users[item1]
            similar_items = []
            
            for item2 in top_items:
                if item1 == item2:
                    continue
                
                # Find common users
                item2_users = item_users[item2]
                common_users = set(item1_users.keys()) & set(item2_users.keys())
                
                if len(common_users) < 5:  # Need at least 5 common ratings
                    continue
                
                # Calculate cosine similarity
                dot_product = sum(item1_users[u] * item2_users[u] for u in common_users)
                norm1 = math.sqrt(sum(r**2 for r in item1_users.values()))
                norm2 = math.sqrt(sum(r**2 for r in item2_users.values()))
                
                if norm1 * norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    if similarity > 0.1:  # Keep only positive similarities above threshold
                        similar_items.append((item2, similarity))
            
            # Keep top 50 most similar items
            similar_items.sort(key=lambda x: x[1], reverse=True)
            similarities[item1] = similar_items[:50]
        
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
                'num_items': model.get('num_items', 0)
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
- Implementation: Pure Python (no external ML libraries required)

## Usage
This model is automatically loaded by the LENS recommendation service.
"""
        
        with open(model_path / 'model_card.md', 'w') as f:
            f.write(model_card)
        
        print(f"✓ Saved {model_type} model to {model_path}")
    
    def train_all_models(self):
        """Train all initial models"""
        # Check data exists
        if not self.check_data_exists():
            return False
        
        # Load data
        ratings = self.load_ratings()
        movies = self.load_movies()
        
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
        
        return True

def main():
    """Main function"""
    import sys
    
    # Simple argument parsing
    data_dir = "./data"
    model_dir = "./model_registry"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python train_simple_models.py [data_dir] [model_dir]")
        sys.exit(0)
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_dir = sys.argv[2]
    
    trainer = SimpleModelTrainer(data_dir, model_dir)
    success = trainer.train_all_models()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()