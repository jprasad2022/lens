"""
Batch Trainer Service - Scheduled Model Training Pipeline
"""

import asyncio
import json
import logging
import os
import schedule
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from services.kafka_service import get_kafka_service
from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel
from config.settings import get_settings
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchTrainer:
    """Handles scheduled model training with Kafka integration."""
    
    def __init__(self):
        self.settings = get_settings()
        self.kafka_service = None
        self.training_history = []
        
    async def initialize(self):
        """Initialize services."""
        if self.settings.kafka_enabled:
            self.kafka_service = get_kafka_service()
            await self.kafka_service.initialize()
            logger.info("Kafka service initialized")
    
    async def load_training_data(self) -> pd.DataFrame:
        """Load training data from snapshots and base data."""
        logger.info("Loading training data...")
        
        # Load base MovieLens data
        ratings_path = self.settings.data_path / "ratings.dat"
        base_ratings = pd.read_csv(
            ratings_path,
            sep='::',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Load streaming data from snapshots
        snapshot_path = Path("data/snapshots")
        if snapshot_path.exists():
            # Load watch events
            watch_files = list(snapshot_path.glob("**/rate_*.parquet"))
            if watch_files:
                watch_dfs = [pd.read_parquet(f) for f in watch_files[-10:]]  # Last 10 files
                streaming_ratings = pd.concat(watch_dfs, ignore_index=True)
                
                # Convert to ratings format
                streaming_ratings = streaming_ratings[['user_id', 'movie_id', 'rating', 'ts']]
                streaming_ratings.rename(columns={'ts': 'timestamp'}, inplace=True)
                
                # Combine with base data
                all_ratings = pd.concat([base_ratings, streaming_ratings], ignore_index=True)
                logger.info(f"Combined {len(base_ratings)} base ratings with {len(streaming_ratings)} streaming ratings")
                return all_ratings
        
        return base_ratings
    
    async def train_model(self, model_class, model_name: str, ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """Train a single model and save to registry."""
        logger.info(f"Training {model_name} model...")
        
        try:
            start_time = time.time()
            
            # Initialize and train model
            model = model_class()
            await model.train(ratings_df)
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = {
                "training_time_seconds": training_time,
                "n_users": len(ratings_df['user_id'].unique()),
                "n_movies": len(ratings_df['movie_id'].unique()),
                "n_ratings": len(ratings_df),
                "trained_at": datetime.utcnow().isoformat()
            }
            
            # Save model
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            await self.save_model(model_name, model, version, metrics)
            
            # Send training event to Kafka
            if self.kafka_service:
                await self.kafka_service.produce(
                    topic=f"{self.settings.team_prefix}.model_metrics",
                    key=model_name,
                    value=json.dumps({
                        "model": model_name,
                        "version": version,
                        "metrics": metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                )
            
            return {
                "model": model_name,
                "version": version,
                "metrics": metrics,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            return {
                "model": model_name,
                "status": "failed",
                "error": str(e)
            }
    
    async def save_model(self, model_name: str, model: Any, version: str, metrics: Dict[str, Any]):
        """Save model to registry with metadata."""
        model_dir = self.settings.model_registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model pickle
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "name": model_name,
            "version": version,
            "type": model.__class__.__name__,
            "metrics": metrics,
            "parameters": getattr(model, 'params', {}),
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest pointer
        with open(self.settings.model_registry_path / model_name / "latest.txt", 'w') as f:
            f.write(version)
        
        # Create model card
        model_card = f"""# {model_name.title()} Model

## Version: {version}

### Training Metrics
- Training Time: {metrics['training_time_seconds']:.2f} seconds
- Number of Users: {metrics['n_users']:,}
- Number of Movies: {metrics['n_movies']:,}
- Number of Ratings: {metrics['n_ratings']:,}
- Trained At: {metrics['trained_at']}

### Model Type
{model.__class__.__name__}

### Usage
This model is automatically loaded by the recommendation service.
"""
        
        with open(model_dir / "model_card.md", 'w') as f:
            f.write(model_card)
        
        logger.info(f"Saved {model_name} v{version} to {model_dir}")
    
    async def train_all_models(self):
        """Train all recommendation models."""
        logger.info("Starting batch training pipeline...")
        
        # Load data
        ratings_df = await self.load_training_data()
        
        # Define models to train
        models = [
            (PopularityModel, "popularity"),
            (CollaborativeFilteringModel, "collaborative"),
            (ALSModel, "als")
        ]
        
        # Train each model
        results = []
        for model_class, model_name in models:
            result = await self.train_model(model_class, model_name, ratings_df)
            results.append(result)
            
        # Log summary
        successful = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"Training complete: {successful}/{len(models)} models trained successfully")
        
        # Store history
        self.training_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        })
        
        return results
    
    async def evaluate_drift(self):
        """Check for model drift using recent predictions."""
        logger.info("Evaluating model drift...")
        
        # This would analyze recent prediction performance
        # and trigger retraining if drift detected
        pass
    
    def schedule_training(self):
        """Schedule periodic training jobs."""
        # Schedule daily training at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.train_all_models())
        )
        
        # Schedule drift check every hour
        schedule.every().hour.do(
            lambda: asyncio.create_task(self.evaluate_drift())
        )
        
        logger.info("Training schedule configured:")
        logger.info("- Daily full retraining at 02:00")
        logger.info("- Hourly drift evaluation")
    
    async def run(self):
        """Run the batch trainer service."""
        await self.initialize()
        
        # Run initial training
        await self.train_all_models()
        
        # Schedule periodic training
        self.schedule_training()
        
        # Keep service running
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute

async def main():
    """Main entry point."""
    trainer = BatchTrainer()
    
    # Handle graceful shutdown
    try:
        await trainer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down batch trainer...")
        if trainer.kafka_service:
            await trainer.kafka_service.close()

if __name__ == "__main__":
    asyncio.run(main())