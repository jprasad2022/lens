#!/usr/bin/env python3
"""
Run a simple offline evaluation without Kafka dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment to disable Kafka
os.environ["KAFKA_ENABLED"] = "false"
os.environ["MODEL_PATH"] = "../models"
os.environ["DATA_PATH"] = "../data"

import asyncio
import json
from evaluation.offline_evaluator import OfflineEvaluator
from services.model_service import ModelService


async def run_offline_evaluation():
    """Run offline evaluation with chronological split"""
    
    # Initialize evaluator
    # Get the backend directory path
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(backend_dir, "data")
    
    evaluator = OfflineEvaluator(data_path=data_path)
    evaluator.load_data()
    
    # Perform chronological split
    print("\nPerforming chronological split...")
    train, test = evaluator.chronological_split()
    
    # Initialize and load model
    print("\nLoading recommendation model...")
    model_service = ModelService()
    await model_service.load_models()
    
    # Create a wrapper to match expected interface
    class ModelWrapper:
        def __init__(self, model_service):
            self.model_service = model_service
            
        def recommend(self, user_id, k=10):
            # Use the SVD model for evaluation
            try:
                predictions = self.model_service.svd_model.model.test(
                    [(user_id, movie_id, 0) for movie_id in range(1, 3953)]
                )
                # Sort by predicted rating
                predictions.sort(key=lambda x: x.est, reverse=True)
                # Return top k
                return [{'movie_id': int(pred.iid)} for pred in predictions[:k]]
            except:
                # Fallback to simple recommendations
                return [{'movie_id': i} for i in range(1, k+1)]
    
    # Evaluate model
    print("\nEvaluating model...")
    model = ModelWrapper(model_service)
    results = evaluator.evaluate_model(model)
    subpop_results = evaluator.subpopulation_analysis()
    
    # Generate report
    report = evaluator.generate_report(results, subpop_results)
    print("\n" + report)
    
    # Save results
    with open('offline_evaluation_results.json', 'w') as f:
        json.dump({
            'metrics': results,
            'subpopulations': subpop_results,
            'timestamp': str(evaluator.split_date)
        }, f, indent=2)


async def main():
    """Run offline evaluation only"""
    print("=== Offline Evaluation (No Kafka) ===")
    print()
    
    try:
        await run_offline_evaluation()
        print("\n✓ Offline evaluation completed")
        print("Check offline_evaluation_results.json for metrics")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())