#!/usr/bin/env python3
"""
Run Online Evaluation Pipeline
1. Simulate user interactions
2. Compute online metrics
3. Generate report
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.user_simulator import run_simulation
from evaluation.online_evaluator import run_online_evaluation


async def main():
    """Run complete online evaluation pipeline"""
    
    print("=== Online Evaluation Pipeline ===")
    print()
    
    # Step 1: Run user simulation
    print("Step 1: Simulating user interactions...")
    print("-" * 40)
    
    try:
        await run_simulation(
            num_users=20,  # Simulate 20 users for demo
            num_sessions_per_user=2  # Each user has 2 sessions
        )
        print("\n✓ User simulation completed")
    except Exception as e:
        print(f"\n✗ User simulation failed: {e}")
        return
    
    # Wait a bit for Kafka to process messages
    print("\nWaiting for Kafka to process events...")
    await asyncio.sleep(5)
    
    # Step 2: Run online evaluation
    print("\nStep 2: Computing online evaluation metrics...")
    print("-" * 40)
    
    try:
        await run_online_evaluation()
        print("\n✓ Online evaluation completed")
    except Exception as e:
        print(f"\n✗ Online evaluation failed: {e}")
        return
    
    print("\n=== Evaluation Pipeline Complete ===")
    print("Check online_evaluation_results.json for detailed metrics")


if __name__ == "__main__":
    asyncio.run(main())