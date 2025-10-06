#!/usr/bin/env python3
"""
Probe Script for MovieLens API
Periodically hits /recommend endpoint and logs to Kafka
"""

import os
import sys
import time
import json
import uuid
import random
import argparse
from datetime import datetime
import httpx
from confluent_kafka import Producer

# Configuration from environment
API_URL = os.getenv("API_URL", "http://localhost:8000")
KAFKA_CONFIG = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
}

if KAFKA_CONFIG["security.protocol"] == "SASL_SSL":
    KAFKA_CONFIG.update({
        "sasl.mechanisms": "PLAIN",
        "sasl.username": os.getenv("KAFKA_API_KEY"),
        "sasl.password": os.getenv("KAFKA_API_SECRET"),
    })

# Topics
TEAM_PREFIX = os.getenv("TEAM_PREFIX", "team1")
RECO_REQUESTS_TOPIC = f"{TEAM_PREFIX}.reco_requests"
RECO_RESPONSES_TOPIC = f"{TEAM_PREFIX}.reco_responses"


class MovieLensProber:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self.producer = Producer(KAFKA_CONFIG)
        self.user_ids = list(range(1, 6041))  # MovieLens 1M users
        
    def delivery_report(self, err, msg):
        """Kafka delivery callback"""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def probe_recommend(self, user_id: int, k: int = 20, model: str = None):
        """Probe the recommendation endpoint"""
        request_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        
        # Log request to Kafka
        request_event = {
            "ts": timestamp,
            "user_id": user_id,
            "request_id": request_id,
            "model": model or "default",
            "k": k,
        }
        
        self.producer.produce(
            RECO_REQUESTS_TOPIC,
            key=str(user_id),
            value=json.dumps(request_event),
            callback=self.delivery_report
        )
        
        # Make API request
        start_time = time.time()
        try:
            params = {"k": k}
            if model:
                params["model"] = model
                
            response = self.client.get(f"{API_URL}/recommend/{user_id}", params=params)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log response to Kafka
            movie_ids = []
            if response.status_code == 200:
                data = response.json()
                movie_ids = [m["id"] for m in data.get("recommendations", [])]
            
            response_event = {
                "ts": int(time.time() * 1000),
                "user_id": user_id,
                "request_id": request_id,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "k": len(movie_ids),
                "movie_ids": movie_ids,
                "model_version": data.get("model_info", {}).get("version", "unknown") if response.status_code == 200 else "error",
                "cached": data.get("cached", False) if response.status_code == 200 else False,
            }
            
            self.producer.produce(
                RECO_RESPONSES_TOPIC,
                key=str(user_id),
                value=json.dumps(response_event),
                callback=self.delivery_report
            )
            
            return response.status_code, latency_ms
            
        except Exception as e:
            print(f"Error probing user {user_id}: {e}")
            
            # Log error response
            response_event = {
                "ts": int(time.time() * 1000),
                "user_id": user_id,
                "request_id": request_id,
                "status": 500,
                "latency_ms": (time.time() - start_time) * 1000,
                "k": 0,
                "movie_ids": [],
                "model_version": "error",
                "cached": False,
                "error": str(e),
            }
            
            self.producer.produce(
                RECO_RESPONSES_TOPIC,
                key=str(user_id),
                value=json.dumps(response_event),
                callback=self.delivery_report
            )
            
            return 500, None
    
    def run_probe_cycle(self, num_requests: int = 10, delay: float = 1.0):
        """Run a cycle of probes"""
        print(f"Running {num_requests} probes with {delay}s delay...")
        
        models = ["popularity", "collaborative", "als", None]  # None = default
        stats = {"success": 0, "errors": 0, "total_latency": 0}
        
        for i in range(num_requests):
            # Random user and model
            user_id = random.choice(self.user_ids)
            model = random.choice(models)
            k = random.choice([10, 20, 50])
            
            print(f"\nProbe {i+1}/{num_requests}: user={user_id}, model={model}, k={k}")
            
            status, latency = self.probe_recommend(user_id, k, model)
            
            if status == 200:
                stats["success"] += 1
                if latency:
                    stats["total_latency"] += latency
            else:
                stats["errors"] += 1
            
            # Flush Kafka producer
            self.producer.flush()
            
            # Delay between requests
            if i < num_requests - 1:
                time.sleep(delay)
        
        # Print summary
        print("\n" + "="*50)
        print("Probe Summary:")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {stats['success']} ({stats['success']/num_requests*100:.1f}%)")
        print(f"Errors: {stats['errors']} ({stats['errors']/num_requests*100:.1f}%)")
        if stats['success'] > 0:
            avg_latency = stats['total_latency'] / stats['success']
            print(f"Average latency: {avg_latency:.1f}ms")
        print("="*50)
    
    def close(self):
        """Cleanup"""
        self.client.close()
        self.producer.flush()


def main():
    parser = argparse.ArgumentParser(description="Probe MovieLens API")
    parser.add_argument("--runs", type=int, default=10, help="Number of probe requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Interval for continuous mode (seconds)")
    
    args = parser.parse_args()
    
    prober = MovieLensProber()
    
    try:
        if args.continuous:
            print(f"Running continuous probes every {args.interval} seconds...")
            while True:
                prober.run_probe_cycle(args.runs, args.delay)
                print(f"\nSleeping for {args.interval} seconds...\n")
                time.sleep(args.interval)
        else:
            prober.run_probe_cycle(args.runs, args.delay)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        prober.close()


if __name__ == "__main__":
    main()