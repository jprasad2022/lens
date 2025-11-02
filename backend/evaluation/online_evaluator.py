"""
Online Evaluation Metrics Calculator
Consumes Kafka events to compute online success metrics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set
from collections import defaultdict
import pandas as pd
import numpy as np

from aiokafka import AIOKafkaConsumer
from config.settings import get_settings

settings = get_settings()


class OnlineEvaluator:
    """Computes online evaluation metrics from Kafka events"""
    
    def __init__(self):
        self.consumer = None
        self.recommendations = {}  # request_id -> recommendation details
        self.interactions = defaultdict(list)  # request_id -> list of interactions
        
    async def start(self):
        """Start Kafka consumer"""
        self.consumer = AIOKafkaConsumer(
            'reco_responses',
            'user_interactions',
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id='online_evaluator',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest'
        )
        await self.consumer.start()
    
    async def stop(self):
        """Stop Kafka consumer"""
        if self.consumer:
            await self.consumer.stop()
    
    async def consume_events(self, duration_minutes: int = 60):
        """Consume events for specified duration"""
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        async for message in self.consumer:
            # Store recommendation responses
            if message.topic == 'reco_responses':
                data = message.value
                request_id = data.get('request_id')
                if request_id:
                    self.recommendations[request_id] = {
                        'user_id': data.get('user_id'),
                        'movie_ids': data.get('movie_ids', []),
                        'model_version': data.get('model_version'),
                        'timestamp': data.get('timestamp'),
                        'latency_ms': data.get('latency_ms')
                    }
            
            # Store user interactions
            elif message.topic == 'user_interactions':
                data = message.value
                request_id = data.get('recommendation_request_id')
                if request_id and data.get('event_type') == 'movie_watched':
                    self.interactions[request_id].append({
                        'movie_id': data.get('movie_id'),
                        'rank': data.get('recommendation_rank'),
                        'watch_timestamp': data.get('watch_timestamp'),
                        'watch_duration': data.get('watch_duration_minutes'),
                        'watch_completion': data.get('watch_completion', 1.0)
                    })
            
            # Check if we've reached the time limit
            if datetime.utcnow() >= end_time:
                break
    
    def calculate_metrics(self) -> Dict:
        """Calculate online evaluation metrics"""
        
        metrics = {
            'total_recommendations': len(self.recommendations),
            'total_interactions': sum(len(v) for v in self.interactions.values()),
            'proxy_success_metrics': {},
            'engagement_metrics': {},
            'latency_metrics': {}
        }
        
        if not self.recommendations:
            return metrics
        
        # Calculate proxy success metrics
        proxy_metrics = self._calculate_proxy_success()
        metrics['proxy_success_metrics'] = proxy_metrics
        
        # Calculate engagement metrics
        engagement_metrics = self._calculate_engagement_metrics()
        metrics['engagement_metrics'] = engagement_metrics
        
        # Calculate latency metrics
        latency_metrics = self._calculate_latency_metrics()
        metrics['latency_metrics'] = latency_metrics
        
        return metrics
    
    def _calculate_proxy_success(self) -> Dict:
        """Calculate proxy success metrics"""
        
        # Define success: user watched at least one recommended movie within 2 hours
        success_window_minutes = 120
        
        successful_sessions = 0
        total_sessions = len(self.recommendations)
        
        click_through_rates = []
        precision_at_k = defaultdict(list)
        
        for request_id, rec_data in self.recommendations.items():
            interactions = self.interactions.get(request_id, [])
            recommended_movies = set(rec_data['movie_ids'])
            
            # Check if any recommended movie was watched
            watched_movies = {i['movie_id'] for i in interactions}
            watched_recommended = watched_movies & recommended_movies
            
            if watched_recommended:
                successful_sessions += 1
            
            # Calculate CTR for this session
            if recommended_movies:
                ctr = len(watched_recommended) / len(recommended_movies)
                click_through_rates.append(ctr)
            
            # Calculate precision@k
            for k in [1, 5, 10]:
                if len(recommended_movies) >= k:
                    top_k_movies = set(rec_data['movie_ids'][:k])
                    watched_in_top_k = len(watched_movies & top_k_movies)
                    precision_at_k[k].append(watched_in_top_k / k)
        
        return {
            'success_rate': successful_sessions / total_sessions if total_sessions > 0 else 0,
            'avg_ctr': np.mean(click_through_rates) if click_through_rates else 0,
            'precision_at_1': np.mean(precision_at_k[1]) if precision_at_k[1] else 0,
            'precision_at_5': np.mean(precision_at_k[5]) if precision_at_k[5] else 0,
            'precision_at_10': np.mean(precision_at_k[10]) if precision_at_k[10] else 0,
        }
    
    def _calculate_engagement_metrics(self) -> Dict:
        """Calculate user engagement metrics"""
        
        watch_durations = []
        watch_completions = []
        ranks_of_watched = []
        
        for interactions in self.interactions.values():
            for interaction in interactions:
                if interaction.get('watch_duration'):
                    watch_durations.append(interaction['watch_duration'])
                if interaction.get('watch_completion'):
                    watch_completions.append(interaction['watch_completion'])
                if interaction.get('rank'):
                    ranks_of_watched.append(interaction['rank'])
        
        return {
            'avg_watch_duration_minutes': np.mean(watch_durations) if watch_durations else 0,
            'avg_watch_completion': np.mean(watch_completions) if watch_completions else 0,
            'median_rank_of_watched': np.median(ranks_of_watched) if ranks_of_watched else 0,
            'total_movies_watched': len(watch_durations)
        }
    
    def _calculate_latency_metrics(self) -> Dict:
        """Calculate system latency metrics"""
        
        latencies = [r['latency_ms'] for r in self.recommendations.values() 
                    if r.get('latency_ms') is not None]
        
        if not latencies:
            return {}
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': max(latencies)
        }
    
    def generate_report(self, metrics: Dict) -> str:
        """Generate evaluation report"""
        
        report = f"""
Online Evaluation Report
========================

Summary
-------
Total Recommendation Sessions: {metrics['total_recommendations']}
Total User Interactions: {metrics['total_interactions']}

Proxy Success Metrics
--------------------
Success Rate (watched any recommended): {metrics['proxy_success_metrics']['success_rate']:.2%}
Average Click-Through Rate: {metrics['proxy_success_metrics']['avg_ctr']:.2%}
Precision@1: {metrics['proxy_success_metrics']['precision_at_1']:.3f}
Precision@5: {metrics['proxy_success_metrics']['precision_at_5']:.3f}
Precision@10: {metrics['proxy_success_metrics']['precision_at_10']:.3f}

Engagement Metrics
-----------------
Average Watch Duration: {metrics['engagement_metrics']['avg_watch_duration_minutes']:.1f} minutes
Average Watch Completion: {metrics['engagement_metrics']['avg_watch_completion']:.2%}
Median Rank of Watched Movies: {metrics['engagement_metrics']['median_rank_of_watched']:.1f}
Total Movies Watched: {metrics['engagement_metrics']['total_movies_watched']}

System Performance
-----------------
Mean Latency: {metrics['latency_metrics'].get('mean_latency_ms', 0):.1f} ms
P95 Latency: {metrics['latency_metrics'].get('p95_latency_ms', 0):.1f} ms
P99 Latency: {metrics['latency_metrics'].get('p99_latency_ms', 0):.1f} ms
"""
        return report


async def run_online_evaluation():
    """Run online evaluation"""
    
    evaluator = OnlineEvaluator()
    await evaluator.start()
    
    print("Starting online evaluation...")
    print("Consuming events for 60 minutes...")
    
    # Consume events
    await evaluator.consume_events(duration_minutes=60)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Generate report
    report = evaluator.generate_report(metrics)
    print(report)
    
    # Save metrics to file
    with open('online_evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    await evaluator.stop()
    print("Evaluation complete! Results saved to online_evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(run_online_evaluation())