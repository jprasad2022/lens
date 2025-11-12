"""
A/B Switch Service - Traffic Routing and Experiment Management
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class ABExperiment:
    """Represents an A/B test experiment."""
    
    def __init__(self, experiment_id: str, name: str, models: Dict[str, float], 
                 start_date: datetime, end_date: Optional[datetime] = None):
        self.experiment_id = experiment_id
        self.name = name
        self.models = models  # {"model_name": allocation_percentage}
        self.start_date = start_date
        self.end_date = end_date or (start_date + timedelta(days=14))
        self.metrics = defaultdict(lambda: defaultdict(list))
        
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        now = datetime.utcnow()
        return self.start_date <= now <= self.end_date
    
    def get_model_for_user(self, user_id: int) -> str:
        """Deterministically assign user to model variant."""
        # Create stable hash for user
        user_hash = int(hashlib.md5(f"{self.experiment_id}:{user_id}".encode()).hexdigest(), 16)
        bucket = (user_hash % 100) / 100.0
        
        # Assign to model based on allocation
        cumulative = 0.0
        for model, allocation in self.models.items():
            cumulative += allocation
            if bucket < cumulative:
                return model
        
        # Fallback to first model
        return list(self.models.keys())[0]
    
    def record_metric(self, model: str, metric_name: str, value: float):
        """Record metric for analysis."""
        self.metrics[model][metric_name].append(value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate experiment statistics."""
        stats_result = {}
        
        for metric_name in set(sum([list(m.keys()) for m in self.metrics.values()], [])):
            metric_stats = {}
            
            # Get data for each model
            model_data = {}
            for model in self.models:
                if metric_name in self.metrics[model]:
                    model_data[model] = self.metrics[model][metric_name]
            
            if len(model_data) < 2:
                continue
            
            # Calculate basic statistics
            for model, values in model_data.items():
                if values:
                    metric_stats[model] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                        "confidence_interval": stats.t.interval(
                            0.95, len(values)-1, 
                            loc=np.mean(values), 
                            scale=stats.sem(values)
                        ) if len(values) > 1 else (np.mean(values), np.mean(values))
                    }
            
            # Perform statistical tests if we have two models
            if len(model_data) == 2:
                models = list(model_data.keys())
                if len(model_data[models[0]]) > 1 and len(model_data[models[1]]) > 1:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(
                        model_data[models[0]], 
                        model_data[models[1]]
                    )
                    
                    metric_stats["statistical_test"] = {
                        "type": "t-test",
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            
            stats_result[metric_name] = metric_stats
        
        return stats_result

class ABSwitchService:
    """Manages A/B testing and traffic routing."""
    
    def __init__(self):
        self.experiments: Dict[str, ABExperiment] = {}
        self.default_model = "popularity"
        self.model_performance = defaultdict(lambda: {
            "requests": 0,
            "errors": 0,
            "total_latency": 0,
            "cache_hits": 0
        })
    
    def create_experiment(self, name: str, models: Dict[str, float], 
                         duration_days: int = 14) -> str:
        """Create a new A/B test experiment."""
        # Validate allocations sum to 1.0
        total_allocation = sum(models.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Model allocations must sum to 1.0, got {total_allocation}")
        
        # Generate experiment ID
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{name.lower().replace(' ', '_')}"
        
        # Create experiment
        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            models=models,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=duration_days)
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment {experiment_id}: {models}")
        
        return experiment_id
    
    def get_model_for_user(self, user_id: int) -> str:
        """Get model assignment for user based on active experiments."""
        # Check active experiments
        active_experiments = [
            exp for exp in self.experiments.values() 
            if exp.is_active()
        ]
        
        if not active_experiments:
            return self.default_model
        
        # Use most recent experiment
        experiment = sorted(active_experiments, key=lambda x: x.start_date)[-1]
        return experiment.get_model_for_user(user_id)
    
    def record_request(self, user_id: int, model: str, latency_ms: float, 
                      cached: bool = False, error: bool = False):
        """Record request metrics for monitoring."""
        # Update model performance
        self.model_performance[model]["requests"] += 1
        self.model_performance[model]["total_latency"] += latency_ms
        if cached:
            self.model_performance[model]["cache_hits"] += 1
        if error:
            self.model_performance[model]["errors"] += 1
        
        # Record in active experiments
        for experiment in self.experiments.values():
            if experiment.is_active() and model in experiment.models:
                experiment.record_metric(model, "latency_ms", latency_ms)
                experiment.record_metric(model, "success_rate", 0.0 if error else 1.0)
                if cached:
                    experiment.record_metric(model, "cache_hit_rate", 1.0)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for a specific experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        statistics = experiment.get_statistics()
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": "active" if experiment.is_active() else "completed",
            "start_date": experiment.start_date.isoformat(),
            "end_date": experiment.end_date.isoformat(),
            "models": experiment.models,
            "statistics": statistics
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        performance = {}
        
        for model, metrics in self.model_performance.items():
            if metrics["requests"] > 0:
                performance[model] = {
                    "requests": metrics["requests"],
                    "error_rate": metrics["errors"] / metrics["requests"],
                    "avg_latency_ms": metrics["total_latency"] / metrics["requests"],
                    "cache_hit_rate": metrics["cache_hits"] / metrics["requests"]
                }
        
        return performance
    
    def recommend_winner(self, experiment_id: str, metric: str = "latency_ms") -> Optional[str]:
        """Recommend winning model based on experiment results."""
        results = self.get_experiment_results(experiment_id)
        
        if metric not in results["statistics"]:
            return None
        
        metric_stats = results["statistics"][metric]
        
        # Find model with best performance (lower is better for latency)
        best_model = None
        best_value = float('inf') if metric == "latency_ms" else float('-inf')
        
        for model, stats in metric_stats.items():
            if model == "statistical_test":
                continue
                
            value = stats["mean"]
            if metric == "latency_ms" and value < best_value:
                best_model = model
                best_value = value
            elif metric != "latency_ms" and value > best_value:
                best_model = model
                best_value = value
        
        # Check if result is statistically significant
        if "statistical_test" in metric_stats:
            if not metric_stats["statistical_test"]["significant"]:
                logger.warning(f"No statistically significant difference found for {metric}")
        
        return best_model
    
    def set_default_model(self, model: str):
        """Set the default model for users not in experiments."""
        self.default_model = model
        logger.info(f"Default model set to: {model}")
    
    def cleanup_old_experiments(self, days: int = 30):
        """Remove experiments older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        to_remove = []
        for exp_id, experiment in self.experiments.items():
            if experiment.end_date < cutoff_date:
                to_remove.append(exp_id)
        
        for exp_id in to_remove:
            del self.experiments[exp_id]
            logger.info(f"Removed old experiment: {exp_id}")
        
        return len(to_remove)

# Global instance
_ab_switch_service = None

def get_ab_switch_service() -> ABSwitchService:
    """Get singleton AB switch service instance."""
    global _ab_switch_service
    if _ab_switch_service is None:
        _ab_switch_service = ABSwitchService()
    return _ab_switch_service