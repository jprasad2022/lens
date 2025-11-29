"""
Fairness Improvements for LENS Recommendation System
Concrete implementations for improving fairness
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random


class FairnessImprovements:
    """Implements concrete fairness improvements for the recommendation system"""
    
    def __init__(self):
        self.exploration_rate = 0.1  # 10% exploration
        self.diversity_weight = 0.3  # 30% weight for diversity
        
    def exposure_fair_ranking(self, recommendations: List[Dict[str, Any]], 
                             item_popularity: Dict[int, float],
                             alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Re-rank recommendations to ensure fair exposure for long-tail items
        
        Args:
            recommendations: Original ranked recommendations
            item_popularity: Dictionary of item_id -> popularity score
            alpha: Balance parameter (0=original ranking, 1=inverse popularity)
        """
        # Calculate exposure scores
        exposure_scores = []
        for i, rec in enumerate(recommendations):
            item_id = rec['id']
            popularity = item_popularity.get(item_id, 1.0)
            
            # Original rank score (higher rank = higher score)
            rank_score = len(recommendations) - i
            
            # Inverse popularity score (less popular = higher score)
            inv_popularity_score = 1.0 / (1.0 + np.log1p(popularity))
            
            # Combined score
            final_score = (1 - alpha) * rank_score + alpha * inv_popularity_score * len(recommendations)
            exposure_scores.append((final_score, rec))
        
        # Re-rank by exposure score
        exposure_scores.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in exposure_scores]
    
    def diversity_reranking(self, recommendations: List[Dict[str, Any]], 
                           k: int = 20,
                           lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) based re-ranking for diversity
        
        Args:
            recommendations: Original recommendations with genres
            k: Number of items to return
            lambda_param: Trade-off between relevance and diversity
        """
        if len(recommendations) <= k:
            return recommendations
            
        # Extract features (genres) for diversity calculation
        def get_features(item):
            return set(item.get('genres', []))
        
        # Initialize with the most relevant item
        reranked = [recommendations[0]]
        candidates = recommendations[1:]
        
        while len(reranked) < k and candidates:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(candidates):
                # Relevance score (based on original ranking)
                relevance = 1.0 - (recommendations.index(candidate) / len(recommendations))
                
                # Diversity score (minimum similarity to selected items)
                candidate_features = get_features(candidate)
                max_sim = 0
                
                for selected in reranked:
                    selected_features = get_features(selected)
                    if candidate_features and selected_features:
                        sim = len(candidate_features & selected_features) / len(candidate_features | selected_features)
                        max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                reranked.append(candidates.pop(best_idx))
        
        return reranked
    
    def demographic_aware_recommendations(self, user_demographics: Dict[str, Any],
                                        recommendations: List[Dict[str, Any]],
                                        demographic_preferences: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Adjust recommendations to counter demographic biases
        
        Args:
            user_demographics: User's demographic info (age, gender, occupation)
            recommendations: Base recommendations
            demographic_preferences: Learned preferences by demographic
        """
        user_gender = user_demographics.get('gender')
        user_age_group = user_demographics.get('age_group')
        
        # Get counter-stereotype boosts
        genre_boosts = {}
        
        # Example: Boost action movies for female users, romance for male users
        if user_gender == 'F':
            genre_boosts['Action'] = 1.2
            genre_boosts['Sci-Fi'] = 1.15
        elif user_gender == 'M':
            genre_boosts['Romance'] = 1.2
            genre_boosts['Drama'] = 1.15
            
        # Age-based diversity boosts
        if user_age_group == '56+':
            genre_boosts['Animation'] = 1.3
            genre_boosts['Thriller'] = 1.2
        elif user_age_group == '18-24':
            genre_boosts['Film-Noir'] = 1.3
            genre_boosts['Documentary'] = 1.2
        
        # Apply boosts
        boosted_recs = []
        for rec in recommendations:
            boost = 1.0
            for genre in rec.get('genres', []):
                boost *= genre_boosts.get(genre, 1.0)
            
            boosted_rec = rec.copy()
            boosted_rec['diversity_boost'] = boost
            boosted_recs.append(boosted_rec)
        
        # Re-sort with boosts
        boosted_recs.sort(key=lambda x: x.get('diversity_boost', 1.0), reverse=True)
        
        return boosted_recs
    
    def exploration_injection(self, recommendations: List[Dict[str, Any]],
                            exploration_pool: List[Dict[str, Any]],
                            exploration_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        Inject exploration items to break filter bubbles
        
        Args:
            recommendations: Current recommendations
            exploration_pool: Pool of diverse items for exploration
            exploration_rate: Percentage of recommendations to replace
        """
        n_explore = int(len(recommendations) * exploration_rate)
        if n_explore == 0 or not exploration_pool:
            return recommendations
        
        # Select positions for exploration (not the top positions)
        positions = list(range(len(recommendations)//3, len(recommendations)))
        explore_positions = random.sample(positions, min(n_explore, len(positions)))
        
        # Replace with exploration items
        result = recommendations.copy()
        used_items = {rec['id'] for rec in recommendations}
        
        for pos in explore_positions:
            # Find an exploration item not already in recommendations
            candidates = [item for item in exploration_pool if item['id'] not in used_items]
            if candidates:
                explore_item = random.choice(candidates)
                result[pos] = explore_item
                used_items.add(explore_item['id'])
        
        return result


class FairnessMonitoring:
    """Monitoring components for fairness metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def calculate_coverage(self, recommendations: List[List[int]], 
                          catalog_size: int) -> float:
        """Calculate catalog coverage metric"""
        unique_items = set()
        for rec_list in recommendations:
            unique_items.update(rec_list)
        return len(unique_items) / catalog_size
    
    def calculate_gini_coefficient(self, item_frequencies: Dict[int, int]) -> float:
        """Calculate Gini coefficient for recommendation distribution"""
        if not item_frequencies:
            return 0.0
            
        frequencies = list(item_frequencies.values())
        frequencies.sort()
        n = len(frequencies)
        cumsum = 0
        
        for i, freq in enumerate(frequencies):
            cumsum += (n - i) * freq
            
        return (n + 1 - 2 * cumsum / sum(frequencies)) / n
    
    def calculate_demographic_parity(self, recommendations_by_group: Dict[str, List[List[int]]]) -> Dict[str, Any]:
        """Calculate demographic parity metrics across groups"""
        group_stats = {}
        
        for group, recs in recommendations_by_group.items():
            # Calculate average diversity
            diversities = []
            for rec_list in recs:
                if len(rec_list) > 1:
                    # Simple diversity: unique items / total items
                    diversity = len(set(rec_list)) / len(rec_list)
                    diversities.append(diversity)
            
            group_stats[group] = {
                'avg_diversity': np.mean(diversities) if diversities else 0,
                'recommendation_count': len(recs)
            }
        
        # Calculate parity metrics
        diversity_values = [stats['avg_diversity'] for stats in group_stats.values()]
        
        return {
            'group_stats': group_stats,
            'diversity_variance': np.var(diversity_values) if diversity_values else 0,
            'min_max_ratio': min(diversity_values) / max(diversity_values) if diversity_values else 1
        }
    
    def monitor_feedback_loop(self, popularity_history: List[Dict[int, float]]) -> Dict[str, Any]:
        """Monitor for feedback loop indicators"""
        if len(popularity_history) < 2:
            return {'trending': False, 'concentration_increasing': False}
        
        # Calculate Gini coefficients over time
        gini_scores = []
        for popularity_dist in popularity_history:
            gini = self.calculate_gini_coefficient(popularity_dist)
            gini_scores.append(gini)
        
        # Check if concentration is increasing
        concentration_increasing = all(gini_scores[i] <= gini_scores[i+1] 
                                     for i in range(len(gini_scores)-1))
        
        # Calculate top-k concentration
        latest_dist = popularity_history[-1]
        sorted_items = sorted(latest_dist.items(), key=lambda x: x[1], reverse=True)
        top_10_percent = int(len(sorted_items) * 0.1)
        top_10_share = sum(item[1] for item in sorted_items[:top_10_percent]) / sum(latest_dist.values())
        
        return {
            'trending': concentration_increasing,
            'concentration_increasing': concentration_increasing,
            'latest_gini': gini_scores[-1] if gini_scores else 0,
            'gini_change': gini_scores[-1] - gini_scores[0] if len(gini_scores) >= 2 else 0,
            'top_10_percent_share': top_10_share
        }


class ColdStartMitigation:
    """Strategies for mitigating cold start problems"""
    
    def __init__(self):
        self.demographic_profiles = {}
        
    def build_demographic_profiles(self, user_data: List[Dict[str, Any]], 
                                  interaction_data: List[Dict[str, Any]]) -> None:
        """Build preference profiles for demographic groups"""
        # Group interactions by demographics
        demo_interactions = defaultdict(list)
        
        for interaction in interaction_data:
            user_id = interaction['user_id']
            user = next((u for u in user_data if u['user_id'] == user_id), None)
            
            if user:
                demo_key = f"{user['gender']}_{user['age_group']}_{user['occupation']}"
                demo_interactions[demo_key].append(interaction)
        
        # Build profiles
        for demo_key, interactions in demo_interactions.items():
            # Calculate genre preferences
            genre_counts = defaultdict(int)
            for interaction in interactions:
                for genre in interaction.get('genres', []):
                    genre_counts[genre] += 1
            
            # Normalize
            total = sum(genre_counts.values())
            if total > 0:
                self.demographic_profiles[demo_key] = {
                    genre: count / total for genre, count in genre_counts.items()
                }
    
    def get_cold_start_recommendations(self, user_demographics: Dict[str, Any],
                                     popular_items: List[Dict[str, Any]],
                                     diverse_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get recommendations for new users based on demographics"""
        # Build demographic key
        demo_key = f"{user_demographics['gender']}_{user_demographics['age_group']}_{user_demographics['occupation']}"
        
        # Get demographic preferences
        demo_preferences = self.demographic_profiles.get(demo_key, {})
        
        # Mix popular, demographic-relevant, and diverse items
        recommendations = []
        
        # 40% demographic-relevant items
        if demo_preferences:
            scored_items = []
            for item in popular_items + diverse_items:
                score = sum(demo_preferences.get(genre, 0) for genre in item.get('genres', []))
                scored_items.append((score, item))
            
            scored_items.sort(key=lambda x: x[0], reverse=True)
            recommendations.extend([item for _, item in scored_items[:8]])
        
        # 40% popular items (but not too popular)
        mid_popular = popular_items[len(popular_items)//4:len(popular_items)//2]
        recommendations.extend(mid_popular[:8])
        
        # 20% diverse exploration
        recommendations.extend(random.sample(diverse_items, min(4, len(diverse_items))))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['id'] not in seen:
                seen.add(rec['id'])
                unique_recs.append(rec)
        
        return unique_recs[:20]


# Example usage and configuration
def get_fairness_configuration() -> Dict[str, Any]:
    """Get recommended fairness configuration"""
    return {
        "exposure_fairness": {
            "enabled": True,
            "alpha": 0.3,  # 30% weight to inverse popularity
            "description": "Boosts long-tail items in rankings"
        },
        "diversity_reranking": {
            "enabled": True,
            "lambda": 0.5,  # Equal weight to relevance and diversity
            "description": "MMR-based diversification of results"
        },
        "demographic_debiasing": {
            "enabled": True,
            "boost_factor": 1.2,
            "description": "Counter-stereotypical content boosting"
        },
        "exploration_injection": {
            "enabled": True,
            "rate": 0.1,  # 10% exploration
            "description": "Random diverse items to break filter bubbles"
        },
        "cold_start_mitigation": {
            "enabled": True,
            "strategy": "demographic_hybrid",
            "description": "Demographic-aware initialization for new users"
        },
        "monitoring": {
            "gini_threshold": 0.8,
            "diversity_threshold": 0.5,
            "demographic_parity_threshold": 0.1,
            "alert_channels": ["slack", "email", "dashboard"]
        }
    }


if __name__ == "__main__":
    # Demonstration of fairness improvements
    print("Fairness Improvements for LENS System")
    print("=" * 50)
    
    # Example recommendations
    example_recs = [
        {"id": 1, "title": "Action Movie 1", "genres": ["Action", "Sci-Fi"], "popularity": 1000},
        {"id": 2, "title": "Romance Movie", "genres": ["Romance", "Drama"], "popularity": 800},
        {"id": 3, "title": "Indie Film", "genres": ["Drama", "Independent"], "popularity": 50},
        {"id": 4, "title": "Blockbuster", "genres": ["Action", "Adventure"], "popularity": 2000},
        {"id": 5, "title": "Art Film", "genres": ["Drama", "Foreign"], "popularity": 30},
    ]
    
    # Initialize improvements
    fairness_improver = FairnessImprovements()
    
    # Test exposure fairness
    item_popularity = {rec["id"]: rec["popularity"] for rec in example_recs}
    fair_recs = fairness_improver.exposure_fair_ranking(example_recs, item_popularity, alpha=0.3)
    
    print("\nExposure Fair Ranking:")
    for i, rec in enumerate(fair_recs):
        print(f"{i+1}. {rec['title']} (popularity: {rec['popularity']})")
    
    # Test diversity reranking
    diverse_recs = fairness_improver.diversity_reranking(example_recs, k=3)
    
    print("\nDiversity Re-ranking (top 3):")
    for i, rec in enumerate(diverse_recs):
        print(f"{i+1}. {rec['title']} - Genres: {', '.join(rec['genres'])}")
    
    # Show configuration
    config = get_fairness_configuration()
    print("\nRecommended Configuration:")
    for component, settings in config.items():
        if isinstance(settings, dict) and 'enabled' in settings:
            print(f"\n{component}:")
            print(f"  Enabled: {settings['enabled']}")
            print(f"  Description: {settings.get('description', 'N/A')}")