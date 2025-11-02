"""
User Interaction Simulator for Online Evaluation
Simulates users watching recommended movies based on relevance
"""

import asyncio
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from services.kafka_service import get_kafka_service
from services.recommendation_service import RecommendationService
from services.rating_statistics_service import RatingStatisticsService
from recommender.model_service import ModelService
from config.settings import get_settings

settings = get_settings()


class UserSimulator:
    """Simulates user interactions with recommendations"""

    def __init__(self, kafka_service: Any, rating_service: RatingStatisticsService):
        self.kafka = kafka_service
        self.rating_service = rating_service

        # Probability of watching based on movie rating
        self.watch_probability = {
            5.0: 0.9,  # 90% chance for 5-star movies
            4.5: 0.75,
            4.0: 0.6,
            3.5: 0.45,
            3.0: 0.3,
            2.5: 0.2,
            2.0: 0.1,
            1.0: 0.05  # 5% chance for 1-star movies
        }

    async def simulate_watch_event(self, user_id: int, movie_id: int, recommendation_rank: int, request_id: str):
        """Simulate whether a user watches a recommended movie"""

        # Get user's historical rating for this movie (if exists)
        user_rating = await self.rating_service.get_user_rating(user_id, movie_id)

        # If user already rated it, use their rating to determine watch probability
        if user_rating:
            watch_prob = self.watch_probability.get(
                round(user_rating * 2) / 2,  # Round to nearest 0.5
                0.3
            )
        else:
            # For unrated movies, use movie's average rating
            movie_stats = await self.rating_service.get_movie_statistics(movie_id)
            if movie_stats and 'vote_average' in movie_stats:
                avg_rating = movie_stats['vote_average']
                watch_prob = self.watch_probability.get(
                    round(avg_rating * 2) / 2,
                    0.3
                )
            else:
                # Default probability for new movies
                watch_prob = 0.3

        # Adjust probability based on recommendation rank
        # Higher ranked items more likely to be watched
        rank_multiplier = 1.0 - (recommendation_rank - 1) * 0.05
        rank_multiplier = max(0.5, rank_multiplier)  # At least 50% of original probability
        watch_prob *= rank_multiplier

        # Simulate the watch decision
        watched = random.random() < watch_prob

        if watched:
            # Generate watch event
            watch_time = random.randint(5, 120)  # Random watch time 5-120 minutes after recommendation

            await self.kafka.produce_interaction_event({
                "event_type": "movie_watched",
                "user_id": user_id,
                "movie_id": movie_id,
                "recommendation_request_id": request_id,
                "recommendation_rank": recommendation_rank,
                "watch_timestamp": (datetime.utcnow() + timedelta(minutes=watch_time)).isoformat(),
                "watch_duration_minutes": random.randint(30, 150),  # Random movie duration
                "watch_completion": random.uniform(0.3, 1.0),  # How much they watched
                "source": "recommendation",
                "simulated": True
            })

            return True

        return False

    async def simulate_user_session(self, user_id: int, num_recommendations: int = 20):
        """Simulate a complete user session"""

        # Get recommendations for the user
        model_service = ModelService()
        await model_service.load_models()
        rec_service = RecommendationService(model_service)

        recommendations = await rec_service.get_recommendations(
            user_id=user_id,
            k=num_recommendations
        )

        request_id = recommendations.get('request_id', str(datetime.utcnow().timestamp()))

        # Track which movies were watched
        watched_movies = []

        # Simulate interaction with each recommendation
        for rank, movie in enumerate(recommendations['recommendations'], 1):
            watched = await self.simulate_watch_event(
                user_id=user_id,
                movie_id=movie['id'],
                recommendation_rank=rank,
                request_id=request_id
            )

            if watched:
                watched_movies.append({
                    'movie_id': movie['id'],
                    'rank': rank,
                    'title': movie.get('title', f'Movie {movie["id"]}')
                })

            # Simulate time between viewing recommendations
            await asyncio.sleep(random.uniform(0.1, 0.5))

        # Log session summary
        session_summary = {
            "event_type": "session_summary",
            "user_id": user_id,
            "request_id": request_id,
            "recommendations_shown": num_recommendations,
            "movies_watched": len(watched_movies),
            "watch_rate": len(watched_movies) / num_recommendations,
            "watched_movies": watched_movies,
            "timestamp": datetime.utcnow().isoformat(),
            "simulated": True
        }

        await self.kafka.produce_interaction_event(session_summary)

        return session_summary


async def run_simulation(num_users: int = 100, num_sessions_per_user: int = 5):
    """Run simulation for multiple users"""

    kafka_service = get_kafka_service()
    await kafka_service.initialize()

    rating_service = RatingStatisticsService()
    await rating_service.initialize()

    simulator = UserSimulator(kafka_service, rating_service)

    # Sample random users from MovieLens dataset
    all_user_ids = list(range(1, 6041))
    selected_users = random.sample(all_user_ids, min(num_users, len(all_user_ids)))

    print(f"Starting simulation for {len(selected_users)} users...")

    for user_id in selected_users:
        print(f"Simulating user {user_id}...")

        for session in range(num_sessions_per_user):
            try:
                summary = await simulator.simulate_user_session(user_id)
                print(f"  Session {session + 1}: {summary['movies_watched']}/{summary['recommendations_shown']} movies watched")

                # Wait between sessions
                await asyncio.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"  Error in session {session + 1}: {e}")

    await kafka_service.close()
    print("Simulation complete!")


if __name__ == "__main__":
    # Run simulation
    asyncio.run(run_simulation(
        num_users=50,  # Simulate 50 users
        num_sessions_per_user=3  # Each user has 3 sessions
    ))
