"""
Rating Statistics Service
Loads and provides rating statistics from MovieLens data
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import asyncio

from config.settings import get_settings

settings = get_settings()


class RatingStatisticsService:
    """Service for managing rating statistics"""

    def __init__(self):
        self.movie_stats: Dict[int, Dict] = {}
        self.user_ratings: Dict[int, Dict[int, float]] = {}  # user_id -> {movie_id: rating}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Load rating data and calculate statistics"""
        async with self._lock:
            if self._initialized:
                return

            ratings_file = Path(settings.data_path) / "ratings.dat"
            if not ratings_file.exists():
                print(f"Warning: Ratings file not found at {ratings_file}")
                return

            print("Loading rating statistics...")

            # Temporary storage for calculations
            ratings_sum = defaultdict(float)
            ratings_count = defaultdict(int)
            ratings_distribution = defaultdict(lambda: defaultdict(int))

            try:
                with open(ratings_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Parse format: UserID::MovieID::Rating::Timestamp
                        parts = line.split('::')
                        if len(parts) == 4:
                            user_id = int(parts[0])
                            movie_id = int(parts[1])
                            rating = float(parts[2])

                            ratings_sum[movie_id] += rating
                            ratings_count[movie_id] += 1
                            ratings_distribution[movie_id][int(rating)] += 1

                            # Store user ratings
                            if user_id not in self.user_ratings:
                                self.user_ratings[user_id] = {}
                            self.user_ratings[user_id][movie_id] = rating

                # Calculate statistics for each movie
                for movie_id in ratings_count:
                    count = ratings_count[movie_id]
                    avg = ratings_sum[movie_id] / count

                    # Calculate rating distribution percentages
                    distribution = {}
                    for rating in range(1, 6):
                        distribution[f"star_{rating}"] = ratings_distribution[movie_id].get(rating, 0)

                    self.movie_stats[movie_id] = {
                        'vote_average': round(avg, 2),
                        'vote_count': count,
                        'rating_sum': ratings_sum[movie_id],
                        'rating_distribution': distribution,
                        'popularity_score': count  # Simple popularity based on number of ratings
                    }

                self._initialized = True
                print(f"Loaded statistics for {len(self.movie_stats)} movies")

                # Print top 10 most rated movies
                top_movies = sorted(self.movie_stats.items(),
                                  key=lambda x: x[1]['vote_count'],
                                  reverse=True)[:10]
                print("Top 10 most rated movies:")
                for movie_id, stats in top_movies:
                    print(f"  Movie {movie_id}: {stats['vote_count']} ratings, "
                          f"avg: {stats['vote_average']}")

            except Exception as e:
                print(f"Error loading rating statistics: {e}")
                raise

    async def get_movie_stats(self, movie_id: int) -> Optional[Dict]:
        """Get rating statistics for a specific movie"""
        if not self._initialized:
            await self.initialize()

        return self.movie_stats.get(movie_id)

    async def get_movies_stats(self, movie_ids: list) -> Dict[int, Dict]:
        """Get rating statistics for multiple movies"""
        if not self._initialized:
            await self.initialize()

        results = {}
        for movie_id in movie_ids:
            stats = self.movie_stats.get(movie_id)
            if stats:
                results[movie_id] = stats
            else:
                # Return default stats for movies without ratings
                results[movie_id] = {
                    'vote_average': 0.0,
                    'vote_count': 0,
                    'rating_sum': 0.0,
                    'rating_distribution': {f"star_{i}": 0 for i in range(1, 6)},
                    'popularity_score': 0
                }

        return results

    def get_popularity_rank(self, movie_id: int) -> Tuple[int, int]:
        """Get popularity rank of a movie (rank, total_movies)"""
        if movie_id not in self.movie_stats:
            return 0, len(self.movie_stats)

        # Sort by vote count (popularity)
        sorted_movies = sorted(self.movie_stats.items(),
                             key=lambda x: x[1]['vote_count'],
                             reverse=True)

        for rank, (mid, _) in enumerate(sorted_movies, 1):
            if mid == movie_id:
                return rank, len(self.movie_stats)

        return 0, len(self.movie_stats)

    async def get_user_rating(self, user_id: int, movie_id: int) -> Optional[float]:
        """Get a specific user's rating for a movie"""
        if not self._initialized:
            await self.initialize()

        return self.user_ratings.get(user_id, {}).get(movie_id)

    async def get_movie_statistics(self, movie_id: int) -> Optional[Dict]:
        """Get statistics for a specific movie (alias for get_movie_stats)"""
        return await self.get_movie_stats(movie_id)
