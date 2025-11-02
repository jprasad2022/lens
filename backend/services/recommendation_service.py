"""
Recommendation Service
Coordinates requests between API and ModelService, and shapes responses.
"""

from typing import Any, Dict, List, Optional

from services.model_service import ModelService


class RecommendationService:
    """High-level recommendation orchestration service."""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service

    async def get_recommendations(
        self,
        user_id: int,
        k: int = 20,
        model_name: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get recommendations and return enriched payload expected by routers."""
        if not model_name:
            from app.state import app_state

            model_name = await app_state.get_current_model(user_id)

        # Fetch recommendation ids from model service
        movie_ids: List[int] = await self.model_service.get_recommendations(
            model_name=model_name,
            user_id=user_id,
            k=k,
            features=features,
        )

        # Get movie metadata
        from app.state import app_state
        if app_state.movie_metadata_service and app_state.rating_statistics_service:
            movies = await app_state.movie_metadata_service.get_movies(movie_ids)
            movie_stats = await app_state.rating_statistics_service.get_movies_stats(movie_ids)

            recommendations = []
            for movie in movies:
                movie_id = movie['id']
                stats = movie_stats.get(movie_id, {})

                # Format release date from year
                release_date = f"{movie.get('year', '')}-01-01" if movie.get('year') else None

                # Create a simple overview from genres
                if movie['genres']:
                    genre_text = ' and '.join(movie['genres'][:2]).lower()
                    overview = f"A {genre_text} film"
                    if movie.get('year'):
                        overview += f" from {movie['year']}"
                    if stats.get('vote_count', 0) > 100:
                        overview += f", rated {stats.get('vote_average', 0)}/5 by {stats.get('vote_count', 0)} users"
                    overview += "."
                else:
                    overview = "No description available."

                recommendations.append({
                    "id": movie_id,
                    "title": movie['title'],
                    "genres": movie['genres'],
                    "release_date": release_date,
                    "vote_average": stats.get('vote_average', 0.0) * 2,  # Convert 5-star to 10-scale
                    "vote_count": stats.get('vote_count', 0),
                    "popularity": stats.get('popularity_score', 0),
                    "overview": overview,
                    "year": movie.get('year'),
                    "full_title": movie.get('full_title', movie['title'])
                })
        else:
            # Fallback if metadata service is not available
            recommendations = [
                {
                    "id": movie_id,
                    "title": f"Movie {movie_id}",
                    "genres": [],
                }
                for movie_id in movie_ids
            ]

        # Collect model info if available
        from app.state import app_state
        from datetime import datetime

        # Ensure all required ModelInfo fields are present with correct types
        default_model_info = {
            "name": model_name,
            "version": "latest",
            "type": model_name,
            "trained_at": datetime.utcnow(),
            "metrics": {},
            "parameters": {},
            "active": True,
        }

        raw_info = app_state.active_models.get(model_name, {})
        model_info = {
            "name": raw_info.get("name", default_model_info["name"]),
            "version": raw_info.get("version", default_model_info["version"]),
            "type": raw_info.get("type", default_model_info["type"]),
            "trained_at": raw_info.get("trained_at", default_model_info["trained_at"]),
            "metrics": raw_info.get("metrics", default_model_info["metrics"]),
            "parameters": raw_info.get("parameters", default_model_info["parameters"]),
            "active": raw_info.get("active", default_model_info["active"]),
        }

        return {
            "recommendations": recommendations,
            "model_info": model_info,
            "cached": False,
            "is_personalized": model_name != "popularity",
        }

    async def store_feedback(self, feedback: Dict[str, Any]) -> None:
        """Persist feedback (no-op stub)."""
        # This can be extended to write to Kafka or a datastore
        return None

    async def explain_recommendations(
        self, user_id: int, model_name: str, k: int = 5
    ) -> Dict[str, Any]:
        """Return a simple explanation stub."""
        return {
            "user_id": user_id,
            "model": model_name,
            "top_factors": [
                {"factor": "user_history_similarity", "weight": 0.6},
                {"factor": "item_popularity", "weight": 0.3},
                {"factor": "recent_trends", "weight": 0.1},
            ],
            "k": k,
        }

    async def get_popular_movies(
        self, k: int = 20, genre: Optional[str] = None, year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return popular movies with optional filtering."""
        # Get more movies if filtering is needed
        fetch_k = k * 3 if (genre or year) else k

        # Get popular movie IDs
        movie_ids = await self.model_service.get_recommendations(
            model_name="popularity", user_id=0, k=fetch_k
        )

        # Get movie metadata
        from app.state import app_state
        if app_state.movie_metadata_service and app_state.rating_statistics_service:
            all_movies = await app_state.movie_metadata_service.get_movies(movie_ids)
            movie_stats = await app_state.rating_statistics_service.get_movies_stats(movie_ids)

            # Filter by genre and/or year if specified
            if genre or year:
                filtered_movies = []
                for movie in all_movies:
                    if genre and genre not in movie.get('genres', []):
                        continue
                    if year and movie.get('year') != year:
                        continue
                    movie_id = movie['id']
                    stats = movie_stats.get(movie_id, {})
                    release_date = f"{movie.get('year', '')}-01-01" if movie.get('year') else None

                    filtered_movies.append({
                        "id": movie_id,
                        "title": movie['title'],
                        "genres": movie['genres'],
                        "year": movie.get('year'),
                        "release_date": release_date,
                        "vote_average": stats.get('vote_average', 0.0) * 2,
                        "vote_count": stats.get('vote_count', 0),
                        "popularity": stats.get('popularity_score', 0),
                        "full_title": movie.get('full_title', movie['title'])
                    })
                    if len(filtered_movies) >= k:
                        break
                return filtered_movies
            else:
                # No filtering, return first k movies with metadata
                result_movies = []
                for movie in all_movies[:k]:
                    movie_id = movie['id']
                    stats = movie_stats.get(movie_id, {})
                    release_date = f"{movie.get('year', '')}-01-01" if movie.get('year') else None

                    result_movies.append({
                        "id": movie_id,
                        "title": movie['title'],
                        "genres": movie['genres'],
                        "year": movie.get('year'),
                        "release_date": release_date,
                        "vote_average": stats.get('vote_average', 0.0) * 2,
                        "vote_count": stats.get('vote_count', 0),
                        "popularity": stats.get('popularity_score', 0),
                        "full_title": movie.get('full_title', movie['title'])
                    })
                return result_movies
        else:
            # Fallback without metadata
            return [
                {"id": movie_id, "title": f"Movie {movie_id}", "genres": []}
                for movie_id in movie_ids[:k]
            ]




