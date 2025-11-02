"""
Movie Metadata Service
Loads and provides movie information from MovieLens data
"""

import re
from typing import Dict, Optional, List
from pathlib import Path
import asyncio

from config.settings import get_settings

settings = get_settings()


class MovieMetadataService:
    """Service for managing movie metadata"""
    
    def __init__(self):
        self.movies: Dict[int, Dict] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Load movie data from movies.dat file"""
        async with self._lock:
            if self._initialized:
                return
            
            movies_file = Path(settings.data_path) / "movies.dat"
            if not movies_file.exists():
                print(f"Warning: Movies file not found at {movies_file}")
                return
            
            print("Loading movie metadata...")
            try:
                with open(movies_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse format: MovieID::Title (Year)::Genre1|Genre2|...
                        parts = line.split('::')
                        if len(parts) == 3:
                            movie_id = int(parts[0])
                            
                            # Extract title and year
                            title_with_year = parts[1]
                            year_match = re.search(r'\((\d{4})\)$', title_with_year)
                            
                            if year_match:
                                year = int(year_match.group(1))
                                title = title_with_year[:year_match.start()].strip()
                            else:
                                year = None
                                title = title_with_year
                            
                            # Parse genres
                            genres = parts[2].split('|') if parts[2] else []
                            
                            self.movies[movie_id] = {
                                'id': movie_id,
                                'title': title,
                                'year': year,
                                'genres': genres,
                                'full_title': title_with_year,
                                'poster_url': None  # Placeholder for poster URL
                            }
                
                self._initialized = True
                print(f"Loaded {len(self.movies)} movies")
                
            except Exception as e:
                print(f"Error loading movie metadata: {e}")
                raise
    
    async def get_movie(self, movie_id: int) -> Optional[Dict]:
        """Get movie information by ID"""
        if not self._initialized:
            await self.initialize()
        
        return self.movies.get(movie_id)
    
    async def get_movies(self, movie_ids: List[int]) -> List[Dict]:
        """Get multiple movies by IDs"""
        if not self._initialized:
            await self.initialize()
        
        movies = []
        for movie_id in movie_ids:
            movie = self.movies.get(movie_id)
            if movie:
                movies.append(movie)
            else:
                # Return placeholder for missing movies
                movies.append({
                    'id': movie_id,
                    'title': f'Movie {movie_id}',
                    'year': None,
                    'genres': [],
                    'full_title': f'Movie {movie_id}',
                    'poster_url': None  # Placeholder for poster URL
                })
        
        return movies
    
    async def search_movies(self, query: str, limit: int = 20) -> List[Dict]:
        """Search movies by title"""
        if not self._initialized:
            await self.initialize()
        
        query_lower = query.lower()
        results = []
        
        for movie in self.movies.values():
            if query_lower in movie['title'].lower():
                results.append(movie)
                if len(results) >= limit:
                    break
        
        return results