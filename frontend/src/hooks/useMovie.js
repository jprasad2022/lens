
import { useState, useCallback } from 'react';
import { movieService } from '@/services/movie.service';

export function useMovie(query) {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const searchMovies = useCallback(async (query) => {
    if (!query) {
      setMovies([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const results = await movieService.searchMovies(query);
      setMovies(results);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  return { movies, loading, error, searchMovies };
}
