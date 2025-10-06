
import { apiService } from './api.service';
import { appConfig } from '@/config/app.config';

class MovieService {
  async searchMovies(query) {
    const response = await apiService.client.get(appConfig.api.endpoints.search, {
      params: { q: query },
    });
    return response.data;
  }

  async rateMovie(userId, movieId, rating) {
    const response = await apiService.client.post(appConfig.api.endpoints.ratings, {
      user_id: userId,
      movie_id: movieId,
      rating: rating,
    });
    return response.data;
  }
}

export const movieService = new MovieService();
