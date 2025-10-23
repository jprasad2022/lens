
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
    const endpoint = appConfig.api.endpoints.userRate.replace('{userId}', userId);
    const response = await apiService.client.post(endpoint, {
      movie_id: movieId,
      rating: rating,
    });
    return response.data;
  }

  async trackWatch(userId, movieId, progress) {
    const endpoint = appConfig.api.endpoints.userWatch.replace('{userId}', userId);
    const response = await apiService.client.post(endpoint, {
      movie_id: movieId,
      progress: progress,
    });
    return response.data;
  }
}

export const movieService = new MovieService();
