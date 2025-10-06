/**
 * API Service
 * Handles all API interactions with retry logic and error handling
 */

import axios from 'axios';
import { appConfig } from '@/config/app.config';
import { authService } from './auth.service';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: appConfig.api.baseUrl,
      timeout: appConfig.api.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for authentication
    this.client.interceptors.request.use(
      async (config) => {
        if (appConfig.features.authentication) {
          const token = await authService.getIdToken();
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Retry logic for 5xx errors
        if (error.response?.status >= 500 && originalRequest._retry < appConfig.api.retryAttempts) {
          originalRequest._retry = (originalRequest._retry || 0) + 1;
          await this.delay(1000 * originalRequest._retry); // Exponential backoff
          return this.client(originalRequest);
        }

        // Handle 401 errors
        if (error.response?.status === 401 && appConfig.features.authentication) {
          await authService.refreshToken();
          const token = await authService.getIdToken();
          if (token) {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return this.client(originalRequest);
          }
        }

        return Promise.reject(this.handleError(error));
      }
    );
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  handleError(error) {
    if (error.response) {
      // Server responded with error
      return {
        status: error.response.status,
        message: error.response.data?.message || error.message,
        data: error.response.data,
      };
    } else if (error.request) {
      // Request made but no response
      return {
        status: 0,
        message: 'Network error - please check your connection',
      };
    } else {
      // Something else happened
      return {
        status: -1,
        message: error.message || 'An unexpected error occurred',
      };
    }
  }

  // Recommendation endpoints
  async getRecommendations(userId, options = {}) {
    const params = {
      k: options.k || appConfig.models.recommendationCount,
      model: options.model || appConfig.models.default,
    };

    const response = await this.client.get(
      `${appConfig.api.endpoints.recommend}/${userId}`,
      { params }
    );

    return response.data;
  }

  async submitFeedback(feedback) {
    const response = await this.client.post(appConfig.api.endpoints.feedback, feedback);
    return response.data;
  }

  // Model management
  async getAvailableModels() {
    const response = await this.client.get(appConfig.api.endpoints.models);
    return response.data;
  }

  async switchModel(modelName) {
    const response = await this.client.post(appConfig.api.endpoints.switchModel, {
      model: modelName,
    });
    return response.data;
  }

  // Health and monitoring
  async checkHealth() {
    const response = await this.client.get(appConfig.api.endpoints.health);
    return response.data;
  }

  async getMetrics() {
    const response = await this.client.get(appConfig.api.endpoints.metrics, {
      responseType: 'text',
    });
    return response.data;
  }

  // User endpoints
  async getUserHistory(userId) {
    const endpoint = appConfig.api.endpoints.userHistory.replace('{userId}', userId);
    const response = await this.client.get(endpoint);
    return response.data;
  }

  async updateUserPreferences(userId, preferences) {
    const endpoint = appConfig.api.endpoints.userPreferences.replace('{userId}', userId);
    const response = await this.client.put(endpoint, preferences);
    return response.data;
  }

  // Admin endpoints
  async triggerRetrain(options = {}) {
    const response = await this.client.post(appConfig.api.endpoints.retrain, options);
    return response.data;
  }

  async getABTestResults() {
    const response = await this.client.get(appConfig.api.endpoints.abTest);
    return response.data;
  }

  async getMonitoringData() {
    const response = await this.client.get(appConfig.api.endpoints.monitoring);
    return response.data;
  }
}

export const apiService = new ApiService();