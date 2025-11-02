'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '@/services/api.service';
import { useApp } from '@/contexts/AppContext';
import RecommendationCard from './RecommendationCard';
import LoadingSpinner from '../ui/LoadingSpinner';
import ErrorMessage from '../ui/ErrorMessage';
import MovieDetailsModal from '../movie/MovieDetailsModal';
import { FiClock, FiCpu, FiRefreshCw, FiFilm } from 'react-icons/fi';

export default function RecommendationGrid({ userId, count = 20 }) {
  const { selectedModel } = useApp();
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['recommendations', userId, selectedModel, count],
    queryFn: () => apiService.getRecommendations(userId, { 
      k: count, 
      model: selectedModel 
    }),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  });

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        {[...Array(count)].map((_, i) => (
          <div key={i} className="card-movie">
            <div className="aspect-[2/3] skeleton rounded-t-xl"></div>
            <div className="p-5 space-y-3">
              <div className="h-6 skeleton rounded w-3/4"></div>
              <div className="h-4 skeleton rounded w-1/2"></div>
              <div className="flex gap-2">
                <div className="h-6 skeleton rounded-full w-16"></div>
                <div className="h-6 skeleton rounded-full w-16"></div>
              </div>
              <div className="h-16 skeleton rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <ErrorMessage
        title="Failed to load recommendations"
        message={error.message}
        onRetry={refetch}
      />
    );
  }

  if (!data || !data.recommendations || data.recommendations.length === 0) {
    return (
      <div className="text-center py-16 card max-w-md mx-auto">
        <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
          <FiFilm className="w-8 h-8 text-gray-400" />
        </div>
        <p className="text-gray-600 dark:text-gray-400 mb-4">No recommendations available.</p>
        <button
          onClick={refetch}
          className="btn-primary inline-flex items-center gap-2"
        >
          <FiRefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div>
      {data.model_info && (
        <div className="mb-8 glass rounded-xl p-6 border border-primary-200 dark:border-primary-800">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                <FiCpu className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">
                  {data.model_info.name} Model
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Version {data.model_info.version} • Updated {new Date(data.model_info.updated_at).toLocaleDateString()}
                </p>
              </div>
            </div>
            {data.latency_ms && (
              <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                <FiClock className="w-4 h-4" />
                <span>{data.latency_ms}ms response time</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        {data.recommendations.map((movie, index) => (
          <div
            key={movie.id}
            className="animate-slide-up"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <RecommendationCard
              movie={movie}
              rank={index + 1}
              model={selectedModel}
              onViewDetails={() => {
                setSelectedMovie(movie);
                setShowModal(true);
              }}
            />
          </div>
        ))}
      </div>

      {data.is_personalized === false && (
        <div className="mt-8 p-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
                General Recommendations
              </h3>
              <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-400">
                These are popular movies. Sign in to get personalized suggestions based on your preferences.
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Single Modal Instance */}
      <MovieDetailsModal 
        movie={selectedMovie}
        isOpen={showModal}
        onClose={() => {
          setShowModal(false);
          setSelectedMovie(null);
        }}
      />
    </div>
  );
}