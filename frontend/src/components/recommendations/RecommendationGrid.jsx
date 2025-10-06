'use client';

import { useQuery } from '@tanstack/react-query';
import { apiService } from '@/services/api.service';
import { useApp } from '@/contexts/AppContext';
import RecommendationCard from './RecommendationCard';
import LoadingSpinner from '../ui/LoadingSpinner';
import ErrorMessage from '../ui/ErrorMessage';

export default function RecommendationGrid({ userId, count = 20 }) {
  const { selectedModel } = useApp();

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
      <div className="flex justify-center items-center min-h-[400px]">
        <LoadingSpinner size="large" />
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
      <div className="text-center py-12">
        <p className="text-gray-600">No recommendations available.</p>
        <button
          onClick={refetch}
          className="mt-4 px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div>
      {data.model_info && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">
                Model: <span className="font-medium">{data.model_info.name}</span>
              </p>
              <p className="text-xs text-gray-500">
                Version: {data.model_info.version} | 
                Updated: {new Date(data.model_info.updated_at).toLocaleDateString()}
              </p>
            </div>
            {data.latency_ms && (
              <p className="text-xs text-gray-500">
                Response time: {data.latency_ms}ms
              </p>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {data.recommendations.map((movie, index) => (
          <RecommendationCard
            key={movie.id}
            movie={movie}
            rank={index + 1}
            model={selectedModel}
          />
        ))}
      </div>

      {data.is_personalized === false && (
        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            These are general recommendations. Sign in for personalized suggestions.
          </p>
        </div>
      )}
    </div>
  );
}