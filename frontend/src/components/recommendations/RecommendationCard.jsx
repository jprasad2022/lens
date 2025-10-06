'use client';

import { useState } from 'react';
import { FiThumbsUp, FiThumbsDown, FiStar } from 'react-icons/fi';
import { apiService } from '@/services/api.service';
import { useAuth } from '@/contexts/AuthContext';

import Image from 'next/image';

export default function RecommendationCard({ movie, rank, model }) {
  const { user } = useAuth();
  const [feedback, setFeedback] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedback = async (type) => {
    if (isSubmitting || feedback) return;

    setIsSubmitting(true);
    try {
      await apiService.submitFeedback({
        user_id: user?.uid || 'anonymous',
        movie_id: movie.id,
        feedback_type: type,
        model_version: model,
        rank_position: rank,
        timestamp: Date.now(),
      });
      setFeedback(type);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200">
      {movie.poster_path && (
        <div className="relative h-64 w-full">
          <Image
            src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
            alt={movie.title}
            layout="fill"
            objectFit="cover"
            className="rounded-t-lg"
          />
        </div>
      )}
      
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <h3 className="text-lg font-semibold line-clamp-2">{movie.title}</h3>
          <span className="text-sm text-gray-500">#{rank}</span>
        </div>
        
        {movie.release_date && (
          <p className="text-sm text-gray-600 mb-2">
            {new Date(movie.release_date).getFullYear()}
          </p>
        )}
        
        {movie.vote_average && (
          <div className="flex items-center mb-2">
            <FiStar className="w-4 h-4 text-yellow-400 mr-1" />
            <span className="text-sm">{movie.vote_average.toFixed(1)}/10</span>
            {movie.vote_count && (
              <span className="text-xs text-gray-500 ml-2">({movie.vote_count} ratings)</span>
            )}
          </div>
        )}
        
        {movie.genres && movie.genres.length > 0 && (
          <div className="mb-3">
            <div className="flex flex-wrap gap-1">
              {movie.genres.slice(0, 3).map((genre, index) => (
                <span 
                  key={index} 
                  className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
                >
                  {genre}
                </span>
              ))}
            </div>
          </div>
        )}
        
        <p className="text-sm text-gray-700 line-clamp-3 mb-4">
          {movie.overview || 'No overview available.'}
        </p>
        
        <div className="flex items-center justify-between pt-3 border-t">
          <span className="text-xs text-gray-500">Was this helpful?</span>
          <div className="flex gap-2">
            <button
              onClick={() => handleFeedback('positive')}
              disabled={isSubmitting || feedback}
              className={`p-2 rounded transition-colors ${
                feedback === 'positive'
                  ? 'bg-green-100 text-green-600'
                  : 'hover:bg-gray-100 text-gray-600'
              } disabled:opacity-50`}
            >
              <FiThumbsUp className="w-4 h-4" />
            </button>
            <button
              onClick={() => handleFeedback('negative')}
              disabled={isSubmitting || feedback}
              className={`p-2 rounded transition-colors ${
                feedback === 'negative'
                  ? 'bg-red-100 text-red-600'
                  : 'hover:bg-gray-100 text-gray-600'
              } disabled:opacity-50`}
            >
              <FiThumbsDown className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}