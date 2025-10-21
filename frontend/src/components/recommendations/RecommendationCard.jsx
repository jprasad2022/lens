'use client';

import { useState } from 'react';
import { FiThumbsUp, FiThumbsDown, FiStar, FiPlay, FiBookmark, FiInfo, FiFilm } from 'react-icons/fi';
import { apiService } from '@/services/api.service';
import { useAuth } from '@/contexts/AuthContext';
import Image from 'next/image';

export default function RecommendationCard({ movie, rank, model }) {
  const { user } = useAuth();
  const [feedback, setFeedback] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);

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

  const getRatingColor = (rating) => {
    if (rating >= 8) return 'bg-green-500';
    if (rating >= 6) return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  const placeholderImage = `https://via.placeholder.com/500x750/1a1a1a/666?text=${encodeURIComponent(movie.title)}`;

  return (
    <div className="card-movie group">
      {/* Movie Poster */}
      <div className="relative aspect-[2/3] overflow-hidden bg-gray-900">
        {!imageError && movie.poster_path ? (
          <Image
            src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
            alt={movie.title}
            layout="fill"
            objectFit="cover"
            className="group-hover:scale-110 transition-transform duration-500"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
            <FiFilm className="w-16 h-16 text-gray-600" />
          </div>
        )}
        
        {/* Overlay on hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
          <div className="space-y-3 transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
            <button className="w-full btn-primary text-sm flex items-center justify-center gap-2">
              <FiPlay className="w-4 h-4" />
              Watch Trailer
            </button>
            <button className="w-full btn-ghost text-sm flex items-center justify-center gap-2">
              <FiInfo className="w-4 h-4" />
              More Info
            </button>
          </div>
        </div>

        {/* Rank Badge */}
        <div className="absolute top-3 left-3 bg-gradient-to-br from-primary-600 to-secondary-600 text-white text-sm font-bold px-3 py-1.5 rounded-full shadow-lg">
          #{rank}
        </div>

        {/* Bookmark Button */}
        <button
          onClick={() => setIsBookmarked(!isBookmarked)}
          className={`absolute top-3 right-3 p-2 rounded-full backdrop-blur-sm transition-all duration-200 ${
            isBookmarked 
              ? 'bg-yellow-500 text-white shadow-glow' 
              : 'bg-black/30 text-white hover:bg-black/50'
          }`}
        >
          <FiBookmark className={`w-4 h-4 ${isBookmarked ? 'fill-current' : ''}`} />
        </button>
      </div>
      
      {/* Movie Details */}
      <div className="p-5">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white line-clamp-1 mb-2 group-hover:text-primary-600 transition-colors">
          {movie.title}
        </h3>
        
        <div className="flex items-center justify-between mb-3">
          {movie.release_date && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {new Date(movie.release_date).getFullYear()}
            </p>
          )}
          
          {movie.vote_average && (
            <div className="flex items-center gap-1.5">
              <div className={`w-8 h-8 ${getRatingColor(movie.vote_average)} rounded-full flex items-center justify-center`}>
                <span className="text-xs font-bold text-white">{movie.vote_average.toFixed(1)}</span>
              </div>
              {movie.vote_count && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {movie.vote_count.toLocaleString()} votes
                </span>
              )}
            </div>
          )}
        </div>
        
        {movie.genres && movie.genres.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {movie.genres.slice(0, 3).map((genre, index) => (
              <span 
                key={index} 
                className="genre-tag"
              >
                {genre}
              </span>
            ))}
          </div>
        )}
        
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2 mb-4">
          {movie.overview || `A ${movie.genres?.[0]?.toLowerCase() || 'film'} from ${new Date(movie.release_date).getFullYear() || 'unknown year'}, rated ${movie.vote_average?.toFixed(1) || 'N/A'}/10 by ${movie.vote_count?.toLocaleString() || '0'} users.`}
        </p>
        
        {/* Feedback Section */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
              Was this helpful?
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => handleFeedback('positive')}
                disabled={isSubmitting || feedback}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  feedback === 'positive'
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 scale-110'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400 hover:scale-110'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
                title="This was helpful"
              >
                <FiThumbsUp className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleFeedback('negative')}
                disabled={isSubmitting || feedback}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  feedback === 'negative'
                    ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 scale-110'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400 hover:scale-110'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
                title="This wasn't helpful"
              >
                <FiThumbsDown className="w-4 h-4" />
              </button>
            </div>
          </div>
          {feedback && (
            <p className="text-xs text-center mt-2 text-gray-500 dark:text-gray-400 animate-slide-up">
              Thanks for your feedback!
            </p>
          )}
        </div>
      </div>
    </div>
  );
}