'use client';

import { FiX, FiStar, FiCalendar, FiFilm } from 'react-icons/fi';
import Image from 'next/image';

export default function MovieDetailsModal({ movie, isOpen, onClose }) {
  if (!isOpen || !movie) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative bg-white dark:bg-gray-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl animate-slide-up">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-black/20 hover:bg-black/30 rounded-full text-white transition-colors"
        >
          <FiX className="w-5 h-5" />
        </button>
        
        <div className="flex flex-col md:flex-row">
          {/* Poster */}
          <div className="md:w-1/3 bg-gray-100 dark:bg-gray-900">
            {(movie.poster_path || movie.poster_url) && movie.poster_path !== null ? (
              <Image
                src={`https://image.tmdb.org/t/p/w500${movie.poster_path || movie.poster_url}`}
                alt={movie.title}
                width={500}
                height={750}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full min-h-[400px] flex items-center justify-center">
                <FiFilm className="w-20 h-20 text-gray-400" />
              </div>
            )}
          </div>
          
          {/* Details */}
          <div className="flex-1 p-6 md:p-8 overflow-y-auto max-h-[90vh]">
            <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-4">
              {movie.title}
            </h2>
            
            <div className="flex flex-wrap gap-4 mb-6">
              {movie.year && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <FiCalendar className="w-4 h-4" />
                  <span>{movie.year}</span>
                </div>
              )}
              
              {movie.vote_average && (
                <div className="flex items-center gap-2">
                  <FiStar className="w-4 h-4 text-yellow-500" />
                  <span className="font-medium text-gray-900 dark:text-white">
                    {movie.vote_average.toFixed(1)}
                  </span>
                  {movie.vote_count && (
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      ({movie.vote_count.toLocaleString()} votes)
                    </span>
                  )}
                </div>
              )}
            </div>
            
            {movie.genres && movie.genres.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Genres</h3>
                <div className="flex flex-wrap gap-2">
                  {movie.genres.map((genre, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full text-sm"
                    >
                      {genre}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {(movie.overview || movie.genres) && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Overview</h3>
                <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                  {movie.overview || `A ${movie.genres?.join(' and ')?.toLowerCase() || 'film'} from ${movie.year || 'unknown year'}.`}
                </p>
              </div>
            )}
            
            {/* Additional Info */}
            <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Movie ID: {movie.id}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}