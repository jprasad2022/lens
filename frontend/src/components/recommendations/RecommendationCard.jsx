'use client';

import { useState } from 'react';
import { FiStar, FiPlay } from 'react-icons/fi';
import Image from 'next/image';

export default function RecommendationCard({ movie, rank, onViewDetails }) {
  const [imageError, setImageError] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const getRatingColor = (rating) => {
    if (rating >= 8) return 'text-green-500';
    if (rating >= 6) return 'text-yellow-500';
    return 'text-orange-500';
  };

  return (
    <div 
      className="card-movie group cursor-pointer"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Movie Poster */}
      <div className="relative aspect-[2/3] overflow-hidden bg-gray-200 dark:bg-gray-800 rounded-lg">
        {!imageError && (movie.poster_path || movie.poster_url) && movie.poster_path !== null ? (
          <Image
            src={`https://image.tmdb.org/t/p/w500${movie.poster_path || movie.poster_url}`}
            alt={movie.title}
            layout="fill"
            objectFit="cover"
            className="transition-transform duration-300 group-hover:scale-105"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-300 to-gray-400 dark:from-gray-700 dark:to-gray-800 flex items-center justify-center">
            <span className="text-xs text-gray-600 dark:text-gray-400 text-center px-2">{movie.title}</span>
          </div>
        )}
        
        {/* Simple Overlay on hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-4">
          <button 
            onClick={onViewDetails}
            className="w-full btn-primary btn-sm flex items-center justify-center gap-2"
          >
            <FiPlay className="w-4 h-4" />
            View Details
          </button>
        </div>

        {/* Simple Rank Badge */}
        {rank <= 3 && (
          <div className="absolute top-2 left-2 bg-primary-600 text-white text-xs font-bold px-2 py-1 rounded-full">
            #{rank}
          </div>
        )}
      </div>
      
      {/* Simplified Movie Details */}
      <div className="p-3">
        <h3 className="font-semibold text-gray-900 dark:text-white line-clamp-1 mb-1">
          {movie.title}
        </h3>
        
        <div className="flex items-center justify-between text-sm">
          {movie.year && (
            <span className="text-gray-500 dark:text-gray-400">
              {movie.year}
            </span>
          )}
          
          {movie.vote_average && (
            <div className="flex items-center gap-1">
              <FiStar className={`w-3.5 h-3.5 ${getRatingColor(movie.vote_average)}`} />
              <span className={`font-medium ${getRatingColor(movie.vote_average)}`}>
                {movie.vote_average.toFixed(1)}
              </span>
            </div>
          )}
        </div>
        
        {/* Only show genres on hover */}
        {isHovered && movie.genres && movie.genres.length > 0 && (
          <div className="mt-2 animate-slide-up">
            <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-1">
              {movie.genres.slice(0, 2).join(' • ')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}