'use client';

import { useState } from 'react';
import { FiPlay, FiCheck } from 'react-icons/fi';
import { movieService } from '@/services/movie.service';
import { useAuth } from '@/contexts/AuthContext';

export default function WatchButton({ movieId, movieTitle }) {
  const { user } = useAuth();
  const [isWatched, setIsWatched] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleWatch = async () => {
    if (!user) {
      alert('Please log in to track watched movies.');
      return;
    }

    setIsLoading(true);
    try {
      // Track as fully watched (progress = 1.0)
      await movieService.trackWatch(user.uid || user.id, movieId, 1.0);
      setIsWatched(true);
      
      // Show success message (optional)
      console.log(`Marked "${movieTitle}" as watched`);
    } catch (error) {
      console.error('Failed to track watch event:', error);
      alert('Failed to mark as watched. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <button
      onClick={handleWatch}
      disabled={isLoading || isWatched}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
        ${isWatched 
          ? 'bg-green-600 text-white cursor-default' 
          : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'
        }
        ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      {isWatched ? (
        <>
          <FiCheck className="w-5 h-5" />
          <span>Watched</span>
        </>
      ) : (
        <>
          <FiPlay className="w-5 h-5" />
          <span>Mark as Watched</span>
        </>
      )}
    </button>
  );
}