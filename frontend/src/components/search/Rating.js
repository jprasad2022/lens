
'use client';

import { useState } from 'react';
import { movieService } from '@/services/movie.service';
import { useAuth } from '@/contexts/AuthContext';

export default function Rating({ movieId }) {
  const { user } = useAuth();
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);

  const handleRating = async (value) => {
    if (!user) {
      // Handle case where user is not logged in
      alert('Please log in to rate movies.');
      return;
    }
    setRating(value);
    try {
      await movieService.rateMovie(user.id, movieId, value);
      // Optionally, show a success message
    } catch (error) {
      // Handle error
      console.error('Failed to submit rating', error);
    }
  };

  return (
    <div className="flex items-center">
      {[...Array(5)].map((_, index) => {
        const starValue = index + 1;
        return (
          <button
            key={starValue}
            onClick={() => handleRating(starValue)}
            onMouseEnter={() => setHover(starValue)}
            onMouseLeave={() => setHover(0)}
            className={`text-2xl ${starValue <= (hover || rating) ? 'text-yellow-400' : 'text-gray-300'}`}>
            ★
          </button>
        );
      })}
    </div>
  );
}
