
'use client';

import { useSearchParams } from 'next/navigation';
import { useMovie } from '@/hooks/useMovie';
import SearchResults from '@/components/search/SearchResults';
import { useEffect } from 'react';

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get('q');
  const { movies, loading, error, searchMovies } = useMovie();

  useEffect(() => {
    if (query) {
      searchMovies(query);
    }
  }, [query, searchMovies]);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-4">Search Results for &quot;{query}&quot;</h1>
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error.message}</p>}
      <SearchResults movies={movies} />
    </div>
  );
}
