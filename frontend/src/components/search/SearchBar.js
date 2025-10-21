'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { FiSearch, FiX } from 'react-icons/fi';

export default function SearchBar() {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const router = useRouter();
  const inputRef = useRef(null);

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
      setQuery('');
      inputRef.current?.blur();
    }
  };

  const handleClear = () => {
    setQuery('');
    inputRef.current?.focus();
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === '/' && e.ctrlKey) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <form onSubmit={handleSearch} className="w-full">
      <div className={`relative transition-all duration-200 ${
        isFocused ? 'scale-105' : ''
      }`}>
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <FiSearch className={`h-5 w-5 transition-colors duration-200 ${
            isFocused ? 'text-primary-500' : 'text-gray-400'
          }`} />
        </div>
        <input
          ref={inputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder="Search movies..."
          className="input-search w-full"
        />
        {query && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute inset-y-0 right-0 pr-10 flex items-center group"
          >
            <FiX className="h-5 w-5 text-gray-400 group-hover:text-gray-600 transition-colors" />
          </button>
        )}
        <button
          type="submit"
          className={`absolute inset-y-0 right-0 px-3 flex items-center group ${
            query ? 'text-primary-600 hover:text-primary-700' : 'text-gray-400'
          } transition-colors`}
        >
          <div className="p-1 rounded-full group-hover:bg-primary-50 dark:group-hover:bg-primary-900/30 transition-colors">
            <FiSearch className="h-5 w-5" />
          </div>
        </button>
      </div>
      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 text-center">
        <span className="hidden md:inline">Press </span>
        <kbd className="px-2 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 dark:bg-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600 rounded">
          Ctrl+/
        </kbd>
        <span className="hidden md:inline"> to search</span>
      </div>
    </form>
  );
}