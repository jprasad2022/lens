'use client';

import { useState } from 'react';
import { useApp } from '@/contexts/AppContext';
import RecommendationGrid from '@/components/recommendations/RecommendationGrid';
import ModelSelector from '@/components/ui/ModelSelector';
import UserInput from '@/components/ui/UserInput';
import { appConfig } from '@/config/app.config';
import { FiFilm, FiStar, FiTrendingUp, FiZap } from 'react-icons/fi';

export default function HomePage() {
  const { health } = useApp();
  const [userId, setUserId] = useState('');
  const [recommendationCount, setRecommendationCount] = useState(appConfig.models.recommendationCount);

  const demoUsers = [
    { id: '1', name: 'Alex (Action Lover)', genres: 'Action, Thriller' },
    { id: '100', name: 'Sam (Sci-Fi Fan)', genres: 'Sci-Fi, Adventure' },
    { id: '500', name: 'Morgan (Drama Buff)', genres: 'Drama, Romance' },
    { id: '1000', name: 'Jordan (Comedy Enthusiast)', genres: 'Comedy, Family' },
  ];

  return (
    <>
      {/* Hero Section */}
      <section className="hero-gradient text-white">
        <div className="container mx-auto px-4 py-20">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-white/20 backdrop-blur-sm rounded-2xl mb-6 animate-float">
              <FiFilm className="w-10 h-10" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 animate-slide-up">
              Discover Your Next
              <span className="block text-yellow-300 mt-2">Favorite Movie</span>
            </h1>
            <p className="text-xl md:text-2xl text-white/90 mb-8 animate-slide-up animation-delay-200">
              Personalized recommendations powered by advanced machine learning
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 animate-slide-up animation-delay-400">
              <div className="glass rounded-xl p-6 text-center">
                <FiStar className="w-8 h-8 mx-auto mb-3 text-yellow-300" />
                <h3 className="font-semibold mb-1">100K+ Movies</h3>
                <p className="text-sm text-white/80">Vast collection to explore</p>
              </div>
              <div className="glass rounded-xl p-6 text-center">
                <FiTrendingUp className="w-8 h-8 mx-auto mb-3 text-yellow-300" />
                <h3 className="font-semibold mb-1">Smart AI</h3>
                <p className="text-sm text-white/80">Multiple ML models</p>
              </div>
              <div className="glass rounded-xl p-6 text-center">
                <FiZap className="w-8 h-8 mx-auto mb-3 text-yellow-300" />
                <h3 className="font-semibold mb-1">Instant Results</h3>
                <p className="text-sm text-white/80">Real-time recommendations</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Health Status Banner */}
        {health && health.status !== 'ok' && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl animate-slide-up">
            <p className="text-red-800 dark:text-red-400 font-medium">
              System Status: {health.status} - {health.message}
            </p>
          </div>
        )}

        {/* Controls Section */}
        <div className="card max-w-6xl mx-auto mb-12 animate-slide-up">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
            <div className="w-2 h-8 bg-gradient-to-b from-primary-600 to-secondary-600 rounded-full mr-3"></div>
            Get Your Recommendations
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* User Selection */}
            <div className="lg:col-span-1">
              <label className="label">
                Select User Profile
              </label>
              <div className="space-y-2 mb-4">
                {demoUsers.map((user) => (
                  <button
                    key={user.id}
                    onClick={() => setUserId(user.id)}
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all duration-200 ${
                      userId === user.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700'
                    }`}
                  >
                    <div className="font-medium text-gray-900 dark:text-white">{user.name}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">{user.genres}</div>
                  </button>
                ))}
              </div>
              
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300 dark:border-gray-700"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-white dark:bg-gray-800 text-gray-500">or</span>
                </div>
              </div>
              
              <UserInput
                value={userId}
                onChange={setUserId}
                placeholder="Enter custom user ID (1-6040)"
                label="Custom User ID"
                className="mt-4"
              />
            </div>
            
            {/* Model and Count Selection */}
            <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
              <ModelSelector />
              
              <div>
                <label className="label">
                  Number of Recommendations
                </label>
                <select
                  value={recommendationCount}
                  onChange={(e) => setRecommendationCount(Number(e.target.value))}
                  className="select"
                >
                  {[10, 20, 30, 50].map(count => (
                    <option key={count} value={count}>
                      {count} movies
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  Choose how many movies to display
                </p>
              </div>
            </div>
          </div>

          {!userId && (
            <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <FiFilm className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
                    Get Started
                  </h3>
                  <p className="mt-1 text-sm text-blue-700 dark:text-blue-400">
                    Select a user profile above or enter a custom user ID to see personalized recommendations
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recommendations Section */}
        {userId && (
          <div className="animate-slide-up">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
              Recommended for You
            </h2>
            <RecommendationGrid 
              userId={userId} 
              count={recommendationCount} 
            />
          </div>
        )}

        {/* Feature Flags Info */}
        {appConfig.ui.debugMode && (
          <div className="mt-12 p-6 bg-gray-100 dark:bg-gray-800 rounded-xl">
            <h3 className="font-semibold mb-3 text-gray-900 dark:text-white">Debug Mode</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {Object.entries(appConfig.features).map(([key, value]) => (
                <div key={key} className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    value ? 'bg-green-500' : 'bg-gray-400'
                  }`}></div>
                  <span className="text-gray-600 dark:text-gray-400">{key}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}