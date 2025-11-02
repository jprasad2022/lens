'use client';

import { useState } from 'react';
import { useApp } from '@/contexts/AppContext';
import RecommendationGrid from '@/components/recommendations/RecommendationGrid';
import ModelSelector from '@/components/ui/ModelSelector';
import UserInput from '@/components/ui/UserInput';
import { appConfig } from '@/config/app.config';
import { FiFilm, FiUser, FiChevronRight } from 'react-icons/fi';

export default function HomePage() {
  const { health } = useApp();
  const [userId, setUserId] = useState('');
  const [recommendationCount, setRecommendationCount] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const demoUsers = [
    { id: '1', name: 'Alex', description: 'Likes action movies' },
    { id: '100', name: 'Sam', description: 'Enjoys sci-fi films' },
    { id: '500', name: 'Morgan', description: 'Prefers dramas' },
    { id: '1000', name: 'Jordan', description: 'Comedy fan' },
  ];

  const quickSelectUser = (selectedUserId) => {
    setUserId(selectedUserId);
    // Scroll to recommendations after selection
    setTimeout(() => {
      document.getElementById('recommendations')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Simplified Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-600/10 to-secondary-600/10"></div>
        <div className="container mx-auto px-4 py-12 relative">
          <div className="max-w-2xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              Find Your Next Movie
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
              Get personalized recommendations in seconds
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="container mx-auto px-4 pb-12">
        {/* Health Status Banner */}
        {health && health.status !== 'ok' && (
          <div className="mb-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
            <p className="text-amber-800 dark:text-amber-400 text-sm">
              System Status: {health.status} - {health.message}
            </p>
          </div>
        )}

        {/* Quick Start Section */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 text-center">
              Who are you watching as today?
            </h2>
            
            {/* Demo User Quick Select */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {demoUsers.map((user) => (
                <button
                  key={user.id}
                  onClick={() => quickSelectUser(user.id)}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 hover:scale-105 ${
                    userId === user.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 shadow-lg'
                      : 'border-gray-200 dark:border-gray-600 hover:border-primary-300 dark:hover:border-primary-600 bg-gray-50 dark:bg-gray-700/50'
                  }`}
                >
                  <FiUser className="w-8 h-8 mx-auto mb-3 text-primary-600 dark:text-primary-400" />
                  <div className="font-medium text-gray-900 dark:text-white">{user.name}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">{user.description}</div>
                </button>
              ))}
            </div>

            {/* Custom User Option */}
            <div className="text-center">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="inline-flex items-center text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-colors"
              >
                <span>Advanced options</span>
                <FiChevronRight className={`ml-1 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
              </button>
            </div>

            {/* Advanced Options (Hidden by default) */}
            {showAdvanced && (
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700 animate-slide-up">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <UserInput
                      value={userId}
                      onChange={setUserId}
                      placeholder="1-6040"
                      label="Custom User ID"
                    />
                  </div>
                  
                  <ModelSelector simplified />
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Number of Movies
                    </label>
                    <select
                      value={recommendationCount}
                      onChange={(e) => setRecommendationCount(Number(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value={10}>10 movies</option>
                      <option value={20}>20 movies</option>
                      <option value={30}>30 movies</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Get Recommendations Button (for custom user) */}
            {userId && !demoUsers.find(u => u.id === userId) && (
              <div className="mt-6 text-center">
                <button
                  onClick={() => document.getElementById('recommendations')?.scrollIntoView({ behavior: 'smooth' })}
                  className="inline-flex items-center px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
                >
                  <FiFilm className="mr-2" />
                  Get Recommendations
                </button>
              </div>
            )}
          </div>

          {/* Help Text */}
          {!userId && (
            <p className="text-center text-gray-500 dark:text-gray-400 mt-4 text-sm">
              Select a demo user above to get started instantly
            </p>
          )}
        </div>

        {/* Recommendations Section */}
        {userId && (
          <div id="recommendations" className="max-w-7xl mx-auto animate-slide-up">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
              Your Recommendations
            </h2>
            <RecommendationGrid 
              userId={userId} 
              count={recommendationCount} 
            />
          </div>
        )}

        {/* Debug Mode - Much smaller */}
        {appConfig.ui.debugMode && (
          <details className="mt-12 max-w-4xl mx-auto">
            <summary className="cursor-pointer text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
              Debug Mode
            </summary>
            <div className="mt-2 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                {Object.entries(appConfig.features).map(([key, value]) => (
                  <div key={key} className="flex items-center">
                    <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                      value ? 'bg-green-500' : 'bg-gray-400'
                    }`}></div>
                    <span className="text-gray-600 dark:text-gray-400">{key}</span>
                  </div>
                ))}
              </div>
            </div>
          </details>
        )}
      </div>
    </div>
  );
}