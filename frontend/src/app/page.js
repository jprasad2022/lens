'use client';

import { useState } from 'react';
// import { useAuth } from '@/contexts/AuthContext';
import { useApp } from '@/contexts/AppContext';
import RecommendationGrid from '@/components/recommendations/RecommendationGrid';
import ModelSelector from '@/components/ui/ModelSelector';
import UserInput from '@/components/ui/UserInput';
import { appConfig } from '@/config/app.config';

export default function HomePage() {
  // const { user } = useAuth();
  const { health } = useApp();
  const [userId, setUserId] = useState('');
  const [recommendationCount, setRecommendationCount] = useState(appConfig.models.recommendationCount);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Health Status Banner */}
      {health && health.status !== 'ok' && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">
            System Status: {health.status} - {health.message}
          </p>
        </div>
      )}

      {/* Header Section */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Movie Recommendations
        </h1>
        <p className="text-lg text-gray-600">
          Get personalized movie recommendations powered by machine learning
        </p>
      </div>

      {/* Controls Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <UserInput
            value={userId}
            onChange={setUserId}
            placeholder="Enter user ID"
            label="User ID"
          />
          
          <ModelSelector />
          
          <div>
            <label className="label">
              Recommendations
            </label>
            <select
              value={recommendationCount}
              onChange={(e) => setRecommendationCount(Number(e.target.value))}
              className="input"
            >
              {[10, 20, 30, 50].map(count => (
                <option key={count} value={count}>
                  {count} movies
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Recommendations Section */}
      {userId ? (
        <RecommendationGrid 
          userId={userId} 
          count={recommendationCount} 
        />
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-600 mb-4">
            Enter a user ID to get recommendations
          </p>
          <p className="text-sm text-gray-500">
            Try user IDs: 1, 100, 500, or any number between 1-6040
          </p>
        </div>
      )}

      {/* Feature Flags Info */}
      {appConfig.ui.debugMode && (
        <div className="mt-8 p-4 bg-gray-100 rounded-lg">
          <h3 className="font-semibold mb-2">Active Features:</h3>
          <div className="text-sm text-gray-600 space-y-1">
            {Object.entries(appConfig.features).map(([key, value]) => (
              <div key={key}>
                {key}: {value ? '✓' : '✗'}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}