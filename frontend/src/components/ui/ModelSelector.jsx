'use client';

import { useApp } from '@/contexts/AppContext';
import toast from 'react-hot-toast';
import { FiCpu, FiZap, FiTarget, FiTrendingUp } from 'react-icons/fi';

export default function ModelSelector({ simplified = false }) {
  const { selectedModel, availableModels, setSelectedModel } = useApp();

  const modelInfo = {
    popularity: {
      icon: FiTrendingUp,
      name: 'Popularity',
      description: 'Most popular movies based on ratings',
      color: 'from-yellow-500 to-orange-600'
    },
    collaborative: {
      icon: FiTarget,
      name: 'Collaborative Filtering',
      description: 'Based on similar users\' preferences',
      color: 'from-blue-500 to-indigo-600'
    },
    als: {
      icon: FiCpu,
      name: 'ALS (Matrix Factorization)',
      description: 'Advanced matrix factorization algorithm',
      color: 'from-purple-500 to-pink-600'
    },
    content: {
      icon: FiZap,
      name: 'Content-Based',
      description: 'Based on movie features and genres',
      color: 'from-green-500 to-teal-600'
    },
    hybrid: {
      icon: FiCpu,
      name: 'Hybrid Model',
      description: 'Combines multiple algorithms',
      color: 'from-indigo-500 to-blue-600'
    },
    neural: {
      icon: FiTrendingUp,
      name: 'Neural Network',
      description: 'Deep learning recommendations',
      color: 'from-red-500 to-pink-600'
    }
  };

  const handleModelChange = async (modelValue) => {
    if (modelValue === selectedModel) return;
    
    try {
      await setSelectedModel(modelValue);
      const info = modelInfo[modelValue] || {};
      toast.success(`Switched to ${info.name || modelValue}`, {
        icon: '🎬',
        duration: 2000,
      });
    } catch (error) {
      toast.error('Failed to switch model');
      console.error('Model switch error:', error);
    }
  };

  // Simplified version shows only 3 main models
  const simplifiedModels = ['popularity', 'collaborative', 'als'];
  const modelsToShow = simplified 
    ? availableModels.filter(m => simplifiedModels.includes(m))
    : availableModels;

  if (simplified) {
    // Simplified dropdown version for advanced options
    return (
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Algorithm
        </label>
        <select
          value={selectedModel}
          onChange={(e) => handleModelChange(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
          {modelsToShow.map((model) => {
            const info = modelInfo[model] || {};
            return (
              <option key={model} value={model}>
                {info.name || model}
              </option>
            );
          })}
        </select>
      </div>
    );
  }

  // Full version with buttons
  return (
    <div>
      <label className="label mb-3">
        Recommendation Algorithm
      </label>
      <div className="space-y-2">
        {modelsToShow.map((model) => {
          const info = modelInfo[model] || {};
          const Icon = info.icon || FiCpu;
          const isSelected = model === selectedModel;
          
          return (
            <button
              key={model}
              onClick={() => handleModelChange(model)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-all duration-200 ${
                isSelected
                  ? 'border-primary-500 bg-gradient-to-r ' + info.color + ' text-white shadow-lg'
                  : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700 bg-white dark:bg-gray-800'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg ${
                  isSelected 
                    ? 'bg-white/20' 
                    : 'bg-gradient-to-br ' + info.color + ' text-white'
                }`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className={`font-medium ${
                    isSelected ? 'text-white' : 'text-gray-900 dark:text-white'
                  }`}>
                    {info.name || model}
                  </div>
                  <div className={`text-sm ${
                    isSelected ? 'text-white/80' : 'text-gray-600 dark:text-gray-400'
                  }`}>
                    {info.description || 'Standard recommendation model'}
                  </div>
                </div>
                {isSelected && (
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400">
        Each algorithm provides different recommendation strategies
      </p>
    </div>
  );
}