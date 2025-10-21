/**
 * Application Configuration
 * Centralized configuration management following the insurance-rag-app pattern
 */

export const appConfig = {
  // API Configuration
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    timeout: 30000,
    retryAttempts: 3,
    endpoints: {
      // Recommendation endpoints
      recommend: '/recommend',
      search: '/search',
      ratings: '/ratings',
      health: '/healthz',
      metrics: '/metrics',
      feedback: '/feedback',
      
      // Model management
      models: '/models',
      switchModel: '/models/switch',
      
      // User endpoints
      userHistory: '/users/{userId}/history',
      userPreferences: '/users/{userId}/preferences',
      
      // Admin endpoints
      retrain: '/admin/retrain',
      abTest: '/admin/ab-test',
      monitoring: '/admin/monitoring',
    },
  },
  
  // Feature Flags
  features: {
    abTesting: process.env.NEXT_PUBLIC_ENABLE_A_B_TESTING === 'true',
    monitoring: process.env.NEXT_PUBLIC_ENABLE_MONITORING === 'true',
    offlineEval: process.env.NEXT_PUBLIC_ENABLE_OFFLINE_EVAL === 'true',
    kafkaEnabled: process.env.NEXT_PUBLIC_KAFKA_ENABLED === 'true',
    authentication: !!process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  },
  
  // Model Configuration
  models: {
    default: process.env.NEXT_PUBLIC_DEFAULT_MODEL || 'popularity',
    available: ['popularity', 'collaborative', 'als'],
    recommendationCount: parseInt(process.env.NEXT_PUBLIC_RECOMMENDATION_COUNT || '20'),
  },
  
  // UI Configuration
  ui: {
    theme: 'light',
    animationsEnabled: true,
    debugMode: process.env.NODE_ENV === 'development',
    toastDuration: 4000,
  },
  
  // Monitoring Configuration
  monitoring: {
    metricsRefreshInterval: 5000, // 5 seconds
    loggingEnabled: process.env.NODE_ENV === 'development',
    errorReporting: process.env.NODE_ENV === 'production',
  },
};

// Validate configuration at startup
export function validateConfig() {
  const required = [
    'NEXT_PUBLIC_API_URL',
  ];
  
  const missing = required.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    console.warn(`Missing required environment variables: ${missing.join(', ')}`);
  }
  
  return missing.length === 0;
}