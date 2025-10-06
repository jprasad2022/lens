'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import { apiService } from '@/services/api.service';
import { appConfig } from '@/config/app.config';

const AppContext = createContext({
  selectedModel: appConfig.models.default,
  availableModels: [],
  metrics: null,
  health: null,
  setSelectedModel: () => {},
  refreshMetrics: () => {},
  checkHealth: () => {},
});

export function AppProvider({ children }) {
  const [selectedModel, setSelectedModel] = useState(appConfig.models.default);
  const [availableModels, setAvailableModels] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);

  // Load available models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const models = await apiService.getAvailableModels();
        // Ensure models is an array
        setAvailableModels(Array.isArray(models) ? models : []);
      } catch (error) {
        console.error('Failed to load models:', error);
        // Fallback to config models
        setAvailableModels(appConfig.models.available);
      }
    };

    loadModels();
  }, []);

  // Monitor health status
  useEffect(() => {
    const checkHealthStatus = async () => {
      try {
        const healthData = await apiService.checkHealth();
        setHealth(healthData);
      } catch (error) {
        setHealth({ status: 'error', message: error.message });
      }
    };

    checkHealthStatus();
    const interval = setInterval(checkHealthStatus, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, []);

  // Monitor metrics if enabled
  useEffect(() => {
    if (!appConfig.features.monitoring) return;

    const fetchMetrics = async () => {
      try {
        const metricsData = await apiService.getMetrics();
        setMetrics(parsePrometheusMetrics(metricsData));
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, appConfig.monitoring.metricsRefreshInterval);

    return () => clearInterval(interval);
  }, []);

  const handleSetSelectedModel = async (modelName) => {
    try {
      await apiService.switchModel(modelName);
      setSelectedModel(modelName);
    } catch (error) {
      console.error('Failed to switch model:', error);
      throw error;
    }
  };

  const refreshMetrics = async () => {
    if (!appConfig.features.monitoring) return;
    
    try {
      const metricsData = await apiService.getMetrics();
      setMetrics(parsePrometheusMetrics(metricsData));
    } catch (error) {
      console.error('Failed to refresh metrics:', error);
    }
  };

  const checkHealth = async () => {
    try {
      const healthData = await apiService.checkHealth();
      setHealth(healthData);
      return healthData;
    } catch (error) {
      const errorHealth = { status: 'error', message: error.message };
      setHealth(errorHealth);
      return errorHealth;
    }
  };

  const value = {
    selectedModel,
    availableModels,
    metrics,
    health,
    setSelectedModel: handleSetSelectedModel,
    refreshMetrics,
    checkHealth,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

// Helper function to parse Prometheus metrics
function parsePrometheusMetrics(metricsText) {
  const lines = metricsText.split('\n');
  const metrics = {};

  lines.forEach(line => {
    if (line.startsWith('#') || !line.trim()) return;

    const match = line.match(/^(\w+)(?:\{([^}]*)\})?\s+(.+)$/);
    if (match) {
      const [, name, labels, value] = match;
      if (!metrics[name]) {
        metrics[name] = [];
      }
      metrics[name].push({
        labels: labels ? Object.fromEntries(
          labels.split(',').map(kv => {
            const [k, v] = kv.split('=');
            return [k, v.replace(/"/g, '')];
          })
        ) : {},
        value: parseFloat(value),
      });
    }
  });

  return metrics;
}