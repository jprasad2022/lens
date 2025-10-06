'use client';

import { useApp } from '@/contexts/AppContext';
import { FiActivity, FiClock, FiTrendingUp, FiUsers } from 'react-icons/fi';
import MetricCard from './MetricCard';
import LatencyChart from './LatencyChart';
import { appConfig } from '@/config/app.config';

export default function MetricsDashboard() {
  const { metrics, refreshMetrics } = useApp();

  if (!appConfig.features.monitoring) {
    return (
      <div className="text-center py-8 text-gray-600">
        Monitoring is disabled in this environment
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="flex justify-center items-center py-8">
        <div className="text-gray-600">Loading metrics...</div>
      </div>
    );
  }

  // Extract key metrics
  const totalRequests = metrics.recommend_requests_total
    ?.reduce((sum, m) => sum + m.value, 0) || 0;
  
  const successRequests = metrics.recommend_requests_total
    ?.filter(m => m.labels.status === '200')
    ?.reduce((sum, m) => sum + m.value, 0) || 0;
  
  const errorRequests = metrics.recommend_requests_total
    ?.filter(m => m.labels.status === '500')
    ?.reduce((sum, m) => sum + m.value, 0) || 0;
  
  const avgLatency = metrics.recommend_latency_seconds_sum?.[0]?.value /
    metrics.recommend_latency_seconds_count?.[0]?.value || 0;

  const successRate = totalRequests > 0 
    ? ((successRequests / totalRequests) * 100).toFixed(1) 
    : '0.0';

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">System Metrics</h2>
        <button
          onClick={refreshMetrics}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Requests"
          value={totalRequests.toLocaleString()}
          icon={<FiActivity className="w-6 h-6" />}
          trend={successRate > 95 ? 'up' : 'down'}
        />
        
        <MetricCard
          title="Success Rate"
          value={`${successRate}%`}
          icon={<FiTrendingUp className="w-6 h-6" />}
          trend={successRate > 99 ? 'up' : successRate > 95 ? 'neutral' : 'down'}
        />
        
        <MetricCard
          title="Avg Latency"
          value={`${(avgLatency * 1000).toFixed(0)}ms`}
          icon={<FiClock className="w-6 h-6" />}
          trend={avgLatency < 0.8 ? 'up' : avgLatency < 1.5 ? 'neutral' : 'down'}
        />
        
        <MetricCard
          title="Error Rate"
          value={`${((errorRequests / totalRequests) * 100 || 0).toFixed(2)}%`}
          icon={<FiUsers className="w-6 h-6" />}
          trend={errorRequests === 0 ? 'up' : 'down'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LatencyChart metrics={metrics} />
        
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium mb-4">Model Performance</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-sm text-gray-600">Personalized Responses</span>
              <span className="text-sm font-medium">
                {metrics.personalized_recommendations_ratio?.[0]?.value 
                  ? `${(metrics.personalized_recommendations_ratio[0].value * 100).toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-sm text-gray-600">Cache Hit Rate</span>
              <span className="text-sm font-medium">
                {metrics.cache_hit_ratio?.[0]?.value
                  ? `${(metrics.cache_hit_ratio[0].value * 100).toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-sm text-gray-600">Active Models</span>
              <span className="text-sm font-medium">
                {metrics.active_models_count?.[0]?.value || 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}