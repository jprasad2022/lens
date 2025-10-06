'use client';

import MetricsDashboard from '@/components/monitoring/MetricsDashboard';
import ABTestResults from '@/components/monitoring/ABTestResults';
import SystemHealth from '@/components/monitoring/SystemHealth';
import { appConfig } from '@/config/app.config';

export default function MonitoringPage() {
  if (!appConfig.features.monitoring) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Monitoring Disabled
          </h1>
          <p className="text-gray-600">
            Monitoring features are not enabled in this environment.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          System Monitoring
        </h1>
        <p className="text-gray-600">
          Real-time metrics and performance monitoring for the recommendation system
        </p>
      </div>

      <div className="space-y-8">
        {/* System Health */}
        <SystemHealth />

        {/* Metrics Dashboard */}
        <MetricsDashboard />

        {/* A/B Test Results */}
        {appConfig.features.abTesting && <ABTestResults />}
      </div>
    </div>
  );
}