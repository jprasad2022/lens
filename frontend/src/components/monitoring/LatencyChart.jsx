'use client';

import { useEffect, useRef } from 'react';
import { Chart as ChartJS, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
);

export default function LatencyChart({ metrics }) {
  const chartRef = useRef(null);

  // Extract latency histogram data
  const latencyBuckets = metrics.recommend_latency_seconds_bucket || [];
  
  // Process histogram buckets
  const bucketData = latencyBuckets
    .filter(m => m.labels.le !== '+Inf')
    .map(m => ({
      le: parseFloat(m.labels.le) * 1000, // Convert to ms
      count: m.value,
    }))
    .sort((a, b) => a.le - b.le);

  // Calculate percentiles
  const totalCount = metrics.recommend_latency_seconds_count?.[0]?.value || 1;
  
  const data = {
    labels: bucketData.map(b => `${b.le}ms`),
    datasets: [
      {
        label: 'Cumulative Requests',
        data: bucketData.map(b => (b.count / totalCount) * 100),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Response Time Distribution',
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.parsed.y.toFixed(1)}% of requests`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (value) => `${value}%`,
        },
      },
      x: {
        title: {
          display: true,
          text: 'Response Time',
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-medium mb-4">Latency Distribution</h3>
      <div className="h-64">
        {bucketData.length > 0 ? (
          <Line ref={chartRef} data={data} options={options} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            No latency data available
          </div>
        )}
      </div>
      
      {/* Key percentiles */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
        <div className="text-center">
          <p className="text-gray-600">P50 (Median)</p>
          <p className="font-medium">
            {calculatePercentile(bucketData, totalCount, 0.5)}ms
          </p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">P95</p>
          <p className="font-medium">
            {calculatePercentile(bucketData, totalCount, 0.95)}ms
          </p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">P99</p>
          <p className="font-medium">
            {calculatePercentile(bucketData, totalCount, 0.99)}ms
          </p>
        </div>
      </div>
    </div>
  );
}

function calculatePercentile(bucketData, totalCount, percentile) {
  const target = totalCount * percentile;
  let cumulative = 0;
  
  for (const bucket of bucketData) {
    cumulative = bucket.count;
    if (cumulative >= target) {
      return bucket.le;
    }
  }
  
  return bucketData[bucketData.length - 1]?.le || 0;
}