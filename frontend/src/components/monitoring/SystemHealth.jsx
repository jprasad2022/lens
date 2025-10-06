import React from 'react';

export default function SystemHealth() {
  const services = [
    { name: 'API', status: 'healthy', latency: '45ms' },
    { name: 'Redis Cache', status: 'healthy', latency: '2ms' },
    { name: 'Kafka', status: 'healthy', latency: '12ms' },
    { name: 'Model Service', status: 'healthy', latency: '150ms' },
  ];

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">System Health</h2>
      <div className="space-y-3">
        {services.map((service) => (
          <div key={service.name} className="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span className="font-medium">{service.name}</span>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">{service.latency}</span>
              <span className={`px-2 py-1 rounded text-xs ${
                service.status === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {service.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}