import { FiTrendingUp, FiTrendingDown, FiMinus } from 'react-icons/fi';

export default function MetricCard({ title, value, icon, trend = 'neutral' }) {
  const trendIcons = {
    up: <FiTrendingUp className="w-4 h-4 text-green-600" />,
    down: <FiTrendingDown className="w-4 h-4 text-red-600" />,
    neutral: <FiMinus className="w-4 h-4 text-gray-400" />,
  };

  const trendColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-600',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold mt-2 ${trendColors[trend]}`}>
            {value}
          </p>
        </div>
        <div className="flex-shrink-0">
          <div className="p-3 bg-gray-50 rounded-full">
            {icon}
          </div>
        </div>
      </div>
      <div className="flex items-center mt-4">
        {trendIcons[trend]}
        <span className={`ml-2 text-sm ${trendColors[trend]}`}>
          {trend === 'up' ? 'Good' : trend === 'down' ? 'Needs attention' : 'Stable'}
        </span>
      </div>
    </div>
  );
}